"""Selection policy: pre-filter by safety, rank by predicted delta, return top-N.

Ranking is a precedence tuple rather than one blended number, for the reason
:mod:`gitm.playbook.match` gives: terms answering different questions should not
be collapsed into a scalar where one can quietly outvote another. Gate first,
then evidence quality, then magnitude, then a deterministic tie-break.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.history import History, record_for
from gitm.optimizer.preconditions import GateContext, applicable
from gitm.optimizer.replay import predict_delta
from gitm.tracer.schema import Trace


@dataclass
class RankedCandidate:
    spec: InterventionSpec
    predicted_delta: float
    rejected_reason: str | None = None
    #: Where ``predicted_delta``'s effect estimate came from: ``"prior"`` for
    #: the spec's hand-authored ``expected_delta_mean``, ``"measured"`` for a
    #: delta this lever recorded on this GPU. Carried for the same reason
    #: ``rejected_reason`` is: a number is worth less without what produced it.
    delta_source: str = "prior"
    #: Ranked below every undemoted candidate, but never removed. A lever whose
    #: record both won and lost has not come out neutral — it behaved differently
    #: under conditions the record does not capture, so it is the weaker bet
    #: while that holds. The demotion lifts by itself once the record stops
    #: disagreeing: it describes the evidence, not the lever.
    demoted: bool = False


@dataclass
class Policy:
    """Greedy by predicted delta with safety pre-filter."""

    require_qualification_commit: bool = False
    skip_high_risk: bool = False
    #: Score a lever from what it measured before, where there is a record for
    #: this GPU. Off by default because it changes which experiments run, and
    #: that should be a decision someone made rather than one that arrived
    #: with an upgrade.
    use_history: bool = False


def select_interventions(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    top_n: int = 5,
    *,
    ctx: GateContext | None = None,
    history: History | None = None,
    gpu_sku: str | None = None,
) -> list[RankedCandidate]:
    """Rank the library for this trace, rejected candidates last.

    ``history`` is passed in rather than read from disk here, so ranking stays a
    pure function of what it is given and a caller can rank against a record it
    has already filtered. It is consulted only when ``policy.use_history`` is on
    *and* ``gpu_sku`` names the box: a result measured on an H100 says nothing
    about an MI355X, and scoring one from the other is the mistake the record's
    GPU key exists to prevent. No SKU therefore means no substitution, not a
    guess at which box the record came from.
    """
    use_history = policy.use_history and history is not None and gpu_sku is not None
    candidates: list[RankedCandidate] = []

    for spec in library:
        reason: str | None = None
        if ctx is not None:
            ok, why = applicable(spec, ctx)
            if not ok:
                reason = f"not_applicable: {why}"
        if reason is None and policy.skip_high_risk and spec.safety.tier == "high_risk":
            reason = "policy.skip_high_risk"
        elif reason is None and (spec.safety.requires_qualification_commit and not policy.require_qualification_commit):
            reason = "safety.requires_qualification_commit"
        record = (
            record_for(history, spec.name, gpu_sku=gpu_sku)
            if use_history and reason is None
            else None
        )
        # A record with no usable delta is still a record: it says the lever was
        # tried and how it fared, but carries no number to rank on. The prior
        # stands in that case and only the demotion applies.
        measured = record.mean_delta if record is not None else None
        delta = predict_delta(trace, spec, delta_mean=measured) if reason is None else 0.0
        candidates.append(RankedCandidate(
            spec=spec,
            predicted_delta=delta,
            rejected_reason=reason,
            delta_source="measured" if measured is not None else "prior",
            demoted=bool(record is not None and record.conflicted),
        ))

    # Four terms, in this order and for these reasons:
    #
    # 1. Rejected. The gate's answer is categorical and comes first.
    # 2. Not worth a run. A lever whose estimate is zero or negative is not a
    #    candidate whatever the evidence behind it says, so it sorts below every
    #    lever that might help. This sits above the demotion because a lever
    #    measured at -9% every time is a worse bet than one that is merely
    #    inconsistent, and ranking the known loser higher would spend the run on
    #    a result already in hand.
    # 3. Demoted. Among levers that might help, prefer the one whose record does
    #    not disagree with itself.
    # 4. Magnitude, then name for a deterministic order.
    candidates.sort(
        key=lambda c: (
            c.rejected_reason is not None,
            c.predicted_delta <= 0.0,
            c.demoted,
            -c.predicted_delta,
            c.spec.name,
        )
    )
    return candidates[:top_n]
