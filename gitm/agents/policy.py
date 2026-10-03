"""Selection policy: pre-filter by safety, rank by predicted delta, return top-N.

Ranking is a precedence tuple rather than one blended number, for the reason
:mod:`gitm.playbook.match` gives: terms answering different questions should not
be collapsed into a scalar where one can quietly outvote another. Gate first,
then evidence quality, then magnitude, then a deterministic tie-break.

``recoverable`` adds a second pre-filter beside the safety one, and it is the
only place the trace decides *whether* a lever is a candidate rather than just
how it scores. The two are different questions: ``predict_delta`` asks how much
of the step a lever touches, and an op-scoped lever aimed at a region already
running at its predicted floor scores well on that and can recover nothing. It
is a filter and not a term in the sort on purpose — ``recoverable`` is a
duration and ``expected_delta_mean`` is a fraction of the step, so a product of
them is not a quantity anything measures (:mod:`gitm.agents.targeting` declines
the same combination for the same reason). Dropping a lever that provably cannot
help needs no new arithmetic; ordering the survivors by how much they might help
would, and that stays an open question.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _at_its_floor(
    spec: InterventionSpec, recoverable: Mapping[str, float | None]
) -> str | None:
    """Why this lever cannot recover anything, or ``None`` if it might.

    The question is narrower than "does this lever touch a region at its floor".
    It is "does this lever's *gain* come from making that region faster" — only
    then does the region's floor bound what the lever can deliver. So the lever
    has to have said so, via ``recovers_kernel_time``.

    That rules out most of the catalogue, correctly. Five of the six levers
    scoped to ``attn_score_value`` work through cache capacity, host swap or
    avoided recomputation rather than through faster attention kernels, and none
    of them need attention to be above its floor to pay off. ``applies_to_kernels``
    answers which kernels a lever touches, which is what coverage needs; reading
    it as a claim about mechanism is a different and wrong question, and would
    reject those five on a sound measurement.

    A ``whole_step`` lever is never ruled out: it reshapes the step itself —
    batch shape, admission order, graph capture — so no per-op gap speaks to it.

    Then, per op, three states, and the lever survives any of them:

    * **present and positive** — the region is over its floor. Keep.
    * **present and ``None``** — the op's layers disagree and the gap cannot be
      judged (see :func:`~gitm.optimizer.monitor.recoverable_by_op`). Keep: an
      unanswered question is not a no.
    * **absent** — no kernel of that op was classified in this window. Keep.
      Absence is ambiguous between "did not run" and "ran but the classifier
      could not name it", and on a trace where most kernels match no graph op the
      second is the common case. Rejecting on absence would discard levers for a
      reason that is about graph coverage rather than about the lever.

    A lever is dropped only when **every** op it names was measured, soundly, at
    or under its predicted floor. Every op it names, not every op that happened
    to be in the map: one op at its floor beside another that was never judged is
    partial evidence, and the catalogue does carry multi-op levers
    (``quantization_awq`` names five) where that distinction decides the outcome.
    """
    if spec.whole_step or not spec.applies_to_kernels:
        return None
    if not spec.recovers_kernel_time:
        return None
    judged = [(op, recoverable.get(op)) for op in spec.applies_to_kernels]
    if any(gap is None or gap > 0 for _, gap in judged):
        return None
    return ", ".join(op for op, _ in judged) + " at predicted floor"


def select_interventions(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    top_n: int = 5,
    *,
    ctx: GateContext | None = None,
    history: History | None = None,
    gpu_sku: str | None = None,
    fingerprint: str | None = None,
    recoverable: Mapping[str, float | None] | None = None,
) -> list[RankedCandidate]:
    """Rank the library for this trace, rejected candidates last.

    ``history`` is passed in rather than read from disk here, so ranking stays a
    pure function of what it is given and a caller can rank against a record it
    has already filtered. It is consulted only when ``policy.use_history`` is on
    *and* ``gpu_sku`` names the box: a result measured on an H100 says nothing
    about an MI355X, and scoring one from the other is the mistake the record's
    GPU key exists to prevent. No SKU therefore means no substitution, not a
    guess at which box the record came from.

    ``recoverable`` maps op to seconds above its predicted floor
    (:func:`gitm.optimizer.monitor.recoverable_by_op`). Passed in rather than
    derived, for the same reason ``history`` is: ranking stays a pure function of
    what it is given. Omit it and nothing is gated on the trace, which is the
    behaviour every caller had before.
    """
    use_history = policy.use_history and history is not None and gpu_sku is not None
    candidates: list[RankedCandidate] = []

    for spec in library:
        reason: str | None = None
        if ctx is not None:
            ok, why = applicable(spec, ctx)
            if not ok:
                reason = f"not_applicable: {why}"
        if reason is None and recoverable is not None:
            at_floor = _at_its_floor(spec, recoverable)
            if at_floor is not None:
                reason = f"no_recoverable_time: {at_floor}"
        if reason is None and policy.skip_high_risk and spec.safety.tier == "high_risk":
            reason = "policy.skip_high_risk"
        elif reason is None and (spec.safety.requires_qualification_commit and not policy.require_qualification_commit):
            reason = "safety.requires_qualification_commit"
        record = (
            record_for(history, spec.name, gpu_sku=gpu_sku, fingerprint=fingerprint)
            if use_history and reason is None
            else None
        )
        # A record with no usable delta is still a record: it says the lever was
        # tried and how it fared, but carries no number to rank on. The prior
        # stands in that case and only the demotion applies.
        measured = record.mean_delta if record is not None else None
        # A measured delta is the A/B's end-to-end ``speedup - 1``: the answer to
        # the question predict_delta *estimates*, already net of how much of the
        # step the lever touches. Scaling it by coverage again discounted a proven
        # result by the lever's scope — a +10% win on a lever scoped to 20% of the
        # trace ranked as +2%, below an untested 5% prior with full coverage, and a
        # measured win on a lever with an empty scope ranked as exactly zero.
        if reason is not None:
            delta = 0.0
        elif measured is not None:
            delta = measured
        else:
            delta = predict_delta(trace, spec)
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
