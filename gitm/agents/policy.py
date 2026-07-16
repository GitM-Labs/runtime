"""Selection policy: pre-filter by safety, rank by predicted delta, return top-N."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.preconditions import GateContext, applicable
from gitm.optimizer.replay import is_kernel_attributable, predict_delta
from gitm.tracer.schema import Trace


@dataclass
class RankedCandidate:
    spec: InterventionSpec
    predicted_delta: float
    rejected_reason: str | None = None


@dataclass
class Policy:
    """Greedy by predicted delta with safety pre-filter."""

    require_qualification_commit: bool = False
    skip_high_risk: bool = False


def select_interventions(
    trace: Trace,
    library: Iterable[InterventionSpec],
    policy: Policy,
    top_n: int = 5,
    *,
    ctx: GateContext | None = None,
) -> list[RankedCandidate]:
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
        delta = predict_delta(trace, spec) if reason is None else 0.0
        candidates.append(RankedCandidate(spec=spec, predicted_delta=delta, rejected_reason=reason))

    def _rank(cs: list[RankedCandidate]) -> list[RankedCandidate]:
        return sorted(cs, key=lambda c: (-c.predicted_delta, c.spec.name))

    eligible = [c for c in candidates if c.rejected_reason is None]
    rejected = sorted(
        (c for c in candidates if c.rejected_reason is not None), key=lambda c: c.spec.name
    )

    # Balance the slate across the two mechanism classes instead of taking a flat
    # top-N. Their predicted deltas are NOT commensurable: a kernel-attributable spec
    # is scored against the time spent in the kernels it names, a scheduler knob
    # against the GPU's idle time (see predict_delta). Ranking them on one axis lets
    # whichever class happens to own the bigger denominator sweep every slot — which
    # is precisely what happened: the first vllm-decode report's five claims were the
    # five specs with no kernels declared, and fp8 KV was never even measured.
    targeted = _rank([c for c in eligible if is_kernel_attributable(c.spec)])
    broad = _rank([c for c in eligible if not is_kernel_attributable(c.spec)])

    slate: list[RankedCandidate] = []
    for i in range(max(len(targeted), len(broad))):
        if i < len(targeted):
            slate.append(targeted[i])
        if i < len(broad):
            slate.append(broad[i])
        if len(slate) >= top_n:
            break

    return (slate + rejected)[:top_n]
