"""Selection policy: pre-filter by safety, rank by predicted delta, return top-N."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.replay import predict_delta
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
) -> list[RankedCandidate]:
    candidates: list[RankedCandidate] = []

    for spec in library:
        reason: str | None = None
        if policy.skip_high_risk and spec.safety.tier == "high_risk":
            reason = "policy.skip_high_risk"
        elif spec.safety.requires_qualification_commit and not policy.require_qualification_commit:
            reason = "safety.requires_qualification_commit"
        delta = predict_delta(trace, spec) if reason is None else 0.0
        candidates.append(RankedCandidate(spec=spec, predicted_delta=delta, rejected_reason=reason))

    candidates.sort(
        key=lambda c: (c.rejected_reason is not None, -c.predicted_delta, c.spec.name)
    )
    return candidates[:top_n]
