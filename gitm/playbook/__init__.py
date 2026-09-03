"""Playbook schema — the contract between detection and apply.

Deliverable 4 of the validation infrastructure. A playbook row says *this knob,
on this model revision and this GPU, under this workload regime, moved these
numbers by this much, and here is everything needed to check it.*

    from gitm.playbook import Playbook, lookup, UNCALIBRATED_POLICY

    result = lookup(playbook, query_identity, UNCALIBRATED_POLICY)
    if result.route_to_discovery:
        ...          # conservative discovery; the handoff is defined, not the mode
    else:
        apply(result.row.identity.knobs)

Matching is a **split**: exact equality on the categorical fields (model,
revision, GPU, environment, source_kind, concurrency), log2-ratio distance
combined with L-infinity on the numeric regime axes. The distance threshold is
**not calibrated yet** and the shipped policy says so — see
:mod:`gitm.playbook.match`.

CPU-only. ``python -m gitm.playbook --selftest`` is the check.
"""

from gitm.playbook.match import (
    AXIS_METRICS,
    DEFAULT_AXES,
    RATE_AXIS_DECISION,
    UNCALIBRATED_POLICY,
    AxisTolerance,
    Candidate,
    MatchPolicy,
    MatchResult,
    MatchStatus,
    RegimeDistance,
    dispersion_distance,
    log2_ratio,
    lookup,
    regime_distance,
)
from gitm.playbook.schema import (
    PENDING_ADIT,
    SCHEMA,
    THROUGHPUT_METRIC,
    EnvCapture,
    Evidence,
    Invalidation,
    MeasuredDelta,
    Playbook,
    PlaybookRow,
    Provenance,
    RowIdentity,
    row_from_runs,
)

__all__ = [
    "AXIS_METRICS",
    "DEFAULT_AXES",
    "PENDING_ADIT",
    "RATE_AXIS_DECISION",
    "SCHEMA",
    "THROUGHPUT_METRIC",
    "UNCALIBRATED_POLICY",
    "AxisTolerance",
    "Candidate",
    "EnvCapture",
    "Evidence",
    "Invalidation",
    "MatchPolicy",
    "MatchResult",
    "MatchStatus",
    "MeasuredDelta",
    "Playbook",
    "PlaybookRow",
    "Provenance",
    "RegimeDistance",
    "RowIdentity",
    "dispersion_distance",
    "log2_ratio",
    "lookup",
    "regime_distance",
    "row_from_runs",
]
