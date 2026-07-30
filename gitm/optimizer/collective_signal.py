"""Collective (NCCL) time → causal attribution.

The kernel-level residuals explain *compute*; on a multi-GPU run a large share
of step time can instead be collective communication — all-reduces and
all-gathers synchronizing partial results. Those are ordinary CUDA kernels on
the timeline, so the busy-fraction math counts them as work: a run blocked on
communication reads as a well-utilized GPU with nothing to fix.

:func:`collective_causes` turns the comm stats already computed by
:func:`gitm.importers.node_rollup.device_comm_stats` into ranked
:class:`CollectiveCause` hypotheses, so communication enters attribution the
same way engine-scheduler signals do (see
:mod:`gitm.optimizer.scheduler_attribution`) and each cause names the levers it
argues for.

The measurement that carries the information is **exposed** communication —
comm that does *not* overlap compute. Overlapped comm is hidden behind useful
work and costs nothing; only exposed comm is a bottleneck. Both are already
computed by ``device_comm_stats``; this module only ranks them.

Like the scheduler causes, these are rule-based severity-ranked observations
grounded in what the timeline shows, deliberately not dressed up as statistical
tests over a handful of kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.importers.node_rollup import DeviceCommStats, device_comm_stats
from gitm.tracer.schema import Trace

# Levers a communication bottleneck argues for. Unlike the scheduler knobs these
# are topology-level and structural (they need an engine rebuild), which is why
# the notes say "consider" rather than promising a hot-swap.
_TOPOLOGY_KNOBS = [
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "distributed_executor_backend",
]


def worst_device_comm(trace: Trace) -> DeviceCommStats | None:
    """Comm stats for the device with the most *exposed* communication.

    ``device_comm_stats`` assumes a single-device trace — the importer path
    splits per device before calling it. A live CUPTI capture of a multi-GPU run
    holds every device's kernels in one trace, and calling it directly there is
    quietly wrong: one GPU's compute overlaps another GPU's communication, so
    exposed comm collapses toward zero and a communication-bound run reports
    clean. Splitting by ``device_id`` first keeps the overlap math inside a
    single device, where "was compute running while this GPU communicated?" is a
    real question.

    The worst device is the one reported because a collective is a barrier: the
    step waits on the GPU that spent longest exposed.
    """
    kernels = trace.kernels()
    if not kernels:
        return None
    by_dev: dict[int, list] = {}
    for k in kernels:
        by_dev.setdefault(k.device_id, []).append(k)
    if len(by_dev) == 1:
        return device_comm_stats(trace)
    per_device = [
        device_comm_stats(trace.model_copy(update={"events": evs}))
        for _, evs in sorted(by_dev.items())
    ]
    return max(per_device, key=lambda c: c.exposed_comm_share_of_wall)


@dataclass
class CollectiveCause:
    """One communication-level causal hypothesis, ranked by ``severity`` (0..1).

    Mirrors :class:`~gitm.optimizer.scheduler_attribution.SchedulerCause` field
    for field so both kinds of cause serialize and rank identically.
    """

    signal: str  # e.g. "exposed_collective"
    effect: str  # the symptom it explains
    severity: float  # 0..1, how strongly the condition held
    note: str  # human-readable cause → implied fix
    motivates_knobs: list[str]  # library knobs this cause argues for


def collective_causes(
    stats: DeviceCommStats | None,
    *,
    exposed_floor: float = 0.05,
    dominant_floor: float = 0.20,
) -> list[CollectiveCause]:
    """Rank communication causes from comm stats (empty when there is no comm).

    ``exposed_floor`` is the share of wall time spent in comm that no compute
    was hiding; ``dominant_floor`` is the share of *busy* time that is comm at
    all. Conservative defaults — a few percent of exposed comm is normal even on
    a healthy run.
    """
    if stats is None or stats.comm_ns <= 0:
        return []

    causes: list[CollectiveCause] = []

    # Exposed comm — the GPU had nothing else to run while communicating. This is
    # the actionable one: it is time that overlap could have reclaimed.
    if stats.exposed_comm_share_of_wall > exposed_floor:
        over = stats.exposed_comm_share_of_wall - exposed_floor
        causes.append(
            CollectiveCause(
                signal="exposed_collective",
                effect="step time inflated by non-overlapped communication",
                # Normalize against the remaining headroom above the floor, so a
                # run that is *entirely* exposed comm saturates at 1.0.
                severity=min(1.0, over / max(1.0 - exposed_floor, 1e-6)),
                note=(
                    f"{stats.exposed_comm_share_of_wall:.0%} of wall time is collective "
                    "communication with no compute overlapping it. Consider a smaller "
                    "tensor_parallel_size (less cross-GPU traffic per step) or an "
                    "executor backend that overlaps comm with compute."
                ),
                motivates_knobs=list(_TOPOLOGY_KNOBS),
            )
        )

    # Comm-dominant — a large fraction of everything the GPU did was communication,
    # overlapped or not. Argues the parallelism topology is over-split.
    if stats.comm_share_of_busy > dominant_floor:
        over = stats.comm_share_of_busy - dominant_floor
        causes.append(
            CollectiveCause(
                signal="collective_dominant",
                effect="GPU busy time dominated by communication, not compute",
                severity=min(1.0, over / max(1.0 - dominant_floor, 1e-6)),
                note=(
                    f"{stats.comm_share_of_busy:.0%} of GPU busy time is collective "
                    "kernels. The parallelism topology may be over-split for this "
                    "model size — a lower tensor_parallel_size trades communication "
                    "for per-GPU memory."
                ),
                motivates_knobs=list(_TOPOLOGY_KNOBS),
            )
        )

    causes.sort(key=lambda c: c.severity, reverse=True)
    return causes
