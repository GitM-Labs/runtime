"""Scheduler-stats → causal attribution (the engine-signal half of the tracer adapter).

The CUPTI trace explains kernel-level effects; the engine scheduler explains the
*scheduling* causes behind them — preemptions that force KV recompute, decode
batches left half-empty (launch-bound), an admission backlog that caps
throughput, KV-cache pressure. :func:`scheduler_causes` turns a
:class:`~gitm.tracer.vllm_stats.SchedulerStatsSummary` into ranked
:class:`SchedulerCause` hypotheses so those engine signals enter causal
attribution alongside the kernel-level Granger hypotheses — and each cause names
the library knobs it motivates, linking attribution to intervention selection.

These are rule-based, severity-ranked observations grounded in vLLM scheduler
semantics, deliberately *not* dressed up as Granger p-values: the honest signal
is "this scheduler condition held, here is how strongly, and here is what it
implies", not a false statistical test over a handful of samples.
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.tracer.vllm_stats import SchedulerStatsSummary


@dataclass
class SchedulerCause:
    """One scheduler-level causal hypothesis, ranked by ``severity`` (0..1)."""

    signal: str  # e.g. "kv_cache_preemption"
    effect: str  # the decode symptom it explains
    severity: float  # 0..1, how strongly the condition held over the window
    note: str  # human-readable cause → implied fix
    motivates_knobs: list[str]  # library knobs this cause argues for


def scheduler_causes(
    summary: SchedulerStatsSummary | None,
    *,
    occupancy_floor: float = 0.6,
    cache_pressure: float = 0.9,
) -> list[SchedulerCause]:
    """Rank scheduler-level causes from a stats summary (empty when no samples).

    Thresholds are conservative defaults: occupancy below ``occupancy_floor`` is
    "under-filled", KV-cache above ``cache_pressure`` is "pressured".
    """
    if summary is None or summary.n_samples == 0:
        return []

    causes: list[SchedulerCause] = []

    # Preemptions force KV recompute — a direct, expensive decode tax.
    if summary.total_preemptions:
        causes.append(
            SchedulerCause(
                signal="kv_cache_preemption",
                effect="decode throughput (recompute after preemption)",
                severity=min(1.0, summary.total_preemptions / 10.0),
                note=(
                    f"{summary.total_preemptions} preemption(s) over the window forced KV "
                    "recompute; raise gpu_memory_utilization / add swap_space, or lower "
                    "max_num_seqs to fit the working set."
                ),
                motivates_knobs=[
                    "gpu_memory_utilization", "swap_space", "kv_cache_dtype", "max_num_seqs"
                ],
            )
        )

    # Under-filled decode batches → launch-bound, GPU starved of parallel work.
    if summary.mean_batch_occupancy is not None and summary.mean_batch_occupancy < occupancy_floor:
        deficit = (occupancy_floor - summary.mean_batch_occupancy) / occupancy_floor
        causes.append(
            SchedulerCause(
                signal="under_filled_batch",
                effect="decode is launch-bound (small batches)",
                severity=min(1.0, deficit),
                note=(
                    f"mean batch occupancy {summary.mean_batch_occupancy:.0%} < "
                    f"{occupancy_floor:.0%}: decode batches are under-filled. Raise "
                    "max_num_seqs / max_num_batched_tokens, or capture CUDA graphs "
                    "(enforce_eager=false) to cut per-step launch overhead."
                ),
                motivates_knobs=[
                    "max_num_seqs", "max_num_batched_tokens", "enforce_eager",
                    "enable_chunked_prefill",
                ],
            )
        )

    # Admission backlog: more waiting than running → scheduler can't admit fast
    # enough. Guard on ``is not None`` (not truthiness): peak_running == 0 with a
    # non-empty queue is the *worst* backlog (nothing admitted at all), and must
    # not be silently dropped by a falsy-zero check.
    if (
        summary.peak_queue_depth is not None
        and summary.peak_running is not None
        and summary.peak_queue_depth > summary.peak_running
    ):
        ratio = (summary.peak_queue_depth - summary.peak_running) / max(summary.peak_running, 1)
        causes.append(
            SchedulerCause(
                signal="admission_backlog",
                effect="throughput ceiling (requests waiting, not running)",
                severity=min(1.0, ratio),
                note=(
                    f"peak queue depth {summary.peak_queue_depth} exceeded peak running "
                    f"{summary.peak_running}: admission is the bottleneck. Raise max_num_seqs "
                    "/ max_num_batched_tokens to admit more concurrently."
                ),
                motivates_knobs=["max_num_seqs", "max_num_batched_tokens"],
            )
        )

    # KV-cache near full → preemption risk; fewer bytes per token helps.
    if summary.peak_gpu_cache_usage is not None and summary.peak_gpu_cache_usage > cache_pressure:
        over = (summary.peak_gpu_cache_usage - cache_pressure) / max(1.0 - cache_pressure, 1e-6)
        causes.append(
            SchedulerCause(
                signal="kv_cache_pressure",
                effect="preemption risk (KV cache near full)",
                severity=min(1.0, over),
                note=(
                    f"peak KV-cache usage {summary.peak_gpu_cache_usage:.0%}: cache is "
                    "pressured. fp8 KV cache or a larger block size frees capacity before "
                    "preemption kicks in."
                ),
                motivates_knobs=["kv_cache_dtype", "block_size", "gpu_memory_utilization"],
            )
        )

    causes.sort(key=lambda c: c.severity, reverse=True)
    return causes
