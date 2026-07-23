"""
Utilization metrics from a trace: HFU, modeled MFU, MBU, and stall fraction.

These are the customer-facing "how full is the GPU" numbers, computed from the
captured :class:~gitm.tracer.schema.Trace plus the hardware peaks:

busy_fraction — union of kernel intervals over wall time. Pure-timestamp,
  always available; the complement is GPU-idle/stall.
stall_breakdown — that GPU-idle complement split by cause (sync-wait,
  transfer-bound, launch-latency, idle) from overlapping sync/copy events. Pure
  timestamps; the four fractions sum to ``stall_fraction``.
HFU (Hardware FLOP Utilization) — achieved FLOP/s ÷ peak FLOP/s. Needs a
  per-kernel FLOP model (passed in); ``None`` without one.
MFU (Model FLOP Utilization) — HFU with recompute/overhead removed, i.e.
  the useful model FLOPs. Modeled as ``HFU * (1 - recompute_fraction)``.
MBU (Memory Bandwidth Utilization) — achieved bytes/s ÷ peak bandwidth,
  from memcpy bytes plus any per-kernel byte attribution.

HFU vs MFU is deliberately explicit: HFU counts every FLOP the silicon issued,
MFU only the FLOPs that advanced the model. Reporting both makes "the GPU is
busy" (high HFU) vs "the GPU is doing useful work" (high MFU) separable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from gitm.tracer.schema import KernelEvent, Trace

FlopsModel = Callable[[KernelEvent], float]


@dataclass(frozen=True)
class HardwarePeak:
    """Vendor peak rates for the SKU under test (dense, not sparse)."""

    name: str
    peak_flops: float  # FLOP/s
    peak_bw_bytes_s: float  # bytes/s


@dataclass(frozen=True)
class MetricsResult:
    n_kernels: int
    wall_s: float
    busy_fraction: float
    stall_fraction: float
    stall_breakdown: dict[str, float]  # sync_wait/transfer_bound/launch_latency/idle
    achieved_flops_per_s: float | None
    achieved_bw_bytes_s: float
    hfu: float | None
    mfu: float | None
    mbu: float
    recompute_fraction: float


# Gaps shorter than this with no overlapping sync or copy are attributed to
# CPU-side kernel-launch/dispatch latency rather than genuine device idle. ~20 µs
# comfortably covers CUDA launch/enqueue latency without swallowing real idle
# stretches; a heuristic, kept as a named knob.
_LAUNCH_LATENCY_NS = 20_000


def _merge_intervals(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping [start, end) intervals into a sorted, disjoint list."""
    # Materialize once; avoid re-sorting generators that are already nearly ordered.
    items = list(spans) if not isinstance(spans, list) else spans
    if not items:
        return []
    items.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = [items[0]]
    for start, end in items[1:]:
        last_s, last_e = merged[-1]
        if start <= last_e:
            if end > last_e:
                merged[-1] = (last_s, end)
        else:
            merged.append((start, end))
    return merged


def _merged_busy_ns(kernels: list[KernelEvent]) -> int:
    """Union length of kernel [start, end) intervals across all streams."""
    return sum(e - s for s, e in _merge_intervals((k.start_ns, k.end_ns) for k in kernels))


def _idle_gaps(kernels: list[KernelEvent], duration_ns: int) -> list[tuple[int, int]]:
    """GPU-idle intervals: complement of merged kernel-busy time in [0, duration).

    Sibling of :func:`_merged_busy_ns` (which returns only the total busy length)
    — here we keep the gap boundaries so each idle stretch can be attributed to a
    stall cause. Covers the leading gap before the first kernel, the gaps between
    kernel clusters, and the trailing gap after the last kernel. Intervals are
    clipped to ``[0, duration_ns)``.
    """
    if duration_ns <= 0:
        return []
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _merge_intervals((k.start_ns, k.end_ns) for k in kernels):
        s = min(max(start, 0), duration_ns)
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, min(max(end, 0), duration_ns))
        if cursor >= duration_ns:
            break
    if cursor < duration_ns:
        gaps.append((cursor, duration_ns))
    return gaps


def _clip_union_ns(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Total length of ``spans`` clipped to [lo, hi), overlaps counted once."""
    clipped = [(max(s, lo), min(e, hi)) for s, e in spans if min(e, hi) > max(s, lo)]
    return sum(e - s for s, e in _merge_intervals(clipped))


def _classify_stalls(
    gaps: list[tuple[int, int]],
    memcpys: list,
    syncs: list,
    wall_ns: int,
) -> dict[str, float]:
    """Attribute each idle gap to a stall cause, as fractions of wall time.

    Per gap: time overlapped by a memcpy is ``transfer_bound`` (copy engine busy,
    compute idle); time overlapped by a sync but not a copy is ``sync_wait``; the
    uncovered remainder is ``launch_latency`` for short gaps (dispatch latency),
    else ``idle``. The four fractions sum to ``stall_fraction``.
    """
    ns = {"sync_wait": 0, "transfer_bound": 0, "launch_latency": 0, "idle": 0}
    if wall_ns <= 0:
        return {k: 0.0 for k in ns}
    # Fast path: no memcpy/sync overlays — just classify gaps by length.
    if not memcpys and not syncs:
        for g0, g1 in gaps:
            residual = g1 - g0
            ns["launch_latency" if residual <= _LAUNCH_LATENCY_NS else "idle"] += residual
        return {k: v / wall_ns for k, v in ns.items()}
    cp = [(e.start_ns, e.end_ns) for e in memcpys]
    sy = [(e.start_ns, e.end_ns) for e in syncs]
    for g0, g1 in gaps:
        transfer = _clip_union_ns(cp, g0, g1)
        transfer_or_sync = _clip_union_ns(cp + sy, g0, g1)
        ns["transfer_bound"] += transfer
        ns["sync_wait"] += transfer_or_sync - transfer  # sync time not already a copy
        residual = (g1 - g0) - transfer_or_sync
        ns["launch_latency" if (g1 - g0) <= _LAUNCH_LATENCY_NS else "idle"] += residual
    return {k: v / wall_ns for k, v in ns.items()}


def compute_metrics(
        trace: Trace,
        peak: HardwarePeak,
        *,
        flops_model: FlopsModel | None = None,
        recompute_fraction: float = 0.0,
) -> MetricsResult:
    """Compute utilization metrics for ``trace`` against ``peak``.

    ``flops_model`` maps a kernel to its issued FLOPs (e.g. from a GEMM-shape
    table); without it HFU/MFU are ``None`` (we don't invent FLOPs). MBU is
    always computed from observed byte movement.
    """
    if not 0.0 <= recompute_fraction < 1.0:
        raise ValueError(f"recompute_fraction must be in [0, 1), got {recompute_fraction}")

    kernels = trace.kernels()
    memcpys = [e for e in trace.events if getattr(e, "kind", None) == "memcpy"]
    syncs = [e for e in trace.events if getattr(e, "kind", None) == "sync"]
    wall_s = trace.duration_ns / 1e9
    busy_fraction = (_merged_busy_ns(kernels) / trace.duration_ns) if trace.duration_ns else 0.0
    gaps = _idle_gaps(kernels, trace.duration_ns)
    stall_breakdown = _classify_stalls(gaps, memcpys, syncs, trace.duration_ns)

    achieved_flops: float | None = None
    hfu: float | None = None
    mfu: float | None = None
    if flops_model is not None:
        total_flops = sum(flops_model(k) for k in kernels)
        achieved_flops = total_flops / wall_s if wall_s else 0.0
        if peak.peak_flops > 0:
            hfu = achieved_flops / peak.peak_flops
            mfu = hfu * (1.0 - recompute_fraction)

    bytes_moved = sum(e.bytes for e in memcpys)
    for k in kernels:
        bytes_moved += (k.bytes_read or 0) + (k.bytes_written or 0)
    achieved_bw = bytes_moved / wall_s if wall_s else 0.0
    mbu = achieved_bw / peak.peak_bw_bytes_s if peak.peak_bw_bytes_s > 0 else 0.0

    return MetricsResult(
        n_kernels=len(kernels),
        wall_s=wall_s,
        busy_fraction=busy_fraction,
        stall_fraction=max(0.0, 1.0 - busy_fraction),
        stall_breakdown=stall_breakdown,
        achieved_flops_per_s=achieved_flops,
        achieved_bw_bytes_s=achieved_bw,
        hfu=hfu,
        mfu=mfu,
        mbu=mbu,
        recompute_fraction=recompute_fraction,
    )





