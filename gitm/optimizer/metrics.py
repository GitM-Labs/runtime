"""
Utilization metrics from a trace: HFU, modeled MFU, MBU, and stall fraction.

These are the customer-facing "how full is the GPU" numbers, computed from the
captured :class:~gitm.tracer.schema.Trace plus the hardware peaks:

busy_fraction — union of kernel intervals over wall time. Pure-timestamp,
  always available; the complement is GPU-idle/stall.
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

from collections.abc import Callable
from dataclasses import dataclass

from gitm.tracer.schema import KernelEvent, MemcpyEvent, Trace

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
    achieved_flops_per_s: float | None
    achieved_bw_bytes_s: float
    hfu: float | None
    mfu: float | None
    mbu: float
    recompute_fraction: float


def _merged_busy_ns(kernels: list[KernelEvent]) -> int:
    """Union length of kernel [start, end) intervals across all streams."""
    spans = sorted((k.start_ns, k.end_ns) for k in kernels)
    total = 0
    cur_start: int | None = None
    cur_end = 0
    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


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
    wall_s = trace.duration_ns / 1e9
    busy_fraction = (_merged_busy_ns(kernels) / trace.duration_ns) if trace.duration_ns else 0.0

    achieved_flops: float | None = None
    hfu: float | None = None
    mfu: float | None = None
    if flops_model is not None:
        total_flops = sum(flops_model(k) for k in kernels)
        achieved_flops = total_flops / wall_s if wall_s else 0.0
        if peak.peak_flops > 0:
            hfu = achieved_flops / peak.peak_flops
            mfu = hfu * (1.0 - recompute_fraction)

    bytes_moved = sum(e.bytes for e in trace.events if isinstance(e, MemcpyEvent))
    for k in kernels:
        bytes_moved += (k.bytes_read or 0) + (k.bytes_written or 0)
    achieved_bw = bytes_moved / wall_s if wall_s else 0.0
    mbu = achieved_bw / peak.peak_bw_bytes_s if peak.peak_bw_bytes_s > 0 else 0.0

    return MetricsResult(
        n_kernels=len(kernels),
        wall_s=wall_s,
        busy_fraction=busy_fraction,
        stall_fraction=max(0.0, 1.0 - busy_fraction),
        achieved_flops_per_s=achieved_flops,
        achieved_bw_bytes_s=achieved_bw,
        hfu=hfu,
        mfu=mfu,
        mbu=mbu,
        recompute_fraction=recompute_fraction,
    )


    


