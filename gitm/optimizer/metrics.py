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

import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def _overlap_sweep(spans: list[tuple[int, int]], gaps: list[tuple[int, int]]) -> list[int]:
    """Length of ``spans`` (sorted, disjoint) inside each of ``gaps`` (sorted by start).

    Linear when the gaps are disjoint; gaps that overlap each other are still answered
    correctly but rescan the spans they share.
    """
    out: list[int] = []
    i, n = 0, len(spans)
    for g0, g1 in gaps:
        # A span ending at or before this gap's start cannot reach a later gap either.
        while i < n and spans[i][1] <= g0:
            i += 1
        covered = 0
        j = i
        while j < n and spans[j][0] < g1:
            piece = min(spans[j][1], g1) - max(spans[j][0], g0)
            if piece > 0:
                covered += piece
            j += 1
        out.append(covered)
    return out


def _stall_ns_per_gap(gaps: list[tuple[int, int]], memcpys: list, syncs: list) -> np.ndarray:
    """Per gap, sorted by start: columns ``start, transfer, sync, launch, idle`` in ns.

    One sweep over the sorted gaps and the merged records, so for disjoint gaps the
    cost is linear in gaps plus records after the sorts rather than their product.
    An int64 array, one row per gap: a list of tuples costs about three times as much.
    (Object dtype, holding exact Python integers, when a timestamp does not fit int64.)
    """
    try:
        ordered = np.array(sorted(gaps), dtype=np.int64).reshape(-1, 2)
    except OverflowError:
        # A timestamp past int64 (a corrupt header's sentinel): exact Python integers.
        ordered = np.array(sorted(gaps), dtype=object).reshape(-1, 2)
    dtype = ordered.dtype
    start, length = ordered[:, 0], ordered[:, 1] - ordered[:, 0]
    # Fast path: no memcpy/sync overlays — just classify gaps by length.
    if not memcpys and not syncs:
        transfer = covered = np.zeros(len(ordered), dtype=dtype)
    else:
        # Empty and reversed records cover nothing; dropped before the merge.
        cp = [(e.start_ns, e.end_ns) for e in memcpys if e.end_ns > e.start_ns]
        sy = [(e.start_ns, e.end_ns) for e in syncs if e.end_ns > e.start_ns]
        pairs = ordered.tolist()
        transfer = np.array(_overlap_sweep(_merge_intervals(cp), pairs), dtype=dtype)
        covered = np.array(_overlap_sweep(_merge_intervals(cp + sy), pairs), dtype=dtype)
    residual = length - covered
    # The 20 us test is on the whole gap; covered - transfer is sync time not already a copy.
    short = length <= _LAUNCH_LATENCY_NS
    return np.column_stack([start, transfer, covered - transfer,
                            np.where(short, residual, 0), np.where(short, 0, residual)])


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
    # The column order of _stall_ns_per_gap.
    keys = ("transfer_bound", "sync_wait", "launch_latency", "idle")
    if wall_ns <= 0:
        return dict.fromkeys(keys, 0.0)
    totals = _stall_ns_per_gap(gaps, memcpys, syncs)[:, 1:].sum(axis=0)
    return {k: int(v) / wall_ns for k, v in zip(keys, totals, strict=True)}


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


def _strict_json(value):
    """A copy that strict JSON can carry: non-finite floats become ``None``."""
    if isinstance(value, Mapping):
        return {k: _strict_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_strict_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_atomic(path: Path, text: str) -> None:
    """Write beside ``path`` and rename into place, so a reader never sees half a file.

    The temporary file is created exclusively under a random name: a guessable name
    would follow a symlink planted in a shared run directory.
    """
    fh = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False, encoding="utf-8")
    try:
        with fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
            # The exclusive temp file is 0600; give it the mode a plain write would have.
            umask = os.umask(0)
            os.umask(umask)
            os.fchmod(fh.fileno(), 0o666 & ~umask)
        os.replace(fh.name, path)
    except BaseException:
        Path(fh.name).unlink(missing_ok=True)
        raise
