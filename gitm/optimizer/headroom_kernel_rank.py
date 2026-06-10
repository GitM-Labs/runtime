"""Headroom + per-kernel ROI — where optimization actually pays off.

Two complementary readouts the runtime produces on every run:

* **GPU headroom** (from state telemetry): compute headroom (1 - mean util),
  memory headroom (free at peak), and concurrency headroom (the fraction of
  kernel-time that runs serialized and could be overlapped).
* **Kernel ROI** (from the event trace): kernel families ranked by *estimated
  recoverable time* = cost x headroom.
  The model: if every invocation of a family ran at the family's own
  best-observed rate (a low percentile, default p10), recoverable time is
  ``sum(max(0, dur - floor))`` over its calls. Upper bound on intra-kernel
  headroom, not a guarantee; excludes cross-kernel concurrency savings.

This is optimizer logic (it prioritises which kernels an intervention should
target), so it lives in ``gitm.optimizer`` and is exported for the run loop.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

# Mangled CUDA kernel names -> a small set of stable families, so durations
# aggregate per kernel *type* rather than per template instantiation. Shared
# with attribution grouping.
_NOISE = {
    "detail", "kernel", "void", "const", "unsigned", "int", "long", "float",
    "double", "global", "device", "functor", "impl", "internal", "type",
    "types", "common", "native", "operator", "policy", "dispatch", "agent",
}



def kernel_family(name: str) -> str:
    """Collapse a mangled CUDA kernel name to a stable ``lib.func`` family."""
    lib = ("cub" if "cub" in name else "cudf" if "cudf" in name
            else "thrust" if "thrust" in name else "k")
    toks = [t for t in re.findall(r"[a-z][a-z_]{3,}", name) if t not in _NOISE and t != lib]
    return f"{lib}.{toks[0] if toks else 'anon'}"



def _percentile(sorted_xs: list[int], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    idx = round((q / 100.0) * (len(sorted_xs) - 1))
    return float(sorted_xs[idx])






@dataclass
class KernelROI:
    family: str
    calls: int
    total_ns: int 
    median_ns: float
    floor_ns: float
    recoverable_ns: float 
    share: float # total_ns / all kernel time
    recoverable_share: float # recoverable_ns / all kernel time


def kernel_roi(name_durations, floor_pct: float = 10.0) -> list[KernelROI]:
    """Rank kernel families by recoverable time (cost x headroom), descending.

    ``name_durations`` is an iterable of ``(kernel_name, duration_ns)``.
    """
    by_fam: dict[str, list[int]] = {}
    for name, dur in name_durations:
        if dur is None or dur <= 0:
            continue
        by_fam.setdefault(kernel_family(name), []).append(int(dur))
    total = sum(sum(v) for v in by_fam.values()) or 1

    rows: list[KernelROI] = []
    for fam, ds in by_fam.items():
        s = sorted(ds)
        tot = sum(s)
        floor = _percentile(s, floor_pct)
        med = _percentile(s, 50)
        recoverable = sum(max(0.0, d - floor) for d in s)
        rows.append(KernelROI(
            family=fam, calls=len(s), total_ns=tot, median_ns=med, floor_ns=floor,
            recoverable_ns=recoverable, share=tot / total, recoverable_share=recoverable / total,
        ))
    rows.sort(key=lambda r: -r.recoverable_ns)
    return rows


def render_roi_table(rows: list[KernelROI], *, floor_pct: float = 10.0, top: int = 20) -> str:
    """Human-readable ROI table (used in stdout and the provenance report)."""
    total = sum(r.total_ns for r in rows) or 1
    us, ms = 1000.0, 1_000_000.0
    out = [f"kernel ROI (floor p{floor_pct:.0f}, total kernel time {total / ms:.1f} ms):", ""]
    hdr = (f"{'#':>2}  {'family':<22} {'calls':>7} {'total_ms':>9} {'%rt':>6} "
            f"{'med_us':>8} {'p10_us':>8} {'recover_ms':>11} {'%rt':>6}")
    out += [hdr, "-" * len(hdr)]
    for i, r in enumerate(rows[:top], 1):
        out.append(
            f"{i:>2}  {r.family:<22} {r.calls:>7} {r.total_ns / ms:>9.1f} {100 * r.share:>5.1f}% "
            f"{r.median_ns / us:>8.1f} {r.floor_ns / us:>8.1f} "
            f"{r.recoverable_ns / ms:>11.2f} {100 * r.recoverable_share:>5.1f}%"
        )
    all_rec = sum(r.recoverable_ns for r in rows)
    out += ["-" * len(hdr), "",
            f"intra-kernel headroom (every call to p{floor_pct:.0f}): "
            f"{all_rec / ms:.1f} ms = {100 * all_rec / total:.1f}% of kernel time"]
    return "\n".join(out)


@dataclass
class GpuHeadroom:
    mean_util_pct: float
    peak_util_pct: float
    compute_headroom_pct: float # 100 - mean util (coarse, time-based)
    peak_mem_used_bytes: int
    mem_total_bytes: int
    mem_free_at_peak_bytes: int
    serialized_concurrency_fraction: float # fraction of kernel-time with no overlap
    n_samples: int

def gpu_headroom(samples, serialized_concurrency_fraction: float = 0.0) -> GpuHeadroom | None:
    """Summarise GPU headroom from a list of telemetry sample dicts.

    Each sample is a dict with ``util_pct`` / ``mem_used_bytes`` /
    ``mem_total_bytes`` (the canonical :class:`gitm.telemetry.Sample` fields).
    Returns ``None`` if there are no usable samples.
    """
    utils = [float(s["util_pct"]) for s in samples if s.get("util_pct") is not None]
    mems = [int(s["mem_used_bytes"]) for s in samples if s.get("mem_used_bytes") is not None]
    total = next((int(s["mem_total_bytes"]) for s in samples if s.get("mem_total_bytes")), 0)
    if not utils and not mems:
        return None
    mean_u = sum(utils) / len(utils) if utils else 0.0
    peak_u = max(utils) if utils else 0.0
    peak_m = max(mems) if mems else 0
    return GpuHeadroom(
        mean_util_pct=mean_u,
        peak_util_pct=peak_u,
        compute_headroom_pct=max(0.0, 100.0 - mean_u),
        peak_mem_used_bytes=peak_m,
        mem_total_bytes=total,
        mem_free_at_peak_bytes=max(0, total - peak_m),
        serialized_concurrency_fraction=serialized_concurrency_fraction,
        n_samples=len(samples),
    )

def live_gpu_headroom():
    """One-shot live snapshot via the telemetry backend (no running workload required)."""
    from gitm.telemetry.backends import discover_backends

    backends = discover_backends()
    out = []
    for b in backends:
        for idx in range(b.device_count()):
            s = b.sample(idx)
            out.append({
                "gpu_index": idx,
                "util_pct": s.util_pct,
                "mem_used_bytes": s.mem_used_bytes,
                "mem_total_bytes": s.mem_total_bytes,
                "power_w": s.power_w,
                "sm_clock_mhz": s.sm_clock_mhz,
                "throttle": s.throttle_reasons.name if s.throttle_reasons else "NONE",
            })
    return out
