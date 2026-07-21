"""Blind headroom report: how far a workload sits above its roofline floor.

"Blind" = computed from the trace + the planner ceiling only, with no source,
weights, or data access. For each run it answers three things:

ceiling distance — the recoverable fraction between observed wall time and
  the predicted roofline floor (0 = already at the floor);
gap by stall class — that recoverable fraction split across idle/stall,
  memory-bound, and compute-bound, from the metrics module;
already-optimized flag — set when the distance is within a small threshold,
  so we never bill for headroom that isn't there.

This is the packaging step: planner ceiling + attribution gap +
metrics → one report-ready object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from gitm.optimizer.metrics import MetricsResult
from gitm.tracer.schema import Trace

# Sources whose event plane lacks NVML state and per-kernel DRAM counters.
_IMPORT_SOURCES = frozenset({"nsys-import", "torch-import"})

_CAVEAT_MBU = (
    "MBU computed from memcpy traffic only; per-kernel DRAM counters are not "
    "present in profiler exports."
)
_CAVEAT_NO_STATE = (
    "No device state plane (power, clocks, throttling); throttle-induced stalls "
    "appear as idle."
)
_CAVEAT_CATALOGUE = (
    "Predicted floor uses catalogue peak rates for the reported SKU; unvalidated "
    "against live telemetry."
)
_CAVEAT_TORCH_SYNC = (
    "Sync events absent from this trace format; sync-wait vs launch-latency "
    "attribution is coarse."
)


@dataclass
class HeadroomReport:
    workload: str
    sku: str | None
    predicted_floor_s: float
    observed_s: float
    ceiling_distance: float  # recoverable fraction in [0, 1)
    already_optimized: bool
    gap_by_stall_class: dict[str, float] = field(default_factory=dict)
    busy_fraction: float = 0.0
    mbu: float = 0.0
    hfu: float | None = None
    confidence: Literal["full", "trace-only"] = "full"
    caveats: list[str] = field(default_factory=list)
    # When True, memory/compute split is indicative (HFU unavailable / import path).
    indicative_mem_compute_split: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _import_caveats(source: str) -> list[str]:
    caveats = [_CAVEAT_MBU, _CAVEAT_NO_STATE, _CAVEAT_CATALOGUE]
    if source == "torch-import":
        caveats.append(_CAVEAT_TORCH_SYNC)
    return caveats


def build_headroom(
    trace: Trace,
    *,
    predicted_floor_s: float,
    metrics: MetricsResult,
    workload: str,
    sku: str | None = None,
    optimized_threshold: float = 0.10,
) -> HeadroomReport:
    """Assemble the headroom report from the trace, planner floor, and metrics."""
    observed_s = trace.duration_ns / 1e9
    if observed_s > 0 and predicted_floor_s > 0:
        ceiling_distance = max(0.0, (observed_s - predicted_floor_s) / observed_s)
    else:
        ceiling_distance = 0.0
    already_optimized = ceiling_distance < optimized_threshold

    # Split the recoverable distance across stall classes. Idle is the GPU-idle
    # fraction; the busy remainder is attributed memory- vs compute-bound by which
    # utilization dominates (MBU vs HFU). Shares scale to the ceiling distance.
    idle = metrics.stall_fraction
    hfu = metrics.hfu or 0.0
    denom = metrics.mbu + hfu
    # Stall split with HFU=None keeps the existing 50/50 fallback.
    mem_share = (metrics.mbu / denom) if denom > 0 else 0.5
    busy = max(0.0, 1.0 - idle)
    shares = {
        "idle_stall": idle,
        "memory_bound": busy * mem_share,
        "compute_bound": busy * (1.0 - mem_share),
    }
    gap = {k: round(v * ceiling_distance, 4) for k, v in shares.items()}

    source = getattr(trace, "source", "cupti") or "cupti"
    if source in _IMPORT_SOURCES:
        confidence: Literal["full", "trace-only"] = "trace-only"
        caveats = _import_caveats(source)
        indicative = True  # HFU typically None; mem/compute split is indicative
    else:
        confidence = "full"
        caveats = []
        indicative = metrics.hfu is None

    return HeadroomReport(
        workload=workload,
        sku=sku,
        predicted_floor_s=predicted_floor_s,
        observed_s=observed_s,
        ceiling_distance=round(ceiling_distance, 4),
        already_optimized=already_optimized,
        gap_by_stall_class=gap,
        busy_fraction=round(metrics.busy_fraction, 4),
        mbu=round(metrics.mbu, 4),
        hfu=round(metrics.hfu, 4) if metrics.hfu is not None else None,
        confidence=confidence,
        caveats=caveats,
        indicative_mem_compute_split=indicative,
    )


def render_headroom_md(r: HeadroomReport) -> str:
    """Render the headroom report as a compact markdown block."""
    lines = [
        f"## Blind headroom — {r.workload}" + (f" on {r.sku}" if r.sku else ""),
        "",
        f"- Observed wall: {r.observed_s * 1e3:.3f} ms",
        f"- Predicted floor: {r.predicted_floor_s * 1e3:.3f} ms",
        f"- **Ceiling distance (recoverable): {r.ceiling_distance:.1%}**",
        f"- Already optimized: {'yes' if r.already_optimized else 'no'}",
        f"- Busy/MBU/HFU: {r.busy_fraction:.1%} / {r.mbu:.1%} / "
        + (f"{r.hfu:.1%}" if r.hfu is not None else "n/a"),
        "",
        "Gap by stall class:",
    ]
    for cls, frac in r.gap_by_stall_class.items():
        lines.append(f"  - {cls}: {frac:.1%}")
    if r.already_optimized:
        lines.append("")
        lines.append("> Flagged already-optimized — no headroom to bill.")
    return "\n".join(lines) + "\n"

