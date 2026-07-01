"""Hardware peaks and utilization metrics (HFU / MFU / MBU).

The planner's gate-context (:mod:`gitm.planner.context`) carries a
:class:`HardwarePeak` for the box it ran on — the dense fp16/bf16 tensor-core
FLOP/s and HBM bandwidth used as denominators for the three utilization
fractions below:

* **HFU** — Hardware FLOP Utilization: achieved FLOP/s ÷ peak FLOP/s.
* **MFU** — Model FLOP Utilization: useful model FLOPs ÷ (peak FLOP/s × seconds).
* **MBU** — Memory Bandwidth Utilization: achieved bytes/s ÷ peak bytes/s.

Every helper returns ``None`` when it cannot be computed honestly (no peak for
the SKU, zero/negative denominators, missing inputs) rather than a misleading
number — an unknown SKU leaves utilization *unreported*, never wrong. Decode is
overwhelmingly memory-bound, so MBU is the headline number for ``vllm-decode``;
HFU/MFU are reported alongside for the compute-bound prefill phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwarePeak:
    """Vendor dense peaks for one GPU SKU, the denominators for HFU/MFU/MBU.

    ``peak_flops`` is dense fp16/bf16 tensor-core throughput (FLOP/s);
    ``peak_bw_bytes_s`` is HBM bandwidth (bytes/s). Both are conservative
    catalogue figures — see ``gitm.planner.context._PEAKS``.
    """

    name: str
    peak_flops: float
    peak_bw_bytes_s: float


def _ratio(achieved: float | None, peak: float | None) -> float | None:
    """``achieved / peak`` when both are usable, else ``None`` (never a fake 0)."""
    if achieved is None or peak is None or peak <= 0 or achieved < 0:
        return None
    return achieved / peak


def hfu(achieved_flops_per_s: float | None, peak: HardwarePeak | None) -> float | None:
    """Hardware FLOP Utilization: achieved FLOP/s ÷ peak FLOP/s."""
    return _ratio(achieved_flops_per_s, peak.peak_flops if peak else None)


def mfu(
    model_flops: float | None,
    elapsed_s: float | None,
    peak: HardwarePeak | None,
) -> float | None:
    """Model FLOP Utilization: useful model FLOPs ÷ (peak FLOP/s × seconds).

    ``model_flops`` is the *useful* work (e.g. the predicted-graph FLOP count for
    the step), not every FLOP the hardware issued — so MFU ≤ HFU.
    """
    if model_flops is None or elapsed_s is None or elapsed_s <= 0:
        return None
    return _ratio(model_flops / elapsed_s, peak.peak_flops if peak else None)


def mbu(achieved_bytes_per_s: float | None, peak: HardwarePeak | None) -> float | None:
    """Memory Bandwidth Utilization: achieved bytes/s ÷ peak HBM bytes/s.

    The headline number for memory-bound decode: KV-cache + weight reads dominate
    the step, so MBU near 1.0 means the box is bandwidth-saturated and the win is
    in moving fewer bytes (fp8 KV, smaller weights), not more FLOPs.
    """
    return _ratio(achieved_bytes_per_s, peak.peak_bw_bytes_s if peak else None)
