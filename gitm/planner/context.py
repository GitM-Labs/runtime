"""Planner gate-context: the deployment facts levers are matched against.

The precondition gate (:mod:`gitm.optimizer.preconditions`) and the metrics
module (:mod:`gitm.optimizer.metrics`) both need ground truth about this box:
which SKU, what dtype, how big the KV cache, how many GPUs, NVLink or not, and
the hardware peak FLOP/bandwidth. This module assembles that once, from NVML and
the live engine, so downstream code never guesses.

Everything degrades cleanly: no NVML → read ``GITM_GPU_SKU``; unknown SKU →
``None`` peaks (HFU/MFU simply stay unreported rather than wrong). Engine
introspection is duck-typed so it survives vLLM version drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from gitm.optimizer.metrics import HardwarePeak
from gitm.optimizer.preconditions import GateContext
from gitm.planner.roofline import HardwareSpec

# Dense fp16/bf16 tensor-core peaks (FLOP/s) and HBM bandwidth (bytes/s) by SKU
# substring. Conservative vendor figures; used for HFU/MFU/MBU denominators
# (below) and, via :func:`hardware_spec_for`, for the roofline prediction
# itself. "L40" must stay ordered before "L4" (see peak_for_sku) since "l4" is
# a substring of "l40" and the first substring match wins.
_PEAKS: dict[str, tuple[float, float]] = {
    # "GB200" must precede "B200": the latter is a substring of the former, and
    # the first substring match wins. Both are the same silicon per GPU, so the
    # ordering is cosmetic here — it stops mattering the moment they diverge.
    # Blackwell Ultra raises dense fp4 by ~1.67x and doubles HBM, but leaves
    # bf16/fp8 and — decisively for decode — memory bandwidth unchanged at
    # 8 TB/s. A memory-bound decode step therefore sees essentially none of the
    # uplift, which is only visible if these are separate entries.
    "GB300": (2250e12, 8000e9),
    "B300": (2250e12, 8000e9),
    "GB200": (2250e12, 8000e9),
    "B200": (2250e12, 8000e9),
    "H100": (989e12, 3350e9),
    "H200": (989e12, 4800e9),
    "A100-SXM": (312e12, 2039e9),
    "A100": (312e12, 1555e9),  # PCIe / 40GB fallback
    "L40": (181e12, 864e9),
    "L4": (121e12, 300e9),
    "T4": (65e12, 320e9),  # the common free-tier Colab GPU
    "V100": (125e12, 900e9),
}

# Low-precision tensor-core peaks (FLOP/s), keyed by the same SKU substrings as
# ``_PEAKS``. A SKU absent here has no fp8/fp4 path *or* no catalogue entry yet;
# both resolve identically in :func:`gitm.planner.roofline.resolve_peak`, which
# falls back up the precision ladder and flags the prediction.
#
# Dense figures, no 2:4 sparsity — the sparsity-doubled numbers in vendor
# marketing do not apply to a dense decode step, and using them would halve every
# predicted compute time and so double the apparent headroom.
#
_QUANT_PEAKS: dict[str, dict[str, float]] = {
    # Blackwell Ultra: 15 PFLOPS dense fp4, with fp8/bf16 carried over from B200.
    "GB300": {"fp8": 4500e12, "fp4": 15000e12},
    "B300": {"fp8": 4500e12, "fp4": 15000e12},
    "GB200": {"fp8": 4500e12, "fp4": 9000e12},
    "B200": {"fp8": 4500e12, "fp4": 9000e12},
    # Hopper has fp8 tensor cores; it has no fp4 path (MXFP4 runs dequantised
    # through Marlin, which is why an fp4 checkpoint traced on H100/H200 prices
    # against fp8 and still shows a compute-bound expert GEMM).
    "H100": {"fp8": 1979e12},
    "H200": {"fp8": 1979e12},
}


# CUDA-core FP32 peaks (FLOP/s), same substring keys. Not a tensor-core rate:
# an fp32 op in a graph is there because the *model* asked for fp32 — a MoE router
# under ``moe_router_dtype: "float32"``, a softmax accumulation — and those run on
# the FP32 pipe, not on the tensor cores. Vendor "TF32 tensor core" figures are an
# order of magnitude higher and pricing a router against one would make the node
# disappear from the table.
#
# A SKU absent here keeps the dataclass default (an A100's 19.5 TF/s), which is
# low for anything newer and so under-reports rather than over-reports headroom —
# but on an H200 it is 3.4x low, which is enough to move a small fp32 node's bound
# label, so the SKUs this planner actually targets are listed.
_FP32_PEAKS: dict[str, float] = {
    "GB300": 80e12,
    "B300": 80e12,
    "GB200": 80e12,
    "B200": 80e12,
    "H100": 67e12,
    "H200": 67e12,
    "A100": 19.5e12,
    "L40": 90e12,
    "L4": 30e12,
    "T4": 8.1e12,
    "V100": 15.7e12,
}


def fp32_peak_for_sku(sku: str | None) -> float:
    """CUDA-core FP32 peak for a SKU (substring match), else ``0.0``."""
    if not sku:
        return 0.0
    for key, peak in _FP32_PEAKS.items():
        if key.lower() in sku.lower():
            return peak
    return 0.0


# Per-GPU bidirectional NVLink bandwidth (bytes/s), same substring keys. Used to
# price the collectives a sharded graph emits. A SKU absent here leaves the spec
# at 0.0, which makes the sharded planner report collectives as unpriced instead
# of predicting them as free — an unpriced node is a visible gap, a free one is a
# wrong ceiling.
_INTERCONNECT: dict[str, float] = {
    "GB300": 1800e9,  # NVLink 5
    "B300": 1800e9,
    "GB200": 1800e9,
    "B200": 1800e9,
    "H100": 900e9,  # NVLink 4
    "H200": 900e9,
    "A100": 600e9,  # NVLink 3
}


def quant_peaks_for_sku(sku: str | None) -> dict[str, float]:
    """Low-precision peaks for a SKU string (substring match), else empty."""
    if not sku:
        return {}
    for key, peaks in _QUANT_PEAKS.items():
        if key.lower() in sku.lower():
            return peaks
    return {}


def interconnect_bw_for_sku(sku: str | None) -> float:
    """Per-GPU bidirectional NVLink bandwidth for a SKU, else ``0.0``."""
    if not sku:
        return 0.0
    for key, bw in _INTERCONNECT.items():
        if key.lower() in sku.lower():
            return bw
    return 0.0


@dataclass
class PlannerContext:
    """The assembled deployment facts for one run.

    ``gate`` is what the precondition gate matches levers against; ``peak`` is
    the SKU's dense peaks (``None`` on an unknown SKU). ``sku``/``num_gpus`` are
    surfaced for the report.
    """

    gate: GateContext
    peak: HardwarePeak | None
    sku: str | None
    num_gpus: int


def _query_nvml() -> tuple[str | None, int | None]:
    """(SKU name, device count) via NVML in a single init/shutdown cycle.

    Returns ``(None, None)`` when NVML/pynvml is unavailable. One cycle for both
    queries — they describe the same device set, so there's no reason to init,
    shutdown, and re-init.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            name_s = name.decode() if isinstance(name, bytes) else str(name)
            return name_s, int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None, None


def peak_for_sku(sku: str | None) -> HardwarePeak | None:
    """Look up dense peaks for a SKU string (substring match), else None."""
    if not sku:
        return None
    for key, (flops, bw) in _PEAKS.items():
        if key.lower() in sku.lower():
            return HardwarePeak(name=sku, peak_flops=flops, peak_bw_bytes_s=bw)
    return None


def hardware_spec_for(peak: HardwarePeak | None) -> HardwareSpec:
    """Roofline :class:`HardwareSpec` for the detected GPU peak.

    Falls back to ``HardwareSpec()`` (A100-SXM4-80GB) when the SKU wasn't
    recognized (unknown NVML name, ``GITM_GPU_SKU`` unset, no GPU) — the same
    default ``predict_graph`` silently used everywhere before this existed.
    ``peak_flops`` covers fp16/bf16; fp8/fp4 come from ``_QUANT_PEAKS`` when the
    SKU has them, and stay ``0.0`` otherwise so ``resolve_peak`` can fall back
    and mark the prediction rather than pricing an fp4 GEMM at the bf16 rate.
    The fp32 peak comes from ``_FP32_PEAKS`` — the CUDA-core rate, since a model
    that declares an fp32 op (a MoE router under ``moe_router_dtype``) runs it on
    that pipe. A SKU without an entry keeps the dataclass default.
    """
    if peak is None:
        return HardwareSpec()
    quant = quant_peaks_for_sku(peak.name)
    fp32 = fp32_peak_for_sku(peak.name)
    defaults = HardwareSpec()
    return HardwareSpec(
        name=peak.name,
        peak_flops_fp16_per_s=peak.peak_flops,
        peak_flops_bf16_per_s=peak.peak_flops,
        peak_flops_fp32_per_s=fp32 or defaults.peak_flops_fp32_per_s,
        peak_flops_fp8_per_s=quant.get("fp8", 0.0),
        peak_flops_fp4_per_s=quant.get("fp4", 0.0),
        peak_mem_bw_bytes_per_s=peak.peak_bw_bytes_s,
        interconnect_bw_bytes_per_s=interconnect_bw_for_sku(peak.name),
    )


def _engine_dtype(engine: Any) -> str | None:
    if engine is None:
        return None
    for path in ("model_config.dtype", "dtype"):
        obj: Any = engine
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            s = str(obj).lower()
            for dt in ("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"):
                if dt in s:
                    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(dt, dt)
    return None


def _engine_kv_len(engine: Any) -> int | None:
    if engine is None:
        return None
    for path in ("cache_config.max_model_len", "model_config.max_model_len", "max_model_len"):
        obj: Any = engine
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, int):
            return obj
    return None


def build_planner_context(
    engine: Any = None,
    *,
    workload: str = "vllm-decode",
    num_gpus: int | None = None,
) -> PlannerContext:
    """Assemble the gate context + hardware peaks for this run.

    ``GITM_GPU_SKU`` overrides NVML (useful in CI / on a box without pynvml).
    """
    env_sku = os.environ.get("GITM_GPU_SKU")
    # Only touch NVML if something it provides is actually missing.
    nvml_name = nvml_count = None
    if env_sku is None or num_gpus is None:
        nvml_name, nvml_count = _query_nvml()
    sku = env_sku or nvml_name
    n = num_gpus or nvml_count or 1
    peak = peak_for_sku(sku)
    dtype = _engine_dtype(engine)
    kv_len = _engine_kv_len(engine)

    gate = GateContext(
        workload=workload,
        dtype=dtype,
        hardware=sku,
        kv_cache_len=kv_len,
        num_gpus=n,
        has_collective=n > 1,
        has_interconnect=n > 1,  # refined later by NVLink/IB probe
    )
    return PlannerContext(gate=gate, peak=peak, sku=sku, num_gpus=n)
