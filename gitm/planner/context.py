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

# Dense fp16/bf16 tensor-core peaks (FLOP/s) and HBM bandwidth (bytes/s) by SKU
# substring. Conservative vendor figures; used for HFU/MFU/MBU denominators.
_PEAKS: dict[str, tuple[float, float]] = {
    "H100": (989e12, 3350e9),
    "H200": (989e12, 4800e9),
    "A100-SXM": (312e12, 2039e9),
    "A100": (312e12, 1555e9),  # PCIe / 40GB fallback
    "L40": (181e12, 864e9),
    "L4": (121e12, 300e9),
    "V100": (125e12, 900e9),
}


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
