"""Roofline-based per-operation predictions.

For each op we compute:

    t_compute = flops / peak_flops_per_s
    t_memory  = bytes / peak_mem_bw
    t_pred    = max(t_compute, t_memory)

with a vendor-specific efficiency band ``(eff_lo, eff_hi)``: a kernel within
that band is "as expected". Residuals outside the band drive attribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    """Peak achievable rates for a target GPU.

    Numbers below are illustrative defaults for A100-SXM4-80GB. Real values
    land in a vendor catalogue at ``gitm/planner/catalogue.yaml`` (roadmap).
    """

    name: str = "A100-SXM4-80GB"
    peak_flops_fp16_per_s: float = 312e12
    peak_flops_bf16_per_s: float = 312e12
    peak_flops_fp32_per_s: float = 19.5e12
    peak_mem_bw_bytes_per_s: float = 2_039e9
    eff_lo: float = 0.55
    eff_hi: float = 0.95


@dataclass(frozen=True)
class ModelSpec:
    """Model shape relevant to the decode roofline.

    Defaults match Llama-2-7B. GQA modeled via ``num_kv_heads``.
    """

    name: str = "llama-2-7b"
    hidden: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    num_kv_heads: int = 32  # < n_heads when GQA
    head_dim: int = 128
    intermediate: int = 11008
    dtype_bytes: int = 2  # fp16 / bf16
    vocab: int = 32000


@dataclass(frozen=True)
class BatchConfig:
    """Decode batch shape — prompt length already paid; we predict per-step."""

    batch: int = 1
    prompt_len: int = 128
    kv_cache_len: int = 128  # tokens already in KV-cache when decode starts


@dataclass(frozen=True)
class RooflinePrediction:
    op: str
    flops: float
    bytes: float
    t_compute_s: float
    t_memory_s: float
    t_pred_s: float
    bound: str  # "compute" | "memory"


def roofline(
    op: str,
    flops: float,
    bytes_moved: float,
    hw: HardwareSpec,
    dtype: str = "fp16",
) -> RooflinePrediction:
    """Compute the roofline prediction for a single op."""
    if dtype in ("fp16", "bf16"):
        peak_flops = hw.peak_flops_fp16_per_s
    else:
        peak_flops = hw.peak_flops_fp32_per_s
    t_c = flops / peak_flops if peak_flops > 0 else 0.0
    t_m = bytes_moved / hw.peak_mem_bw_bytes_per_s if hw.peak_mem_bw_bytes_per_s > 0 else 0.0
    bound = "compute" if t_c >= t_m else "memory"
    return RooflinePrediction(
        op=op,
        flops=flops,
        bytes=bytes_moved,
        t_compute_s=t_c,
        t_memory_s=t_m,
        t_pred_s=max(t_c, t_m),
        bound=bound,
    )
