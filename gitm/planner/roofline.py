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

    Defaults match Llama-2-7B (dense). GQA modeled via ``num_kv_heads``.

    Mixture-of-experts is opt-in: leave ``num_experts`` at 0 and every field
    below behaves exactly as a dense model, so existing callers are unaffected.
    Set it (with ``experts_per_token``) and the FFN switches to the MoE model
    described in :func:`distinct_experts` — compute scaling with the *activated*
    experts, weight traffic with the *distinct* ones.
    """

    name: str = "llama-2-7b"
    hidden: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    num_kv_heads: int = 32  # < n_heads when GQA
    head_dim: int = 128
    intermediate: int = 11008
    dtype_bytes: int = 2  # fp16 / bf16 — activations
    vocab: int = 32000

    # --- Mixture-of-experts (0 experts => dense FFN, i.e. unchanged) ---------
    #: Routed experts per MoE layer. 0 disables every MoE term below.
    num_experts: int = 0
    #: Experts each token is routed to (top-k). Clamped to ``num_experts``.
    experts_per_token: int = 0
    #: Per-expert FFN width. ``None`` falls back to ``intermediate`` — MoE models
    #: usually make each expert much narrower than a dense FFN of the same size.
    moe_intermediate: int | None = None
    #: Always-active experts (Qwen/DeepSeek-style shared expert). Their weights
    #: are fetched every step regardless of routing, and every token pays them.
    shared_experts: int = 0
    #: Width of one shared expert; ``None`` falls back to ``moe_intermediate``.
    shared_expert_intermediate: int | None = None
    #: Bytes per *weight* element. ``None`` falls back to ``dtype_bytes``. Split
    #: out because quantized MoE checkpoints (fp8/int4 weights, bf16 activations)
    #: are the common case, and MoE decode is dominated by weight traffic — using
    #: the activation width for weights would overstate it by 2x or more.
    weight_dtype_bytes: int | None = None

    @property
    def is_moe(self) -> bool:
        """True when the FFN should be modeled as a mixture of experts."""
        return self.num_experts > 0 and self.experts_per_token > 0

    @property
    def w_bytes(self) -> int:
        """Bytes per weight element (falls back to the activation dtype)."""
        return self.weight_dtype_bytes or self.dtype_bytes

    @property
    def expert_intermediate(self) -> int:
        """Per-routed-expert FFN width (falls back to the dense width)."""
        return self.moe_intermediate or self.intermediate

    @property
    def shared_intermediate(self) -> int:
        """Per-shared-expert FFN width (falls back to the routed width)."""
        return self.shared_expert_intermediate or self.expert_intermediate

    @property
    def top_k(self) -> int:
        """Routed experts per token, clamped to what actually exists."""
        return min(self.experts_per_token, self.num_experts) if self.is_moe else 0


def distinct_experts(batch: int, num_experts: int, top_k: int) -> float:
    """Expected number of *distinct* experts a batch of ``batch`` tokens activates.

    This is the term that makes MoE decode different from a dense FFN. Compute
    scales with the experts each token activates (``batch * top_k`` — linear),
    but an expert's weights are read from HBM **once** no matter how many tokens
    in the step route to it. So weight traffic scales with the size of the
    *union* of selected experts, which saturates:

        distinct(B) = E * (1 - (1 - k/E)^B)

    Under uniform routing, an expert is missed by one token with probability
    ``(1 - k/E)`` and by all ``B`` with that raised to ``B``. Two limits matter:

    * ``B * k << E`` -> ``distinct ~= B * k`` (linear; few collisions), and
    * ``B`` large    -> ``distinct -> E`` (every expert touched, so the step has
      fetched the *whole* model and MoE's bandwidth advantage over a dense model
      of the same total size is gone).

    The knee sits near ``B ~= E / k``. Because compute keeps growing linearly
    past it while bytes flatten, arithmetic intensity rises with batch — which is
    why MoE decode is memory-bandwidth-bound at low batch and only becomes
    compute-bound at large batch.

    Uniform routing is an assumption, not a measurement. Real routers are skewed,
    and skew makes tokens *collide* on hot experts, so the true distinct count is
    at or below this estimate — the prediction errs toward more traffic (slower),
    never toward an optimistic floor. Deviation detection only needs a stable,
    directionally-correct reference, and the expert-imbalance invariant is what
    measures the skew itself.

    Returns a float (an expectation, not a count). ``0.0`` when there is nothing
    to route.
    """
    if num_experts <= 0 or top_k <= 0 or batch <= 0:
        return 0.0
    k = min(top_k, num_experts)
    # k == num_experts collapses the base to 0.0, giving distinct == E for any
    # batch >= 1, which is correct: every token already touches every expert.
    p_expert_missed_by_all = (1.0 - k / num_experts) ** batch
    return num_experts * (1.0 - p_expert_missed_by_all)


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
