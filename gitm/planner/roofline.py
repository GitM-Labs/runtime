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
    #: Leading layers that keep a *dense* FFN (DeepSeek ``first_k_dense_replace``).
    #: MoE models commonly leave the first block(s) dense; modeling them as MoE
    #: overstates both their weight footprint and their traffic.
    first_dense_layers: int = 0
    #: Among the remaining layers, every ``moe_layer_step``-th one is MoE and the
    #: rest stay dense (Qwen ``decoder_sparse_step``). 1 = every layer is MoE.
    moe_layer_step: int = 1
    #: Hybrid attention: every ``full_attn_layer_step``-th layer uses softmax
    #: attention over a growing KV cache; the rest use linear/recurrent attention
    #: (gated DeltaNet, Mamba) whose state is *constant* in sequence length.
    #: 1 = every layer is full attention, i.e. a conventional transformer.
    full_attn_layer_step: int = 1

    @property
    def is_moe(self) -> bool:
        """True when *any* layer's FFN should be modeled as a mixture of experts."""
        return self.num_experts > 0 and self.experts_per_token > 0

    def is_moe_layer(self, layer: int) -> bool:
        """Whether layer index ``layer`` uses the mixture FFN rather than a dense one.

        Real MoE checkpoints are not uniformly sparse: DeepSeek keeps the first
        ``first_k_dense_replace`` layers dense, and Qwen places MoE blocks every
        ``decoder_sparse_step`` layers. Treating every layer as MoE inflates the
        predicted weight footprint (and therefore the ceiling) by whatever
        fraction is actually dense.
        """
        if not self.is_moe or layer < self.first_dense_layers:
            return False
        step = max(self.moe_layer_step, 1)
        return (layer - self.first_dense_layers) % step == 0

    @property
    def n_moe_layers(self) -> int:
        """How many layers actually carry the mixture FFN."""
        return sum(1 for i in range(self.n_layers) if self.is_moe_layer(i))

    def is_full_attention_layer(self, layer: int) -> bool:
        """Whether layer ``layer`` uses softmax attention over a KV cache.

        Hybrid models (Qwen3-Next-style gated DeltaNet, Mamba/Jamba) interleave a
        few full-attention layers among many linear-attention ones. The two have
        fundamentally different memory behaviour at decode:

        * **full attention** re-reads a KV cache that grows with context, so its
          traffic scales with ``kv_cache_len``;
        * **linear attention** carries a fixed-size recurrent state per sequence,
          so its traffic is *constant* in sequence length.

        Modeling every layer as full attention overstates KV traffic by the ratio
        of context length to state size — at 16k context that is over an order of
        magnitude, and it is why a hybrid model can serve long contexts with only
        a few percent of KV-cache utilisation.
        """
        step = max(self.full_attn_layer_step, 1)
        return layer % step == 0

    @property
    def n_full_attention_layers(self) -> int:
        """How many layers use softmax attention over a growing KV cache."""
        return sum(1 for i in range(self.n_layers) if self.is_full_attention_layer(i))

    @property
    def is_hybrid_attention(self) -> bool:
        """True when some layers use linear/recurrent attention instead of KV."""
        return max(self.full_attn_layer_step, 1) > 1

    @property
    def linear_attn_state_elems(self) -> int:
        """Recurrent-state elements per sequence for one linear-attention layer.

        Gated DeltaNet (and linear attention generally) keeps a ``[head_dim,
        head_dim]`` state matrix per head instead of a per-token KV cache, so the
        state is ``n_heads * head_dim^2`` and does **not** grow with context.

        An approximation: architectures vary in whether the linear-attention
        heads share the softmax heads' dimensions. It is the right *shape* — flat
        in sequence length rather than linear in it — which is what makes the
        prediction directionally correct where treating it as KV does not.
        """
        return self.n_heads * self.head_dim * self.head_dim

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

    # --- parameter accounting -------------------------------------------------
    # The "35B-A3B" naming convention: total parameters vs the ones a single
    # token actually multiplies against. FLOPs follow *active*, checkpoint size
    # and (at saturation) weight traffic follow *total* — conflating them is the
    # 10x error that makes an MoE ceiling meaningless.

    @property
    def _attn_params_per_layer(self) -> int:
        qkv = self.hidden * (self.n_heads + 2 * self.num_kv_heads) * self.head_dim
        out = self.n_heads * self.head_dim * self.hidden
        return qkv + out

    def _dense_ffn_params(self, width: int) -> int:
        """gate + up + down for an FFN of the given intermediate width."""
        return 3 * self.hidden * width

    @property
    def total_params(self) -> int:
        """Every weight in the checkpoint, including all experts.

        Embedding and LM head are counted separately (untied); a tied-embedding
        model has ``vocab * hidden`` fewer, which is under a percent for the
        large-vocab models this matters for.
        """
        n_moe = self.n_moe_layers
        n_dense = self.n_layers - n_moe
        total = self.n_layers * self._attn_params_per_layer
        total += n_dense * self._dense_ffn_params(self.intermediate)
        if n_moe:
            per_moe = (
                self.num_experts * self._dense_ffn_params(self.expert_intermediate)
                + self.shared_experts * self._dense_ffn_params(self.shared_intermediate)
                + self.hidden * self.num_experts  # router
            )
            total += n_moe * per_moe
        return total + 2 * self.vocab * self.hidden  # embedding + lm_head

    @property
    def active_params(self) -> int:
        """Weights one token actually multiplies against on a decode step.

        For a dense model this equals :attr:`total_params`. For a mixture only
        ``top_k`` of ``num_experts`` participate per token (plus shared experts),
        which is what makes a 35B model cost 3B of compute per token.
        """
        n_moe = self.n_moe_layers
        n_dense = self.n_layers - n_moe
        active = self.n_layers * self._attn_params_per_layer
        active += n_dense * self._dense_ffn_params(self.intermediate)
        if n_moe:
            per_moe = (
                self.top_k * self._dense_ffn_params(self.expert_intermediate)
                + self.shared_experts * self._dense_ffn_params(self.shared_intermediate)
                + self.hidden * self.num_experts  # router runs for every token
            )
            active += n_moe * per_moe
        return active + 2 * self.vocab * self.hidden

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
