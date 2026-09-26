"""Roofline-based per-operation predictions.

For each op we compute:

    t_compute = flops / peak_flops_per_s
    t_memory  = bytes / peak_mem_bw
    t_pred    = max(t_compute, t_memory)

with a vendor-specific efficiency band ``(eff_lo, eff_hi)``: a kernel within
that band is "as expected". Residuals outside the band drive attribution.

The peak must match the op's dtype. A checkpoint that runs fp8 linears and
fp4 experts priced against a bf16 peak understates its own ceiling by 2-4x, and
an understated ceiling reads as recoverable headroom that isn't there — the one
error the headroom report exists to avoid making. ``roofline`` therefore resolves
a peak per dtype and records which peak it actually used, so a catalogue miss
surfaces as ``peak_is_fallback`` rather than as a confident wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantFormat:
    """How one tensor class is *stored*: a payload plus the scales riding with it.

    Storage is one of four questions a quantised checkpoint raises, and the only
    one a checkpoint answers by itself. The other three depend on the engine and
    the SKU (see :class:`WeightExecution`): what the MACs run in, what is resident
    after load, and what extra traffic the kernel generates. Keeping them apart is
    what lets an NVFP4 checkpoint be priced correctly on a part with no FP4 path.

    Bytes per element are exact, not fitted:

        bytes = payload_bits / 8 + scale_bytes / block_elems

    ``tensor_scales`` counts per-matrix fp32 scalars (NVFP4's ``weight_scale_2``
    and ``input_scale``, an fp8 cache's ``k_scale``/``v_scale``). They are 8 bytes
    against a matrix of ~10^7 weights, so they are carried for the record and left
    out of the per-element figure.
    """

    name: str
    payload_bits: int
    #: Elements sharing one block scale. 0 means no per-block scale at all.
    block_elems: int = 0
    #: Width of one block scale in bytes: 1 for e4m3/e8m0, 2 for bf16, 4 for fp32.
    scale_bytes: float = 0.0
    tensor_scales: int = 0
    #: What a *native* kernel multiplies in — the tensor-core family the format
    #: was designed for. Whether a given SKU has that path is decided elsewhere.
    native_compute: str = "fp16"
    source: str = ""

    @property
    def payload_bytes(self) -> float:
        return self.payload_bits / 8.0

    @property
    def scale_bytes_per_elem(self) -> float:
        return self.scale_bytes / self.block_elems if self.block_elems else 0.0

    @property
    def bytes_per_elem(self) -> float:
        return self.payload_bytes + self.scale_bytes_per_elem

    @property
    def scale_overhead(self) -> float:
        """Fraction of this format's bytes that are scales rather than payload."""
        return self.scale_bytes_per_elem / self.bytes_per_elem


#: Every storage format the catalogue entries use, with where its layout is read.
QUANT_FORMATS: dict[str, QuantFormat] = {
    f.name: f
    for f in (
        QuantFormat("fp32", 32, native_compute="fp32"),
        QuantFormat("fp16", 16),
        QuantFormat("bf16", 16),
        # DeepSeek-V3 / GLM-5.2-FP8: ``weight_block_size: [128, 128]``, one fp32
        # ``weight_scale_inv`` per block. 4 / 16,384 = 0.000244 B per weight.
        QuantFormat(
            "fp8_block128", 8, 128 * 128, 4.0, native_compute="fp8",
            source="zai-org/GLM-5.2-FP8 config.json quantization_config "
                   "weight_block_size [128, 128]; one F32 weight_scale_inv per block",
        ),
        # OCP Microscaling v1.0 §5.2: 32 elements share one E8M0 (1 byte) scale.
        QuantFormat(
            "mxfp8", 8, 32, 1.0, native_compute="fp8",
            source="OCP Microscaling Formats (MX) v1.0, Table 1: MXFP8 k=32, E8M0 scale",
        ),
        QuantFormat(
            "mxfp4", 4, 32, 1.0, native_compute="fp4",
            source="OCP Microscaling Formats (MX) v1.0, Table 1: MXFP4 k=32, E8M0 scale",
        ),
        # DeepSeek-V4's unlabelled "fp4" experts: MXFP4's bytes, but no engine
        # has been pinned for them, so no execution rule claims to know the
        # backend and the peak stays on the ladder.
        QuantFormat(
            "fp4", 4, 32, 1.0, native_compute="fp4",
            source="DeepSeek-V4 catalogue label; bytes as MXFP4, backend unpinned",
        ),
        # Read off the K2.6 shard headers: gate_proj.weight U8 [2048, 3584] (two
        # e2m1 per byte) beside weight_scale F8_E4M3 [2048, 448] (7168 / 16), plus
        # F32 [] weight_scale_2 and input_scale per matrix.
        QuantFormat(
            "nvfp4", 4, 16, 1.0, tensor_scales=2, native_compute="fp4",
            source="nvidia/Kimi-K2.6-NVFP4 hf_quant_config.json group_size 16; "
                   "shard headers U8 [2048,3584] + F8_E4M3 [2048,448]",
        ),
        # compressed-tensors pack-quantized, symmetric (no zero point): int4
        # ``weight_packed`` with a bf16 ``weight_scale`` per group of 32.
        QuantFormat(
            "int4_g32", 4, 32, 2.0, native_compute="fp16",
            source="moonshotai/Kimi-K2.5 config.json quantization_config "
                   "num_bits 4, group_size 32, symmetric, pack-quantized",
        ),
        # Cache-side formats. A generic fp8 KV cache carries one k_scale and one
        # v_scale per layer and nothing per element; DeepSeek's fp8_ds_mla layout
        # keeps one fp32 scale per 128 latent elements (656 B per 576-dim entry
        # once the bf16 RoPE key is added — see ``kv_elem_bytes``).
        QuantFormat(
            "fp8_tensor", 8, tensor_scales=2, native_compute="fp8",
            source="vLLM v0.19.1 quantization/kv_cache.py: per-layer k_scale/v_scale",
        ),
        QuantFormat(
            "fp8_ds_mla", 8, 128, 4.0, native_compute="fp8",
            source="vLLM v0.19.1 attention/mla_attention.py: fp8_ds_mla on sparse MLA",
        ),
        # Activation-side block fp8: dynamic, one fp32 scale per 128 channels of
        # each row (``GroupShape(1, weight_block_size[0])``), not one per row.
        QuantFormat(
            "fp8_group128", 8, 128, 4.0, native_compute="fp8",
            source="vLLM v0.19.1 quantization/fp8.py:317; utils/fp8_utils.py:931-932",
        ),
    )
}

#: Names the catalogue and older call sites use. Each keeps the byte cost it had
#: before the table existed: ``fp8`` is DeepSeek-style block fp8 and ``int4`` is
#: W4A16 group 32.
_WEIGHT_ALIASES: dict[str, str] = {
    "fp8": "fp8_block128",
    "e4m3": "fp8_block128",
    "fp8_e4m3": "fp8_block128",
    "float8_e4m3fn": "fp8_block128",
    "int4": "int4_g32",
    "w4a16": "int4_g32",
    "float32": "fp32",
    "float16": "fp16",
    "half": "fp16",
}

#: A KV cache is not a weight: ``fp8`` there means one scale per layer, not one
#: per 128x128 block.
_KV_ALIASES: dict[str, str] = {"fp8": "fp8_tensor", "fp8_e4m3": "fp8_tensor", "e4m3": "fp8_tensor"}


def quant_format(dtype: str) -> QuantFormat | None:
    """The storage format a weight dtype label names, or ``None`` if unknown."""
    d = dtype.lower()
    return QUANT_FORMATS.get(_WEIGHT_ALIASES.get(d, d))


# Bytes of HBM traffic per stored weight, including the quantisation scales that
# ride alongside the payload. Derived from ``QUANT_FORMATS`` so there is one
# owner for every layout. Scales are a real fraction of the bytes a decode step
# moves: NVFP4's are 11% of expert traffic, larger than several effects the
# monitor is expected to resolve.
_WEIGHT_BYTES: dict[str, float] = {
    **{name: f.bytes_per_elem for name, f in QUANT_FORMATS.items()},
    **{alias: QUANT_FORMATS[name].bytes_per_elem for alias, name in _WEIGHT_ALIASES.items()},
}


def weight_bytes(dtype: str) -> float:
    """Bytes per *stored* weight for ``dtype``, scales included.

    Unknown dtypes fall back to bf16 (2 bytes) — the conservative direction,
    since over-counting weight traffic predicts a *slower* floor and so cannot
    manufacture headroom. What a kernel actually streams can differ from what is
    stored; :func:`resolve_execution` answers that per SKU.
    """
    return _WEIGHT_BYTES.get(dtype.lower(), 2.0)


def kv_elem_bytes(dtype: str) -> float:
    """Bytes per cached KV element for a cache dtype label.

    Distinct from :func:`weight_bytes` because the same label means a different
    layout on the two sides. ``fp8`` weights carry a 128x128 block scale
    (1.000244 B); an ``fp8`` cache carries one scale per layer (1.0 B). Pricing
    the cache with the weight constant overstated GLM-5.2's fp8 KV by 10 B per
    token, the gap its design note flags against 52,608.
    """
    d = dtype.lower()
    fmt = QUANT_FORMATS.get(_KV_ALIASES.get(d, d))
    return fmt.bytes_per_elem if fmt is not None else weight_bytes(d)


@dataclass(frozen=True)
class HardwareSpec:
    """Peak achievable rates for a target GPU.

    Numbers below are illustrative defaults for A100-SXM4-80GB. Real values
    land in a vendor catalogue at ``gitm/planner/catalogue.yaml`` (roadmap).

    ``peak_flops_fp8_per_s`` / ``peak_flops_fp4_per_s`` are ``0.0`` when the SKU
    has no such tensor-core path (A100 predates both) *or* when the catalogue
    simply doesn't carry the figure. Both cases mean the same thing to
    :func:`roofline` — fall back to a dtype we do have and say so.
    """

    name: str = "A100-SXM4-80GB"
    peak_flops_fp16_per_s: float = 312e12
    peak_flops_bf16_per_s: float = 312e12
    peak_flops_fp32_per_s: float = 19.5e12
    peak_flops_fp8_per_s: float = 0.0
    peak_flops_fp4_per_s: float = 0.0
    peak_mem_bw_bytes_per_s: float = 2_039e9
    # Per-GPU bidirectional interconnect bandwidth, used to price collectives.
    # ``0.0`` means unknown, which makes a sharded graph refuse to guess rather
    # than predict a free all-to-all.
    interconnect_bw_bytes_per_s: float = 0.0
    # Wall time to issue one dependent kernel, for work whose cost is the *number
    # of launches* rather than the arithmetic in them. ~2 us is a CUDA-graph
    # replay figure; eager launch is nearer 5 us. Calibrate from a trace — an
    # iteration count multiplied by a wrong constant is still the right shape,
    # which is more than a pure roofline offers here.
    kernel_launch_overhead_s: float = 2.0e-6
    eff_lo: float = 0.55
    eff_hi: float = 0.95
    #: Tensor-core generation, which decides what a quantised format *executes*
    #: as (see :func:`resolve_execution`). Empty means unknown, and the old
    #: precision-ladder fallback applies.
    arch: str = ""
    #: HBM capacity per GPU, for deployment fit. ``0.0`` means unknown.
    memory_bytes: float = 0.0


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

        * full attention** re-reads a KV cache that grows with context, so its
          traffic scales with ``kv_cache_len``;
        * linear attention carries a fixed-size recurrent state per sequence,
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
class SparseMoEModelSpec:
    """Model shape for a sparse-MoE decode step with compressed, sparse attention.

    :class:`ModelSpec` describes a dense transformer: every weight is read every
    step, attention reads the whole KV cache, and one dtype prices everything.
    None of those hold for a DeepSeek-V4-class checkpoint, and each broken
    assumption is a separate field here:

    * **Experts are conditional.** ``num_experts_per_tok`` of ``n_routed_experts``
      run per token, so FLOPs scale with the batch while *weight traffic* scales
      with how many distinct experts the batch collectively touched — a number
      that saturates at ``n_routed_experts`` and is the dominant cost at decode.
    * **Attention is compressed and selected.** ``compress_ratios`` is per-layer;
      an indexer scores the compressed candidates and keeps ``index_topk``, on
      top of a ``sliding_window`` of recent tokens. Total context length stops
      driving the attention core once selection saturates — it drives the
      *indexer* instead, which is a different node with a different bound.
    * **Precision is per-tensor-class.** Experts, linears, and the KV cache each
      carry their own dtype, and each prices against its own peak.
    """

    name: str = "deepseek-v4-flash"
    hidden: int = 4096
    n_layers: int = 43
    n_heads: int = 64
    num_kv_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    vocab: int = 129280

    # Mixture of experts
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048

    # Sparse / compressed attention
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    sliding_window: int = 128
    # Per-layer KV compression. 0 or 1 == uncompressed (full attention). Indexed
    # by layer; layers past the end of the tuple reuse the last entry.
    compress_ratios: tuple[int, ...] = ()

    # Multi-token prediction (speculative decoding head)
    num_nextn_predict_layers: int = 1

    # Manifold-Constrained Hyper-Connections (paper §2.2). ``hc_width`` is n_hc,
    # the factor the residual stream is widened by — 1 disables every mHC term.
    # Two instances per transformer block (one around attention, one around the
    # MoE), each projecting a flattened n_hc*d residual state down to the three
    # small mappings A (1 x n_hc), B (n_hc x n_hc) and C (n_hc x 1), then
    # projecting B onto the doubly-stochastic manifold by Sinkhorn-Knopp.
    hc_width: int = 1
    hc_sinkhorn_iters: int = 0
    hc_blocks_per_layer: int = 2

    # Leading layers that route to experts by a hash of the token id rather than
    # a learned router (paper §2.1) — those layers run no router GEMM.
    num_hash_layers: int = 0

    # Low-rank state update on a subset of layers (DeepSeek "DSpark").
    dspark_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256

    # Precision, per tensor class
    weight_dtype: str = "fp8"  # attention + router + lm_head linears
    expert_dtype: str = "fp4"  # routed + shared expert weights
    kv_dtype: str = "fp8"  # KV cache and index keys
    act_dtype: str = "bf16"  # activations between ops

    def compress_ratio(self, layer: int) -> int:
        """KV compression ratio for ``layer`` (0/1 == uncompressed)."""
        if not self.compress_ratios:
            return 0
        if layer < len(self.compress_ratios):
            return self.compress_ratios[layer]
        return self.compress_ratios[-1]

    @property
    def compression_levels(self) -> tuple[int, ...]:
        """The distinct compression rates this checkpoint interleaves, ascending.

        DeepSeek-V4 uses two: ``m`` for Compressed Sparse Attention and
        ``m' >> m`` for Heavily Compressed Attention. Derived from the config
        rather than hardcoded, so a checkpoint with different rates — or only
        one — still classifies.
        """
        return tuple(sorted({r for r in self.compress_ratios if r > 1}))

    def attention_kind(self, layer: int) -> str:
        """``"swa"`` | ``"csa"`` | ``"hca"`` — which attention this layer runs.

        The three differ in ways that change the cost model qualitatively, not
        just numerically (paper §2.3):

        * **swa** — uncompressed and windowed. Only the sliding-window branch.
        * **csa** — compress by ``m``, then *sparse* attention: a lightning
          indexer scores the compressed entries and keeps ``index_topk``. Read is
          bounded, so it is flat in context once selection saturates.
        * **hca** — compress by ``m' >> m`` and attend **densely** over every
          compressed entry. No indexer. Read is ``kv_len / m'``, so unlike CSA it
          *grows with context* — HCA trades a bigger compression rate for not
          having to select.

        The largest compression level is HCA; anything else compressed is CSA.
        """
        r = self.compress_ratio(layer)
        if r == 0:
            return "swa"
        levels = self.compression_levels
        if len(levels) >= 2 and r == levels[-1]:
            return "hca"
        return "csa"

    @property
    def q_head_dim(self) -> int:
        """Per-head query width: the nope part plus the RoPE part."""
        return self.head_dim + self.qk_rope_head_dim

    @property
    def kv_latent_dim(self) -> int:
        """Bytes-per-token-per-layer worth of KV state, in elements.

        With ``num_kv_heads == 1`` the latent is shared across every query head —
        that sharing is the entire point of the compressed-KV design, and folding
        it into ``n_heads`` (as a GQA model would) overstates decode KV traffic
        by ``n_heads``x.
        """
        return self.num_kv_heads * (self.head_dim + self.qk_rope_head_dim)


@dataclass(frozen=True)
class ShardingConfig:
    """How one model is spread across ranks, from the perspective of one rank.

    The planner predicts *per-rank* cost, because that is what a rank's kernels
    are observed doing. Two sharding modes exist for the experts and they are
    routinely conflated:

    * **TP-sharded experts** (``ep == 1``) — every rank holds a slice of *every*
      expert, cut along the intermediate dimension. Perfectly load-balanced, and
      the cross-rank cost is an all-reduce of hidden states.
    * **Expert parallel** (``ep > 1``) — every rank holds *whole* experts,
      ``n_routed_experts / ep`` of them, and tokens are shipped to whichever rank
      owns their expert. The cross-rank cost is an all-to-all.

    Both divide per-rank expert weight traffic by the same degree, so **EP versus
    TP is a collective trade, not a memory trade** — a distinction worth having in
    the graph, because the lever catalog would otherwise rank them as if one saved
    HBM traffic the other doesn't.

    ``ep_imbalance`` is where EP's real cost lives. Under TP every rank does
    exactly 1/tp of the work; under EP a step waits for whichever rank drew the
    most selected experts, and at low batch that skew is large. It defaults to
    1.0 (perfect balance) and is meant to be *calibrated from a trace* rather
    than predicted — inventing a distribution here would be a guess dressed as a
    model.
    """

    tp: int = 1
    ep: int = 1
    dp: int = 1
    ep_imbalance: float = 1.0

    @property
    def expert_shards(self) -> int:
        """Ranks the expert weights are divided across."""
        return self.ep if self.ep > 1 else self.tp

    @property
    def world_size(self) -> int:
        return self.tp * self.dp


@dataclass(frozen=True)
class BatchConfig:
    """The shape of one engine step.

    Defaults describe a pure decode step, which is what every caller wanted
    before prefill was modelled. Setting ``prefill_tokens`` adds the other phase;
    under chunked prefill a single step routinely carries both, and vLLM fuses
    them into one forward pass, so their costs are additive within an op rather
    than separate nodes.

    The prefill fields mirror ``vllm.v1.metrics.perf.ExecutionContext``
    deliberately. That shape was arrived at independently and its byte and FLOP
    totals agree with this planner to 3% on a real H200 capture, so matching it
    means a measured step can parameterise a prediction directly instead of
    being approximated by a batch size somebody chose.
    """

    batch: int = 1
    prompt_len: int = 128
    kv_cache_len: int = 128  # tokens already in KV-cache when decode starts
    # Multi-token prediction: draft positions proposed per step, and the fraction
    # the verifier keeps. A step costs the drafted work regardless; only accepted
    # tokens count as output, so per-token cost divides by ``tokens_per_step``.
    speculative_tokens: int = 0
    acceptance_rate: float = 0.0

    # --- prefill (0 => a pure decode step, i.e. previous behaviour) ----------
    #: Query tokens being prefilled this step. Bounded by
    #: ``--max-num-batched-tokens`` (8192 by default), not by prompt length: a
    #: long prompt is split across steps.
    prefill_tokens: int = 0
    #: Sum over prefilling requests of the context already cached *before* this
    #: chunk. Non-zero only for the second and later chunks of a split prompt.
    prefill_context: int = 0
    #: How many distinct requests those tokens belong to. Needed because only the
    #: final token of a prompt needs logits — charging ``lm_head`` for every
    #: prefill token overstates it by the chunk size, which at 8192 tokens and a
    #: 248k vocabulary is the largest single error available to make here.
    prefill_requests: int = 1

    @property
    def is_prefill(self) -> bool:
        return self.prefill_tokens > 0

    @property
    def positions_per_step(self) -> int:
        """Sequence positions the model actually computes in one step."""
        return self.batch * (1 + max(0, self.speculative_tokens))

    @property
    def logits_rows(self) -> int:
        """Rows the vocabulary projection actually computes.

        One per prefilling request plus every decode position — vLLM's
        ``num_logits_tokens``. A prefill chunk of 8192 tokens produces one row,
        not 8192.
        """
        return (self.prefill_requests if self.is_prefill else 0) + self.positions_per_step

    def attention_qk_pairs(self, window: int = 0) -> float:
        """Query-key pairs the attention core evaluates this step.

        Decode contributes ``batch x kv_len``: one query against the whole cache.

        Prefill contributes ``P x C + P(P+1)/2``: every chunk token attends to
        all previously cached context, plus a causal prefix within the chunk.
        The second term is the quadratic one, and it is why prefill is compute-
        bound where decode is memory-bound — at 8192 tokens it is 33.6M pairs
        against a decode step's 8192.

        ``window`` caps how far back any one query may look, for sliding-window
        layers. It changes the *asymptotics*, not just the constant: the causal
        triangle becomes a band, so the quadratic term collapses to a linear one.
        At P=8192 and W=128 that is 1,040,448 pairs against 33,558,528 — a
        **32x overcount** if the window is ignored, on the term that decides
        whether prefill is compute-bound. ``0`` means no window, which reproduces
        the unwindowed expression exactly.
        """
        if window > 0:
            decode = float(self.positions_per_step * min(self.kv_cache_len, window))
        else:
            decode = float(self.positions_per_step * self.kv_cache_len)
        if not self.is_prefill:
            return decode

        p, ctx = self.prefill_tokens, self.prefill_context
        if window <= 0:
            return decode + p * ctx + p * (p + 1) / 2.0
        if ctx >= window:
            # Every chunk token already has a full window behind it.
            return decode + float(p * window)
        # ``m`` tokens ramp up from ``ctx+1`` to the window; the rest run flat.
        m = min(p, window - ctx)
        return decode + m * ctx + m * (m + 1) / 2.0 + (p - m) * window

    @property
    def tokens_per_step(self) -> float:
        """Accepted output tokens per step — the denominator for per-token cost.

        Always at least ``batch``: the non-speculative token is verified, not
        drafted, so it is never rejected.

        The speculative term is a **prefix chain**, not a product. A verifier
        walks the draft in order and stops at the first rejection, so draft token
        *k* is kept only if 1…*k*-1 were also kept: the expectation is
        ``sum(alpha**i for i in 0..D)``, not ``1 + D*alpha``. The two are far
        apart where it matters — at D=5, alpha=0.5 the linear form claims 3.5
        accepted tokens against a real 1.97, overstating throughput 1.8x and
        putting break-even at less than a third of its true value.

        This models a single-chain verifier (EAGLE/MTP-style), which is what every
        family here drafts with. A tree-attention scheme that verifies several
        candidate continuations at once accepts more than one chain and would need
        its own term.
        """
        d = max(0, self.speculative_tokens)
        a = self.acceptance_rate
        return self.batch * sum(a ** i for i in range(d + 1))


@dataclass(frozen=True)
class RooflinePrediction:
    op: str
    flops: float
    bytes: float
    t_compute_s: float
    t_memory_s: float
    t_pred_s: float
    bound: str  # "compute" | "memory" | "launch"
    # Which dtype the op runs in, and which peak was actually available to price
    # it. They differ only on a catalogue miss; see ``peak_is_fallback``.
    dtype: str = "fp16"
    peak_dtype: str = "fp16"
    peak_flops_per_s: float = 0.0
    # Set when the op's cost model is a documented approximation rather than a
    # derivation from published shapes — carried through to the report so an
    # estimate is never read as a measurement.
    estimated: bool = False
    #: Dependent kernel launches this op costs. Non-zero only for iterative work
    #: that cannot be overlapped with itself; see ``bound == "launch"``.
    serial_launches: int = 0
    #: The dtype the MACs ran in once the stored ``dtype`` was resolved against
    #: the SKU (``bf16`` for Marlin NVFP4 on H200). Empty means "same as dtype".
    compute_dtype: str = ""

    @property
    def peak_is_fallback(self) -> bool:
        """True when the op's dtype had no peak in the catalogue.

        A fallback prediction is still usable, but its ceiling is wrong in a
        known direction (too low for a dtype faster than the fallback), so the
        report must not present it as a clean roofline.
        """
        return _canon_dtype(self.compute_dtype or self.dtype) != self.peak_dtype


# ── Execution: what a stored format becomes on a given SKU ────────────────────
#
# Every rule below was read from vLLM v0.19.1 (commit b1388b1f), the engine the
# Kimi K2.6 case pinned. Paths are relative to ``vllm/model_executor/layers/``.
# A backend choice is an engine fact, not a checkpoint fact: a different vLLM
# version, or an env override, can move it, which is why the backend is named on
# every result and can be forced with ``backend=``.

Backend = Literal["native", "marlin", "emulation"]


class UnsupportedExecution(ValueError):
    """The pinned engine has no kernel for this format on this architecture."""


@dataclass(frozen=True)
class WeightExecution:
    """A stored format resolved against a SKU and an engine backend.

    Four quantities a single bytes-per-weight number conflates:

    * ``resident_bytes`` — HBM held per weight after load. Decides *fit*.
    * ``streamed_bytes`` — HBM read per weight each time the GEMM runs. Decides
      the *memory* term. Equal to storage unless the backend re-lays it out.
    * ``temp_bytes`` — scratch written and read back per weight per use, when a
      backend materialises a dequantised copy (emulation). Pure overhead.
    * ``compute_dtype`` — the tensor-core rate the MACs run at. Decides the
      *compute* term and the ridge. Marlin stores 4 bits and multiplies in bf16.

    ``act_format`` is the format the activation is quantised into ahead of the
    GEMM (``None`` for weight-only W4A16 paths): a separate kernel on the
    activation, priced per element in :func:`act_quant_bytes`.

    ``pad_hidden``/``pad_inter`` are the multiples the backend rounds an expert's
    hidden and intermediate dims up to at load; see :func:`expert_pad_factor`.
    """

    stored: QuantFormat
    backend: str
    compute_dtype: str
    resident_bytes: float
    streamed_bytes: float
    temp_bytes: float = 0.0
    act_format: QuantFormat | None = None
    pad_hidden: int = 1
    pad_inter: int = 1
    #: True when a rule is inferred rather than read from engine source.
    estimated: bool = False
    source: str = ""

    @property
    def bytes_per_use(self) -> float:
        """HBM bytes per weight each time the GEMM runs: streamed plus scratch."""
        return self.streamed_bytes + self.temp_bytes

    @property
    def is_upcast(self) -> bool:
        """True when the MACs run wider than the stored payload's native path."""
        return _canon_dtype(self.compute_dtype) != _canon_dtype(self.stored.native_compute)


@dataclass(frozen=True)
class _Rule:
    backend: str
    compute: str
    act: str | None = None
    pad_hidden: int = 1
    pad_inter: int = 1
    temp_bytes: float = 0.0
    estimated: bool = False
    source: str = ""


_MARLIN_W4A16_FP4 = (
    "fused_moe/fused_marlin_moe.py:567 (sm75+); quantization/utils/"
    "marlin_utils_fp4.py:135 (bf16/fp16 activations only), :335-344 (payload "
    "repacked, K*N/2 bytes kept), :84-110 (e4m3 scales stay 1 byte)"
)
_MARLIN_W4A16_INT4 = (
    "quantization/compressed_tensors/compressed_tensors_moe.py:176-197 "
    "(WNA16 Marlin MoE), :1249 (bf16 group scales), :1159-1161 (symmetric, no "
    "zero points); _custom_ops.py:1238-1242 (repack keeps K*N/2 bytes)"
)
_BLOCK_FP8 = (
    "quantization/fp8.py:317 (activation group 128), utils/fp8_utils.py:1512-1518 "
    "(fp32 scale per 128x128 block); fused_moe/flashinfer_cutlass_moe.py:161-168 "
    "(block fp8 MoE on sm90)"
)

#: (stored format, arch) -> what vLLM v0.19.1 runs by default.
_EXECUTION_RULES: dict[tuple[str, str], _Rule] = {
    ("fp8_block128", "hopper"): _Rule("native", "fp8", "fp8_group128", source=_BLOCK_FP8),
    ("fp8_block128", "blackwell"): _Rule("native", "fp8", "fp8_group128", source=_BLOCK_FP8),
    ("fp8_block128", "cdna4"): _Rule(
        "native", "fp8", "fp8_group128", estimated=True,
        source="fused_moe/oracle/fp8.py:68-80 lists AITER first; ROCm kernels not read",
    ),
    # NVFP4. Blackwell: TRT-LLM FP4 grouped GEMM, W4A4 with one scaled_fp4_quant
    # per MoE input. Hopper: no FP4 tensor cores and no W4A8 for NVFP4
    # (marlin_utils_fp4.py:305-307), so Marlin W4A16.
    ("nvfp4", "blackwell"): _Rule(
        "native", "fp4", "nvfp4",
        source="fused_moe/oracle/nvfp4.py:140-146; experts/trtllm_nvfp4_moe.py:87 "
               "(sm100 family); _custom_ops.py:78,1633 (activation e2m1 + e4m3/16)",
    ),
    ("nvfp4", "hopper"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_FP4),
    ("nvfp4", "ampere"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_FP4),
    ("nvfp4", "ada"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_FP4),
    # MXFP4. The default MoE backend on Blackwell is FLASHINFER_TRTLLM_MXFP4_BF16:
    # bf16 activations, so the MACs are bf16 even where fp4 tensor cores exist.
    ("mxfp4", "blackwell"): _Rule(
        "native", "bf16", pad_hidden=256, pad_inter=256,
        source="fused_moe/oracle/mxfp4.py:174-182 (TRTLLM_MXFP4_BF16 first), "
               ":386-388 (hidden and intermediate padded to 256); "
               "experts/trtllm_mxfp4_moe.py:81",
    ),
    ("mxfp4", "hopper"): _Rule(
        "native", "bf16", estimated=True,
        source="fused_moe/gpt_oss_triton_kernels_moe.py:564 (Triton on sm90-sm10x); "
               "the upcast and any scale padding live in triton_kernels, not read",
    ),
    ("mxfp4", "cdna4"): _Rule(
        "native", "fp4", estimated=True,
        source="fused_moe/oracle/mxfp4.py:174-182 lists CK second; ROCm kernels not read",
    ),
    # MXFP8: Blackwell only. There is no sm90 path in this version.
    ("mxfp8", "blackwell"): _Rule(
        "native", "fp8", "mxfp8",
        source="fused_moe/oracle/mxfp8.py:18-22,44-45; quantization/mxfp8.py:77-78",
    ),
    ("int4_g32", "hopper"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_INT4),
    ("int4_g32", "blackwell"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_INT4),
    ("int4_g32", "ampere"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_INT4),
    ("int4_g32", "cdna4"): _Rule(
        "native", "bf16", estimated=True,
        source="W4A16 by construction; the ROCm MoE kernel was not read",
    ),
}

#: Architectures that have no kernel for a format in vLLM v0.19.1.
_UNSUPPORTED: dict[tuple[str, str], str] = {
    ("mxfp8", "hopper"): "quantization/mxfp8.py:77-78 and modelopt.py:1501-1503 "
                         "require sm100",
}

#: Forced backends (``backend=``), per format. Marlin is the upcast path every
#: 4-bit format has on sm75+; emulation dequantises the whole matrix each forward.
_FORCED: dict[tuple[str, str], _Rule] = {
    ("nvfp4", "marlin"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_FP4),
    ("int4_g32", "marlin"): _Rule("marlin", "bf16", source=_MARLIN_W4A16_INT4),
    ("mxfp4", "marlin"): _Rule(
        "marlin", "bf16", pad_hidden=256, pad_inter=128,
        source="fused_moe/oracle/mxfp4.py:336-343 (VLLM_MXFP4_USE_MARLIN), :380-385 "
               "(hidden padded to 256, intermediate to 128); quantization/utils/"
               "marlin_utils_fp4.py:122 (e8m0 scales stay 1 byte)",
    ),
    # nvfp4_emulation_utils.py:57-65,140: the payload is unpacked to an fp32
    # M x K tensor (write 4), scaled into a second fp32 tensor (read 4, write 4),
    # cast to bf16 (read 4, write 2) and read by the matmul (read 2): 20 B per
    # weight of scratch on top of the 0.5625 B read from storage.
    ("nvfp4", "emulation"): _Rule(
        "emulation", "bf16", temp_bytes=20.0,
        source="quantization/utils/nvfp4_emulation_utils.py:57-65,130-141",
    ),
    # mxfp8_utils.py:71-82 follows the same fp32-then-bf16 chain.
    ("mxfp8", "emulation"): _Rule(
        "emulation", "bf16", temp_bytes=20.0, estimated=True,
        source="quantization/utils/mxfp8_utils.py:71-82,157",
    ),
}


#: With no rule for the SKU, an 8-bit format is assumed to run as designed, with
#: its dynamic activation quantisation. 4-bit formats are not: whether they run
#: W4A4 or W4A16 is exactly what an unpinned backend leaves open.
_NATIVE_ACT: dict[str, str] = {"fp8_block128": "fp8_group128", "mxfp8": "mxfp8"}


def resolve_execution(
    dtype: str, hw: HardwareSpec, *, backend: str | None = None
) -> WeightExecution:
    """What a weight stored as ``dtype`` executes as on ``hw``.

    Unknown formats and unknown architectures fall back to the precision ladder
    of :func:`_ladder_peak` with ``backend="unknown"`` and ``estimated=True``: the
    storage bytes are still exact, and only the compute rate is a guess.

    Raises :class:`UnsupportedExecution` when the pinned engine has no kernel at
    all (MXFP8 on Hopper), because pricing a checkpoint that will not load would
    be a prediction for a deployment that cannot exist.
    """
    fmt = quant_format(dtype)
    if fmt is None:
        # An unrecognised label keeps weight_bytes' conservative 2 B/weight, and
        # says it is guessing rather than claiming a native bf16 path.
        return WeightExecution(
            stored=QuantFormat(dtype.lower(), 16), backend="unknown",
            compute_dtype=_ladder_peak(hw, dtype)[1],
            resident_bytes=weight_bytes(dtype), streamed_bytes=weight_bytes(dtype),
            estimated=True,
        )
    if backend is not None and backend != "native":
        rule = _FORCED.get((fmt.name, backend))
        if rule is None:
            raise UnsupportedExecution(f"no {backend!r} backend for {fmt.name}")
    else:
        why = _UNSUPPORTED.get((fmt.name, hw.arch))
        if why is not None:
            raise UnsupportedExecution(f"{fmt.name} has no kernel on {hw.arch}: {why}")
        rule = _EXECUTION_RULES.get((fmt.name, hw.arch))
    if rule is None:
        native = fmt.payload_bits >= 16
        act = _NATIVE_ACT.get(fmt.name)
        return WeightExecution(
            stored=fmt,
            backend="native" if native else "unknown",
            compute_dtype=fmt.name if native else _ladder_peak(hw, fmt.native_compute)[1],
            resident_bytes=fmt.bytes_per_elem,
            streamed_bytes=fmt.bytes_per_elem,
            act_format=QUANT_FORMATS[act] if act else None,
            estimated=not native,
        )
    return WeightExecution(
        stored=fmt,
        backend=rule.backend,
        compute_dtype=rule.compute,
        resident_bytes=fmt.bytes_per_elem,
        streamed_bytes=fmt.bytes_per_elem,
        temp_bytes=rule.temp_bytes,
        act_format=QUANT_FORMATS[rule.act] if rule.act else None,
        pad_hidden=rule.pad_hidden,
        pad_inter=rule.pad_inter,
        estimated=rule.estimated,
        source=rule.source,
    )


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def expert_pad_factor(
    ex: WeightExecution, hidden: int, inter: int, shards: int = 1
) -> float:
    """Resident-and-streamed bytes multiplier from backend shape padding, per rank.

    vLLM splits an expert across tensor-parallel ranks along the intermediate
    dim first and pads what each rank holds afterwards (``fused_moe/layer.py:426``
    computes ``intermediate_size_per_partition = intermediate_size // tp_size``,
    then ``:537-538`` calls ``maybe_roundup_sizes(hidden_size,
    intermediate_size_per_partition)``). The hidden dim is not split for the
    expert GEMM, so it pads on its full width.

    Padding the whole matrix and then dividing is not the same thing. A width of
    2,880 across eight ranks: pad-then-split gives 3,072 / 8 = 384 per rank, and
    384 is not a multiple of 256, so the kernel pads it again to 512. Split-then-
    pad gives 360 -> 512 directly. The first order under-counts that rank's
    expert bytes by 25%. Kimi (7,168 x 2,048, TP8: 256 per rank) and GLM-5.2
    (6,144 x 2,048) are exact multiples either way, which is why the error was
    invisible on the models this was first tested on.
    """
    n = max(1, shards)
    inter_per_rank = inter / n
    ph = _round_up(hidden, ex.pad_hidden)
    pi = _round_up(int(inter_per_rank), ex.pad_inter) if ex.pad_inter > 1 else inter_per_rank
    return (ph * pi) / float(hidden * inter_per_rank)


def act_quant_bytes(ex: WeightExecution, rows: float, elems: float, act_b: float) -> float:
    """HBM bytes of the activation-quantisation kernel ahead of the GEMM.

    Reads the wide activation once and writes the quantised copy with its block
    scales. Zero for weight-only (W4A16) execution, where no such kernel runs.
    """
    if ex.act_format is None:
        return 0.0
    return rows * elems * (act_b + ex.act_format.bytes_per_elem)


@dataclass(frozen=True)
class TierTraffic:
    """Bytes one GEMM moves, by where they come from.

    ``weight_payload`` and ``weight_scales`` split the streamed weight so the
    scale overhead of a format is visible as its own line; ``temporary`` is
    scratch a backend writes and reads back; ``interconnect`` never touches HBM
    and answers to the link ridge instead.
    """

    weight_payload: float = 0.0
    weight_scales: float = 0.0
    activations: float = 0.0
    temporary: float = 0.0
    interconnect: float = 0.0

    @property
    def hbm(self) -> float:
        return self.weight_payload + self.weight_scales + self.activations + self.temporary


def linear_traffic(
    rows: float, k: int, n: int, ex: WeightExecution, act_b: float
) -> TierTraffic:
    """Per-tier bytes for a ``(rows, k) @ (k, n)`` projection under ``ex``.

    Weights are read once per use regardless of ``rows``, which is why at decode
    the weight format sets the floor and the activation width barely matters.
    """
    weights = k * n
    return TierTraffic(
        weight_payload=weights * ex.stored.payload_bytes,
        weight_scales=weights * (ex.streamed_bytes - ex.stored.payload_bytes),
        activations=act_b * rows * (k + n),
        temporary=weights * ex.temp_bytes,
    )


def ridge(hw: HardwareSpec, dtype: str, *, tier: str = "hbm") -> float:
    """FLOP/byte at which work in ``dtype`` stops being bound by ``tier``.

    ``dtype`` is the *stored* label; the rate is the one it executes at (Marlin
    NVFP4 on H200 answers to 989/4.8 = 206, not to an fp4 or fp8 figure). The
    ``link`` tier prices bytes that cross NVLink/xGMI, whose ridge is 5.3x the HBM
    one on H200: a collective is never compute-bound, and a GEMM fed over the link
    would need 5x the intensity to hide it.
    """
    peak, _ = resolve_peak(hw, dtype)
    bw = hw.interconnect_bw_bytes_per_s if tier == "link" else hw.peak_mem_bw_bytes_per_s
    return peak / bw if bw > 0 else 0.0


def critical_rows(
    dtype: str, hw: HardwareSpec, k: int, n: int, *,
    act_b: float = 2.0, backend: str | None = None,
) -> float:
    """Rows per weight matrix at which a GEMM crosses from memory- to compute-bound.

    With weights streamed once per use, a ``(r, k) @ (k, n)`` GEMM has

        AI(r) = 2 r k n / (w k n + a r (k + n))

    where ``w`` is bytes per weight per use and ``a`` the activation width.
    Setting AI = R (the ridge) and solving for r:

        r* = R w k n / (2 k n - R a (k + n))  ~=  R w / 2   for k, n >> R a

    So the knee is set by the *execution* ridge times the *streamed* bytes: NVFP4
    through Marlin on H200 turns compute-bound near 206 x 0.5625 / 2 = 58 rows
    per expert, and at 67 once a Kimi expert's own activation traffic is counted
    (k=7168, n=2048: the ``R a (k + n)`` term is 13% of ``2 k n``). Block fp8 on
    the same part runs at the fp8 rate and needs ~412 x 1.0 / 2 = 206. For MoE, rows per
    expert is ``batch * top_k / distinct_experts``, so this is the batch that makes
    an expert bank compute-bound. ``inf`` when no finite row count gets there.
    """
    ex = resolve_execution(dtype, hw, backend=backend)
    r = ridge(hw, dtype) if backend is None else (
        _ladder_peak(hw, ex.compute_dtype)[0] / hw.peak_mem_bw_bytes_per_s
    )
    denom = 2.0 * k * n - r * act_b * (k + n)
    return r * ex.bytes_per_use * k * n / denom if denom > 0 else float("inf")


def _canon_dtype(dtype: str) -> str:
    d = dtype.lower()
    # int4 (W4A16 pack-quantised) canonicalises to fp16: the weights are
    # dequantised into bf16 MACs, so the fp16/bf16 tensor-core rate is the
    # op's *correct* ceiling, not a fallback — mapping it here keeps
    # ``peak_is_fallback`` from flagging a prediction that is right.
    if d in ("bf16", "float16", "fp16", "half", "int4", "int4_g32", "w4a16"):
        return "fp16"
    if d in ("fp4", "mxfp4", "nvfp4"):
        return "fp4"
    if d in ("fp8", "e4m3", "e5m2", "fp8_e4m3", "fp8_e5m2", "float8_e4m3fn",
             "fp8_block128", "fp8_group128", "fp8_tensor", "fp8_ds_mla", "mxfp8"):
        return "fp8"
    if d in ("fp32", "float32", "tf32"):
        return "fp32"
    return d


def _execution_dtype(hw: HardwareSpec, dtype: str) -> str:
    """The dtype ``dtype``'s MACs actually run in on ``hw``, or ``dtype`` itself.

    Only a known architecture and a known quantised format resolve; anything
    else keeps the label, so the ladder below decides and flags it as before.
    """
    fmt = quant_format(dtype)
    if not hw.arch or fmt is None or fmt.payload_bits >= 16:
        return dtype
    try:
        ex = resolve_execution(dtype, hw)
    except UnsupportedExecution:
        return dtype
    return dtype if ex.backend == "unknown" else ex.compute_dtype


def resolve_peak(hw: HardwareSpec, dtype: str) -> tuple[float, str]:
    """(peak FLOP/s, the dtype that peak belongs to) for ``dtype`` on ``hw``.

    ``dtype`` is the *stored* label. On a SKU whose architecture is known it is
    first resolved to what the pinned engine executes (:func:`resolve_execution`):
    NVFP4 or MXFP4 on Hopper runs Marlin/Triton W4A16, so its rate is bf16 —
    not fp4, which Hopper lacks, and not fp8, which the old ladder picked and
    which overstated the expert GEMM's compute ceiling 2x.
    """
    return _ladder_peak(hw, _execution_dtype(hw, dtype))


def _ladder_peak(hw: HardwareSpec, dtype: str) -> tuple[float, str]:
    """The peak for ``dtype``'s tensor-core family, falling back up the ladder.

    fp4 → fp8 → fp16: a missing low-precision peak means the catalogue is
    incomplete, not that the op is free. Only for a format/arch pair with no
    execution rule; ``resolve_peak`` resolves the rest first. The fallback is
    safe only when the op really runs at the requested precision: when the
    engine upcasts (Marlin runs 4-bit weights at bf16), fp8 is 2x *too fast*,
    which is the error ``_EXECUTION_RULES`` exists to remove.
    """
    d = _canon_dtype(dtype)
    if d == "fp32":
        return hw.peak_flops_fp32_per_s, "fp32"
    if d == "fp4":
        if hw.peak_flops_fp4_per_s > 0:
            return hw.peak_flops_fp4_per_s, "fp4"
        if hw.peak_flops_fp8_per_s > 0:
            return hw.peak_flops_fp8_per_s, "fp8"
        return hw.peak_flops_fp16_per_s, "fp16"
    if d == "fp8":
        if hw.peak_flops_fp8_per_s > 0:
            return hw.peak_flops_fp8_per_s, "fp8"
        return hw.peak_flops_fp16_per_s, "fp16"
    return hw.peak_flops_fp16_per_s, "fp16"


def roofline(
    op: str,
    flops: float,
    bytes_moved: float,
    hw: HardwareSpec,
    dtype: str = "fp16",
    *,
    estimated: bool = False,
    serial_launches: int = 0,
) -> RooflinePrediction:
    """Compute the roofline prediction for a single op.

    ``serial_launches`` adds a third bound for work whose cost is the number of
    *dependent* kernel launches rather than the arithmetic inside them. A
    roofline cannot see this: 20 Sinkhorn iterations over a 4x4 matrix move a few
    hundred bytes and do a few hundred FLOPs, so both classic terms round to
    zero, while the wall time is 20 launches deep and cannot be overlapped
    because each iteration consumes the previous one's output. Ignoring it does
    not make the prediction slightly optimistic — it makes it absent.
    """
    compute_dtype = _execution_dtype(hw, dtype)
    peak_flops, peak_dtype = _ladder_peak(hw, compute_dtype)
    t_c = flops / peak_flops if peak_flops > 0 else 0.0
    t_m = bytes_moved / hw.peak_mem_bw_bytes_per_s if hw.peak_mem_bw_bytes_per_s > 0 else 0.0
    t_l = max(0, serial_launches) * hw.kernel_launch_overhead_s
    t_pred = max(t_c, t_m, t_l)
    bound = "launch" if t_l >= max(t_c, t_m) and t_l > 0 else (
        "compute" if t_c >= t_m else "memory"
    )
    return RooflinePrediction(
        op=op,
        flops=flops,
        bytes=bytes_moved,
        t_compute_s=t_c,
        t_memory_s=t_m,
        t_pred_s=t_pred,
        bound=bound,
        dtype=dtype,
        peak_dtype=peak_dtype,
        peak_flops_per_s=peak_flops,
        estimated=estimated,
        serial_launches=max(0, serial_launches),
        compute_dtype=compute_dtype,
    )
