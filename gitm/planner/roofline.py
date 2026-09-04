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

# Bytes of HBM traffic per stored weight, including the quantisation scales that
# ride alongside the payload. Scales are a real fraction of the bytes a decode
# step moves — at fp4 they are 6% of expert traffic, which is larger than several
# effects the monitor is expected to resolve.
_WEIGHT_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    # One fp32 scale per 128x128 block (DeepSeek-style block quantisation).
    "fp8": 1.0 + 4.0 / (128 * 128),
    # MXFP4: one e8m0 (1 byte) scale per 32 values.
    "mxfp4": 0.5 + 1.0 / 32,
    # NVFP4: one e4m3 (1 byte) scale per 16 values.
    "nvfp4": 0.5 + 1.0 / 16,
    "fp4" :  0.5 + 1.0 / 32,
}


def weight_bytes(dtype: str) -> float:
    """Bytes moved per stored weight for ``dtype``, scales included.

    Unknown dtypes fall back to bf16 (2 bytes) — the conservative direction,
    since over-counting weight traffic predicts a *slower* floor and so cannot
    manufacture headroom.
    """
    return _WEIGHT_BYTES.get(dtype.lower(), 2.0)


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

    @property
    def peak_is_fallback(self) -> bool:
        """True when the op's dtype had no peak in the catalogue.

        A fallback prediction is still usable, but its ceiling is wrong in a
        known direction (too low for a dtype faster than the fallback), so the
        report must not present it as a clean roofline.
        """
        return _canon_dtype(self.dtype) != self.peak_dtype


def _canon_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("bf16", "float16", "fp16", "half"):
        return "fp16"
    if d in ("fp4", "mxfp4", "nvfp4"):
        return "fp4"
    if d in ("fp8", "e4m3", "e5m2"):
        return "fp8"
    if d in ("fp32", "float32", "tf32"):
        return "fp32"
    return d


def resolve_peak(hw: HardwareSpec, dtype: str) -> tuple[float, str]:
    """(peak FLOP/s, the dtype that peak belongs to) for ``dtype`` on ``hw``.

    Falls back down the precision ladder — fp4 → fp8 → fp16 — because a missing
    low-precision peak means the catalogue is incomplete, not that the op is
    free. Falling back *upward* in precision understates the ceiling, which is
    the safe direction: it under-reports headroom rather than inventing it.
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
    peak_flops, peak_dtype = resolve_peak(hw, dtype)
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
    )
