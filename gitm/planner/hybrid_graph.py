# Predicted execution graph for a hybrid linear-attention MoE decode step.

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    distinct_experts,
    roofline,
    weight_bytes,
)

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

#: Chunk width the chunked gated-delta-rule kernels tile prefill with.
#:
#: A kernel property, not a model property — it comes from the
#: flash-linear-attention implementation vLLM ships, not from ``config.json`` —
#: which is why it lives here rather than on the spec. It sets the trade the
#: chunked form makes: arithmetic rises with C while state traffic falls by C,
#: so a wrong value moves this node's bytes proportionally.
GDN_CHUNK = 64


@dataclass(frozen=True)
class HybridMoEModelSpec:
    """Model shape for a hybrid linear-attention MoE decode step.

    **The defaults below are a small reference shape, not any real checkpoint.**
    They exist so the class is constructible and so tests have something to vary
    against; they are deliberately far too small for a deployment to be mistaken
    for one. Real checkpoints live in ``gitm/planner/models/*.yaml`` and load via
    :func:`gitm.planner.model_catalogue.load_spec`, or are read from a
    checkpoint's own ``config.json`` by :func:`spec_from_hf_config`.

    The separation is the point. When these defaults were one checkpoint's
    numbers, ``HybridMoEModelSpec()`` silently described that model, and every
    derived figure — weight footprint, KV rate, predicted step time — came out
    plausible while answering a question nobody asked. The comments below cite
    Qwen3.6-35B-A3B as the worked example precisely because its values differ
    from these defaults in ways that matter.
    """

    name: str = "hybrid-moe-reference"
    hidden: int = 512
    n_layers: int = 4
    vocab: int = 1024

    # ── attention (the full_attention layers) ────────────────────────────────
    n_heads: int = 4
    num_kv_heads: int = 2
    #: Per-head width. Real checkpoints in this family do **not** set it to
    #: ``hidden / n_heads``: Qwen3.6 runs 16 heads at 256 against a 2048 hidden
    #: size, so its query projection widens to 4096 rather than partitioning.
    #: A reader that derives this by division is wrong by 2x there.
    head_dim: int = 64
    #: Fraction of ``head_dim`` that carries rotary embedding. Real values are
    #: often well below 1.0 (0.25 on Qwen3.6); the remainder passes through
    #: unrotated, so charging RoPE over the full head is a 4x overcount.
    partial_rotary_factor: float = 1.0
    #: Qwen3.5+ gates the attention output with a projection of the same width as
    #: the query. It is a real weight matrix and a real GEMM; omitting it
    #: under-predicts the attention projection by a third.
    attn_output_gate: bool = False

    # ── gated DeltaNet (the linear_attention layers) ─────────────────────────
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 4
    linear_key_head_dim: int = 32
    linear_value_head_dim: int = 32
    linear_conv_kernel_dim: int = 4
    #: The recurrent state's storage dtype, independent of the model dtype.
    #: ``float32`` on this family while the weights are bf16 — and at low batch
    #: the state read-modify-write is the largest memory term the linear layers
    #: have, so collapsing the two halves it.
    ssm_state_dtype: str = "fp32"
    #: Channels the causal depthwise convolution maintains state for.
    #:
    #: **Fitted, not read, on the checkpoints that set it.** The architectural
    #: reading is ``q + k + v`` channels (10,240 on Qwen3.6). The value that
    #: reproduces vLLM's own page arithmetic there is 4,096 — the value width
    #: alone:
    #:
    #:     attention page = 1056 tokens x 2,048 B          = 2,162,688 B
    #:     recurrent      = 32 x 128 x 128 x 4 B           = 2,097,152 B
    #:     conv           = 4,096 x (4-1) x 4 B            =    49,152 B
    #:     ratio          = 2,162,688 / 2,146,304 = 1.00763  ->  0.76% pad
    #:
    #: which is the "Padding mamba page size by 0.76%" the engine logs at
    #: startup. Three significant figures of agreement is strong evidence but it
    #: is still an inference from one observation, so nodes that depend on it are
    #: flagged ``estimated``. Verify against ``Qwen3_5MoeText`` in transformers
    #: before treating it as read.
    conv_dim: int = 128

    # ── mixture of experts ───────────────────────────────────────────────────
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 256
    #: Width of the always-active shared expert. ``0`` disables it.
    shared_expert_intermediate_size: int = 0

    # ── layer schedule ───────────────────────────────────────────────────────
    #: Per-layer type, straight from the checkpoint. Authoritative when present:
    #: the phase matters and an interval alone does not carry it. Qwen3.6 places
    #: full attention at layers 3, 7, 11 ... — i.e. ``layer % 4 == 3``, not
    #: ``== 0``, so a modulo rule guessed from the interval puts all ten
    #: full-attention layers in the wrong place while producing a total that
    #: looks entirely plausible.
    layer_types: tuple[str, ...] = ()
    #: Fallback when ``layer_types`` is absent: every ``n``-th layer is full
    #: attention, counting so that the *last* layer is one. ``1`` means every
    #: layer is full attention.
    full_attention_interval: int = 2

    # ── multi-token prediction ───────────────────────────────────────────────
    mtp_num_hidden_layers: int = 0

    # ── precision ────────────────────────────────────────────────────────────
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    act_dtype: str = "bf16"

    # ── derived shapes ───────────────────────────────────────────────────────

    def layer_kind(self, layer: int) -> str:
        """``"full_attention"`` | ``"linear_attention"`` for ``layer``.

        Prefers the checkpoint's explicit schedule. The interval fallback is
        phased so the final layer is full attention, which is how these
        checkpoints are laid out — a naive ``layer % n == 0`` inverts the
        assignment and silently swaps 30 layers for 10.
        """
        if self.layer_types:
            if layer < len(self.layer_types):
                return self.layer_types[layer]
            return self.layer_types[-1]
        n = max(1, self.full_attention_interval)
        if n == 1:
            return FULL_ATTENTION
        return FULL_ATTENTION if (layer + 1) % n == 0 else LINEAR_ATTENTION

    def is_full_attention(self, layer: int) -> bool:
        return self.layer_kind(layer) == FULL_ATTENTION

    @property
    def n_full_attention_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.is_full_attention(i))

    @property
    def n_linear_attention_layers(self) -> int:
        return self.n_layers - self.n_full_attention_layers

    @property
    def q_dim(self) -> int:
        """Query projection width — ``n_heads * head_dim``, which may exceed hidden."""
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """Width of K (and separately V) after projection."""
        return self.num_kv_heads * self.head_dim

    @property
    def rope_dim(self) -> int:
        """Dimensions per head that actually carry rotary embedding."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_query_dim(self) -> int:
        """GDN query width. Query follows the *value* head count on this family."""
        return self.linear_num_value_heads * self.linear_key_head_dim

    @property
    def recurrent_state_elems(self) -> int:
        """Recurrent-state elements per sequence, per linear layer.

        One ``[key_head_dim, value_head_dim]`` matrix per value head. Flat in
        sequence length — the single fact that separates this family from every
        KV-cached architecture.
        """
        return (
            self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
        )

    @property
    def conv_state_elems(self) -> int:
        """Convolution-state elements per sequence, per linear layer.

        A causal depthwise convolution of width ``k`` needs the previous ``k-1``
        inputs per channel.
        """
        return self.conv_dim * max(0, self.linear_conv_kernel_dim - 1)

    @property
    def linear_state_bytes_per_layer(self) -> float:
        """Bytes of recurrent + convolution state per sequence, per linear layer."""
        sw = weight_bytes(self.ssm_state_dtype)
        return (self.recurrent_state_elems + self.conv_state_elems) * sw

    @property
    def linear_state_bytes_per_sequence(self) -> float:
        """Bytes of linear-attention state a sequence costs, across all layers.

        Paid once per sequence regardless of context length. Compare against
        :func:`kv_bytes_per_token` multiplied by the context — on this
        checkpoint the crossover sits in the low thousands of tokens, past which
        the fixed state is the cheaper half of the model despite being three
        quarters of the layers.
        """
        return self.n_linear_attention_layers * self.linear_state_bytes_per_layer

    @property
    def top_k(self) -> int:
        return min(self.num_experts_per_tok, self.num_experts)


def kv_bytes_per_token(spec: HybridMoEModelSpec) -> float:
    """KV bytes each additional token of context costs, across the whole model.

    **Only the full-attention layers contribute.** The linear layers hold a
    fixed-size state that does not grow, which is why this is ``n_full`` and not
    ``n_layers`` — the difference is 4x on this checkpoint and it sets the
    concurrency ceiling.
    """
    per_layer = 2.0 * spec.kv_dim * weight_bytes(spec.kv_dtype)  # K and V
    return spec.n_full_attention_layers * per_layer


def attention_page_bytes(spec: HybridMoEModelSpec, block_size: int) -> float:
    """Bytes one attention block occupies for a single layer.

    Exposed because the hybrid allocator equalises this against the linear
    layers' state page, and that equalisation is externally observable: vLLM
    logs the block size it picked and the padding it applied. Reproducing both
    from the config is the cheapest available check that the state arithmetic in
    this module is right.
    """
    return block_size * 2.0 * spec.kv_dim * weight_bytes(spec.kv_dtype)


def mamba_page_bytes(spec: HybridMoEModelSpec) -> float:
    """Bytes one linear-attention layer's state occupies for a single sequence."""
    return spec.linear_state_bytes_per_layer


def model_weight_bytes(
    spec: HybridMoEModelSpec, sharding: ShardingConfig | None = None
) -> float:
    """Resident weight bytes on one rank.

    Decides whether a deployment shape fits at all, which the timing graph
    cannot. Experts dominate: 256 of them per layer at three ``hidden x
    moe_intermediate`` matrices each is the great majority of the checkpoint,
    and it is the term that makes a 35B model activate 3B per token.
    """
    sh = sharding or ShardingConfig()
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)
    ww = weight_bytes(spec.weight_dtype)
    h = spec.hidden
    inter = spec.moe_intermediate_size

    experts = spec.n_layers * spec.num_experts * 3 * h * inter * ww / es
    shared = (
        spec.n_layers * 3 * h * spec.shared_expert_intermediate_size * ww / tp
        if spec.shared_expert_intermediate_size > 0
        else 0.0
    )
    router = spec.n_layers * h * spec.num_experts * ww  # replicated

    gate = spec.q_dim if spec.attn_output_gate else 0
    attn_per_layer = (h * (spec.q_dim + gate + 2 * spec.kv_dim) + spec.q_dim * h) / tp

    lin_per_layer = (
        h * (spec.linear_query_dim + spec.linear_key_dim + 2 * spec.linear_value_dim)
        + spec.linear_value_dim * h
    ) / tp

    n_full = spec.n_full_attention_layers
    n_lin = spec.n_linear_attention_layers
    embed = 2.0 * spec.vocab * h / tp  # untied input embedding + lm_head

    return (
        experts
        + shared
        + router
        + (n_full * attn_per_layer + n_lin * lin_per_layer + embed) * ww
    )


def _linear(rows: float, k: int, n: int, act_b: float, w_b: float) -> tuple[float, float]:
    """(flops, bytes) for a ``(rows, k) @ (k, n)`` projection.

    Bytes count the activation in, the weights, and the activation out. At decode
    ``rows`` is small and the weight term dominates.
    """
    return 2.0 * rows * k * n, act_b * rows * k + w_b * k * n + act_b * rows * n


def _emit_full_attention(
    add, spec: HybridMoEModelSpec, *, rows: float, cache_entries: float,
    qk_pairs: float, tp: int, aw: float, ww: float,
) -> None:
    """Softmax attention over a growing KV cache — the 10-layer minority.

    ``rows`` is every query token the step computes, decode positions and
    prefill chunk together; ``qk_pairs`` is how many query-key products the core
    evaluates, which is *not* ``rows x kv_len`` once prefill is involved (see
    :attr:`BatchConfig.attention_qk_pairs`); ``cache_entries`` is how many cached
    positions are read, counted once per sequence rather than once per row.
    """
    heads = max(1, spec.n_heads // tp)
    kv_heads = max(1, spec.num_kv_heads // tp)
    gate = spec.q_dim if spec.attn_output_gate else 0

    # Q, K, V and the output gate come out of one fused projection in vLLM, so
    # they are one node. Splitting them would put four predicted kernels where
    # one runs and leave three permanently unmatched.
    out_width = (spec.q_dim + gate) // tp + 2 * (spec.kv_dim // max(1, tp))
    f, b = _linear(rows, spec.hidden, out_width, aw, ww)
    add("qkv_proj", f, b, spec.weight_dtype)

    # QK-norm plus *partial* RoPE. Only ``partial_rotary_factor`` of each head
    # rotates; charging the whole head is a 4x overcount on this family.
    normed = rows * (heads * spec.head_dim + kv_heads * spec.head_dim)
    roped = rows * (heads + kv_heads) * spec.rope_dim
    add(
        "attn_qnorm_rope_insert",
        3.0 * normed + 6.0 * roped,
        2.0 * normed * aw + 2.0 * roped * aw + rows * 2.0 * kv_heads
        * spec.head_dim * weight_bytes(spec.kv_dtype),
        spec.act_dtype,
    )

    # The core. FLOPs follow the query-key pairs — quadratic within a prefill
    # chunk, linear at decode. Traffic follows the cached entries read, which is
    # what makes a wide step amortise cache reads across many queries and is the
    # reason prefill flips this node from memory-bound to compute-bound.
    kw = weight_bytes(spec.kv_dtype)
    qk = 2.0 * qk_pairs * heads * spec.head_dim
    pv = 2.0 * qk_pairs * heads * spec.head_dim
    add(
        "attn_score_value",
        qk + pv,
        cache_entries * 2.0 * kv_heads * spec.head_dim * kw,
        spec.weight_dtype,
    )

    f, b = _linear(rows, spec.q_dim // tp, spec.hidden, aw, ww)
    add("attn_out_proj", f, b, spec.weight_dtype)


def _emit_linear_attention(
    add, spec: HybridMoEModelSpec, *, positions: float, sequences: int,
    prefill_tokens: float, prefill_requests: int, tp: int, aw: float, ww: float,
) -> None:
    """Gated DeltaNet — the 30-layer majority, flat in context length.

    Three nodes, because three distinct kernels run and they have different
    bounds: an input projection (a GEMM), a causal depthwise convolution over a
    ``k``-wide window (elementwise, tiny), and the recurrent state update (a
    read-modify-write of a fixed matrix, which is the term that dominates at low
    batch).
    """
    q = spec.linear_query_dim // tp
    k = spec.linear_key_dim // tp
    v = spec.linear_value_dim // tp
    sw = weight_bytes(spec.ssm_state_dtype)

    # Every query token flows through the projections and the convolution,
    # whichever phase produced it. Only the recurrent state below distinguishes
    # them, because only its *algorithm* differs.
    rows = positions + prefill_tokens

    # q, k, v and the output gate z, plus the two per-head scalars (beta, alpha)
    # that gate the delta rule. vLLM fuses these into in_proj_qkvz / in_proj_ba.
    scalars = 2 * (spec.linear_num_value_heads // max(1, tp))
    f, b = _linear(rows, spec.hidden, q + k + 2 * v + scalars, aw, ww)
    add("linattn_in_proj", f, b, spec.weight_dtype)

    # Short causal convolution. The state is ``k-1`` previous inputs per channel,
    # read and written every step. Both terms are small; it is emitted separately
    # because a distinct kernel runs (``causal_conv1d_update``) and folding it
    # into the recurrent node would leave that kernel unattributable.
    conv_ch = spec.conv_dim // max(1, tp)
    conv_state = conv_ch * max(0, spec.linear_conv_kernel_dim - 1)
    add(
        "linattn_conv",
        2.0 * rows * conv_ch * spec.linear_conv_kernel_dim,
        2.0 * sequences * conv_state * sw + 2.0 * rows * conv_ch * aw,
        spec.act_dtype,
        estimated=True,  # conv_dim is fitted; see HybridMoEModelSpec.conv_dim
    )

    # The delta-rule state update. **This is the node that makes the family
    # different**, and it is also the one op whose *algorithm* changes between
    # phases rather than just its size.
    #
    # Decode runs the recurrent form (`fused_recurrent_gated_delta_rule`): one
    # sequential step per token, reading and writing the whole
    # ``[key_head_dim, value_head_dim]`` state per head per sequence. Its size
    # does not depend on accumulated context, which is what makes this family
    # cheap at long context.
    #
    # Prefill runs the *chunked* form — `chunk_scaled_dot_kkt_fwd`,
    # `solve_tril`, `recompute_w_u`, `chunk_gated_delta_rule_fwd_h`,
    # `chunk_fwd_o`, all of which appear in real captures. It rewrites the scan
    # as matrix products over chunks of ``GDN_CHUNK``:
    #
    #   * intra-chunk: K K^T then A V, costing ``2 P C (d_k + d_v)`` per head
    #   * inter-chunk: the state read-out and rank-C update, ``4 P d_k d_v``
    #   * the state is touched once per *chunk*, not once per token
    #
    # That last point is the whole reason prefill is not simply P decode steps:
    # state traffic falls by a factor of C (64x here). Modelling prefill as a
    # wide decode would overstate this node's bytes by that factor and make the
    # thirty GDN layers look like the bottleneck they are not.
    heads = max(1, spec.linear_num_value_heads // tp)
    d_k, d_v = spec.linear_key_head_dim, spec.linear_value_head_dim
    state_elems = heads * d_k * d_v

    flops = 4.0 * positions * state_elems          # decode: per-token scan
    # decode touches the state once per sequence; prefill once per chunk
    state_touches = float(sequences)
    if prefill_tokens > 0:
        c = float(GDN_CHUNK)
        flops += heads * (2.0 * prefill_tokens * c * (d_k + d_v)
                          + 4.0 * prefill_tokens * d_k * d_v)
        state_touches += prefill_requests * math.ceil(prefill_tokens / c)

    add(
        "linattn_recurrent",
        flops,
        # read the state, write it back; fp32 while the model is bf16
        2.0 * state_touches * state_elems * sw
        + (positions + prefill_tokens) * (q + k + 2 * v) * aw,
        # The *activation* dtype, not the state dtype. `ssm_state_dtype` governs
        # how the state is stored — it is already applied to the byte term above
        # via `sw` — but the arithmetic is tensor-core matmuls in the model
        # dtype. Passing fp32 here selects the fp32 FLOP peak, which the SKU
        # catalogue does not populate, so it falls back to the A100 dataclass
        # default of 19.5 TF/s against an H200's 989. That 50x penalty made this
        # node read as 57% of a prefill step when it is nearer 5%.
        spec.act_dtype,
    )

    f, b = _linear(rows, v, spec.hidden, aw, ww)
    add("attn_out_proj", f, b, spec.weight_dtype)


def _emit_moe(
    add, spec: HybridMoEModelSpec, *, positions: float, sh: ShardingConfig,
    aw: float, ww: float,
) -> None:
    """Router, shared expert, routed experts — identical in kind to the sparse-MoE
    family, so the node names are deliberately the same."""
    h = spec.hidden
    inter = spec.moe_intermediate_size
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)

    f, b = _linear(positions, h, spec.num_experts, aw, ww)
    add("moe_router", f, b, spec.weight_dtype)

    per_expert_weights = 3.0 * h * inter
    per_position_flops = 6.0 * h * inter

    if spec.shared_expert_intermediate_size > 0:
        s_inter = spec.shared_expert_intermediate_size
        add(
            "moe_shared",
            6.0 * h * s_inter * positions / tp,
            3.0 * h * s_inter * ww / tp
            + aw * (positions * h * 2 + positions * s_inter * 2 / tp),
            spec.weight_dtype,
        )

    # The saturating set-union: FLOPs scale with positions x top_k, weight
    # traffic with how many *distinct* experts the batch collectively woke.
    distinct = distinct_experts(int(positions), spec.num_experts, spec.num_experts_per_tok)
    skew = sh.ep_imbalance if sh.ep > 1 else 1.0
    add(
        "moe_routed",
        per_position_flops * positions * spec.top_k * skew / es,
        per_expert_weights * distinct * ww * skew / es
        + aw * (positions * h * 2 + positions * inter * 2 * spec.top_k / es),
        spec.weight_dtype,
    )


def _emit_layer(
    g: Graph,
    spec: HybridMoEModelSpec,
    hw: HardwareSpec,
    layer: int,
    *,
    positions: float,
    sequences: int,
    kv_len: int,
    sh: ShardingConfig,
    batch: BatchConfig,
    prefix: str = "",
) -> None:
    """Append one block's predicted nodes to ``g``.

    ``positions`` is how many sequence positions this layer computes; ``sequences``
    is how many distinct caches (KV or recurrent state) those positions read from.
    They differ under speculative decoding, and the difference is why multi-token
    verification helps a memory-bound decode: the state is read once however many
    positions ride on it.

    ``batch`` carries the prefill half of the step. Under chunked prefill one
    step routinely runs both phases in a single fused forward, so their costs are
    summed within each op rather than emitted as separate nodes — one node per op
    per layer keeps residuals comparable with a decode-only capture.
    """
    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    tp = max(1, sh.tp)

    def add(
        op: str, flops: float, byts: float, dtype: str,
        *, estimated: bool = False, serial_launches: int = 0,
    ) -> None:
        name = f"{prefix}{op}"
        g.nodes.append(
            PredictedNode(
                name, layer,
                roofline(
                    name, flops, byts, hw, dtype,
                    estimated=estimated, serial_launches=serial_launches,
                ),
            )
        )

    # Every query token the step pushes through the projections and the FFN:
    # decode positions plus whatever prefill chunk rides along with them.
    rows = positions + batch.prefill_tokens

    if spec.is_full_attention(layer):
        # Cached entries read, counted once per sequence rather than once per
        # row. Prefill re-reads its own chunk as it is written, which is why the
        # chunk appears here as well as its prior context.
        cache_entries = float(sequences * kv_len)
        if batch.is_prefill:
            cache_entries += batch.prefill_context + batch.prefill_tokens
        _emit_full_attention(
            add, spec, rows=rows, cache_entries=cache_entries,
            qk_pairs=batch.attention_qk_pairs, tp=tp, aw=aw, ww=ww,
        )
    else:
        _emit_linear_attention(
            add, spec, positions=positions, sequences=sequences,
            prefill_tokens=batch.prefill_tokens,
            prefill_requests=batch.prefill_requests, tp=tp, aw=aw, ww=ww,
        )

    _emit_moe(add, spec, positions=rows, sh=sh, aw=aw, ww=ww)

    if tp > 1:
        # Priced against the interconnect, not HBM. Bandwidth-only, so optimistic
        # at decode message sizes; a SKU with no interconnect figure leaves these
        # at zero, which Graph.has_unpriced_collectives surfaces.
        link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)
        name = f"{prefix}tp_all_reduce"
        g.nodes.append(
            PredictedNode(
                name, layer,
                roofline(
                    name, 0.0,
                    2.0 * (2.0 * (tp - 1) / tp) * rows * spec.hidden * aw,
                    link, spec.act_dtype, estimated=True,
                ),
            )
        )


def predict_hybrid_graph(
    model: HybridMoEModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
    *,
    with_mtp: bool = False,
) -> Graph:
    """Emit a predicted execution graph for one hybrid-MoE decode step, per rank.

    ``with_mtp`` is off by default even when the checkpoint declares an MTP head.
    vLLM builds it only under a speculative config, and a node no kernel can
    match is worse than an absent one: it reads as a permanently negative
    residual for an op that never ran.
    """
    spec = model or HybridMoEModelSpec()
    hw = hw or HardwareSpec()
    batch = batch or BatchConfig()
    sh = sharding or ShardingConfig()

    # Refuse shardings the model cannot take. Head counts floor-divide
    # throughout, so an indivisible split silently prices a whole path at zero
    # work — a cheap, confident, completely wrong graph.
    if spec.n_heads % max(1, sh.tp) != 0:
        raise ValueError(
            f"tensor-parallel size {sh.tp} does not divide {spec.n_heads} attention "
            "heads — every head-sharded op would floor to zero work"
        )
    if spec.linear_num_value_heads % max(1, sh.tp) != 0:
        raise ValueError(
            f"tensor-parallel size {sh.tp} does not divide "
            f"{spec.linear_num_value_heads} linear-attention value heads"
        )
    if spec.rope_dim > spec.head_dim:
        raise ValueError(
            f"partial_rotary_factor {spec.partial_rotary_factor} yields a rotary "
            f"width larger than head_dim ({spec.head_dim})"
        )
    if spec.n_layers <= 0:
        raise ValueError("n_layers must be positive — an empty model predicts nothing")

    g = Graph(model=spec, hw=hw, batch=batch, sharding=sh)  # type: ignore[arg-type]
    positions = batch.positions_per_step
    sequences = batch.batch
    kv_len = batch.kv_cache_len

    for layer in range(spec.n_layers):
        _emit_layer(
            g, spec, hw, layer,
            positions=positions, sequences=sequences, kv_len=kv_len, sh=sh,
            batch=batch,
        )

    if with_mtp:
        # Draft layers run one position per sequence. Deliberately not given
        # distinct op names — an MTP layer's qkv projection is a qkv projection;
        # what makes it the draft head is its layer index.
        for i in range(spec.mtp_num_hidden_layers):
            _emit_layer(
                g, spec, hw, spec.n_layers + i,
                positions=sequences, sequences=sequences, kv_len=kv_len, sh=sh,
                # The draft head runs one position per sequence and never
                # prefills: it proposes tokens, it does not ingest a prompt.
                batch=replace(batch, prefill_tokens=0),
            )

    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    # Only the final token of a prompt needs logits, so an 8192-token prefill
    # chunk contributes ONE row here, not 8192. Charging every prefill token
    # against a 248k-row vocabulary projection is the single largest overcount
    # available on this path — it would make lm_head dominate a prefill step.
    f, b = _linear(batch.logits_rows, spec.hidden, spec.vocab // max(1, sh.tp), aw, ww)
    g.nodes.append(
        PredictedNode("lm_head", None, roofline("lm_head", f, b, hw, spec.weight_dtype))
    )
    return g


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The text sub-config, or the config itself.

    These checkpoints ship a multimodal wrapper whose top level carries only
    ``architectures``, the vision tower and the token ids; every shape the decode
    graph needs sits under ``text_config``. A reader that looks for
    ``num_hidden_layers`` at the top level finds nothing.
    """
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def is_hybrid_moe_config(cfg: dict[str, Any]) -> bool:
    """True for the hybrid linear-attention MoE checkpoints this module models.

    The discriminator is a *mixed* layer schedule: routed experts alone describe
    a Mixtral, and linear attention alone describes a Mamba. Only a checkpoint
    that interleaves both — declared either as an explicit ``layer_types`` list
    containing more than one kind, or as a ``full_attention_interval`` greater
    than one — has the two-asymptotic structure this graph exists to price.
    """
    text = _text_config(cfg)
    routed = text.get("num_experts") and text.get("num_experts_per_tok")
    if not routed:
        return False
    types = text.get("layer_types")
    if isinstance(types, list | tuple) and len(set(types)) > 1:
        return True
    return bool(text.get("full_attention_interval", 1) and
                int(text.get("full_attention_interval", 1)) > 1)


def spec_from_hf_config(
    cfg: dict[str, Any], *, name: str | None = None
) -> HybridMoEModelSpec:
    """Build a :class:`HybridMoEModelSpec` from a HuggingFace ``config.json``.

    Reads the checkpoint's declared shape so the graph cannot drift from the
    model. Descends into ``text_config`` for the multimodal wrappers these
    checkpoints ship with.
    """
    text = _text_config(cfg)

    def _int(key: str, default: int) -> int:
        v = text.get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _req(key: str) -> int:
        """A field with no safe default.

        Falling back to some other checkpoint's value here is how a graph ends
        up confidently describing a model it never read. These are shapes every
        checkpoint in the family declares, so absence means the config is not
        what the caller thinks it is — which is worth an exception rather than a
        plausible number.
        """
        v = text.get(key)
        if v is None:
            raise ValueError(
                f"config declares no {key!r}; refusing to substitute a default, "
                "which would predict a model this checkpoint is not"
            )
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"config field {key!r} is not an integer: {v!r}") from None

    types = text.get("layer_types")
    layer_types: tuple[str, ...] = (
        tuple(str(t) for t in types) if isinstance(types, list | tuple) else ()
    )

    dtype = str(text.get("dtype") or text.get("torch_dtype") or "bf16").lower()
    if dtype.startswith("float16") or dtype == "half":
        dtype = "fp16"
    elif dtype.startswith("bfloat"):
        dtype = "bf16"
    elif dtype.startswith("float32"):
        dtype = "fp32"

    ssm = str(text.get("mamba_ssm_dtype", "fp32")).lower()
    ssm = "fp32" if ssm.startswith("float32") or ssm == "fp32" else ssm

    rotary = text.get("partial_rotary_factor")
    if rotary is None:
        params = text.get("rope_parameters")
        if isinstance(params, dict):
            rotary = params.get("partial_rotary_factor")

    hidden = _req("hidden_size")
    n_heads = _req("num_attention_heads")
    n_value_heads = _req("linear_num_value_heads")
    value_head_dim = _req("linear_value_head_dim")

    return HybridMoEModelSpec(
        name=name or str(cfg.get("_name_or_path") or "hybrid-moe"),
        hidden=hidden,
        n_layers=_req("num_hidden_layers"),
        vocab=_req("vocab_size"),
        n_heads=n_heads,
        num_kv_heads=_int("num_key_value_heads", n_heads),
        # HuggingFace's own convention when head_dim is omitted. Derived rather
        # than defaulted: the division is wrong for checkpoints that widen the
        # query projection, but it is at least a function of *this* config.
        head_dim=_int("head_dim", hidden // max(1, n_heads)),
        partial_rotary_factor=float(rotary if rotary is not None else 1.0),
        attn_output_gate=bool(text.get("attn_output_gate", False)),
        linear_num_key_heads=_int("linear_num_key_heads", n_value_heads),
        linear_num_value_heads=n_value_heads,
        linear_key_head_dim=_int("linear_key_head_dim", value_head_dim),
        linear_value_head_dim=value_head_dim,
        linear_conv_kernel_dim=_int("linear_conv_kernel_dim", 4),
        ssm_state_dtype=ssm,
        # Expressed in terms of *this* checkpoint's value width rather than a
        # constant, so a differently-shaped sibling scales with it instead of
        # silently inheriting the width that happened to fit Qwen3.6.
        conv_dim=n_value_heads * value_head_dim,
        num_experts=_req("num_experts"),
        num_experts_per_tok=_req("num_experts_per_tok"),
        moe_intermediate_size=_req("moe_intermediate_size"),
        shared_expert_intermediate_size=_int("shared_expert_intermediate_size", 0),
        layer_types=layer_types,
        full_attention_interval=_int("full_attention_interval", 1),
        mtp_num_hidden_layers=_int("mtp_num_hidden_layers", 0),
        weight_dtype=dtype,
        kv_dtype=dtype,
        act_dtype=dtype,
    )
