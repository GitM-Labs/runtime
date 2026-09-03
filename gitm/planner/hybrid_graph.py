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
#: Softmax attention over a KV cache capped at ``sliding_window`` recent
#: tokens. Neither of the other two: priced as linear its KV read vanishes
#: (a fixed recurrent state is not a cache), priced as full it is overstated
#: by ``kv_len / window`` — 64x at 8k context on a 128-token window. It runs
#: the *same kernels* as full attention, so it emits the same node names and
#: ``classify_op`` needs no new rule.
SLIDING_ATTENTION = "sliding_attention"

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

    #: Per-head width of **V**, when it differs from K's ``head_dim``. MiMo runs
    #: K at 192 and V at 128, which makes ``o_proj``'s input ``n_heads *
    #: v_head_dim`` (8192) rather than ``q_dim`` (12288) — a 1.5x *overstatement*
    #: of that projection if left conflated, not the understatement one might
    #: expect. ``None`` means "same as head_dim", under which every expression
    #: below reduces exactly to what it was before this field existed.
    v_head_dim: int | None = None

    # ── sliding-window attention (the third layer kind) ──────────────────────
    #: Tokens a windowed layer may attend to. ``0`` means no window, which is
    #: what every non-windowed checkpoint wants and leaves the arithmetic
    #: unchanged.
    sliding_window: int = 0
    #: Windowed layers may carry their own GQA geometry. MiMo runs 8 KV heads on
    #: its sliding-window layers against 4 on its full-attention ones — and the
    #: KV rate is the quantity decode is bound by, so a single head count for the
    #: model is wrong on 39 of 48 layers. ``None`` inherits the full-attention
    #: value.
    swa_num_kv_heads: int | None = None
    swa_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    #: Windowed layers commonly use a second, much smaller RoPE base (MiMo: 1e4
    #: against 1e7). Carried because it is a real model fact worth recording, but
    #: it prices nothing: a second sin/cos table changes no byte count at decode.
    swa_rope_theta: float = 0.0
    #: SWA layers add a learned per-head bias to the softmax denominator
    #: (``add_swa_attention_sink_bias``). Also carried and also free — it is one
    #: elementwise add already inside the attention kernel. Recorded so the spec
    #: describes the model, not so a node gets invented for it.
    swa_attention_sink: bool = False

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

    #: Width of the dense FFN on ``dense_layers`` and the draft head. Falls back
    #: to ``moe_intermediate_size`` when unset, but real checkpoints make the
    #: dense block far wider than one expert: MiMo runs 16384 against an expert's
    #: 2048, an 8x difference on the only node in that layer that matters.
    dense_intermediate_size: int = 0

    #: Layers whose FFN is a plain dense MLP rather than a mixture. DeepSeek and
    #: MiMo both leave the first block dense; charging it a 256-expert router and
    #: an expert bank it does not have is pure invention. Empty means every layer
    #: is MoE, which is what this family did before the field existed.
    dense_layers: frozenset[int] = frozenset()

    # ── multi-token prediction ───────────────────────────────────────────────
    mtp_num_hidden_layers: int = 0
    #: Attention kind for the draft layers. ``None`` keeps the old behaviour of
    #: running off the end of ``layer_types`` and taking its last entry — which
    #: is accidentally right when the final layer matches the draft head and
    #: silently wrong when it does not. MiMo's last layer is full attention while
    #: its draft head is windowed, so the fallback prices an O(1)-in-S head with
    #: a cache growing in S.
    mtp_layer_kind: str | None = None
    #: Draft layers whose FFN is dense. Separate from ``dense_layers`` because
    #: those index the backbone; these index the draft stack.
    mtp_dense: bool = False

    # ── collectives ──────────────────────────────────────────────────────────
    #: All-reduces a tensor-parallel layer performs. The conventional transformer
    #: fires two (post-attention, post-FFN); this family has always priced *two
    #: reductions' worth of bytes* while emitting *one node* for them. Raising
    #: this splits the bytes across that many nodes — the total is invariant, the
    #: node count is not — so a trace with 96 NCCL kernels aligns to 96 predicted
    #: nodes instead of billing half of them as unmodelled. Defaults to 1 (the
    #: long-standing behaviour); raise it per checkpoint once a trace confirms.
    all_reduces_per_layer: int = 1

    # ── precision ────────────────────────────────────────────────────────────
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    act_dtype: str = "bf16"
    #: Per-op precision, for checkpoints that run more than one width inside a
    #: single block. MiMo runs three: fp8 ``qkv_proj``, **bf16** ``attn_out_proj``
    #: (listed in ``quantization_config.ignored_layers``, and carrying no scale
    #: tensor anywhere in the index), **fp32** ``moe_router`` (an explicit
    #: ``.float()`` cast). Pricing bf16 weights at one byte understates that
    #: projection 2x, and it is the second-largest projection byte term in the
    #: block.
    #:
    #: A tuple of pairs rather than a mapping because this dataclass is frozen
    #: and therefore hashable — a ``dict`` field makes ``hash(spec)`` raise.
    op_dtype_overrides: tuple[tuple[str, str], ...] = ()

    # ── derived shapes ───────────────────────────────────────────────────────

    def layer_kind(self, layer: int) -> str:
        """``"full_attention"`` | ``"linear_attention"`` for ``layer``.

        Prefers the checkpoint's explicit schedule. The interval fallback is
        phased so the final layer is full attention, which is how these
        checkpoints are laid out — a naive ``layer % n == 0`` inverts the
        assignment and silently swaps 30 layers for 10.
        """
        if layer >= self.n_layers and self.mtp_layer_kind is not None:
            # A draft layer. Taking ``layer_types[-1]`` here reads the *backbone's
            # last* layer, which is a different mechanism on any checkpoint whose
            # draft head does not match its final block. MiMo's last layer is full
            # attention and its draft head is windowed, so the fallback prices a
            # head that is O(1) in context as one that grows with it.
            return self.mtp_layer_kind
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

    def is_sliding_attention(self, layer: int) -> bool:
        return self.layer_kind(layer) == SLIDING_ATTENTION

    def reads_kv_cache(self, layer: int) -> bool:
        """Whether ``layer`` re-reads a KV cache at all.

        True for both softmax kinds, false only for the recurrent one. *Cache or
        no cache* is the distinction this answers; *how much of the cache* is
        :meth:`attention_window`'s business. Conflating the two is what makes a
        windowed layer priced as linear lose its KV read entirely — a whole
        mechanism silently costed at zero.
        """
        return self.layer_kind(layer) in (FULL_ATTENTION, SLIDING_ATTENTION)

    def attention_window(self, layer: int) -> int:
        """Cached tokens ``layer`` may attend to; ``0`` means unbounded.

        The same window :func:`gitm.planner.moe_graph.effective_kv_tokens`
        already prices for the sparse-MoE family — one derivation, not two that
        drift apart.
        """
        if self.layer_kind(layer) != SLIDING_ATTENTION:
            return 0
        # A windowed layer whose config forgot the window is not a window layer,
        # it is global attention, and it reads everything. Returning 0 says
        # "unbounded" to the caller, which is the conservative answer; the
        # alternative reads as ``min(kv_len, 0)`` and prices the layer at zero.
        return max(0, self.sliding_window)

    def attn_geometry(self, layer: int) -> tuple[int, int, int, int]:
        """``(n_heads, kv_heads, head_dim, v_head_dim)`` for ``layer``.

        Windowed layers may carry their own GQA shape — MiMo runs 8 KV heads on
        them against 4 on its full-attention layers — and the per-layer KV width
        *is* the decode KV rate. Every ``swa_*`` field falls back to the
        full-attention value, so a checkpoint that sets none of them gets
        byte-identical arithmetic to before this method existed.
        """
        head_dim = self.head_dim
        v_head_dim = self.value_head_dim
        kv_heads = self.num_kv_heads
        if self.is_sliding_attention(layer):
            if self.swa_head_dim is not None:
                head_dim = self.swa_head_dim
            if self.swa_v_head_dim is not None:
                v_head_dim = self.swa_v_head_dim
            if self.swa_num_kv_heads is not None:
                kv_heads = self.swa_num_kv_heads
        return self.n_heads, kv_heads, head_dim, v_head_dim

    def kv_entry_elems(self, layer: int) -> int:
        """Elements one cached position costs in ``layer`` — K plus V.

        A sum, not ``2 * kv_dim``: K carries ``head_dim`` and V carries
        ``v_head_dim``, which on MiMo are 192 and 128. They coincide on every
        checkpoint that leaves ``v_head_dim`` unset, which is why this reduces to
        the old expression there.
        """
        _, kv_heads, head_dim, v_head_dim = self.attn_geometry(layer)
        return kv_heads * (head_dim + v_head_dim)

    @property
    def n_full_attention_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.is_full_attention(i))

    @property
    def n_sliding_attention_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.is_sliding_attention(i))

    @property
    def n_linear_attention_layers(self) -> int:
        return (
            self.n_layers
            - self.n_full_attention_layers
            - self.n_sliding_attention_layers
        )

    @property
    def q_dim(self) -> int:
        """Query projection width — ``n_heads * head_dim``, which may exceed hidden."""
        return self.n_heads * self.head_dim

    @property
    def value_head_dim(self) -> int:
        """Per-head V width, defaulting to ``head_dim`` when undeclared."""
        return self.v_head_dim if self.v_head_dim is not None else self.head_dim

    @property
    def o_proj_in(self) -> int:
        """Input width of the output projection — ``n_heads * v_head_dim``.

        **Not** ``q_dim``. Attention returns one ``v_head_dim``-wide result per
        query head, so on MiMo this is 64x128 = 8192 where ``q_dim`` is
        64x192 = 12288. Using ``q_dim`` therefore **overstates** this projection
        by 1.5x — the opposite direction from the intuition that a narrower V
        means fewer bytes, and worth naming because the sign decides how a
        residual on ``attn_out_proj`` reads: at 1.5x over-priced, a *near-zero*
        residual is a kernel running 1.5x slower than the hardware allows.

        Identical to ``q_dim`` whenever ``v_head_dim`` is unset.
        """
        return self.n_heads * self.value_head_dim

    @property
    def kv_dim(self) -> int:
        """Width of K (and separately V) after projection.

        Answers for the *full-attention* layers, and is kept for callers that
        predate per-layer geometry. Per-layer widths come from
        :meth:`attn_geometry`.
        """
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
    def dense_ffn_width(self) -> int:
        """FFN width for a dense block, falling back to the expert width."""
        return self.dense_intermediate_size or self.moe_intermediate_size

    def dtype_for(self, op: str, default: str) -> str:
        """Precision ``op`` runs in, falling back to ``default``.

        An empty override table returns ``default`` for everything — exactly what
        this family did before mixed precision existed in it.
        """
        for name, dtype in self.op_dtype_overrides:
            if name == op:
                return dtype
        return default

    @property
    def top_k(self) -> int:
        return min(self.num_experts_per_tok, self.num_experts)


def kv_bytes_per_token(spec: HybridMoEModelSpec) -> float:
    """KV bytes each additional token of context costs, across the whole model.

    **Only the full-attention layers contribute.** The other two kinds are both
    flat in context for different reasons — a linear layer holds a fixed-size
    recurrent state, a windowed layer holds a fixed-size buffer — and neither
    grows. Counting either here inflates the rate and caps concurrency far below
    what the hardware allows.

    A per-layer sum rather than ``n_full x per_layer``: K and V may be different
    widths, and on a checkpoint that declares neither they coincide, so this
    reduces to the old expression exactly.

    On MiMo the split is the single most important structural property of the
    model: 9 full-attention layers grow with context and 39 windowed ones do
    not, giving ``11,520 x S + 12.78M`` elements rather than a rate 4x higher.
    """
    kw = weight_bytes(spec.kv_dtype)
    return sum(
        spec.kv_entry_elems(i) * kw
        for i in range(spec.n_layers)
        if spec.is_full_attention(i)
    )


def kv_fixed_bytes_per_sequence(spec: HybridMoEModelSpec) -> float:
    """KV bytes a sequence costs however long its context grows.

    The sliding-window layers: each holds ``sliding_window`` tokens and nothing
    more, so their cost is paid once per sequence rather than per token. Charging
    them per-token is what makes a windowed model look like it cannot serve long
    contexts when in fact windowing is precisely how it does.

    Zero for a checkpoint with no windowed layers, which is every checkpoint this
    family carried before MiMo.
    """
    kw = weight_bytes(spec.kv_dtype)
    return sum(
        spec.attention_window(i) * spec.kv_entry_elems(i) * kw
        for i in range(spec.n_layers)
        if spec.is_sliding_attention(i)
    )


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

    # Layers with a dense FFN carry no expert bank and no router. Counting them
    # as MoE is not a small error: the expert term is the great majority of the
    # checkpoint, so one dense layer priced as a mixture adds ~1.6 GB that is not
    # there.
    n_moe = sum(1 for i in range(spec.n_layers) if i not in spec.dense_layers)
    n_dense_ffn = spec.n_layers - n_moe

    experts = n_moe * spec.num_experts * 3 * h * inter * ww / es
    shared = (
        n_moe * 3 * h * spec.shared_expert_intermediate_size * ww / tp
        if spec.shared_expert_intermediate_size > 0
        else 0.0
    )
    router = n_moe * h * spec.num_experts * ww  # replicated
    dense_ffn = n_dense_ffn * 3 * h * spec.dense_ffn_width * ww / tp

    gate = spec.q_dim if spec.attn_output_gate else 0

    def attn_bytes(layer: int) -> float:
        """Attention weights for one layer, at that layer's own geometry."""
        n_heads, n_kv, head_dim, v_head_dim = spec.attn_geometry(layer)
        qkv = h * (n_heads * head_dim + gate + n_kv * (head_dim + v_head_dim))
        # ``o_proj`` maps n_heads x v_head_dim back to hidden — not q_dim x
        # hidden. See ``HybridMoEModelSpec.o_proj_in``.
        out = n_heads * v_head_dim * h
        return (qkv + out) / tp

    lin_per_layer = (
        h * (spec.linear_query_dim + spec.linear_key_dim + 2 * spec.linear_value_dim)
        + spec.linear_value_dim * h
    ) / tp

    attn = sum(
        attn_bytes(i) for i in range(spec.n_layers) if spec.reads_kv_cache(i)
    )
    n_lin = spec.n_linear_attention_layers
    embed = 2.0 * spec.vocab * h / tp  # untied input embedding + lm_head

    return (
        experts
        + shared
        + router
        + dense_ffn
        + (attn + n_lin * lin_per_layer + embed) * ww
    )


def _linear(rows: float, k: int, n: int, act_b: float, w_b: float) -> tuple[float, float]:
    """(flops, bytes) for a ``(rows, k) @ (k, n)`` projection.

    Bytes count the activation in, the weights, and the activation out. At decode
    ``rows`` is small and the weight term dominates.
    """
    return 2.0 * rows * k * n, act_b * rows * k + w_b * k * n + act_b * rows * n


def _emit_full_attention(
    add, spec: HybridMoEModelSpec, *, rows: float, cache_entries: float,
    qk_pairs: float, tp: int, aw: float, ww: float, layer: int = 0,
) -> None:
    """Softmax attention over a KV cache — full-attention and windowed layers.

    ``rows`` is every query token the step computes, decode positions and
    prefill chunk together; ``qk_pairs`` is how many query-key products the core
    evaluates, which is *not* ``rows x kv_len`` once prefill is involved (see
    :attr:`BatchConfig.attention_qk_pairs`); ``cache_entries`` is how many cached
    positions are read, counted once per sequence rather than once per row.
    """
    # Per-layer geometry: a windowed layer may run a different KV head count and
    # a different V width from a full-attention one in the same model. Every
    # ``swa_*`` field falls back to the full-attention value, so a checkpoint
    # that declares none of them lands on exactly the previous arithmetic.
    n_heads, n_kv, head_dim, v_head_dim = spec.attn_geometry(layer)
    heads = max(1, n_heads // tp)
    kv_heads = max(1, n_kv // tp)
    gate = spec.q_dim if spec.attn_output_gate else 0

    # Q, K, V and the output gate come out of one fused projection in vLLM, so
    # they are one node. Splitting them would put four predicted kernels where
    # one runs and leave three permanently unmatched.
    #
    # K is ``head_dim`` wide and V is ``v_head_dim`` wide — a sum, not ``2 x``
    # one of them. They coincide unless the checkpoint declares ``v_head_dim``.
    q_width = n_heads * head_dim
    kv_width = (n_kv * head_dim + n_kv * v_head_dim) // max(1, tp)
    out_width = (q_width + gate) // tp + kv_width
    f, b = _linear(rows, spec.hidden, out_width, aw, ww)
    add("qkv_proj", f, b, spec.dtype_for("qkv_proj", spec.weight_dtype))

    # QK-norm plus *partial* RoPE. Only ``partial_rotary_factor`` of each head
    # rotates; charging the whole head is a 4x overcount on this family.
    rope_dim = int(head_dim * spec.partial_rotary_factor)
    normed = rows * (heads * head_dim + kv_heads * head_dim)
    roped = rows * (heads + kv_heads) * rope_dim
    add(
        "attn_qnorm_rope_insert",
        3.0 * normed + 6.0 * roped,
        2.0 * normed * aw + 2.0 * roped * aw
        + rows * kv_heads * (head_dim + v_head_dim) * weight_bytes(spec.kv_dtype),
        spec.act_dtype,
    )

    # The core. FLOPs follow the query-key pairs — quadratic within a prefill
    # chunk, linear at decode. Traffic follows the cached entries read, which is
    # what makes a wide step amortise cache reads across many queries and is the
    # reason prefill flips this node from memory-bound to compute-bound.
    kw = weight_bytes(spec.kv_dtype)
    # QK contracts over ``head_dim``; PV contracts over ``v_head_dim``. Equal
    # unless the checkpoint splits them.
    qk = 2.0 * qk_pairs * heads * head_dim
    pv = 2.0 * qk_pairs * heads * v_head_dim
    add(
        "attn_score_value",
        qk + pv,
        cache_entries * kv_heads * (head_dim + v_head_dim) * kw,
        spec.dtype_for("attn_score_value", spec.weight_dtype),
    )

    # ``o_proj``'s input is ``n_heads * v_head_dim`` — one V-width result per
    # query head — NOT ``q_dim``. On MiMo that is 8192 against q_dim's 12288, so
    # using q_dim OVERSTATES this projection by 1.5x. The sign matters: an
    # over-priced node makes a near-zero residual look healthy when the kernel is
    # actually running 1.5x slower than the hardware allows.
    f, b = _linear(rows, spec.o_proj_in // tp, spec.hidden, aw, ww)
    add("attn_out_proj", f, b, spec.dtype_for("attn_out_proj", spec.weight_dtype))


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
    add("linattn_in_proj", f, b, spec.dtype_for("linattn_in_proj", spec.weight_dtype))

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
    add("attn_out_proj", f, b, spec.dtype_for("attn_out_proj", spec.weight_dtype))


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

    # The router runs in fp32 on several checkpoints (an explicit ``.float()``
    # cast) while the block around it is fp8. It is a small GEMM, but fp32 has
    # its own — far lower — peak, so pricing it at the weight dtype picks the
    # wrong ceiling for it.
    f, b = _linear(positions, h, spec.num_experts, aw, ww)
    add("moe_router", f, b, spec.dtype_for("moe_router", spec.weight_dtype))

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


def _emit_dense_mlp(
    add, spec: HybridMoEModelSpec, *, positions: float, tp: int,
    aw: float, ww: float,
) -> None:
    """A plain gated FFN, for the layers that are not a mixture.

    Both MiMo's layer 0 and its three draft layers carry one of these. Charging
    them ``_emit_moe`` invents a 256-way router and an expert bank neither has —
    and because the expert term dominates a decode step, the invention is not a
    rounding error but the largest single node in the layer.

    Named ``mlp_gate_up`` / ``mlp_down`` to match the dense graph
    (:func:`gitm.planner.graph.predict_graph`): it is the same op in the same
    place, so residuals stay comparable across model families.
    """
    h = spec.hidden
    ff = spec.dense_ffn_width // max(1, tp)
    add(
        "mlp_gate_up",
        2.0 * 2.0 * positions * h * ff,
        aw * (positions * h + 2.0 * positions * ff) + ww * (2.0 * h * ff),
        spec.dtype_for("mlp_gate_up", spec.weight_dtype),
    )
    add(
        "mlp_down",
        2.0 * positions * ff * h,
        aw * (positions * ff + positions * h) + ww * (ff * h),
        spec.dtype_for("mlp_down", spec.weight_dtype),
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

    if spec.reads_kv_cache(layer):
        # A windowed layer reads a KV cache like a full-attention one, but only
        # the last ``window`` entries of it. Priced as linear its cache read
        # would vanish; priced as unbounded it is overstated by kv_len/window —
        # 64x at 8k context on a 128-token window.
        window = spec.attention_window(layer)
        cached = min(kv_len, window) if window > 0 else kv_len
        # Cached entries read, counted once per sequence rather than once per
        # row. Prefill re-reads its own chunk as it is written, which is why the
        # chunk appears here as well as its prior context — both capped by the
        # window when there is one.
        cache_entries = float(sequences * cached)
        if batch.is_prefill:
            chunk = batch.prefill_context + batch.prefill_tokens
            cache_entries += min(chunk, window) if window > 0 else chunk
        _emit_full_attention(
            add, spec, rows=rows, cache_entries=cache_entries,
            qk_pairs=batch.attention_qk_pairs(window), tp=tp, aw=aw, ww=ww,
            layer=layer,
        )
    else:
        _emit_linear_attention(
            add, spec, positions=positions, sequences=sequences,
            prefill_tokens=batch.prefill_tokens,
            prefill_requests=batch.prefill_requests, tp=tp, aw=aw, ww=ww,
        )

    if layer in spec.dense_layers or (layer >= spec.n_layers and spec.mtp_dense):
        _emit_dense_mlp(add, spec, positions=rows, tp=tp, aw=aw, ww=ww)
    else:
        _emit_moe(add, spec, positions=rows, sh=sh, aw=aw, ww=ww)

    if tp > 1:
        # Priced against the interconnect, not HBM. Bandwidth-only, so optimistic
        # at decode message sizes; a SKU with no interconnect figure leaves these
        # at zero, which Graph.has_unpriced_collectives surfaces.
        link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)
        name = f"{prefix}tp_all_reduce"
        # A conventional TP layer fires two reductions (post-attention,
        # post-FFN). This family has always priced two reductions' *bytes* — the
        # leading 2.0 that used to sit in the expression below — while emitting
        # ONE node for them. Splitting the bytes across ``all_reduces_per_layer``
        # nodes leaves the total invariant and makes the node count match the
        # kernel count, so a trace with 96 NCCL kernels stops billing half of
        # them as unmodelled work.
        count = max(1, spec.all_reduces_per_layer)
        per_node = (2.0 / count) * (2.0 * (tp - 1) / tp) * rows * spec.hidden * aw
        for _ in range(count):
            g.nodes.append(
                PredictedNode(
                    name, layer,
                    roofline(
                        name, 0.0, per_node,
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
    # Only meaningful when the model actually has recurrent layers. A windowed
    # hybrid has none, and refusing its sharding over a head count that governs
    # no kernel would reject a perfectly valid deployment.
    if spec.n_linear_attention_layers and spec.linear_num_value_heads % max(1, sh.tp) != 0:
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

    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    lm_dtype = spec.dtype_for("lm_head", spec.weight_dtype)

    def emit_lm_head(rows: float, depends_on: tuple[int, ...] = ()) -> int:
        """Append the vocabulary projection and return its node index."""
        f, b = _linear(rows, spec.hidden, spec.vocab // max(1, sh.tp), aw, ww)
        g.nodes.append(
            PredictedNode(
                "lm_head", None, roofline("lm_head", f, b, hw, lm_dtype),
                depends_on=depends_on,
            )
        )
        return len(g.nodes) - 1

    # Only the final token of a prompt needs logits, so an 8192-token prefill
    # chunk contributes ONE row here, not 8192. Charging every prefill token
    # against a 248k-row vocabulary projection is the single largest overcount
    # available on this path — it would make lm_head dominate a prefill step.
    last = emit_lm_head(batch.logits_rows)

    if with_mtp:
        # Draft layers run one position per sequence. Deliberately not given
        # distinct op names — an MTP layer's qkv projection is a qkv projection;
        # what makes it the draft head is its layer index.
        #
        # **One lm_head per stage.** Each draft stage must sample a token before
        # the next stage can consume it, so every stage runs its own vocabulary
        # projection. Emitting a single lm_head for the whole step leaves D of
        # them unmodelled — on a 152k-vocabulary model at TP4 that is ~312 MB of
        # weight traffic per missing stage, every step.
        #
        # The chain is also the one place in this graph where a dependency edge
        # is *required* rather than incidental: stage i+1 genuinely cannot begin
        # until stage i has produced a token. Recording it lets a positive
        # residual here be read as architectural serialisation rather than as
        # recoverable scheduling slack.
        for i in range(spec.mtp_num_hidden_layers):
            first = len(g.nodes)
            _emit_layer(
                g, spec, hw, spec.n_layers + i,
                positions=sequences, sequences=sequences, kv_len=kv_len, sh=sh,
                # The draft head runs one position per sequence and never
                # prefills: it proposes tokens, it does not ingest a prompt.
                batch=replace(batch, prefill_tokens=0),
            )
            # The stage's first node waits on the previous stage's logits.
            if first < len(g.nodes):
                g.nodes[first].depends_on = (last,)
            last = emit_lm_head(sequences, depends_on=(len(g.nodes) - 1,))

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


#: Checkpoints spell the same two facts differently. Aliased rather than
#: special-cased so a third checkpoint with a fourth spelling is one line.
_EXPERT_KEYS = ("num_experts", "n_routed_experts")
_SCHEDULE_KEYS = ("layer_types", "hybrid_layer_pattern")


def _first(cfg: dict[str, Any], keys: tuple[str, ...]):
    """The first of ``keys`` this config declares, or ``None``."""
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return None


def _schedule(cfg: dict[str, Any]) -> tuple[str, ...]:
    """The per-layer attention schedule, normalised to this module's names.

    Two spellings, and **they disagree about polarity**:

    * ``layer_types`` — already strings (``full_attention`` /
      ``linear_attention``), used as-is.
    * ``hybrid_layer_pattern`` — integers, and **1 means WINDOWED, 0 means
      FULL** (``modeling_mimo_v2.py:400``: ``is_swa_layer = pattern[i] == 1``).
      That is the opposite of the intuitive reading, and inverting it silently
      swaps 39 windowed layers for 9 — producing a graph with the right node
      count, a plausible total, and every per-layer residual compared against
      the wrong mechanism.
    """
    types = cfg.get("layer_types")
    if isinstance(types, list | tuple) and types:
        return tuple(str(t) for t in types)
    pattern = cfg.get("hybrid_layer_pattern")
    if isinstance(pattern, list | tuple) and pattern:
        return tuple(
            SLIDING_ATTENTION if int(p) == 1 else FULL_ATTENTION for p in pattern
        )
    return ()


def is_hybrid_moe_config(cfg: dict[str, Any]) -> bool:
    """True for the hybrid-schedule MoE checkpoints this module models.

    The discriminator is a *mixed* layer schedule: routed experts alone describe
    a Mixtral, and one uniform attention kind describes a conventional MoE. Only
    a checkpoint that interleaves two attention mechanisms — linear with full
    (Qwen3.6) or windowed with full (MiMo) — has the two-asymptotic structure
    this graph exists to price.

    ``n_routed_experts`` is accepted alongside ``num_experts``, which matters
    because **DeepSeek-V4 declares it too** and ``detect_family`` consults this
    predicate first. V4 declares no schedule under either spelling and no
    ``full_attention_interval``, so it still falls through to the sparse-MoE
    family — but that is a property worth a test rather than a comment.
    """
    text = _text_config(cfg)
    routed = _first(text, _EXPERT_KEYS) and text.get("num_experts_per_tok")
    if not routed:
        return False
    if len(set(_schedule(text))) > 1:
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

    def _req_any(keys: tuple[str, ...]) -> int:
        """A required field that more than one checkpoint spells differently.

        Raises naming the canonical spelling *and* the aliases, so the message
        says what to look for rather than only what was absent.
        """
        v = _first(text, keys)
        if v is None:
            raise ValueError(
                f"config declares no {keys[0]!r} (nor any of {list(keys[1:])}); "
                "refusing to substitute a default, which would predict a model "
                "this checkpoint is not"
            )
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"config field {keys[0]!r} is not an integer: {v!r}") from None

    layer_types = _schedule(text)

    # ``moe_layer_freq`` marks which layers carry a mixture; a falsy entry is a
    # dense block. MiMo leaves layer 0 dense. Charging it a 256-expert router and
    # an expert bank it does not have is the same error as G7's draft head, and
    # the expert term dominates the layer.
    freq = text.get("moe_layer_freq")
    dense_layers = (
        frozenset(i for i, v in enumerate(freq) if not v)
        if isinstance(freq, list | tuple)
        else frozenset()
    )

    dtype = str(text.get("dtype") or text.get("torch_dtype") or "bf16").lower()
    if dtype.startswith("float16") or dtype == "half":
        dtype = "fp16"
    elif dtype.startswith("bfloat"):
        dtype = "bf16"
    elif dtype.startswith("float32"):
        dtype = "fp32"

    # Weights and activations are genuinely different widths on a quantised
    # checkpoint, and this reader used to conflate them because the only member
    # of the family was bf16 throughout. MiMo stores fp8 weights under a
    # bfloat16 ``dtype``, so reading the torch dtype for weights prices the
    # entire checkpoint at 2 bytes and doubles the footprint — 617 GB against a
    # 315 GB index. ``quantization_config.quant_method`` is where the weight
    # width actually lives; the torch dtype governs activations.
    q = cfg.get("quantization_config") or text.get("quantization_config") or {}
    weight_dtype = str(q.get("quant_method") or dtype).lower()

    ssm = str(text.get("mamba_ssm_dtype", "fp32")).lower()
    ssm = "fp32" if ssm.startswith("float32") or ssm == "fp32" else ssm

    rotary = text.get("partial_rotary_factor")
    if rotary is None:
        params = text.get("rope_parameters")
        if isinstance(params, dict):
            rotary = params.get("partial_rotary_factor")

    hidden = _req("hidden_size")
    n_heads = _req("num_attention_heads")

    # Linear-attention geometry exists only on the gated-DeltaNet checkpoints. A
    # windowed hybrid has no recurrent layers at all, so demanding these fields
    # would refuse a config that is simply a different member of the family.
    #
    # The discriminator is the *schedule*, deliberately not the presence of the
    # fields themselves: keying off ``linear_num_value_heads`` would make the
    # check disable itself exactly when the field is missing, which is the one
    # case it exists to catch. Absent a schedule the fields stay required, since
    # nothing has established that this checkpoint has no recurrent layers.
    has_linear = (not layer_types) or LINEAR_ATTENTION in layer_types
    if has_linear:
        n_value_heads = _req("linear_num_value_heads")
        value_head_dim = _req("linear_value_head_dim")
    else:
        n_value_heads, value_head_dim = 1, 1

    def _opt_int(key: str) -> int | None:
        """A field whose absence means "same as the full-attention layers"."""
        v = text.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    # A windowed checkpoint may declare the window under either spelling.
    window = _opt_int("sliding_window") or _opt_int("sliding_window_size") or 0

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
        v_head_dim=_opt_int("v_head_dim"),
        sliding_window=window,
        swa_num_kv_heads=_opt_int("swa_num_key_value_heads"),
        swa_head_dim=_opt_int("swa_head_dim"),
        swa_v_head_dim=_opt_int("swa_v_head_dim"),
        swa_rope_theta=float(text.get("swa_rope_theta") or 0.0),
        swa_attention_sink=bool(text.get("add_swa_attention_sink_bias", False)),
        num_experts=_req_any(_EXPERT_KEYS),
        num_experts_per_tok=_req("num_experts_per_tok"),
        moe_intermediate_size=_req("moe_intermediate_size"),
        shared_expert_intermediate_size=_int("shared_expert_intermediate_size", 0),
        dense_intermediate_size=_int("intermediate_size", 0),
        dense_layers=dense_layers,
        layer_types=layer_types,
        full_attention_interval=_int("full_attention_interval", 1),
        # MiMo ships three draft layers but declares them nowhere in
        # ``config.json`` — they are only visible as tensors in the safetensors
        # index. Defaulting to 0 from a bare config is the honest answer: the
        # catalogue entry carries the real count, and inventing one here would
        # predict a head this reader never saw.
        mtp_num_hidden_layers=_int("mtp_num_hidden_layers", 0),
        weight_dtype=weight_dtype,
        # The cache dtype is a *serving* decision, not a model fact: no
        # checkpoint declares it. The torch dtype is the reading the config
        # supports, and it is the conservative one — an fp8 cache would halve
        # this. Tracked as an open question rather than assumed away.
        kv_dtype=dtype,
        act_dtype=dtype,
    )
