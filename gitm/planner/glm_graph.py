"""Predicted execution graph for a GLM-5.2-class (``glm_moe_dsa``) decode step.

A fork of :mod:`gitm.planner.moe_graph`, specialised for ZhipuAI's
``GlmMoeDsaForCausalLM``. Both families are sparse-MoE with a lightning indexer,
but the attention differs in kind, not degree, so a shared spec would carry a
field for each that is dead on the other — the exact "a default silently
activates" hazard the roofline module warns about. The two graphs therefore live
apart, and share only the canonical op names (so residuals stay comparable) and
the :func:`~gitm.planner.roofline.distinct_experts` union term (one owner, no
drift).

What GLM-5.2 is, read from ``config.json`` and the checkpoint's own tensor index
(``model.safetensors.index.json``), never from a trace:

**Attention is MLA + DeepSeek Sparse Attention, with no per-layer compression.**
Unlike DeepSeek-V4's CSA/HCA schedule (``compress_ratios``), every GLM layer runs
the *same* attention: a compressed KV latent (``kv_lora_rank=512``, shared across
all 64 query heads) plus a lightning indexer that scores the whole history and
keeps ``index_topk=2048`` positions for the core. There is no ``m``/``m'`` split,
no HCA, no sliding window. So the V4 fields that encode that schedule are simply
absent here.

**IndexShare — the mechanism this fork exists to price.** The checkpoint declares
``indexer_types`` per layer: ``full`` or ``shared``. A ``full`` layer computes its
own indexer (projection + score) and selects the top-k; the next three ``shared``
layers *reuse that selection* and run no indexer at all. This is not an inference:
the ``shared`` layers physically **carry no indexer tensors** in the weight map.
Only 21 of 78 layers run the indexer; 57 skip it. ``index_topk_freq=4`` is the
period of that grouping (one full + three shared), and ``index_skip_topk_offset``
its offset. Pricing every layer's indexer at full rate — as a naive reading of
``index_topk`` would — overstates the indexer's share of the step roughly 4x and
mis-ranks it against the MoE weight traffic that actually dominates decode.

**The MLP schedule is dense-then-sparse.** ``first_k_dense_replace=3``: the first
three layers run a conventional dense FFN (``intermediate_size=12288``) with no
router and no experts; the remaining 75 are MoE (256 routed experts, top-8, one
shared, ``moe_intermediate_size=2048``). Modelling the dense layers as MoE would
invent a router GEMM and expert traffic that the weight map shows are not there.

**Three precisions in one block, and which op runs in which is read from the
checkpoint.** ``zai-org/GLM-5.2`` is bf16 (1.507 TB on disk, no
``quantization_config``); ``zai-org/GLM-5.2-FP8`` is the vendor's *recommended*
deployment (753.33 GB, e4m3, 128x128 block-scaled) and its
``quantization_config.modules_to_not_convert`` names what stays wide. On the FP8
checkpoint the backbone GEMMs — ``q_a``/``q_b``/``kv_a``/``kv_b``, **``o_proj``**,
the dense FFN, the shared expert and all 256 routed experts — are fp8, while
``lm_head``, ``embed_tokens``, the MTP ``eh_proj``, every norm and — the outlier
worth naming — the **lightning indexer's projections** are bf16. The router is
fp32 on both checkpoints (``moe_router_dtype: "float32"``, a field of the *base*
config, so it is a model fact and not a quantisation choice).

That is three widths inside one attention block, which one ``weight_dtype`` per
spec cannot say. :attr:`GlmMoeDsaModelSpec.op_dtype_overrides` says it per op, and
it is load-bearing in both directions: pricing ``lm_head`` at 1 byte/weight
understates 154,880 x 6,144 of real traffic, and pricing the indexer at fp8
understates the one node whose cost grows with context. Note the inversion against
the fp8-backbone models this planner has seen before — here ``o_proj`` is *inside*
the quantised set and the *indexer* is outside it.

Known limits, stated rather than hidden:

* **Prefill is modelled, and it is not decode with a bigger M.** DSA makes the two
  phases disagree about what ``index_topk`` buys: at decode the core reads at most
  ``index_topk`` cached entries, so attention is flat in context; at prefill every
  query in the chunk selects a *different* top-k, so their union is the whole
  history and the core streams the entire cache once per request. Top-k bounds
  prefill FLOPs, not prefill bytes. See :func:`core_qk_pairs` /
  :func:`core_read_entries`.
* **Uniform routing.** :func:`~gitm.planner.roofline.distinct_experts` assumes a
  balanced router; real skew touches fewer distinct experts, moving *less* traffic
  than predicted — the conservative direction.
* **Expert-parallel skew is calibrated, not predicted.**
  :attr:`ShardingConfig.ep_imbalance` stays at 1.0 until a trace measures it.
* **Collectives are bandwidth-plus-launch**: a ring latency floor of one launch
  each, which is what a 262 kB decode all-reduce is actually bounded by. Still
  flagged ``estimated``, and still reported unpriced when the SKU carries no
  interconnect bandwidth.
* **The MTP economics are a prediction, not a measurement.** Acceptance rate is a
  serving observable; the graph prices the *cost* of D drafts and a 1+D verify and
  leaves the payoff to :attr:`BatchConfig.tokens_per_step`.
"""

from __future__ import annotations

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

FULL_INDEXER = "full"
SHARED_INDEXER = "shared"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


@dataclass(frozen=True)
class GlmMoeDsaModelSpec:
    """Model shape for a GLM-5.2-class (``glm_moe_dsa``) decode step.

    **The defaults below are a small reference shape, not any real checkpoint** —
    deliberately far too small to be mistaken for GLM-5.2. Real checkpoints live in
    ``gitm/planner/models/*.yaml`` (``family: glm_moe_dsa``) or are read from a
    config by :func:`spec_from_hf_config`. The comments cite GLM-5.2 as the worked
    example precisely where its values differ from these defaults in ways that
    matter.
    """

    name: str = "glm-moe-dsa-reference"
    hidden: int = 512
    n_layers: int = 6
    vocab: int = 2048

    # ── MLA attention ────────────────────────────────────────────────────────
    n_heads: int = 8
    #: Query down-projection rank. The query goes ``hidden -> q_lora_rank ->
    #: n_heads * q_head_dim``; the middle rank is a real, replicated matrix.
    q_lora_rank: int = 256
    #: Compressed KV latent width. The cache holds **one** latent per token per
    #: layer, shared across every query head — deriving KV traffic from
    #: ``num_key_value_heads * head_dim`` instead is the classic MLA error and
    #: overstates it by ``n_heads`` (64x on GLM-5.2).
    kv_lora_rank: int = 128
    #: Per-head query/key width carrying no rotary embedding.
    qk_nope_head_dim: int = 96
    #: Per-head query/key width carrying rotary embedding. The decoupled RoPE key
    #: is shared (one per token, MQA-style) alongside the latent in the cache.
    qk_rope_head_dim: int = 32
    #: Per-head value width. On GLM-5.2 this (256) differs from ``qk_nope`` (192),
    #: so the score and the value read use different per-head widths.
    v_head_dim: int = 128

    # ── DeepSeek Sparse Attention indexer ────────────────────────────────────
    index_n_heads: int = 16
    index_head_dim: int = 64
    #: Positions the indexer keeps for the attention core. The core read is
    #: bounded by this once history exceeds it, so attention stops growing with
    #: context while the *indexer scan* keeps growing — a different node, a
    #: different bound.
    index_topk: int = 512
    #: Period of the IndexShare grouping: one ``full`` layer that computes the
    #: selection, then ``index_topk_freq - 1`` ``shared`` layers that reuse it.
    index_topk_freq: int = 4
    #: Per-layer ``full`` | ``shared``, straight from the checkpoint. Authoritative
    #: when present — the ``shared`` layers carry no indexer weights, so this is
    #: read, not a rule guessed from the frequency. Empty falls back to
    #: :attr:`index_topk_freq` (first layer of each group is ``full``).
    indexer_types: tuple[str, ...] = ()

    # ── mixture of experts ───────────────────────────────────────────────────
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 256
    #: Dense-FFN width for the leading ``first_k_dense_replace`` layers.
    intermediate_size: int = 768
    #: Leading layers that run a dense FFN instead of the mixture (GLM
    #: ``first_k_dense_replace``). Those layers carry no router and no experts.
    first_k_dense_replace: int = 1
    #: Per-layer ``dense`` | ``sparse`` when the checkpoint declares it; overrides
    #: :attr:`first_k_dense_replace` if both are given.
    mlp_layer_types: tuple[str, ...] = ()
    #: Scales routed-expert outputs (GLM ``routed_scaling_factor``). Numerics only —
    #: no effect on the FLOP/byte roofline, carried for completeness.
    routed_scaling_factor: float = 1.0

    # ── multi-token prediction ───────────────────────────────────────────────
    num_nextn_predict_layers: int = 0
    #: The MTP layer reuses the main model's index rather than recomputing it
    #: (GLM ``index_share_for_mtp_iteration``). The tensor exists but the iteration
    #: shares — carried as a headroom lever, not banked into the floor.
    index_share_for_mtp_iteration: bool = True

    # ── precision ────────────────────────────────────────────────────────────
    weight_dtype: str = "bf16"
    expert_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    act_dtype: str = "bf16"
    #: Per-op precision, for the ops that do not run in :attr:`weight_dtype`.
    #: ``(op_name, dtype)`` pairs — a tuple rather than a mapping so the spec
    #: stays hashable, matching how the per-layer schedules are carried.
    #:
    #: One dtype per model is a fiction on this checkpoint. GLM-5.2-FP8 quantises
    #: the backbone GEMMs and the experts to e4m3 but leaves ``lm_head``, the MTP
    #: ``eh_proj`` and the **lightning indexer** in bf16
    #: (``quantization_config.modules_to_not_convert``), and the router is fp32 on
    #: every variant (``moe_router_dtype``). Without this the indexer — the one
    #: attention node whose cost grows with context — is priced at half its real
    #: weight traffic and against a peak it never sees.
    op_dtype_overrides: tuple[tuple[str, str], ...] = ()

    # ── derived shapes / schedule ────────────────────────────────────────────

    @property
    def q_head_dim(self) -> int:
        """Per-head query/key width: the nope part plus the RoPE part."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_entry_dim(self) -> int:
        """Elements one cached KV entry occupies: the latent plus the shared RoPE key.

        One per token per layer, shared across every query head — the whole point
        of MLA. Not multiplied by ``n_heads`` or ``num_kv_heads``.
        """
        return self.kv_lora_rank + self.qk_rope_head_dim

    def dtype_for(self, op: str, default: str) -> str:
        """Precision ``op`` runs in — an override if the checkpoint declares one.

        ``default`` is the op's family dtype (:attr:`weight_dtype` for
        projections, :attr:`expert_dtype` for the mixture), so an entry is needed
        only where the checkpoint departs from it.
        """
        for name, dtype in self.op_dtype_overrides:
            if name == op:
                return dtype
        return default

    def mlp_kind(self, layer: int) -> str:
        """``"dense"`` | ``"sparse"`` for ``layer``'s FFN."""
        if self.mlp_layer_types and layer < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer]
        return DENSE_MLP if layer < self.first_k_dense_replace else SPARSE_MLP

    def is_sparse_mlp(self, layer: int) -> bool:
        return self.mlp_kind(layer) == SPARSE_MLP

    def indexer_kind(self, layer: int) -> str:
        """``"full"`` | ``"shared"`` — whether ``layer`` computes its own index.

        Prefers the checkpoint's explicit ``indexer_types``. The frequency fallback
        makes the first layer of each ``index_topk_freq``-sized group ``full`` and
        the rest ``shared``, which is how GLM-5.2 is laid out past its dense prefix.
        """
        if self.indexer_types and layer < len(self.indexer_types):
            return self.indexer_types[layer]
        n = max(1, self.index_topk_freq)
        return FULL_INDEXER if layer % n == 0 else SHARED_INDEXER

    def is_full_indexer(self, layer: int) -> bool:
        return self.indexer_kind(layer) == FULL_INDEXER

    @property
    def n_full_indexer_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.is_full_indexer(i))

    @property
    def n_sparse_mlp_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.is_sparse_mlp(i))

    @property
    def top_k(self) -> int:
        return min(self.num_experts_per_tok, self.n_routed_experts)


def kv_entry_bytes(spec: GlmMoeDsaModelSpec) -> float:
    """Bytes one cached KV entry occupies.

    The latent plus the decoupled RoPE key. DeepSeek-family checkpoints keep the
    RoPE dimensions in bf16 while the latent may be quantised; on a pure-bf16
    checkpoint both are 2 bytes and this reduces to ``kv_entry_dim * 2``. The split
    is kept so an fp8-KV *serving* config prices the two halves correctly.
    """
    rope = spec.qk_rope_head_dim * weight_bytes("bf16")
    latent = spec.kv_lora_rank * weight_bytes(spec.kv_dtype)
    return latent + rope


def effective_kv_tokens(spec: GlmMoeDsaModelSpec, kv_len: int) -> int:
    """KV positions the attention *core* reads — bounded by the indexer's top-k.

    Every GLM layer runs DSA, so the core reads at most ``index_topk`` selected
    positions regardless of how long the context grows. This is why attention is
    flat in context on this architecture and the term that still grows is the
    indexer scan, not the core.
    """
    if kv_len <= 0:
        return 0
    return min(kv_len, spec.index_topk)


def index_candidates(spec: GlmMoeDsaModelSpec, kv_len: int) -> int:
    """Positions the indexer must score — the whole (uncompressed) history.

    GLM does not compress before selecting, so the indexer scores every past
    token. This is the only term in the attention path that grows with context,
    and it is paid on ``full`` layers alone.
    """
    return max(0, kv_len)


def _capped_prefix_sum(context: int, tokens: int, cap: int) -> float:
    """``sum(min(context + i, cap) for i in 1..tokens)`` in closed form.

    How many positions a causal chunk of ``tokens`` queries actually attends to
    when each query is capped at ``cap`` selected keys. Written out rather than
    looped because a prefill chunk is 8,192 queries and this is called per layer.
    """
    if tokens <= 0:
        return 0.0
    c, n, k = float(context), float(tokens), float(cap)
    # Queries whose own history is still under the cap attend to all of it.
    uncapped = max(0.0, min(n, k - c))
    total = uncapped * (2.0 * c + uncapped + 1.0) / 2.0
    return total + (n - uncapped) * k


def core_qk_pairs(spec: GlmMoeDsaModelSpec, batch: BatchConfig) -> float:
    """Query-key products the attention *core* evaluates this step.

    Deliberately **not** :attr:`BatchConfig.attention_qk_pairs`, which is the
    dense causal count. DSA hands the core at most ``index_topk`` selected
    positions per query, so past 2,048 tokens of context the core is linear in
    context where a dense model is quadratic — the whole point of the
    architecture, and the term a mechanical copy of another family's prefill path
    would get wrong by ``kv_len / index_topk`` (64x at 128K).
    """
    k = spec.index_topk
    pairs = batch.positions_per_step * float(min(batch.kv_cache_len, k))
    if batch.is_prefill:
        pairs += _capped_prefix_sum(batch.prefill_context, batch.prefill_tokens, k)
    return pairs


def core_read_entries(spec: GlmMoeDsaModelSpec, batch: BatchConfig) -> float:
    """Cached KV entries the core actually reads — the phases disagree here.

    **Decode:** one query row per sequence, reading its own top-``index_topk``
    selection. ``sequences x min(kv_len, index_topk)`` — flat in context.

    **Prefill:** every query in the chunk selects a *different* top-k, and their
    union over a few thousand queries is the whole history. The kernel therefore
    streams the entire cache once per request: top-k bounds prefill *FLOPs*, not
    prefill *bytes*. Charging ``P x index_topk`` here instead would understate
    long-context prefill traffic by the ratio of context to 2,048, which is the
    single most inviting error on this architecture.
    """
    entries = batch.batch * float(min(batch.kv_cache_len, spec.index_topk))
    if batch.is_prefill:
        entries += batch.prefill_requests * float(
            batch.prefill_context + batch.prefill_tokens
        )
    return entries


def index_scan_pairs(batch: BatchConfig) -> float:
    """Query-key products the *indexer* scores — the uncompressed history.

    GLM does not compress before selecting, so every past token is scored. This
    is the dense causal count, and it is the term that carries the quadratic at
    prefill and the growth-in-context at decode. Paid on ``full`` layers only,
    which is what IndexShare is worth.
    """
    return batch.attention_qk_pairs


def index_scan_entries(batch: BatchConfig) -> float:
    """Cached index keys read per step: the whole history, once per sequence."""
    entries = float(batch.batch * batch.kv_cache_len)
    if batch.is_prefill:
        entries += batch.prefill_requests * float(
            batch.prefill_context + batch.prefill_tokens
        )
    return entries


def model_weight_bytes(
    spec: GlmMoeDsaModelSpec, sharding: ShardingConfig | None = None
) -> float:
    """Resident weight bytes on one rank.

    Decides whether a deployment shape fits at all, which the timing graph cannot.
    Experts dominate overwhelmingly: 75 sparse layers x 256 experts x three
    ``hidden x moe_intermediate`` matrices is the great majority of the checkpoint,
    and it is what makes a ~753B model activate ~25B per token.

    Validated against ground truth: GLM-5.2 predicts within a few percent of the
    published 1.507 TB checkpoint (``model.safetensors.index.json`` ``total_size``),
    the residual being norms, biases and the MTP head this rolls in coarsely.
    """
    sh = sharding or ShardingConfig()
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)
    ew = weight_bytes(spec.dtype_for("moe_routed", spec.expert_dtype))
    sw = weight_bytes(spec.dtype_for("moe_shared", spec.expert_dtype))
    ww = weight_bytes(spec.weight_dtype)
    rw = weight_bytes(spec.dtype_for("moe_router", spec.weight_dtype))
    iw = weight_bytes(spec.dtype_for("attn_index_proj", spec.weight_dtype))
    lw = weight_bytes(spec.dtype_for("lm_head", spec.weight_dtype))
    h = spec.hidden
    inter = spec.moe_intermediate_size

    n_sparse = spec.n_sparse_mlp_layers + spec.num_nextn_predict_layers
    n_dense = spec.n_layers - spec.n_sparse_mlp_layers
    n_full_idx = spec.n_full_indexer_layers + spec.num_nextn_predict_layers
    n_attn = spec.n_layers + spec.num_nextn_predict_layers

    experts = n_sparse * spec.n_routed_experts * 3 * h * inter * ew / es
    shared_exp = n_sparse * spec.n_shared_experts * 3 * h * inter * sw / tp
    router = n_sparse * h * spec.n_routed_experts * rw  # replicated
    dense_ffn = n_dense * 3 * h * spec.intermediate_size * ww / tp

    # MLA projections, per attention layer. q_a and kv_a are replicated (they
    # produce the shared latent, which has nothing to split); q_b, o_proj and the
    # per-head value up-projection shard on heads.
    attn_per_layer = (
        h * spec.q_lora_rank  # q_a, replicated
        + spec.q_lora_rank * spec.n_heads * spec.q_head_dim / tp  # q_b
        + h * spec.kv_entry_dim  # kv_a (latent + rope key), replicated
        + spec.kv_lora_rank * spec.n_heads * (spec.qk_nope_head_dim + spec.v_head_dim) / tp  # kv_b
        + spec.n_heads * spec.v_head_dim * h / tp  # o_proj
    )
    # Indexer weights live only on ``full`` layers (proven: ``shared`` layers carry
    # none). Replicated across ranks, as vLLM builds the indexer ReplicatedLinear.
    # ``wq_b`` from the query latent, ``wk`` from hidden, and the per-head gate.
    indexer_per_full = (
        spec.q_lora_rank * spec.index_n_heads * spec.index_head_dim
        + h * spec.index_head_dim
        + h * spec.index_n_heads
    )

    # Untied input embedding and vocabulary projection. Both stay wide on the FP8
    # checkpoint (``embed_tokens`` and ``lm_head`` are in ``modules_to_not_convert``),
    # so they are priced at their own width rather than the backbone's — 1.9 GB of
    # the resident footprint that an fp8 read would halve on paper and not on disk.
    embed = 2.0 * spec.vocab * h * lw / tp

    return (
        experts
        + shared_exp
        + router
        + n_full_idx * indexer_per_full * iw
        + embed
        + (n_attn * attn_per_layer + dense_ffn) * ww
    )


def kv_bytes_per_token(spec: GlmMoeDsaModelSpec) -> float:
    """KV bytes each additional token of context costs, across the whole model.

    One MLA latent (plus the indexer key on ``full`` layers) per token per layer.
    Flat across layers — there is no compression schedule to sum over — but the
    indexer key is only cached where an indexer runs.
    """
    kw = weight_bytes(spec.kv_dtype)
    latent = spec.n_layers * kv_entry_bytes(spec)
    index_keys = spec.n_full_indexer_layers * spec.index_head_dim * kw
    return latent + index_keys


def _linear(rows: float, k: int, n: int, act_b: float, w_b: float) -> tuple[float, float]:
    """(flops, bytes) for a ``(rows, k) @ (k, n)`` projection.

    Bytes count the activation in, the weights, and the activation out. At decode
    ``rows`` is small and the weight term dominates — which is why weight dtype,
    not activation dtype, sets the floor for every projection here.
    """
    return 2.0 * rows * k * n, act_b * rows * k + w_b * k * n + act_b * rows * n


def _emit_layer(
    g: Graph,
    spec: GlmMoeDsaModelSpec,
    hw: HardwareSpec,
    layer: int,
    *,
    batch: BatchConfig,
    sh: ShardingConfig,
    prefix: str = "",
    force_full_indexer: bool | None = None,
) -> None:
    """Append one transformer layer's predicted nodes to ``g``.

    ``batch`` is the phase this layer runs in, already adjusted by the caller: a
    decode step, a chunked-prefill step, or a draft stage (prefill stripped). The
    node *set* is identical across all three — what changes is the class of four
    of them, which is why phase is a parameter here rather than a second emitter.

    Three row counts, and conflating any two is a real error:

    ``rows``
        Sequence positions this layer computes — decode positions (one per
        sequence per speculative slot) plus whatever prefill chunk rides along.
        Every projection and every FFN scales with this.
    ``batch.batch``
        Distinct KV caches those rows read from. Cache traffic is charged per
        *sequence*, not per row, which is why verifying 1+D drafted positions in
        one step reads the cache once — and is what makes MTP pay at all on a
        memory-bound step.
    ``batch.prefill_requests``
        Prompts the prefill tokens belong to — the denominator for anything read
        once per request rather than once per token.
    """
    h = spec.hidden
    aw = weight_bytes(spec.act_dtype)
    wd, ed = spec.weight_dtype, spec.expert_dtype
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)

    rows = float(batch.positions_per_step + batch.prefill_tokens)

    def add(
        op: str, flops: float, byts: float, dtype: str,
        *, estimated: bool = False, serial_launches: int = 1,
    ) -> None:
        """Emit one node, at the precision the checkpoint says the op runs in.

        ``serial_launches`` defaults to 1 because every node here *is* one
        dependent kernel launch: it consumes the previous node's output, so its
        wall time cannot fall below the launch overhead however few bytes it
        moves. On a decode step that floor is what the small pointwise and
        routing nodes are actually bounded by, and omitting it does not make the
        prediction slightly optimistic — it makes a whole bound label absent.
        """
        name = f"{prefix}{op}"
        g.nodes.append(
            PredictedNode(
                name, layer,
                roofline(
                    name, flops, byts, hw, spec.dtype_for(op, dtype),
                    estimated=estimated, serial_launches=serial_launches,
                ),
            )
        )

    def w_bytes(op: str, default: str) -> float:
        """Bytes per stored weight for ``op``, after any precision override."""
        return weight_bytes(spec.dtype_for(op, default))

    # ── MLA attention: low-rank query, compressed KV latent ──────────────────
    # q_a and kv_a are replicated across TP ranks: they produce the shared latent,
    # which has nothing to split when there is one KV latent. Every rank pays them
    # in full, so TP's speedup on attention is strictly less than ``tp``.
    f, b = _linear(rows, h, spec.q_lora_rank, aw, w_bytes("attn_q_a", wd))
    add("attn_q_a", f, b, wd)

    f, b = _linear(
        rows, spec.q_lora_rank, spec.n_heads * spec.q_head_dim // tp, aw,
        w_bytes("attn_q_b", wd),
    )
    add("attn_q_b", f, b, wd)

    # The compressed latent plus the decoupled RoPE key, and the cache write for
    # the positions just computed. One projection (no CSA/HCA overlap here).
    f, b = _linear(rows, h, spec.kv_entry_dim, aw, w_bytes("attn_kv_a", wd))
    add("attn_kv_a", f, b + rows * kv_entry_bytes(spec), wd)

    # KV up-projection: reconstruct per-head K_nope and V from the cached latent
    # (W^UK, W^UV). Modelled *unabsorbed* — it runs as its own GEMM and the output
    # projection stays narrow (n_heads * v_head_dim). A serving engine that absorbs
    # MLA folds W^UK into the query and W^UV into attn_out_proj instead, dropping
    # this node and doubling attn_out_proj's input width; that is a serving-path
    # variant, flagged in the catalogue provenance. Kept as a node because the
    # weight physically exists in the checkpoint and model_weight_bytes counts it.
    f, b = _linear(
        rows, spec.kv_lora_rank,
        spec.n_heads * (spec.qk_nope_head_dim + spec.v_head_dim) // tp, aw,
        w_bytes("attn_kv_b", wd),
    )
    add("attn_kv_b", f, b, wd)

    # ── indexer: only ``full`` layers run one ────────────────────────────────
    # ``shared`` layers reuse the group's selection and carry no indexer weights,
    # so they emit no indexer node. Emitting one would put a kernel in the graph
    # that never ran and inflate the indexer's share ~4x.
    runs_indexer = (
        spec.is_full_indexer(layer) if force_full_indexer is None else force_full_indexer
    )
    scan_pairs = index_scan_pairs(batch)
    if runs_indexer and scan_pairs > 0:
        # The indexer query comes off the same latent as the attention query, so
        # only the up-projection is charged (``wq_b``), plus the per-token key
        # (``wk`` — one key per token, MQA-style, not one per index head) and the
        # per-head gate (``weights_proj``). Not divided by ``tp``: vLLM builds the
        # indexer as ReplicatedLinear, so every rank runs the whole thing. bf16 on
        # the FP8 checkpoint — the indexer is named in ``modules_to_not_convert``.
        idx_w = w_bytes("attn_index_proj", wd)
        f_q, b_q = _linear(
            rows, spec.q_lora_rank, spec.index_n_heads * spec.index_head_dim, aw, idx_w
        )
        f_k, b_k = _linear(rows, h, spec.index_head_dim, aw, idx_w)
        f_g, b_g = _linear(rows, h, spec.index_n_heads, aw, idx_w)
        add(
            "attn_index_proj",
            f_q + f_k + f_g,
            b_q + b_k + b_g + rows * spec.index_head_dim * weight_bytes(spec.kv_dtype),
            wd,
        )

        add(
            "attn_index_score",
            # Every one of the 32 index heads scores each candidate against the
            # single shared 128-d key ``wk`` produces per token — MQA-style, which
            # is why the key term below is not multiplied by the head count and
            # this one is. Dropping ``index_n_heads`` here understates the scan 32x
            # and would leave it looking free at every context.
            2.0 * scan_pairs * spec.index_n_heads * spec.index_head_dim,
            # Index keys live in the cache: read once per sequence (or per
            # prefilling request), not per position, and replicated across ranks
            # alongside the KV latent.
            index_scan_entries(batch) * spec.index_head_dim
            * weight_bytes(spec.kv_dtype),
            spec.kv_dtype,
        )

    # ── attention core over the selected positions ──────────────────────────
    # FLOPs follow the *selected* pairs (top-k bounded); bytes follow what the
    # kernel must stream, which at prefill is the whole cache and at decode is one
    # top-k window per sequence. The two do not move together on this
    # architecture, which is why they are separate helpers.
    heads = max(1, spec.n_heads // tp)
    pairs = core_qk_pairs(spec, batch)
    qk = 2.0 * pairs * heads * spec.q_head_dim
    pv = 2.0 * pairs * heads * spec.v_head_dim
    add(
        "attn_score_value",
        qk + pv,
        # One latent, shared by every query head, so *not* multiplied by n_heads.
        # Deliberately not divided by ``tp`` either: a single shared latent cannot
        # be split, so the cache is replicated and every rank reads all of it —
        # tensor parallelism buys no KV bandwidth on this architecture.
        core_read_entries(spec, batch) * kv_entry_bytes(spec),
        wd,
    )

    # RMSNorm on every query head and the single KV latent, plus partial RoPE on
    # the last ``qk_rope_head_dim`` dims and the cache insert — one fused kernel.
    normed = rows * (heads * spec.q_head_dim + spec.kv_entry_dim)
    roped = rows * (heads + 1) * spec.qk_rope_head_dim
    add(
        "attn_qnorm_rope_insert",
        3.0 * normed + 6.0 * roped,
        2.0 * normed * aw + 2.0 * roped * aw,
        spec.act_dtype,
    )

    # Output projection: plain dense from the per-head value space back to hidden.
    # No o_lora/o_groups here (unlike V4), so no ``estimated`` grouping guess. On
    # the FP8 checkpoint this one *is* quantised — absent from
    # ``modules_to_not_convert`` — the opposite of the fp8-backbone checkpoints
    # that keep o_proj wide.
    f, b = _linear(
        rows, spec.n_heads * spec.v_head_dim // tp, h, aw, w_bytes("attn_out_proj", wd)
    )
    add("attn_out_proj", f, b, wd)

    _emit_collective(g, spec, hw, layer, "tp_all_reduce_attn", rows, sh, prefix)

    # ── FFN: dense on the leading layers, mixture on the rest ────────────────
    if not spec.is_sparse_mlp(layer):
        # Dense FFN (first_k_dense_replace). gate+up then down over the wide
        # intermediate. Canonical dense-graph names so residuals stay comparable.
        inter = spec.intermediate_size
        f_gu, b_gu = _linear(rows, h, 2 * inter // tp, aw, w_bytes("mlp_gate_up", wd))
        add("mlp_gate_up", f_gu, b_gu, wd)
        f_d, b_d = _linear(rows, inter // tp, h, aw, w_bytes("mlp_down", wd))
        add("mlp_down", f_d, b_d, wd)
        _emit_collective(g, spec, hw, layer, "tp_all_reduce_mlp", rows, sh, prefix)
        return

    # Router is replicated: every rank scores every expert to know what to keep.
    # fp32 on every GLM-5.2 variant (``moe_router_dtype``) — a model fact, not a
    # quantisation choice, and the reason this op carries its own dtype.
    f, b = _linear(rows, h, spec.n_routed_experts, aw, w_bytes("moe_router", wd))
    add("moe_router", f, b, wd)

    # Sigmoid scoring, the noaux_tc bias correction, top-8 and the renorm. Almost
    # no bytes and almost no arithmetic — but it is the **only data-dependent
    # shape in the step**, so it is the node that decides whether the step can be
    # CUDA-graph captured at all. Kept as its own node for that reason alone.
    add(
        "moe_topk",
        3.0 * rows * spec.n_routed_experts,
        2.0 * rows * spec.n_routed_experts * aw,
        spec.act_dtype,
    )

    inter = spec.moe_intermediate_size
    per_expert_weights = 3.0 * h * inter
    per_position_flops = 6.0 * h * inter  # 2 * (gate + up + down) * h * inter
    ew = w_bytes("moe_routed", ed)

    if spec.n_shared_experts > 0:
        sw = w_bytes("moe_shared", ed)
        add(
            "moe_shared",
            per_position_flops * rows * spec.n_shared_experts / tp,
            per_expert_weights * spec.n_shared_experts * sw / tp
            + aw * (rows * h * 2 + rows * inter * 2 * spec.n_shared_experts / tp),
            ed,
        )

    # Gather rows into expert-major order and scatter the results back. Zero
    # arithmetic in the gather, and at decode a rounding error — but the expanded
    # tensor is ``rows x top_k`` wide, so at an 8,192-token prefill chunk this pair
    # moves hundreds of megabytes per layer for essentially no FLOPs. A graph that
    # folds it into the expert GEMM cannot show that, and it is a real and
    # separately addressable share of prefill traffic.
    expanded = rows * spec.top_k
    add("moe_permute", 0.0, aw * (rows * h + expanded * h) / es, spec.act_dtype)

    # The saturating set-union: FLOPs scale with rows x top_k, weight traffic
    # with how many *distinct* experts the batch woke — shared with the dense-MoE
    # roofline so there is one owner for the term. The argument is *rows*, not
    # sequences: under speculative decoding and under prefill every extra row is
    # another draw on the expert bank, which is exactly why a 1+D verify costs
    # more than a decode without doing more work per token.
    distinct = distinct_experts(
        int(rows), spec.n_routed_experts, spec.num_experts_per_tok
    )
    skew = sh.ep_imbalance if sh.ep > 1 else 1.0
    add(
        "moe_routed",
        per_position_flops * rows * spec.num_experts_per_tok * skew / es,
        per_expert_weights * distinct * ew * skew / es
        + aw * (rows * inter * 2 * spec.num_experts_per_tok / es),
        ed,
    )

    # Weighted scatter-add back to ``rows x hidden``, including the
    # routed_scaling_factor multiply.
    add(
        "moe_combine",
        2.0 * expanded * h,
        aw * (expanded * h + rows * h) / es,
        spec.act_dtype,
    )

    _emit_collective(
        g, spec, hw, layer, "tp_all_reduce_mlp", rows, sh, prefix,
        dispatches_experts=True,
    )


def _emit_collective(
    g: Graph,
    spec: GlmMoeDsaModelSpec,
    hw: HardwareSpec,
    layer: int,
    op: str,
    rows: float,
    sh: ShardingConfig,
    prefix: str,
    *,
    dispatches_experts: bool = False,
) -> None:
    """Emit the cross-rank traffic that closes one sub-block.

    **Two per layer, not one.** A tensor-parallel layer all-reduces after
    ``o_proj`` and again after the FFN combine; folding them into a single node
    with double the payload gets the bytes right and the *count* wrong — and at
    decode payloads a collective is bounded by its ring latency rather than by its
    bytes, so the count is the cost. Under expert parallelism the MoE half
    additionally dispatches and combines across expert ranks.

    ``dispatches_experts`` gates the expert-parallel all-to-all. Only a mixture
    layer dispatches tokens to expert ranks; the three dense-FFN layers compute
    their whole FFN locally and emit the all-reduce alone. Charging them an
    all-to-all would put 5.5 MB of wire traffic per layer on a block that has no
    experts to send anything to.

    ``serial_launches`` is withheld when the SKU carries no interconnect
    bandwidth, so an unpriced collective still predicts zero time and
    :attr:`Graph.has_unpriced_collectives` keeps reporting it. A latency floor
    applied there would quietly convert "we cannot price this" into "it costs two
    microseconds".
    """
    tp = max(1, sh.tp)
    if tp <= 1 and sh.ep <= 1:
        return
    aw = weight_bytes(spec.act_dtype)
    link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)
    priced = hw.interconnect_bw_bytes_per_s > 0

    def add_link(name_op: str, byts: float) -> None:
        name = f"{prefix}{name_op}"
        g.nodes.append(
            PredictedNode(
                name, layer,
                roofline(
                    name, 0.0, byts, link, spec.act_dtype,
                    estimated=True, serial_launches=1 if priced else 0,
                ),
                # Collectives are the one region this graph expects off the
                # compute stream; the stream-concurrency invariant reads this.
                expected_stream_id=1,
            )
        )

    if sh.ep > 1 and dispatches_experts:
        off_rank = (sh.ep - 1) / sh.ep
        add_link(
            "moe_all_to_all",
            2.0 * rows * spec.num_experts_per_tok * spec.hidden * aw * off_rank,
        )
    if tp > 1:
        add_link(op, (2.0 * (tp - 1) / tp) * rows * spec.hidden * aw)


def predict_glm_graph(
    model: GlmMoeDsaModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
) -> Graph:
    """Emit a predicted execution graph for one GLM-5.2-class engine step, per rank.

    One step, three passes, and they are not three graphs:

    **The backbone** runs over every position in the step — decode positions plus
    any prefill chunk riding along under chunked prefill. With speculative
    decoding on, "decode positions" is ``batch x (1 + D)``: **verify is not a new
    graph, it is this one with the row dimension multiplied by 1+D**.

    **The draft chain** is the one genuinely new subgraph. GLM-5.2 has a single
    MTP module (``num_nextn_predict_layers: 1``) invoked ``D`` times serially,
    EAGLE-style — stage *k* cannot start until stage *k-1*'s token id exists — and
    each stage runs its own vocabulary projection. Emitting one draft block and
    one ``lm_head`` for a ``D``-deep chain understates it by ``D``, and the
    ``lm_head`` term is not small: it reads the whole untied vocabulary matrix
    per stage regardless of how few rows ride on it.

    **The epilogue** projects only the rows that need logits —
    :attr:`BatchConfig.logits_rows`, which is one row per prefilling *request*
    plus every decode position. Charging every prefill token would overstate a
    154,880-wide projection by the chunk size.

    With ``sharding`` left at its default the graph is whole-model; given a real
    sharding it predicts what *one rank* does.
    """
    spec = model or GlmMoeDsaModelSpec()
    hw = hw or HardwareSpec()
    batch = batch or BatchConfig()
    sh = sharding or ShardingConfig()

    # Refuse a sharding the model cannot take. Head counts floor-divide throughout,
    # so ``tp > n_heads`` silently prices the whole attention path at zero work.
    if spec.n_heads % max(1, sh.tp) != 0:
        raise ValueError(
            f"tensor-parallel size {sh.tp} does not divide {spec.n_heads} attention "
            "heads — every head-sharded op would floor to zero work"
        )
    if spec.qk_rope_head_dim > spec.q_head_dim:
        raise ValueError(
            f"qk_rope_head_dim ({spec.qk_rope_head_dim}) exceeds the query head width "
            f"({spec.q_head_dim}) — the RoPE slice cannot exceed the head it slices"
        )
    if spec.n_layers <= 0:
        raise ValueError("n_layers must be positive — an empty model predicts nothing")

    g = Graph(model=spec, hw=hw, batch=batch, sharding=sh)  # type: ignore[arg-type]

    for layer in range(spec.n_layers):
        _emit_layer(g, spec, hw, layer, batch=batch, sh=sh)

    aw = weight_bytes(spec.act_dtype)
    lm_w = weight_bytes(spec.dtype_for("lm_head", spec.weight_dtype))
    lm_dtype = spec.dtype_for("lm_head", spec.weight_dtype)

    def add_lm_head(rows: float, layer: int | None) -> None:
        f, b = _linear(rows, spec.hidden, spec.vocab // max(1, sh.tp), aw, lm_w)
        g.nodes.append(
            PredictedNode(
                "lm_head", layer,
                roofline("lm_head", f, b, hw, lm_dtype, serial_launches=1),
            )
        )

    add_lm_head(batch.logits_rows, None)

    # ── the draft chain ──────────────────────────────────────────────────────
    # The MTP module is invoked once per drafted token, serially. Each stage sees
    # one row per sequence (it proposes for the sequence, not for the verify rows)
    # and carries no prefill: a draft head proposes continuations, it does not
    # ingest a prompt.
    #
    # ``index_share_for_mtp_iteration`` says the iteration reuses the main model's
    # selection rather than recomputing it, and the weight map agrees — the MTP
    # block carries **no** ``self_attn.indexer.*`` tensors, exactly like the 57
    # ``shared`` layers. So the draft is emitted as a shared block; banking the
    # projection and the scan into the floor would price a scan the runtime skips.
    #
    # What the draft is *not* is a smaller copy of the model. The MTP block carries
    # a full ``mlp.experts.*`` bank in the checkpoint, so every draft stage draws on
    # a 256-expert mixture — the draft's cost is dominated by expert weight traffic
    # it pays ``D`` times over, not by its arithmetic.
    # Gated on there being decode positions at all: on a pure-prefill step the
    # draft head does not run — it proposes continuations, and there is nothing yet
    # to continue. Emitting it anyway put 18 zero-work nodes in the graph whose only
    # cost was their launches, which is a launch facet made of kernels that never
    # ran.
    if spec.num_nextn_predict_layers > 0 and batch.positions_per_step > 0:
        draft_batch = replace(batch, prefill_tokens=0, speculative_tokens=0)
        stages = max(1, batch.speculative_tokens)
        for stage in range(stages):
            _emit_layer(
                g, spec, hw, spec.n_layers + stage,
                batch=draft_batch, sh=sh,
                force_full_indexer=not spec.index_share_for_mtp_iteration,
            )
            # ``eh_proj``: the [2*hidden, hidden] fusion of the previous hidden
            # state with the embedding of the token just drafted. bf16 on the FP8
            # checkpoint (named in ``modules_to_not_convert``), and replicated per
            # rank unless the engine shards it.
            eh_w = weight_bytes(spec.dtype_for("mtp_eh_proj", spec.weight_dtype))
            f, b = _linear(draft_batch.batch, 2 * spec.hidden, spec.hidden, aw, eh_w)
            g.nodes.append(
                PredictedNode(
                    "mtp_eh_proj", spec.n_layers + stage,
                    roofline(
                        "mtp_eh_proj", f, b, hw,
                        spec.dtype_for("mtp_eh_proj", spec.weight_dtype),
                        serial_launches=1,
                    ),
                )
            )
            # There is no ``mtp.*.lm_head`` in the checkpoint — the draft shares the
            # backbone's, which means it re-reads the same vocabulary weights and
            # gets no cheaper for being a draft.
            add_lm_head(draft_batch.batch, spec.n_layers + stage)

    return g


def is_glm_moe_dsa_config(cfg: dict[str, Any]) -> bool:
    """True for the GLM-5.2-class checkpoints this module models.

    The discriminator is unambiguous and cheap: the checkpoint declares
    ``model_type == "glm_moe_dsa"`` (equivalently ``GlmMoeDsaForCausalLM`` in
    ``architectures``). Both this family and DeepSeek-V4 carry ``index_topk`` and
    ``n_routed_experts``, so a structural test would collide — the model_type is
    the clean separator, and this check must run *before* ``is_sparse_moe_config``
    in :func:`gitm.planner.registry.detect_family`.
    """
    if str(cfg.get("model_type", "")).lower() == "glm_moe_dsa":
        return True
    archs = cfg.get("architectures") or []
    return any("glmmoedsa" in str(a).lower() for a in archs)


#: Which graph op each ``modules_to_not_convert`` entry belongs to. Substring
#: match against the tensor name, first hit wins. Norms and biases are omitted
#: deliberately — they are not nodes in this graph, so a precision for them would
#: have nothing to price.
_UNQUANTISED_OPS: tuple[tuple[str, str], ...] = (
    ("lm_head", "lm_head"),
    ("embed_tokens", "lm_head"),  # untied, but priced together in the epilogue
    ("eh_proj", "mtp_eh_proj"),
    ("indexer", "attn_index_proj"),
    ("indexers_proj", "attn_index_proj"),
    ("mlp.gate", "moe_router"),
)


def _op_dtype_overrides(
    cfg: dict[str, Any], q: dict[str, Any], weight_dtype: str, model_dtype: str
) -> tuple[tuple[str, str], ...]:
    """Per-op precision, read from the checkpoint rather than assumed.

    Two independent sources, and they answer different questions:

    ``quantization_config.modules_to_not_convert``
        *What the quantiser skipped.* Those tensors stay at the model dtype while
        the backbone drops to fp8 — on GLM-5.2-FP8 that is ``lm_head``,
        ``embed_tokens``, the MTP ``eh_proj`` and, the one worth naming, the
        **lightning indexer**. Pricing the indexer at fp8 would halve the weight
        traffic of the one attention node whose cost grows with context.

    ``moe_router_dtype``
        *What the model computes in regardless.* fp32 on every GLM-5.2 variant,
        including the unquantised one, so it is emitted whether or not a
        quantisation config exists.

    Returns an empty tuple when the checkpoint says nothing — an unquantised model
    needs no overrides beyond the router, and inventing entries would put a
    precision in the graph that the checkpoint never declared.
    """
    found: dict[str, str] = {}

    skipped = q.get("modules_to_not_convert") or q.get("ignored_layers") or []
    if isinstance(skipped, (list, tuple)) and weight_dtype != model_dtype:
        for tensor in skipped:
            name = str(tensor).lower()
            for needle, op in _UNQUANTISED_OPS:
                if needle in name:
                    found.setdefault(op, model_dtype)

    router = str(cfg.get("moe_router_dtype") or "").lower()
    if router.startswith("float32") or router == "fp32":
        found["moe_router"] = "fp32"

    return tuple(sorted(found.items()))


def spec_from_hf_config(
    cfg: dict[str, Any], *, name: str | None = None
) -> GlmMoeDsaModelSpec:
    """Build a :class:`GlmMoeDsaModelSpec` from a HuggingFace ``config.json``.

    Reads the checkpoint's declared shape so the graph cannot drift from the model.
    The ``indexer_types`` and ``mlp_layer_types`` arrays are read verbatim — they
    are the IndexShare and dense/sparse schedules, and a rule guessed from the
    frequencies would misplace layers while producing a plausible total.
    """

    def _int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    def _req(key: str) -> int:
        v = cfg.get(key)
        if v is None:
            raise ValueError(
                f"config declares no {key!r}; refusing to substitute a default, "
                "which would predict a model this checkpoint is not"
            )
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"config field {key!r} is not an integer: {v!r}") from None

    n_layers = _req("num_hidden_layers")

    def _types(key: str) -> tuple[str, ...]:
        v = cfg.get(key)
        if not isinstance(v, list | tuple):
            return ()
        # Validate the *raw* length before any slicing: GLM-5.2's schedules are
        # exactly num_hidden_layers long, and an over-long array is a sign the
        # config is not what the caller thinks — worth an error, not a silent
        # truncation that leaves the trailing layers on the last entry's kind.
        if len(v) != n_layers:
            raise ValueError(
                f"{key} has {len(v)} entries for {n_layers} layers — the "
                "schedule must cover the model exactly"
            )
        return tuple(str(t) for t in v)

    dtype = str(cfg.get("dtype") or cfg.get("torch_dtype") or "bf16").lower()
    if dtype.startswith("bfloat"):
        dtype = "bf16"
    elif dtype.startswith("float16") or dtype == "half":
        dtype = "fp16"
    elif dtype.startswith("float32"):
        dtype = "fp32"

    # Quantisation, if the checkpoint declares any. The base GLM-5.2 release
    # carries none — an all-bf16 read matches its 1.507 TB on disk — while
    # GLM-5.2-FP8 declares e4m3 with 128x128 weight blocks. Read rather than
    # assumed in either direction: defaulting to fp8 would manufacture headroom on
    # the bf16 checkpoint, and defaulting to bf16 would double every weight term on
    # the one the vendor actually recommends deploying.
    q = cfg.get("quantization_config") or {}
    weight_dtype = str(q.get("quant_method", dtype)).lower() if q else dtype
    expert_dtype = str(cfg.get("expert_dtype", weight_dtype)).lower()
    overrides = _op_dtype_overrides(cfg, q, weight_dtype, dtype)

    return GlmMoeDsaModelSpec(
        name=name or str(cfg.get("_name_or_path") or cfg.get("model_type") or "glm-moe-dsa"),
        hidden=_req("hidden_size"),
        n_layers=n_layers,
        vocab=_req("vocab_size"),
        n_heads=_req("num_attention_heads"),
        q_lora_rank=_req("q_lora_rank"),
        kv_lora_rank=_req("kv_lora_rank"),
        qk_nope_head_dim=_int("qk_nope_head_dim", _int("head_dim", 128)),
        qk_rope_head_dim=_int("qk_rope_head_dim", 64),
        v_head_dim=_int("v_head_dim", _int("head_dim", 128)),
        index_n_heads=_int("index_n_heads", 32),
        index_head_dim=_int("index_head_dim", 128),
        index_topk=_int("index_topk", 2048),
        index_topk_freq=_int("index_topk_freq", 4),
        indexer_types=_types("indexer_types"),
        n_routed_experts=_req("n_routed_experts"),
        n_shared_experts=_int("n_shared_experts", 1),
        num_experts_per_tok=_req("num_experts_per_tok"),
        moe_intermediate_size=_req("moe_intermediate_size"),
        intermediate_size=_int("intermediate_size", 12288),
        first_k_dense_replace=_int("first_k_dense_replace", 0),
        mlp_layer_types=_types("mlp_layer_types"),
        routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0) or 1.0),
        num_nextn_predict_layers=_int("num_nextn_predict_layers", 0),
        index_share_for_mtp_iteration=bool(cfg.get("index_share_for_mtp_iteration", True)),
        weight_dtype=weight_dtype,
        expert_dtype=expert_dtype,
        op_dtype_overrides=overrides,
        # The config declares no cache dtype; bf16 is the model fact. A served
        # deployment may pick fp8 — that is a serving decision, set at deploy time.
        kv_dtype=dtype,
        act_dtype=dtype,
    )
