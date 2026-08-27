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

**Precision is bf16 throughout.** The published checkpoint carries no
``quantization_config`` — the on-disk size (1.507 TB) matches an all-bf16 read.
Any fp8 the KV cache or experts pick up is a *serving* decision, not a model fact,
so — unlike the V4 reader, which defaults to fp4 experts — this graph prices bf16
and records quantisation as a deployment lever in provenance. Defaulting to fp4
here would deflate the dominant expert term ~3.8x and manufacture headroom that
the checkpoint does not have.

Known limits, stated rather than hidden:

* **Decode only.** As with every other graph family; the chunked-prefill indexer
  path has different asymptotics.
* **Uniform routing.** :func:`~gitm.planner.roofline.distinct_experts` assumes a
  balanced router; real skew touches fewer distinct experts, moving *less* traffic
  than predicted — the conservative direction.
* **Expert-parallel skew is calibrated, not predicted.**
  :attr:`ShardingConfig.ep_imbalance` stays at 1.0 until a trace measures it.
* **Collectives are bandwidth-only** (no latency floor), flagged ``estimated``.
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

    # ── precision (bf16 checkpoint; quantisation is a serving decision) ───────
    weight_dtype: str = "bf16"
    expert_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    act_dtype: str = "bf16"

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
    ew = weight_bytes(spec.expert_dtype)
    ww = weight_bytes(spec.weight_dtype)
    h = spec.hidden
    inter = spec.moe_intermediate_size

    n_sparse = spec.n_sparse_mlp_layers + spec.num_nextn_predict_layers
    n_dense = spec.n_layers - spec.n_sparse_mlp_layers
    n_full_idx = spec.n_full_indexer_layers + spec.num_nextn_predict_layers
    n_attn = spec.n_layers + spec.num_nextn_predict_layers

    experts = n_sparse * spec.n_routed_experts * 3 * h * inter * ew / es
    shared_exp = n_sparse * spec.n_shared_experts * 3 * h * inter * ew / tp
    router = n_sparse * h * spec.n_routed_experts * ww  # replicated
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
    indexer_per_full = spec.q_lora_rank * spec.index_n_heads * spec.index_head_dim

    embed = 2.0 * spec.vocab * h / tp  # untied input embedding + lm_head
    return (
        experts
        + shared_exp
        + router
        + (n_attn * attn_per_layer + n_full_idx * indexer_per_full + dense_ffn + embed)
        * ww
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
    positions: float,
    sequences: int,
    kv_len: int,
    sh: ShardingConfig,
    prefix: str = "",
    force_full_indexer: bool | None = None,
) -> None:
    """Append one transformer layer's predicted nodes to ``g``.

    ``positions`` is how many sequence positions this layer computes; ``sequences``
    is how many distinct KV caches those positions read from. They differ under
    speculative decoding, and the difference is why MTP helps a memory-bound
    decode: eight draft positions verified in one step read the KV cache *once*, so
    cache traffic amortises while FLOPs do not.
    """
    h = spec.hidden
    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    ew = weight_bytes(spec.expert_dtype)
    wd, ed = spec.weight_dtype, spec.expert_dtype
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)

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

    # ── MLA attention: low-rank query, compressed KV latent ──────────────────
    # q_a and kv_a are replicated across TP ranks: they produce the shared latent,
    # which has nothing to split when there is one KV latent. Every rank pays them
    # in full, so TP's speedup on attention is strictly less than ``tp``.
    f, b = _linear(positions, h, spec.q_lora_rank, aw, ww)
    add("attn_q_a", f, b, wd)

    f, b = _linear(positions, spec.q_lora_rank, spec.n_heads * spec.q_head_dim // tp, aw, ww)
    add("attn_q_b", f, b, wd)

    # The compressed latent plus the decoupled RoPE key, and the cache write for
    # the positions just computed. One projection (no CSA/HCA overlap here).
    f, b = _linear(positions, h, spec.kv_entry_dim, aw, ww)
    add("attn_kv_a", f, b + positions * kv_entry_bytes(spec), wd)

    # KV up-projection: reconstruct per-head K_nope and V from the cached latent
    # (W^UK, W^UV). Modelled *unabsorbed* — it runs as its own GEMM and the output
    # projection stays narrow (n_heads * v_head_dim). A serving engine that absorbs
    # MLA folds W^UK into the query and W^UV into attn_out_proj instead, dropping
    # this node and doubling attn_out_proj's input width; that is a serving-path
    # variant, flagged in the catalogue provenance. Kept as a node because the
    # weight physically exists in the checkpoint and model_weight_bytes counts it.
    f, b = _linear(
        positions, spec.kv_lora_rank,
        spec.n_heads * (spec.qk_nope_head_dim + spec.v_head_dim) // tp, aw, ww,
    )
    add("attn_kv_b", f, b, wd)

    # ── indexer: only ``full`` layers run one ────────────────────────────────
    # ``shared`` layers reuse the group's selection and carry no indexer weights,
    # so they emit no indexer node. Emitting one would put a kernel in the graph
    # that never ran and inflate the indexer's share ~4x.
    runs_indexer = spec.is_full_indexer(layer) if force_full_indexer is None else force_full_indexer
    cand = index_candidates(spec, kv_len)
    if runs_indexer and cand > 0:
        # The indexer query comes off the same latent as the attention query, so
        # only the up-projection is charged. Not divided by ``tp`` — vLLM builds
        # the indexer as ReplicatedLinear, so every rank runs the whole thing.
        f, b = _linear(
            positions, spec.q_lora_rank, spec.index_n_heads * spec.index_head_dim, aw, ww
        )
        add("attn_index_proj", f, b, wd)

        add(
            "attn_index_score",
            2.0 * positions * spec.index_n_heads * spec.index_head_dim * cand,
            # Index keys live in the cache: read once per sequence, not per
            # position, and replicated across ranks alongside the KV latent.
            sequences * cand * spec.index_head_dim * weight_bytes(spec.kv_dtype),
            spec.kv_dtype,
        )

    # ── attention core over the selected positions ──────────────────────────
    t_eff = effective_kv_tokens(spec, kv_len)
    heads = max(1, spec.n_heads // tp)
    qk = 2.0 * positions * heads * spec.q_head_dim * t_eff
    pv = 2.0 * positions * heads * spec.v_head_dim * t_eff
    add(
        "attn_score_value",
        qk + pv,
        # One latent, shared by every query head, so *not* multiplied by n_heads.
        # Deliberately not divided by ``tp`` either: a single shared latent cannot
        # be split, so the cache is replicated and every rank reads all of it —
        # tensor parallelism buys no KV bandwidth on this architecture.
        sequences * t_eff * kv_entry_bytes(spec),
        wd,
    )

    # RMSNorm on every query head and the single KV latent, plus partial RoPE on
    # the last ``qk_rope_head_dim`` dims and the cache insert — one fused kernel.
    normed = positions * (heads * spec.q_head_dim + spec.kv_entry_dim)
    roped = positions * (heads + 1) * spec.qk_rope_head_dim
    add(
        "attn_qnorm_rope_insert",
        3.0 * normed + 6.0 * roped,
        2.0 * normed * aw + 2.0 * roped * aw,
        spec.act_dtype,
    )

    # Output projection: plain dense from the per-head value space back to hidden.
    # No o_lora/o_groups here (unlike V4), so no ``estimated`` grouping guess.
    f, b = _linear(positions, spec.n_heads * spec.v_head_dim // tp, h, aw, ww)
    add("attn_out_proj", f, b, wd)

    # ── FFN: dense on the leading layers, mixture on the rest ────────────────
    if not spec.is_sparse_mlp(layer):
        # Dense FFN (first_k_dense_replace). gate+up then down over the wide
        # intermediate. Canonical dense-graph names so residuals stay comparable.
        inter = spec.intermediate_size
        f_gu, b_gu = _linear(positions, h, 2 * inter // tp, aw, ww)
        add("mlp_gate_up", f_gu, b_gu, wd)
        f_d, b_d = _linear(positions, inter // tp, h, aw, ww)
        add("mlp_down", f_d, b_d, wd)
        return

    # Router is replicated: every rank scores every expert to know what to keep.
    f, b = _linear(positions, h, spec.n_routed_experts, aw, ww)
    add("moe_router", f, b, wd)

    inter = spec.moe_intermediate_size
    per_expert_weights = 3.0 * h * inter
    per_position_flops = 6.0 * h * inter  # 2 * (gate + up + down) * h * inter

    if spec.n_shared_experts > 0:
        add(
            "moe_shared",
            per_position_flops * positions * spec.n_shared_experts / tp,
            per_expert_weights * spec.n_shared_experts * ew / tp
            + aw * (positions * h * 2 + positions * inter * 2 * spec.n_shared_experts / tp),
            ed,
        )

    # The saturating set-union: FLOPs scale with positions x top_k, weight traffic
    # with how many *distinct* experts the batch woke — shared with the dense-MoE
    # roofline so there is one owner for the term.
    distinct = distinct_experts(
        int(positions), spec.n_routed_experts, spec.num_experts_per_tok
    )
    skew = sh.ep_imbalance if sh.ep > 1 else 1.0
    add(
        "moe_routed",
        per_position_flops * positions * spec.num_experts_per_tok * skew / es,
        per_expert_weights * distinct * ew * skew / es
        + aw * (positions * h * 2 + positions * inter * 2 * spec.num_experts_per_tok / es),
        ed,
    )

    # ── cross-rank traffic ──────────────────────────────────────────────────
    if tp > 1 or sh.ep > 1:
        link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)

        def add_link(op: str, byts: float) -> None:
            name = f"{prefix}{op}"
            g.nodes.append(
                PredictedNode(
                    name, layer,
                    roofline(name, 0.0, byts, link, spec.act_dtype, estimated=True),
                )
            )

        if sh.ep > 1:
            off_rank = (sh.ep - 1) / sh.ep
            add_link(
                "moe_all_to_all",
                2.0 * positions * spec.num_experts_per_tok * h * aw * off_rank,
            )
        if tp > 1:
            add_link("tp_all_reduce", 2.0 * (2.0 * (tp - 1) / tp) * positions * h * aw)


def predict_glm_graph(
    model: GlmMoeDsaModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
) -> Graph:
    """Emit a predicted execution graph for one GLM-5.2-class decode step, per rank.

    The main stack runs over every position in the step (the verified token plus
    any speculative drafts); the MTP head then runs over one position per sequence
    to propose the next draft. With ``sharding`` left at its default the graph is
    whole-model; given a real sharding it predicts what *one rank* does.
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
    positions = batch.positions_per_step
    sequences = batch.batch
    kv_len = batch.kv_cache_len

    for layer in range(spec.n_layers):
        _emit_layer(
            g, spec, hw, layer,
            positions=positions, sequences=sequences, kv_len=kv_len, sh=sh,
        )

    # Multi-token prediction head. ``index_share_for_mtp_iteration`` says the MTP
    # iteration reuses the main model's index rather than recomputing it, so the
    # draft layer is emitted as a ``shared`` (no indexer) block — the tensor exists
    # but the iteration shares it, and banking the projection+scan into the floor
    # would over-predict a node the runtime skips.
    for i in range(spec.num_nextn_predict_layers):
        _emit_layer(
            g, spec, hw, spec.n_layers + i,
            positions=sequences, sequences=sequences, kv_len=kv_len, sh=sh,
            force_full_indexer=not spec.index_share_for_mtp_iteration,
        )

    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    f, b = _linear(positions, spec.hidden, spec.vocab // max(1, sh.tp), aw, ww)
    g.nodes.append(
        PredictedNode("lm_head", None, roofline("lm_head", f, b, hw, spec.weight_dtype))
    )
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

    # Quantisation, if the checkpoint declares any. Absent on the base GLM-5.2
    # release — an all-bf16 read matches the on-disk size — so this defaults to the
    # model dtype rather than to fp4/fp8, which would manufacture headroom.
    q = cfg.get("quantization_config") or {}
    weight_dtype = str(q.get("quant_method", dtype)).lower() if q else dtype
    expert_dtype = str(cfg.get("expert_dtype", weight_dtype)).lower()

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
        # The config declares no cache dtype; bf16 is the model fact. A served
        # deployment may pick fp8 — that is a serving decision, set at deploy time.
        kv_dtype=dtype,
        act_dtype=dtype,
    )
