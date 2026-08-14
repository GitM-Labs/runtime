"""Predicted execution graph for a sparse-MoE decode step.

:func:`gitm.planner.graph.predict_graph` models a dense transformer, where three
things are true that make the roofline easy: every weight is read every step,
attention reads the whole KV cache, and one dtype prices the entire model. A
DeepSeek-V4-class checkpoint breaks all three, and each break is a term the dense
graph has no place to put:

**Expert weight traffic is a set-union problem, not a multiplication.** With
``num_experts_per_tok`` of ``n_routed_experts`` firing per token, FLOPs scale with
``positions x top_k`` — but HBM traffic scales with how many *distinct* experts the
batch collectively touched, which saturates at ``n_routed_experts`` and is the
dominant cost of a decode step. At batch 1 a step reads 6 of 256 experts; by batch
256 it reads essentially all of them for the same per-token FLOPs. The dense model
has no term that saturates, so it mispredicts the low-batch floor by ~40x and then
attributes the difference to a cause. The term itself is
:func:`gitm.planner.roofline.distinct_experts`, shared with the dense MoE roofline
rather than reimplemented here — it is the same union, and two copies of it would
drift.

**Attention is three mechanisms, not one.** Sliding-window layers read a fixed
recent window. CSA layers compress by ``m`` and then *select* ``index_topk``
entries with a lightning indexer, so their core read is bounded and flat in
context while their indexer scan grows. HCA layers compress by ``m' >> m`` and
attend *densely* with no indexer, so their read grows with context without bound
and they dominate the attention cost at long sequence lengths. Folding these into
one op — as a dense graph must — averages a 13x spread at 1M context and
attributes deviation to whichever layer happened to be modelled.

**Precision is per-tensor-class.** Experts run fp4, linears fp8, the KV cache fp8.
The load-bearing half of this is *bytes*, not FLOPs: pricing fp4 expert weights at
bf16 inflates the dominant term 3.3x. The peak-FLOPS half barely moves a decode
step — every node is memory-bound, so the compute ceiling never binds — but it
governs prefill, where it does.

Sharding is modelled per rank (:class:`ShardingConfig`): tensor parallelism splits
heads and dense linears, expert parallelism splits whole experts, and both emit the
collective they actually pay for. Two results fall out that the unsharded graph
could not express — TP does *not* divide KV traffic on a single-KV-head model, and
EP versus TP moves identical weight bytes per rank. Both are documented at the
nodes that produce them.

Known limits, stated rather than hidden:

* **Collectives are bandwidth-only.** No latency floor, so at decode message sizes
  the all-to-all and all-reduce nodes are optimistic; they carry ``estimated``. A
  SKU with no interconnect figure leaves them at zero, which
  :attr:`Graph.has_unpriced_collectives` surfaces rather than banking as free.
* **Uniform routing.** :func:`~gitm.planner.roofline.distinct_experts` assumes a
  balanced router.
  Real routing is skewed, which touches *fewer* distinct experts and so moves less
  weight traffic than predicted — the conservative direction, since it under-reports
  headroom rather than inventing it.
* **Nodes marked** ``estimated`` **are documented approximations** (grouped output
  projection, DSpark, every collective) and carry that flag into the report so an
  estimate is never read as a measurement.
* **Expert-parallel skew is calibrated, not predicted.**
  :attr:`ShardingConfig.ep_imbalance` defaults to perfect balance; the real figure
  comes from a trace. Inventing a routing distribution here would be a guess
  wearing a model's clothes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    SparseMoEModelSpec,
    _canon_dtype,
    distinct_experts,
    roofline,
    weight_bytes,
    weight_bytes_is_fallback,
)

_REQUIRED_POSITIVE_CONFIG_FIELDS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "vocab_size",
    "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
    "index_n_heads", "index_head_dim", "index_topk", "sliding_window",
)


def validate_sparse_moe_config(cfg: dict[str, Any]) -> list[str]:
    """Return fields that cannot be omitted without inventing graph shape/bytes."""
    errors: list[str] = []
    for key in _REQUIRED_POSITIVE_CONFIG_FIELDS:
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{key} must be a declared positive integer, got {value!r}")
    for key in ("qk_rope_head_dim", "n_shared_experts"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{key} must be a declared non-negative integer, got {value!r}")
    n_experts, top_k = cfg.get("n_routed_experts"), cfg.get("num_experts_per_tok")
    if isinstance(n_experts, int) and not isinstance(n_experts, bool) and isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > n_experts:
        errors.append(f"num_experts_per_tok={top_k} exceeds n_routed_experts={n_experts}")
    n_layers, ratios = cfg.get("num_hidden_layers"), cfg.get("compress_ratios")
    if not isinstance(ratios, list | tuple):
        errors.append("compress_ratios must be declared for the sparse-attention graph")
    elif isinstance(n_layers, int) and n_layers > 0 and len(ratios) < n_layers:
        errors.append(f"compress_ratios has {len(ratios)} entries; need at least {n_layers}")
    elif any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in ratios):
        errors.append("compress_ratios must contain non-negative integers")
    q = cfg.get("quantization_config")
    if q is not None and not isinstance(q, dict):
        errors.append("quantization_config must be an object when declared")
    elif isinstance(q, dict) and q.get("quant_method") is not None and weight_bytes_is_fallback(str(q["quant_method"])):
        errors.append(f"quantization_config.quant_method={q['quant_method']!r} is not priceable")
    for key in ("expert_dtype", "torch_dtype"):
        value = cfg.get(key)
        if value is None:
            errors.append(f"{key} must be declared; byte width cannot be guessed")
        elif weight_bytes_is_fallback(str(value)):
            errors.append(f"{key}={value!r} is not priceable")
    return errors


def effective_kv_tokens(spec: SparseMoEModelSpec, layer: int, kv_len: int) -> int:
    """KV positions the attention *core* reads for ``layer`` at ``kv_len`` context.

    Three layer kinds (paper §2.3), and conflating any two of them is the error
    this function exists to avoid:

    **swa — ratio 0.** Not compressed, but not global either: at most
    ``sliding_window`` recent tokens. Reading ``0`` as "uncompressed, therefore
    attends to everything" is wrong by ``kv_len / sliding_window`` — 512x at 64K.

    **csa — the smaller compression rate.** One entry per ``m`` tokens, then a
    lightning indexer scores those entries and keeps ``index_topk``. Bounded, so
    **flat in context** once selection saturates.

    **hca — the larger rate.** One entry per ``m' >> m`` tokens, attended
    **densely** — no indexer, no selection. So its read is ``kv_len / m'`` and
    **grows with context without bound**. HCA buys efficiency from the
    compression rate alone; CSA buys it from selection. That is the whole point
    of interleaving them, and it means the two cannot share a cost model.

    (Ratio 1 means genuinely uncompressed global attention. No layer of this
    checkpoint uses it; it is kept for other architectures.)

    The consequence worth internalising: **HCA is what makes long context
    expensive.** At 64K every layer reads 640 entries and the stack looks flat;
    by 1M the HCA layers read 8,320 each and account for the great majority of
    all attention traffic. A residual that scales with context therefore belongs
    to HCA or to the CSA indexer — never to a CSA core, which is genuinely
    constant.
    """
    if kv_len <= 0:
        return 0
    kind = spec.attention_kind(layer)
    if kind == "swa":
        # An uncompressed layer with no window is not a window layer — it is
        # global attention, and it reads everything. Returning
        # ``min(kv_len, 0) == 0`` there would say attention is free, which is
        # the failure mode a config missing ``sliding_window`` walks straight
        # into: a plausible total with a whole mechanism silently costed at zero.
        if spec.sliding_window <= 0:
            return kv_len
        return min(kv_len, spec.sliding_window)
    r = max(1, spec.compress_ratio(layer))
    compressed = math.ceil(kv_len / r)
    if kind == "hca":
        # Dense over every compressed entry — no selection, so this grows with
        # context. HCA buys its efficiency from the compression rate alone.
        return min(kv_len, spec.sliding_window + compressed)
    # CSA: the indexer keeps at most index_topk of the compressed entries.
    return min(kv_len, spec.sliding_window + min(spec.index_topk, compressed))


def index_candidates(spec: SparseMoEModelSpec, layer: int, kv_len: int) -> int:
    """Positions the indexer must score for ``layer`` — the compressed set.

    Zero for sliding-window layers: a fixed recent window needs no selection, so
    those layers run no indexer at all and the graph emits no indexer node for
    them. Scoring a candidate set there would invent both a kernel and its cost.

    For compressed layers this is unbounded in context length — the node that
    decides whether a million-token deployment is viable, and now the *only* term
    in the attention path that still grows with context.
    """
    if kv_len <= 0 or spec.attention_kind(layer) != "csa":
        return 0
    return math.ceil(kv_len / max(1, spec.compress_ratio(layer)))


def model_weight_bytes(
    spec: SparseMoEModelSpec, sharding: ShardingConfig | None = None
) -> float:
    """Resident weight bytes on one rank.

    Decides whether a deployment shape is possible at all, which the timing graph
    cannot: a config that predicts beautifully and does not fit is not a config.
    Counts experts at ``expert_dtype`` and everything else at ``weight_dtype``,
    since collapsing the two is a ~3x error on the dominant term.

    Validated against ground truth: ``DeepSeek-V4-Flash`` predicts 156 GB against
    a published 160 GB checkpoint (-2.4%), the residual being norms, biases and
    per-tensor metadata this does not enumerate.

    The DSpark variant is the known gap. ``DeepSeek-V4-Flash-0731`` publishes at
    167 GB — 7 GB more than the base checkpoint for three extra layers' worth of
    DSpark — while the low-rank pair modelled here accounts for ~6 MB. The shape
    is not public, so the term below is carried for internal consistency with the
    graph's ``dspark`` node and is *known to be orders of magnitude low*. Treat a
    footprint prediction for that checkpoint as a lower bound, not an estimate.
    """
    sh = sharding or ShardingConfig()
    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)
    ew = weight_bytes(spec.expert_dtype)
    ww = weight_bytes(spec.weight_dtype)
    h, inter = spec.hidden, spec.moe_intermediate_size
    layers = spec.n_layers + spec.num_nextn_predict_layers

    experts = layers * spec.n_routed_experts * 3 * h * inter * ew / es
    shared = layers * spec.n_shared_experts * 3 * h * inter * ew / tp
    attn_per_layer = (
        h * spec.q_lora_rank  # q_a, replicated
        + spec.q_lora_rank * spec.n_heads * spec.q_head_dim / tp
        + h * spec.kv_latent_dim  # kv_a, replicated
        + h * spec.index_n_heads * spec.index_head_dim / tp
        + spec.n_heads * spec.head_dim * spec.o_lora_rank / (spec.o_groups * tp)
        + spec.o_lora_rank * h
        + h * spec.n_routed_experts  # router, replicated
    )
    embed = 2 * spec.vocab * h / tp  # input embedding + untied lm_head
    dspark = len(spec.dspark_layer_ids) * 2 * h * spec.dspark_markov_rank / tp
    return experts + shared + (layers * attn_per_layer + embed + dspark) * ww


def kv_bytes_per_token(spec: SparseMoEModelSpec) -> float:
    """KV bytes that grow with each additional token of context.

    Compression is per-layer, so this is a sum and not a multiplication: twenty
    layers store a quarter of a latent per token and twenty-one store 1/128th.
    Using any single ratio is wrong by up to 128x, and KV footprint is what sets
    the concurrency ceiling.

    **Sliding-window layers are excluded.** They hold a fixed
    ``sliding_window``-sized buffer per sequence however long the context grows,
    so they belong in :func:`kv_fixed_bytes_per_sequence`, not in a per-token
    rate. Counting them here would make footprint appear to grow ~4x faster than
    it does and would cap concurrency far below what the hardware allows.
    """
    kw = weight_bytes(spec.kv_dtype)
    total = 0.0
    for layer in range(spec.n_layers):
        r = spec.compress_ratio(layer)
        if r == 0:
            continue  # bounded buffer, not per-token growth
        # The latent, plus the indexer's key for the same (compressed) position.
        total += (kv_entry_bytes(spec) + spec.index_head_dim * kw) / max(1, r)
    return total


def kv_fixed_bytes_per_sequence(spec: SparseMoEModelSpec) -> float:
    """KV bytes a sequence costs regardless of how long its context grows.

    The sliding-window layers: each holds ``sliding_window`` tokens of latent and
    nothing more, so their cost is paid once per sequence rather than per token.

    In magnitude this is small — on this checkpoint it is worth about 40 tokens
    of context, so it never drives a sizing decision. It exists because the
    *rate* was wrong without it: counting these layers per-token inflated
    :func:`kv_bytes_per_token` by 37% at every context length.
    """
    swa_layers = sum(1 for i in range(spec.n_layers) if spec.compress_ratio(i) == 0)
    return swa_layers * spec.sliding_window * kv_entry_bytes(spec)


def kv_entry_bytes(spec: SparseMoEModelSpec) -> float:
    """Bytes one cached KV entry occupies, across its mixed storage format.

    Paper §2.3.4: the RoPE dimensions are kept in BF16 while the rest is FP8,
    "reducing the KV cache size by nearly half compared with pure BF16". Pricing
    the whole entry at FP8 understates it — on this checkpoint by 11%, since 64
    of the 576 dimensions carry double width.
    """
    rope = spec.qk_rope_head_dim * spec.num_kv_heads
    rest = spec.kv_latent_dim - rope
    return rest * weight_bytes(spec.kv_dtype) + rope * weight_bytes("bf16")


def _linear(rows: float, k: int, n: int, act_b: float, w_b: float) -> tuple[float, float]:
    """(flops, bytes) for a ``(rows, k) @ (k, n)`` projection.

    Bytes count the activation in, the weights, and the activation out. At decode
    ``rows`` is small and the weight term dominates — which is why weight dtype,
    not activation dtype, sets the floor for every projection here.
    """
    return 2.0 * rows * k * n, act_b * rows * k + w_b * k * n + act_b * rows * n


def _emit_layer(
    g: Graph,
    spec: SparseMoEModelSpec,
    hw: HardwareSpec,
    layer: int,
    *,
    positions: float,
    sequences: int,
    kv_len: int,
    sh: ShardingConfig,
    prefix: str = "",
) -> None:
    """Append one transformer layer's predicted nodes to ``g``.

    ``positions`` is how many sequence positions this layer computes; ``sequences``
    is how many distinct KV caches those positions read from. They differ under
    speculative decoding, and the difference is the whole reason MTP helps a
    memory-bound decode: eight draft positions verified in one step read the KV
    cache *once*, so cache traffic amortises across them while FLOPs do not. Using
    one number for both would erase that effect and predict MTP as pure overhead.
    """
    h = spec.hidden
    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    ew = weight_bytes(spec.expert_dtype)
    af = weight_bytes_is_fallback(spec.act_dtype)
    wf = weight_bytes_is_fallback(spec.weight_dtype)
    ef = weight_bytes_is_fallback(spec.expert_dtype)
    kf = weight_bytes_is_fallback(spec.kv_dtype)
    wd, ed = spec.weight_dtype, spec.expert_dtype

    def add(
        op: str, flops: float, byts: float, dtype: str,
        *, byte_fallback: bool = False, estimated: bool = False, serial_launches: int = 0,
    ) -> None:
        name = f"{prefix}{op}"
        g.nodes.append(
            PredictedNode(
                name, layer,
                roofline(
                    name, flops, byts, hw, dtype,
                    estimated=estimated, serial_launches=serial_launches,
                    bytes_are_fallback=byte_fallback,
                ),
            )
        )

    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)

    # ── attention: low-rank query, compressed KV latent ──────────────────────
    # ``q_a`` and ``kv_a`` are replicated across tensor-parallel ranks: they
    # produce the shared latent, which has nothing to split when there is one KV
    # head. Every rank pays for them in full, so TP's speedup on attention is
    # strictly less than ``tp``.
    f, b = _linear(positions, h, spec.q_lora_rank, aw, ww)
    add("attn_q_a", f, b, wd, byte_fallback=af or wf)

    f, b = _linear(positions, spec.q_lora_rank, spec.n_heads * spec.q_head_dim // tp, aw, ww)
    add("attn_q_b", f, b, wd, byte_fallback=af or wf)

    # KV entries and their compression weights, plus the cache write for the
    # positions just computed. CSA computes *two* KV series and two weight series
    # (W^aKV, W^bKV, W^aZ, W^bZ) because its compression windows overlap; HCA
    # computes one of each; a window layer caches raw entries and compresses
    # nothing. Modelling one projection everywhere understates CSA 4x on this op.
    kind = spec.attention_kind(layer)
    n_kv_proj = {"csa": 4, "hca": 2}.get(kind, 1)
    f, b = _linear(positions, h, spec.kv_latent_dim * n_kv_proj, aw, ww)
    add("attn_kv_a", f, b + positions * kv_entry_bytes(spec), wd, byte_fallback=af or wf or kf)

    if kind != "swa":
        # The compressor itself: a row-softmax over the window's weights and the
        # weighted sum that collapses it to one entry. One compressed entry per
        # ``r`` tokens, so at decode this fires on a 1/r duty cycle.
        r = max(1, spec.compress_ratio(layer))
        window = 2 * r if kind == "csa" else r  # CSA windows overlap
        add(
            "attn_kv_compress",
            3.0 * positions * window * spec.kv_latent_dim / r,
            2.0 * positions * window * spec.kv_latent_dim * aw / r,
            spec.act_dtype,
            byte_fallback=af,
        )

    # ── indexer: project the query, then scan the compressed candidate set ───
    # Only CSA layers run one. Window layers need no selection, and HCA attends
    # densely over its (far more aggressively compressed) entries — emitting an
    # indexer for either would put kernels in the graph that never ran.
    cand = index_candidates(spec, layer, kv_len)
    if cand > 0:
        # The indexer query comes off the *same* latent as the attention query
        # (paper eq. 13-14: both are c_t^Q @ W^UQ variants), so only the
        # up-projection is charged here — the down-projection is attn_q_a.
        #
        # Not divided by ``tp``: vLLM builds the indexer's ``wq_b`` and
        # ``weights_proj`` as ReplicatedLinear, so every rank runs the whole
        # indexer. Sharding it here would predict a speedup the deployment does
        # not get, and would under-predict this node by ``tp``.
        f, b = _linear(
            positions, spec.q_lora_rank, spec.index_n_heads * spec.index_head_dim, aw, ww
        )
        add("attn_index_proj", f, b, wd, byte_fallback=af or wf)

        add(
            "attn_index_score",
            2.0 * positions * spec.index_n_heads * spec.index_head_dim * cand,
            # Index keys live in the cache: read once per sequence, not per
            # position, and replicated across ranks alongside the KV latent.
            # vLLM quantises this cache to uint8, so a byte per element — the
            # FP4 in the paper (§2.3.4) is the *arithmetic* precision, which is
            # why the dtype and the byte width disagree here.
            sequences * cand * spec.index_head_dim * 1.0,
            "fp4",
            byte_fallback=kf,
        )

    # ── attention core over the selected positions ──────────────────────────
    t_eff = effective_kv_tokens(spec, layer, kv_len)
    qk = 2.0 * positions * (spec.n_heads // tp) * spec.q_head_dim * t_eff
    pv = 2.0 * positions * (spec.n_heads // tp) * spec.head_dim * t_eff
    add(
        "attn_score_value",
        qk + pv,
        # One latent, shared by every query head. Multiplying by ``n_heads`` here
        # is the classic compressed-KV modelling error and inflates decode KV
        # traffic by 64x on this checkpoint.
        #
        # Deliberately *not* divided by ``tp``: a single KV head cannot be split,
        # so the latent cache is replicated and every rank reads all of it. Tensor
        # parallelism buys no KV bandwidth on this architecture — the opposite of
        # the intuition carried over from GQA models, and a lever-ranking error
        # waiting to happen if the graph divided here.
        sequences * t_eff * kv_entry_bytes(spec),
        wd,
        byte_fallback=kf,
    )

    # RMSNorm on every query head and on the single compressed KV head, plus
    # partial RoPE on the last ``qk_rope_head_dim`` dimensions and the cache
    # insert — one node because vLLM runs them as one kernel
    # (``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``, and
    # ``fused_q_kv_rmsnorm`` for the pair of norms). Emitting a node per
    # mathematical step would put three predicted kernels where one runs, and
    # nothing would align to two of them.
    heads = spec.n_heads // tp
    normed = positions * (heads * spec.q_head_dim + spec.kv_latent_dim)
    roped = positions * (2 * heads + 1) * spec.qk_rope_head_dim
    add(
        "attn_qnorm_rope_insert",
        3.0 * normed + 6.0 * roped,
        2.0 * normed * aw + 2.0 * roped * aw,
        spec.act_dtype,
        byte_fallback=af,
    )

    # Output projection, grouped and low-rank. ``o_groups`` partitions the head
    # dimension so each group carries its own slice of the rank — modelled as an
    # even split, which is the documented reading of the config and not a
    # published shape, hence ``estimated``.
    o_in = spec.n_heads * spec.head_dim // tp
    f_a, b_a = _linear(positions, o_in, max(1, spec.o_lora_rank // spec.o_groups), aw, ww)
    f_b, b_b = _linear(positions, spec.o_lora_rank, h, aw, ww)
    # Named to match the dense graph's output projection: it is the same op in
    # the same place, so residuals for it stay comparable across model families.
    add("attn_out_proj", f_a + f_b, b_a + b_b, wd, byte_fallback=af or wf, estimated=True)

    # ── manifold-constrained hyper-connections ──────────────────────────────
    # Two per block (around attention, around the MoE). The residual stream is
    # n_hc times as wide as the hidden size, so the *activation* traffic here is
    # the cost — the three generated mappings are tiny in weights but every one
    # of them reads a flattened n_hc*d state.
    if spec.hc_width > 1:
        n_hc = spec.hc_width
        wide = n_hc * h
        # A (1 x n_hc), B (n_hc x n_hc) and C (n_hc x 1), all generated from the
        # same normalised state, so one (wide -> 2*n_hc + n_hc**2) projection.
        gen_out = 2 * n_hc + n_hc * n_hc
        # One fused kernel per instance, not one per stage. vLLM's
        # ``mhc_pre_big_fuse_tilelang`` folds the RMSNorm, the Sinkhorn-Knopp
        # projection and the residual mixing together, so the 20 iterations run
        # *inside* a single launch and cost arithmetic rather than launch depth.
        #
        # Worth stating because the unfused reading is not a small error: two
        # dependent launches per iteration, twice per layer, would be ~3.5k
        # serialised launches per step and would dominate a short-context step
        # entirely. The iteration count is real; the launch depth is not.
        iters = max(1, spec.hc_sinkhorn_iters)
        for _ in range(max(1, spec.hc_blocks_per_layer)):
            f_gen, b_gen = _linear(positions, wide, gen_out, aw, ww)
            norm_bytes = 2.0 * positions * wide * aw
            mix_flops = 2.0 * positions * n_hc * n_hc * h
            mix_bytes = 2.0 * positions * wide * aw
            # Sinkhorn arithmetic: alternating row/column normalisation of a
            # per-token n_hc x n_hc matrix, `iters` times, register-resident.
            sinkhorn_flops = 4.0 * positions * n_hc * n_hc * iters
            add(
                "mhc_mix",
                f_gen + mix_flops + sinkhorn_flops,
                b_gen + norm_bytes + mix_bytes,
                wd,
                byte_fallback=af or wf,
                serial_launches=1,
            )

    # ── mixture of experts ──────────────────────────────────────────────────
    # Routing is replicated: every rank scores every expert so it knows what to
    # keep and what to ship. Hash-routed layers pick experts from the token id
    # alone (paper §2.1), so they run no router GEMM at all.
    if layer >= spec.num_hash_layers:
        f, b = _linear(positions, h, spec.n_routed_experts, aw, ww)
        add("moe_router", f, b, wd, byte_fallback=af or wf)

    inter = spec.moe_intermediate_size
    # gate + up + down == three h x inter matrices per expert.
    per_expert_weights = 3.0 * h * inter
    per_position_flops = 6.0 * h * inter  # 2 * (gate + up + down) * h * inter

    if spec.n_shared_experts > 0:
        add(
            "moe_shared",
            per_position_flops * positions * spec.n_shared_experts / tp,
            per_expert_weights * spec.n_shared_experts * ew / tp
            + aw * (positions * h * 2 + positions * inter * 2 * spec.n_shared_experts / tp),
            ed,
            byte_fallback=af or ef,
        )

    # Shared with the dense MoE roofline — one owner for the union term. Note the
    # argument is *positions*, not sequences: under speculative decoding every
    # drafted position routes independently, so drafts widen the expert set the
    # step must fetch.
    distinct = distinct_experts(
        int(positions), spec.n_routed_experts, spec.num_experts_per_tok
    )
    # Under EP a step waits for the rank that drew the most selected experts, so
    # the imbalance factor multiplies the *slowest* rank's share, not the mean.
    skew = sh.ep_imbalance if sh.ep > 1 else 1.0
    add(
        "moe_routed",
        per_position_flops * positions * spec.num_experts_per_tok * skew / es,
        # The union term: arithmetic scales with positions, weight traffic with
        # how many distinct experts the batch woke up — then divided across the
        # ranks that hold them.
        per_expert_weights * distinct * ew * skew / es
        + aw * (positions * h * 2 + positions * inter * 2 * spec.num_experts_per_tok / es),
        ed,
        byte_fallback=af or ef,
    )

    # ── low-rank state update on the tail layers ────────────────────────────
    if layer in spec.dspark_layer_ids:
        rank = spec.dspark_markov_rank
        f_d, b_d = _linear(positions, h, rank // tp, aw, ww)
        f_u, b_u = _linear(positions, rank // tp, h, aw, ww)
        # Coarse: a down-up low-rank pair. The published shape isn't public, so
        # this establishes an order of magnitude and flags itself as an estimate
        # rather than sitting silently in the total.
        add("dspark", f_d + f_u, b_d + b_u, wd, byte_fallback=af or wf, estimated=True)

    # ── cross-rank traffic ──────────────────────────────────────────────────
    # Priced against the interconnect, not HBM, by swapping the bandwidth term.
    # Bandwidth-only: no latency floor, so at decode message sizes these are
    # optimistic and flagged ``estimated``. A zero-bandwidth spec leaves them at
    # t_pred == 0, which ``Graph.has_unpriced_collectives`` surfaces rather than
    # letting a free all-to-all sit in the total.
    if tp > 1 or sh.ep > 1:
        link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)

        def add_link(op: str, byts: float) -> None:
            name = f"{prefix}{op}"
            g.nodes.append(
                PredictedNode(
                    name, layer,
                    roofline(name, 0.0, byts, link, spec.act_dtype, estimated=True, bytes_are_fallback=af),
                )
            )

        if sh.ep > 1:
            # Dispatch each position's hidden state to the ranks owning its
            # experts, then combine the results back. Only the fraction that
            # leaves this rank crosses the link.
            off_rank = (sh.ep - 1) / sh.ep
            add_link(
                "moe_all_to_all",
                2.0 * positions * spec.num_experts_per_tok * h * aw * off_rank,
            )
        if tp > 1:
            # Two all-reduces per layer (post-attention, post-MoE). Ring cost is
            # 2*(tp-1)/tp bytes per participant per reduction.
            add_link("tp_all_reduce", 2.0 * (2.0 * (tp - 1) / tp) * positions * h * aw)


def predict_moe_graph(
    model: SparseMoEModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
) -> Graph:
    """Emit a predicted execution graph for one sparse-MoE decode step, per rank.

    The main stack runs over every position in the step (the verified token plus
    any speculative drafts); the MTP head then runs over one position per
    sequence to propose the next draft.

    With ``sharding`` left at its default the graph is whole-model, as before.
    Given a real sharding it predicts what *one rank* does — which is what that
    rank's kernels are observed doing, and therefore the only thing a residual
    against a per-rank trace can mean.
    """
    spec = model or SparseMoEModelSpec()
    hw = hw or HardwareSpec()
    batch = batch or BatchConfig()
    sh = sharding or ShardingConfig()

    # Refuse a sharding the model cannot actually take. Head counts are floor-
    # divided throughout, so ``tp > n_heads`` silently yields zero heads per rank
    # and prices the entire attention path at zero FLOPs — a cheap, confident,
    # completely wrong graph. vLLM rejects this at construction; so does this.
    if spec.n_heads % max(1, sh.tp) != 0:
        raise ValueError(
            f"tensor-parallel size {sh.tp} does not divide {spec.n_heads} attention "
            "heads — every head-sharded op would floor to zero work"
        )
    if spec.qk_rope_head_dim > spec.head_dim:
        raise ValueError(
            f"qk_rope_head_dim ({spec.qk_rope_head_dim}) exceeds head_dim "
            f"({spec.head_dim}) — the RoPE slice cannot be larger than the head "
            "it is a slice of, and the KV-entry byte split goes negative"
        )

    g = Graph(model=spec, hw=hw, batch=batch, sharding=sh)
    positions = batch.positions_per_step
    sequences = batch.batch
    kv_len = batch.kv_cache_len

    for layer in range(spec.n_layers):
        _emit_layer(
            g, spec, hw, layer,
            positions=positions, sequences=sequences, kv_len=kv_len, sh=sh,
        )

    # Multi-token prediction head: extra layers that draft, running one position
    # per sequence. Their cost is paid every step whether or not the drafts are
    # accepted, which is why ``BatchConfig.acceptance_rate`` belongs in the
    # per-token denominator and not in this node's cost.
    #
    # Deliberately *not* given distinct op names. An MTP layer's q_a projection
    # is a q_a projection; what makes it the draft head is its position in the
    # stack, which ``layer >= n_layers`` already says. Renaming the ops would
    # invent identities that no kernel name can ever match, leaving every MTP
    # node predicted-but-never-observed. Layer index is the disambiguator here,
    # exactly as ``docs/kernel_identity.md`` specifies.
    for i in range(spec.num_nextn_predict_layers):
        _emit_layer(
            g, spec, hw, spec.n_layers + i,
            positions=sequences, sequences=sequences, kv_len=kv_len, sh=sh,
        )

    # Vocabulary projection: logits for every position the step computed, since a
    # speculative step needs a distribution at each drafted position to verify it.
    # Sharded along the vocabulary under TP.
    aw = weight_bytes(spec.act_dtype)
    ww = weight_bytes(spec.weight_dtype)
    af = weight_bytes_is_fallback(spec.act_dtype)
    wf = weight_bytes_is_fallback(spec.weight_dtype)
    f, b = _linear(positions, spec.hidden, spec.vocab // max(1, sh.tp), aw, ww)
    g.nodes.append(
        PredictedNode(
            "lm_head", None,
            roofline("lm_head", f, b, hw, spec.weight_dtype,
                     bytes_are_fallback=af or wf),
        )
    )

    return g


def is_sparse_moe_config(cfg: dict[str, Any]) -> bool:
    """True for the DeepSeek-V4-class checkpoints :func:`predict_moe_graph` models.

    Routed experts alone are not the signal: a Mixtral declares those too but runs
    standard attention, which this graph's compressed-KV latent and indexer nodes
    would mis-price. The discriminator is the sparse-attention machinery — an index
    top-k or a per-layer compression schedule — which only the DSA family carries.
    A config without it belongs to the dense graph, whose FFN already prices a
    mixture.
    """
    routed = cfg.get("n_routed_experts") and cfg.get("num_experts_per_tok")
    sparse_attn = cfg.get("index_topk") or cfg.get("compress_ratios")
    return bool(routed and sparse_attn)


def spec_from_hf_config(cfg: dict[str, Any], *, name: str | None = None) -> SparseMoEModelSpec:
    """Build a :class:`SparseMoEModelSpec` from a HuggingFace ``config.json``.

    Reads the checkpoint's own declared shape so the graph can never drift from
    the model it claims to predict. Quantisation dtypes come from
    ``quantization_config`` (linears) and ``expert_dtype`` (experts), which on a
    mixed-precision checkpoint are genuinely different and must not be collapsed.

    ``compress_ratios`` ships longer than ``num_hidden_layers`` (it carries
    trailing entries for the MTP and padding slots); only the first
    ``num_hidden_layers`` entries index real layers, so the tail is dropped
    rather than silently shifting every layer's ratio.
    """
    errors = validate_sparse_moe_config(cfg)
    if errors:
        raise ValueError("sparse-MoE config is not predictable: " + "; ".join(errors))

    q = cfg.get("quantization_config") or {}
    act_dtype = _canon_dtype(str(cfg["torch_dtype"]))
    weight_dtype = _canon_dtype(str(q.get("quant_method") or act_dtype))
    n_layers = int(cfg["num_hidden_layers"])
    ratios = tuple(int(r) for r in cfg["compress_ratios"][:n_layers])

    return SparseMoEModelSpec(
        name=name or str(cfg.get("model_type", "sparse-moe")),
        hidden=int(cfg["hidden_size"]),
        n_layers=n_layers,
        n_heads=int(cfg["num_attention_heads"]),
        num_kv_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg["head_dim"]),
        qk_rope_head_dim=int(cfg.get("qk_rope_head_dim", 64)),
        q_lora_rank=int(cfg["q_lora_rank"]),
        o_lora_rank=int(cfg["o_lora_rank"]),
        o_groups=int(cfg["o_groups"]),
        vocab=int(cfg["vocab_size"]),
        n_routed_experts=int(cfg["n_routed_experts"]),
        n_shared_experts=int(cfg["n_shared_experts"]),
        num_experts_per_tok=int(cfg["num_experts_per_tok"]),
        moe_intermediate_size=int(cfg["moe_intermediate_size"]),
        index_n_heads=int(cfg["index_n_heads"]),
        index_head_dim=int(cfg["index_head_dim"]),
        index_topk=int(cfg["index_topk"]),
        sliding_window=int(cfg["sliding_window"]),
        compress_ratios=ratios,
        num_nextn_predict_layers=int(cfg.get("num_nextn_predict_layers", 0)),
        # mHC: ``hc_mult`` is n_hc, the residual-stream widening factor.
        hc_width=int(cfg.get("hc_mult", 1) or 1),
        hc_sinkhorn_iters=int(cfg.get("hc_sinkhorn_iters", 0) or 0),
        num_hash_layers=int(cfg.get("num_hash_layers", 0) or 0),
        dspark_layer_ids=tuple(int(i) for i in (cfg.get("dspark_target_layer_ids") or ())),
        dspark_markov_rank=int(cfg.get("dspark_markov_rank", 0)),
        weight_dtype=weight_dtype,
        expert_dtype=_canon_dtype(str(cfg["expert_dtype"])),
        kv_dtype=act_dtype,
        act_dtype=act_dtype,
    )
