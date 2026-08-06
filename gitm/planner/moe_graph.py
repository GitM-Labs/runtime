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

**Context length stops driving the attention core.** Compressed KV plus an indexer
that keeps ``index_topk`` candidates means the attention core reads a bounded
number of tokens no matter how long the sequence gets. What grows with context is
the *indexer scan* — a different node, with a different bound, that the dense graph
would have folded into attention and then blamed on the wrong kernel.

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
    distinct_experts,
    roofline,
    weight_bytes,
)


def effective_kv_tokens(spec: SparseMoEModelSpec, layer: int, kv_len: int) -> int:
    """KV positions the attention *core* reads for ``layer`` at ``kv_len`` context.

    Two layer kinds, and conflating them is the error this function exists to
    avoid:

    **Ratio 0 — sliding-window attention.** The layer is *not* compressed, but it
    is also not global: it reads at most ``sliding_window`` recent tokens. Reading
    ``0`` as "uncompressed, therefore attends to everything" is the natural
    mistake and it is wrong by ``kv_len / sliding_window`` — 512x at 64K, 8192x at
    1M. It is also self-refuting: if any layer read the whole cache, a
    million-token context could not be served at all, which is the headline
    capability of the checkpoint.

    **Ratio > 1 — compressed and selected.** The layer keeps one entry per ``r``
    tokens, an indexer scores those candidates and keeps at most ``index_topk``,
    and the recent ``sliding_window`` tokens are read regardless.

    (Ratio 1 means genuinely uncompressed global attention. No layer of this
    checkpoint uses it; it is kept for other architectures.)

    The consequence worth internalising: **no layer's read grows with context.**
    Once ``kv_len / r`` exceeds ``index_topk``, the core's read is constant, and
    the sliding-window layers were bounded from the start. A layer with ratio 128
    and one with ratio 4 read the same number of tokens at 64K — they differ in
    KV *storage* and in how much the indexer has to scan, not in what attention
    itself costs. So a residual that scales with context belongs to the indexer
    node, and blaming it on attention is the mistake this split prevents.
    """
    if kv_len <= 0:
        return 0
    r = spec.compress_ratio(layer)
    if r == 0:
        return min(kv_len, spec.sliding_window)
    if r == 1:
        return kv_len
    candidates = math.ceil(kv_len / r)
    selected = min(spec.index_topk, candidates)
    return min(kv_len, spec.sliding_window + selected)


def index_candidates(spec: SparseMoEModelSpec, layer: int, kv_len: int) -> int:
    """Positions the indexer must score for ``layer`` — the compressed set.

    Zero for sliding-window layers: a fixed recent window needs no selection, so
    those layers run no indexer at all and the graph emits no indexer node for
    them. Scoring a candidate set there would invent both a kernel and its cost.

    For compressed layers this is unbounded in context length — the node that
    decides whether a million-token deployment is viable, and now the *only* term
    in the attention path that still grows with context.
    """
    if kv_len <= 0:
        return 0
    r = spec.compress_ratio(layer)
    if r == 0:
        return 0
    if r == 1:
        return kv_len
    return math.ceil(kv_len / r)


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
        total += (spec.kv_latent_dim + spec.index_head_dim) * kw / max(1, r)
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
    kw = weight_bytes(spec.kv_dtype)
    swa_layers = sum(1 for i in range(spec.n_layers) if spec.compress_ratio(i) == 0)
    return swa_layers * spec.sliding_window * spec.kv_latent_dim * kw


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
    kw = weight_bytes(spec.kv_dtype)
    wd, ed = spec.weight_dtype, spec.expert_dtype

    def add(op: str, flops: float, byts: float, dtype: str, *, estimated: bool = False) -> None:
        name = f"{prefix}{op}"
        g.nodes.append(
            PredictedNode(name, layer, roofline(name, flops, byts, hw, dtype, estimated=estimated))
        )

    tp = max(1, sh.tp)
    es = max(1, sh.expert_shards)

    # ── attention: low-rank query, compressed KV latent ──────────────────────
    # ``q_a`` and ``kv_a`` are replicated across tensor-parallel ranks: they
    # produce the shared latent, which has nothing to split when there is one KV
    # head. Every rank pays for them in full, so TP's speedup on attention is
    # strictly less than ``tp``.
    f, b = _linear(positions, h, spec.q_lora_rank, aw, ww)
    add("attn_q_a", f, b, wd)

    f, b = _linear(positions, spec.q_lora_rank, spec.n_heads * spec.q_head_dim // tp, aw, ww)
    add("attn_q_b", f, b, wd)

    # KV down-projection, plus the cache write for the positions just computed.
    f, b = _linear(positions, h, spec.kv_latent_dim, aw, ww)
    add("attn_kv_a", f, b + positions * spec.kv_latent_dim * kw, wd)

    # ── indexer: project the query, then scan the compressed candidate set ───
    # Sliding-window layers have no indexer — a fixed recent window needs no
    # selection — so they emit neither node. Emitting them anyway would put two
    # kernels per such layer into the predicted graph that no kernel can ever
    # align to, which reads downstream as "predicted work that never ran".
    cand = index_candidates(spec, layer, kv_len)
    if cand > 0:
        f, b = _linear(positions, h, spec.index_n_heads * spec.index_head_dim // tp, aw, ww)
        add("attn_index_proj", f, b, wd)

        add(
            "attn_index_score",
            2.0 * positions * (spec.index_n_heads // tp) * spec.index_head_dim * cand,
            # Index keys live in the cache: read once per sequence, not per
            # position, and replicated across TP ranks alongside the KV latent.
            sequences * cand * spec.index_head_dim * kw,
            wd,
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
        sequences * t_eff * spec.kv_latent_dim * kw,
        wd,
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
    add("attn_out_proj", f_a + f_b, b_a + b_b, wd, estimated=True)

    # ── mixture of experts ──────────────────────────────────────────────────
    # Routing is replicated: every rank scores every expert so it knows what to
    # keep and what to ship.
    f, b = _linear(positions, h, spec.n_routed_experts, aw, ww)
    add("moe_router", f, b, wd)

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
    )

    # ── low-rank state update on the tail layers ────────────────────────────
    if layer in spec.dspark_layer_ids:
        rank = spec.dspark_markov_rank
        f_d, b_d = _linear(positions, h, rank // tp, aw, ww)
        f_u, b_u = _linear(positions, rank // tp, h, aw, ww)
        # Coarse: a down-up low-rank pair. The published shape isn't public, so
        # this establishes an order of magnitude and flags itself as an estimate
        # rather than sitting silently in the total.
        add("dspark", f_d + f_u, b_d + b_u, wd, estimated=True)

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
                    roofline(name, 0.0, byts, link, spec.act_dtype, estimated=True),
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
    f, b = _linear(positions, spec.hidden, spec.vocab // max(1, sh.tp), aw, ww)
    g.nodes.append(
        PredictedNode("lm_head", None, roofline("lm_head", f, b, hw, spec.weight_dtype))
    )

    return g


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
    q = cfg.get("quantization_config") or {}
    weight_dtype = str(q.get("quant_method", "bf16")).lower()
    n_layers = int(cfg.get("num_hidden_layers", 43))
    ratios = tuple(int(r) for r in (cfg.get("compress_ratios") or ())[:n_layers])

    return SparseMoEModelSpec(
        name=name or str(cfg.get("model_type", "sparse-moe")),
        hidden=int(cfg.get("hidden_size", 4096)),
        n_layers=n_layers,
        n_heads=int(cfg.get("num_attention_heads", 64)),
        num_kv_heads=int(cfg.get("num_key_value_heads", 1)),
        head_dim=int(cfg.get("head_dim", 512)),
        qk_rope_head_dim=int(cfg.get("qk_rope_head_dim", 64)),
        q_lora_rank=int(cfg.get("q_lora_rank", 1024)),
        o_lora_rank=int(cfg.get("o_lora_rank", 1024)),
        o_groups=int(cfg.get("o_groups", 1)),
        vocab=int(cfg.get("vocab_size", 129280)),
        n_routed_experts=int(cfg.get("n_routed_experts", 256)),
        n_shared_experts=int(cfg.get("n_shared_experts", 1)),
        num_experts_per_tok=int(cfg.get("num_experts_per_tok", 6)),
        moe_intermediate_size=int(cfg.get("moe_intermediate_size", 2048)),
        index_n_heads=int(cfg.get("index_n_heads", 64)),
        index_head_dim=int(cfg.get("index_head_dim", 128)),
        index_topk=int(cfg.get("index_topk", 512)),
        sliding_window=int(cfg.get("sliding_window", 0) or 0),
        compress_ratios=ratios,
        num_nextn_predict_layers=int(cfg.get("num_nextn_predict_layers", 0)),
        dspark_layer_ids=tuple(int(i) for i in (cfg.get("dspark_target_layer_ids") or ())),
        dspark_markov_rank=int(cfg.get("dspark_markov_rank", 0)),
        weight_dtype=weight_dtype,
        expert_dtype=str(cfg.get("expert_dtype", weight_dtype)).lower(),
        # vLLM serves this checkpoint with an fp8 KV cache; the config does not
        # declare cache dtype, so it is a serving decision, not a model fact.
        kv_dtype="fp8",
        act_dtype=str(cfg.get("torch_dtype", "bf16")).lower().replace("bfloat16", "bf16"),
    )
