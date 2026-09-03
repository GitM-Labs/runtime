"""The GLM-5.2 (``glm_moe_dsa``) decode graph, pinned against the checkpoint.

Every assertion guards a term that separates GLM-5.2 from the DeepSeek-V4 sparse
family it was forked from, or a wiring seam a plausible-but-wrong graph would slip
through:

* IndexShare — only ``full`` layers emit indexer nodes; ``shared`` layers reuse
  the selection and carry no indexer weights,
* MLA KV traffic scales with the shared latent, never ``n_heads``,
* the dense prefix runs an FFN, not a mixture,
* precision is read per op, not per model — bf16 by default, and on the FP8
  checkpoint fp8 everywhere the quantiser went and bf16 where it did not,
* the predicted footprint matches *both* published checkpoints, 1.507 TB bf16 and
  753.33 GB fp8,
* prefill and decode disagree about what ``index_topk`` buys — it bounds the core
  in both phases but bounds the *bytes* in only one,
* the MTP chain is D stages deep with D vocabulary projections, not one of each,
* ``detect_family`` routes ``glm_moe_dsa`` before the structural sparse-MoE test.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import replace

import pytest
import yaml

from gitm.planner.glm_graph import (
    GlmMoeDsaModelSpec,
    core_read_entries,
    is_glm_moe_dsa_config,
    kv_entry_bytes,
    model_weight_bytes,
    predict_glm_graph,
    spec_from_hf_config,
)
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.registry import detect_family
from gitm.planner.roofline import BatchConfig, ShardingConfig

# GLM-5.2's shape, trimmed to the keys the planner reads. Arrays are the real
# checkpoint's schedules (period-4 IndexShare past a 3-layer dense prefix).
GLM_CONFIG = {
    "model_type": "glm_moe_dsa",
    "architectures": ["GlmMoeDsaForCausalLM"],
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 192,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "indexer_types": (
        ["full", "full", "full"]
        + ["shared", "shared", "shared", "full"] * 18
        + ["shared", "shared", "shared"]
    ),
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "first_k_dense_replace": 3,
    "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
    "routed_scaling_factor": 2.5,
    "moe_router_dtype": "float32",
    "num_nextn_predict_layers": 1,
    "index_share_for_mtp_iteration": True,
    "dtype": "bfloat16",
    "vocab_size": 154880,
}


def _spec() -> GlmMoeDsaModelSpec:
    return spec_from_hf_config(GLM_CONFIG, name="GLM-5.2")


def _ops(g, op: str) -> list:
    return [n for n in g.nodes if n.op == op]


def test_config_reader_reads_schedules_verbatim():
    spec = _spec()
    assert spec.n_layers == 78
    assert spec.n_full_indexer_layers == 21  # 3 dense-prefix + 18 period-4
    assert spec.n_sparse_mlp_layers == 75
    # The dense prefix is dense; a mid-stack layer is sparse.
    assert not spec.is_sparse_mlp(0) and spec.is_sparse_mlp(40)
    # Read verbatim, not derived: layer 2 is 'full' though 2 % 4 != 0.
    assert spec.is_full_indexer(2)
    assert not spec.is_full_indexer(3)


def test_indexshare_shared_layers_emit_no_indexer():
    """The mechanism the fork exists to price: 57 of 78 layers skip the indexer."""
    spec = _spec()
    g = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    n_proj = len(_ops(g, "attn_index_proj"))
    n_score = len(_ops(g, "attn_index_score"))
    # One per full-indexer layer, and the MTP head shares (no indexer node).
    assert n_proj == n_score == spec.n_full_indexer_layers == 21
    # Every layer still runs the attention core over the selected positions.
    assert len(_ops(g, "attn_score_value")) == spec.n_layers


def test_mla_kv_traffic_uses_shared_latent_not_heads():
    """The classic MLA error: charging KV as ``n_heads * head_dim``.

    The cache holds one latent per token, shared across all 64 query heads, so the
    per-entry byte count is ``kv_lora_rank + qk_rope`` — independent of ``n_heads``.
    """
    spec = _spec()
    entry = kv_entry_bytes(spec)
    assert entry == (512 + 64) * 2  # bf16 latent + bf16 rope key
    # A per-head (GQA-style) K+V reading would be tens of times larger.
    gqa_wrong = spec.n_heads * (spec.q_head_dim + spec.v_head_dim) * 2
    assert gqa_wrong > 25 * entry


def test_dense_prefix_runs_ffn_not_mixture():
    spec = _spec()
    g = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    # Exactly the 3 dense layers carry an mlp_gate_up/down; the router runs on the
    # 75 sparse layers plus the sparse MTP head.
    assert len(_ops(g, "mlp_gate_up")) == 3
    # Two per sparse block: the h->256 GEMM, then the fused gating kernel that
    # scores and selects. Same op name because the trace cannot tell them apart
    # (see MOE_LAYER_NODES), so the count is doubled, not the node list.
    assert len(_ops(g, "moe_router")) == 2 * spec.n_sparse_mlp_layers


def test_precision_is_bf16_no_fp4_leak():
    """Forked from a fp4-expert default; a leak would deflate the dominant term."""
    spec = _spec()
    assert spec.weight_dtype == "bf16"
    assert spec.expert_dtype == "bf16"
    assert spec.kv_dtype == "bf16"
    # fp32 even on the unquantised checkpoint: moe_router_dtype is a base-config
    # field, so the router's precision is a model fact, not a deployment one.
    assert spec.dtype_for("moe_router", spec.weight_dtype) == "fp32"


def test_fp8_checkpoint_reads_what_the_quantiser_skipped():
    """One dtype per model is a fiction here; the checkpoint says which ops differ.

    ``modules_to_not_convert`` is the authority, and the interesting entries are
    the ones that invert the usual fp8-backbone layout: ``o_proj`` is quantised
    (absent from the list) while the *indexer* is not.
    """
    cfg = dict(GLM_CONFIG)
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "model.layers.0.input_layernorm",
            "model.layers.47.mlp.gate.e_score_correction_bias",
            "model.layers.74.self_attn.indexers_proj",
            "model.layers.74.self_attn.indexer.k_norm",
            "model.layers.78.eh_proj",
            "lm_head",
        ],
    }
    spec = spec_from_hf_config(cfg, name="GLM-5.2-FP8")
    assert spec.weight_dtype == "fp8" and spec.expert_dtype == "fp8"
    # Quantised: the backbone GEMMs and the experts.
    for op in ("attn_q_b", "attn_kv_b", "attn_out_proj", "moe_routed"):
        assert spec.dtype_for(op, spec.weight_dtype) == "fp8", op
    # Skipped: the vocabulary projection, the MTP fusion, and the indexer.
    for op in ("lm_head", "mtp_eh_proj", "attn_index_proj"):
        assert spec.dtype_for(op, spec.weight_dtype) == "bf16", op
    assert spec.dtype_for("moe_router", spec.weight_dtype) == "fp32"


def test_a_skipped_norm_does_not_imply_a_skipped_projection():
    """``indexer.k_norm`` in the not-convert list says nothing about the GEMM.

    Every fp8 scheme leaves norms wide, so a norm appearing there carries no
    information. A bare ``indexer`` needle matched it and marked the whole indexer
    bf16 — right for GLM-5.2, where ``indexers_proj`` is listed too, and wrong for
    any checkpoint that skipped only the norm.
    """
    cfg = dict(GLM_CONFIG)
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "modules_to_not_convert": ["model.layers.74.self_attn.indexer.k_norm"],
    }
    spec = spec_from_hf_config(cfg)
    assert spec.dtype_for("attn_index_proj", spec.weight_dtype) == "fp8"

    # Name the projection and it is honoured.
    cfg["quantization_config"]["modules_to_not_convert"].append(
        "model.layers.74.self_attn.indexers_proj"
    )
    spec = spec_from_hf_config(cfg)
    assert spec.dtype_for("attn_index_proj", spec.weight_dtype) == "bf16"


def test_embed_tokens_carries_its_own_declared_precision():
    """The untied halves are two tensors and the checkpoint names them separately.

    ``embed_tokens`` used to map onto the ``lm_head`` op, which priced the pair
    together. That is right for GLM-5.2, where both are in
    ``modules_to_not_convert`` — and silently wrong for any checkpoint that
    quantised one and not the other, with ``dtype_for("embed_tokens")`` answering
    fp8 on a model that explicitly does not convert it.
    """
    fp8 = load_spec("glm-5.2-fp8")
    assert fp8.dtype_for("embed_tokens", fp8.weight_dtype) == "bf16"
    assert fp8.dtype_for("lm_head", fp8.weight_dtype) == "bf16"

    # The gather reads the table, so the node runs at the table's width.
    node = [n for n in predict_glm_graph(fp8).nodes if n.op == "embed_tokens"][0]
    assert node.prediction.dtype == "bf16"

    # And the override is load-bearing: quantising the embedding must move the
    # footprint by the size of the table, not by nothing.
    quantised = replace(
        fp8,
        op_dtype_overrides=tuple(
            o for o in fp8.op_dtype_overrides if o[0] != "embed_tokens"
        ),
    )
    table = fp8.vocab * fp8.hidden  # one byte per element saved at fp8
    assert model_weight_bytes(fp8) - model_weight_bytes(quantised) == pytest.approx(
        table, rel=0.01
    )


@pytest.mark.parametrize("shares", [True, False])
def test_footprint_counts_the_indexers_the_graph_emits(shares):
    """The graph and the footprint must agree on how many indexers exist.

    They are computed independently — one walks layers emitting nodes, the other
    sums shapes — so they can disagree silently. They did: the footprint counted
    an indexer for the MTP block while the graph, correctly, emitted none for it,
    because ``index_share_for_mtp_iteration`` means it reuses the main selection
    and carries no indexer tensors. 18.7 MB of weights the checkpoint does not
    have, and a contradiction of the weight-map evidence the note rests on.

    Asserted as an identity rather than a constant, so it holds either way round.
    """
    spec = replace(_spec(), index_share_for_mtp_iteration=shares)
    # With a draft stage actually running — at D=0 the block's weights are
    # resident but none of its kernels launch, so node count and footprint
    # legitimately diverge there and the identity is about the stage that runs.
    g = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096,
                                                  speculative_tokens=1))
    emitted = len(_ops(g, "attn_index_proj"))
    expected = spec.n_full_indexer_layers + (0 if shares else spec.num_nextn_predict_layers)
    assert emitted == expected

    # And the footprint moves by exactly one indexer between the two readings.
    one = (
        spec.q_lora_rank * spec.index_n_heads * spec.index_head_dim
        + spec.hidden * spec.index_head_dim
        + spec.hidden * spec.index_n_heads
    ) * 2  # bf16
    shared_spec = replace(spec, index_share_for_mtp_iteration=True)
    own_spec = replace(spec, index_share_for_mtp_iteration=False)
    assert model_weight_bytes(own_spec) - model_weight_bytes(shared_spec) == pytest.approx(one)


def test_fp8_footprint_matches_published_checkpoint():
    """The same shape arithmetic, checked against a second published precision.

    Two checkpoints agreeing to under half a percent is a stronger check than
    either alone: an error in the shape would have to be precision-proportional
    to survive both.
    """
    published_fp8 = 753_329_940_480  # 141 shards, zai-org/GLM-5.2-FP8
    spec = replace(
        _spec(), weight_dtype="fp8", expert_dtype="fp8",
        op_dtype_overrides=(
            ("attn_index_proj", "bf16"), ("lm_head", "bf16"),
            ("mtp_eh_proj", "bf16"), ("moe_router", "fp32"),
        ),
    )
    assert abs(model_weight_bytes(spec) / published_fp8 - 1.0) < 0.01
    # And the overrides are load-bearing, not decorative: pricing lm_head and the
    # indexer at fp8 loses ~1 GB of real resident weight.
    naive = replace(spec, op_dtype_overrides=())
    assert model_weight_bytes(spec) - model_weight_bytes(naive) > 1e9


def test_prefill_core_streams_the_cache_that_decode_only_samples():
    """``index_topk`` bounds the core's FLOPs in both phases — its bytes in one.

    At decode a sequence reads its own top-2048 selection. At prefill every query
    in the chunk selects a different top-2048 and their union is the whole
    history, so the kernel streams the entire cache. A prefill path copied from a
    dense family would charge ``P x index_topk`` here and understate long-context
    prefill traffic by the ratio of context to 2,048.
    """
    spec = _spec()
    ctx = 65536
    dec = BatchConfig(batch=1, kv_cache_len=ctx)
    pre = BatchConfig(batch=1, kv_cache_len=ctx, prefill_tokens=4096,
                      prefill_context=ctx, prefill_requests=1)
    assert core_read_entries(spec, dec) == spec.index_topk
    # Prefill adds the whole history once per request, not another top-k window.
    assert core_read_entries(spec, pre) == spec.index_topk + ctx + 4096


def test_prefill_scales_projections_by_chunk_but_not_the_epilogue():
    """Rows and logits rows are different numbers, and lm_head follows the second."""
    spec = _spec()
    g = predict_glm_graph(
        spec,
        batch=BatchConfig(batch=1, kv_cache_len=4096, prefill_tokens=8192,
                          prefill_requests=2),
    )
    dec = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))

    def flops(graph, op, layer=None):
        return sum(
            n.prediction.flops for n in graph.nodes
            if n.op == op and (layer is None or n.layer == layer)
        )

    # A backbone projection scales with every row in the step: 1 decode position
    # plus the 8,192-token chunk riding along with it. Read off one layer — the
    # draft stage in the same graph runs at one row and no prefill, which is the
    # point of keeping the two row counts apart.
    assert flops(g, "attn_q_a", layer=3) == pytest.approx(
        8193 * flops(dec, "attn_q_a", layer=3)
    )
    # The vocabulary projection scales with rows that need logits: 1 per
    # prefilling request plus the decode position, so 3 — not 8193. Charging the
    # chunk here is the largest single error available on this path.
    epi_pre = [n for n in g.nodes if n.op == "lm_head" and n.layer is None]
    epi_dec = [n for n in dec.nodes if n.op == "lm_head" and n.layer is None]
    assert epi_pre[0].prediction.flops == pytest.approx(
        3 * epi_dec[0].prediction.flops, rel=1e-6
    )


def test_mtp_chain_is_d_deep_with_its_own_vocab_projection():
    """Verify is the backbone at 1+D rows; the draft is D serial stages.

    ``num_nextn_predict_layers`` is 1 — one *module*, invoked once per drafted
    token. Emitting one draft block and one lm_head for a D-deep chain understates
    the draft by D, and the vocabulary projection is the majority of its bytes.
    """
    spec = _spec()
    d = 5  # the vendor recipe's --speculative-config.num_speculative_tokens
    g = predict_glm_graph(
        spec, batch=BatchConfig(batch=8, kv_cache_len=8192, speculative_tokens=d)
    )
    # One epilogue projection for the verify pass, plus one per draft stage.
    assert len(_ops(g, "lm_head")) == 1 + d
    assert len(_ops(g, "mtp_eh_proj")) == d
    # The backbone still runs once — at 1+D rows, not 1+D times.
    assert len(_ops(g, "attn_q_a")) == spec.n_layers + d

    # The draft carries no indexer: index_share_for_mtp_iteration is true and the
    # MTP block has no indexer tensors in the weight map.
    assert len(_ops(g, "attn_index_proj")) == spec.n_full_indexer_layers

    # And the draft is not a small copy of the model: its expert bank is a full
    # 256-expert mixture, so the chain's cost is weight traffic paid D times.
    assert len(_ops(g, "moe_routed")) == spec.n_sparse_mlp_layers + d


def test_two_collectives_per_layer_not_one():
    """A TP layer all-reduces after o_proj and again after the FFN combine.

    Folding them into one node with double the payload gets the bytes right and
    the count wrong — and at decode payloads a collective is bounded by its ring
    latency, so the count is the cost.
    """
    spec = _spec()
    g = predict_glm_graph(
        spec, batch=BatchConfig(batch=1, kv_cache_len=4096),
        sharding=ShardingConfig(tp=8),
    )
    n_blocks = spec.n_layers  # no draft stage without --spec-tokens
    assert len(_ops(g, "tp_all_reduce_attn")) == n_blocks
    assert len(_ops(g, "tp_all_reduce_mlp")) == n_blocks


def test_dense_layers_dispatch_no_experts():
    """Only a mixture layer sends tokens to expert ranks.

    The three dense-FFN layers compute their whole FFN locally. Charging them an
    expert-parallel all-to-all puts wire traffic on a block with no experts to
    send anything to — and on this model that node is half of prefill, so a
    spurious three layers of it is not a rounding error.
    """
    spec = _spec()
    g = predict_glm_graph(
        spec, batch=BatchConfig(batch=1, kv_cache_len=4096),
        sharding=ShardingConfig(tp=8, ep=8),
    )
    assert len(_ops(g, "moe_all_to_all")) == spec.n_sparse_mlp_layers
    assert not [n for n in g.nodes if n.op == "moe_all_to_all" and n.layer < 3]


def test_unpriced_collectives_stay_visible_under_the_launch_floor():
    """A launch floor must not quietly price a collective the SKU cannot price.

    ``has_unpriced_collectives`` detects a node that moves bytes in zero time. If
    every collective carried a 2 us launch cost, an SKU with no interconnect
    bandwidth would report a priced graph and credit a sharded deployment with a
    nearly-free all-reduce.
    """
    from gitm.planner.roofline import HardwareSpec

    g = predict_glm_graph(
        _spec(), HardwareSpec(interconnect_bw_bytes_per_s=0.0),
        batch=BatchConfig(batch=1, kv_cache_len=4096),
        sharding=ShardingConfig(tp=8),
    )
    assert g.has_unpriced_collectives


def test_footprint_matches_published_checkpoint():
    """Predicted weight bytes within a few percent of the 1.507 TB on disk."""
    published = 1_506_659_919_872  # model.safetensors.index.json total_size
    wb = model_weight_bytes(_spec())
    assert abs(wb / published - 1.0) < 0.03


def test_attention_core_flat_indexer_scan_grows_with_context():
    """DSA: the core is bounded by top-k; only the indexer scan grows."""
    spec = _spec()
    short = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    long = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=131072))

    def total(g, op):
        return sum(n.prediction.t_pred_s for n in g.nodes if n.op == op)

    # Core read is capped at index_topk (2048) — unchanged from 4K to 128K.
    assert total(long, "attn_score_value") == pytest.approx(
        total(short, "attn_score_value"), rel=1e-9
    )
    # The scan scores the whole history, so its *work* grows with context — 32x
    # from 4K to 128K. Asserted on bytes rather than on time: at 4K the scan moves
    # a megabyte across 21 layers and is bounded by its kernel launches, not by
    # its bytes, so predicted time there is a launch floor and cannot grow 32x.
    # That floor is a real property of the node, not an artefact to assert around.
    def total_bytes(g, op):
        return sum(n.prediction.bytes for n in g.nodes if n.op == op)

    assert total_bytes(long, "attn_index_score") == pytest.approx(
        32 * total_bytes(short, "attn_index_score")
    )
    assert total(long, "attn_index_score") > 5 * total(short, "attn_index_score")
    short_scan = [n for n in short.nodes if n.op == "attn_index_score"]
    assert all(n.prediction.bound == "launch" for n in short_scan)


def test_indexer_scans_with_every_index_head():
    """32 query heads against one shared key per token — the head count is a factor.

    ``wk`` produces a single 128-d key per token (MQA-style), which is why the key
    *bytes* carry no head factor and the score *FLOPs* do. Dropping it understates
    the scan 32x and leaves the one node that grows with context looking free.
    """
    spec = _spec()
    batch = BatchConfig(batch=32, kv_cache_len=8192)
    g = predict_glm_graph(spec, batch=batch)
    scan = [n for n in g.nodes if n.op == "attn_index_score"][0]
    pairs = 32 * 8192
    assert scan.prediction.flops == pytest.approx(
        2.0 * pairs * spec.index_n_heads * spec.index_head_dim
    )
    # The cached keys are read once per sequence and are NOT per-head.
    assert scan.prediction.bytes == pytest.approx(pairs * spec.index_head_dim * 2)


def test_no_draft_head_without_a_speculative_config():
    """``num_nextn_predict_layers`` says the block exists, not that it runs.

    Drafting happens only under a speculative config. At D=0 nothing is drafted,
    so no stage is emitted — the weights stay resident (``model_weight_bytes``
    still counts them) and none of their kernels launch. Emitting one anyway
    charged a pure decode step 0.3 ms of drafting a server without
    ``--speculative-config`` never does.
    """
    spec = _spec()
    batch = BatchConfig(batch=32, kv_cache_len=8192)
    g = predict_glm_graph(spec, batch=batch)
    assert not [n for n in g.nodes if n.layer is not None and n.layer >= spec.n_layers]
    assert len(_ops(g, "lm_head")) == 1  # the epilogue only
    assert len(_ops(g, "attn_q_a")) == spec.n_layers

    # The weights are still resident either way — that is the distinction.
    assert model_weight_bytes(spec) > model_weight_bytes(
        replace(spec, num_nextn_predict_layers=0)
    )

    # One stage per drafted token once a speculative config exists.
    g5 = predict_glm_graph(spec, batch=replace(batch, speculative_tokens=5))
    assert len(_ops(g5, "mtp_eh_proj")) == 5


def test_a_pure_prefill_step_runs_no_draft_head():
    """The draft proposes continuations; a prefill chunk has nothing to continue.

    Emitting it anyway puts nodes in the graph that never ran, and since a draft
    stage is almost all launch cost at one row, it shows up as a launch facet made
    of absent kernels.
    """
    spec = _spec()
    g = predict_glm_graph(
        spec,
        batch=BatchConfig(batch=0, kv_cache_len=0, prefill_tokens=8192,
                          prefill_requests=1),
    )
    assert not [n for n in g.nodes if n.layer is not None and n.layer >= spec.n_layers]
    assert len(_ops(g, "mtp_eh_proj")) == 0
    # The backbone still runs, and the epilogue still projects one row per prompt.
    assert len(_ops(g, "attn_q_a")) == spec.n_layers
    assert len(_ops(g, "lm_head")) == 1


#: The node a GLM-5.2 MoE layer lowers to, in issue order. Pinned because the
#: design note's whole low-batch argument is a claim about *how many kernels* a
#: layer is, not just how many bytes it moves — and a graph that quietly folds the
#: pointwise work into the GEMM it precedes reports a decode step as memory-bound
#: when it is launch-bound.
MOE_LAYER_NODES = (
    "rms_norm", "act_quant",
    "attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b",
    "attn_score_value", "attn_qnorm_rope_insert", "attn_out_proj",
    "tp_all_reduce_attn", "rms_norm",
    "moe_router", "moe_router", "act_quant",
    "moe_shared", "moe_permute", "moe_routed", "moe_combine",
    "moe_all_to_all", "tp_all_reduce_mlp",
)


def test_layer_lowers_to_the_documented_node_sequence():
    # The FP8 entry, because two of the 24 nodes are the dynamic activation
    # quantisation the bf16 checkpoint does not run.
    spec = load_spec("glm-5.2-fp8")
    g = predict_glm_graph(
        spec, batch=BatchConfig(batch=32, kv_cache_len=8192),
        sharding=ShardingConfig(tp=8, ep=8),
    )
    shared = tuple(n.op for n in g.nodes if n.layer == 5)  # Ls,sh
    assert shared == MOE_LAYER_NODES

    # A full-indexer layer is the same sequence with two nodes inserted.
    full = tuple(n.op for n in g.nodes if n.layer == 6)  # Ls,f
    assert len(full) == len(shared) + 2
    assert "attn_index_proj" in full and "attn_index_score" in full

    # A dense layer swaps the whole mixture for three nodes, and so is the only
    # block in the model with no data-dependent shape and no expert traffic.
    dense = tuple(n.op for n in g.nodes if n.layer == 0)  # Ld,f
    assert {"act_quant", "mlp_gate_up", "mlp_down"} <= set(dense)
    assert "moe_router" not in dense and "moe_all_to_all" not in dense


def test_every_emitted_op_name_resolves_from_a_kernel_name():
    """A node the pairing cannot receive is a prediction that never gets checked.

    ``classify_op`` is the fallback identity for a capture with no NVTX ranges
    (``docs/kernel_identity.md``), and it matches on the kernel name. An op this
    graph emits that no kernel name can classify to would sit in the predicted
    graph permanently unmatched while the real kernel landed as unmodeled — two
    errors in opposite directions, and the per-op residual diff this whole family
    exists to support would be quietly decorative.

    Collectives are the documented exception: NCCL kernel names carry no hint of
    *which* of a layer's two all-reduces they are, so they are matched by the
    coarse taxonomy rather than by op.
    """
    from gitm.optimizer.deviation import _OP_RULES

    g = predict_glm_graph(
        load_spec("glm-5.2-fp8"),
        batch=BatchConfig(batch=32, kv_cache_len=8192, speculative_tokens=2),
        sharding=ShardingConfig(tp=8, ep=8),
    )
    collectives = {"tp_all_reduce_attn", "tp_all_reduce_mlp", "moe_all_to_all",
                   "logits_all_gather"}
    emitted = {n.op for n in g.nodes} - collectives
    assert emitted <= set(_OP_RULES), sorted(emitted - set(_OP_RULES))


def test_prologue_and_epilogue_are_nodes():
    """The step does not begin at layer 0 or end at the last one.

    A gather, a final norm, the vocabulary projection and — under TP — the logits
    all-gather that has to complete before anything can be sampled.
    """
    spec = _spec()
    g = predict_glm_graph(
        spec, batch=BatchConfig(batch=32, kv_cache_len=8192),
        sharding=ShardingConfig(tp=8),
    )
    ends = [n.op for n in g.nodes if n.layer is None]
    assert ends == ["embed_tokens", "rms_norm", "lm_head", "logits_all_gather"]
    # Without TP there is nothing to gather.
    solo = predict_glm_graph(spec, batch=BatchConfig(batch=32, kv_cache_len=8192))
    assert "logits_all_gather" not in [n.op for n in solo.nodes]


def test_act_quant_exists_only_where_a_gemm_is_actually_fp8():
    """Dynamic activation scaling is a kernel the bf16 checkpoint does not run.

    ``activation_scheme: "dynamic"`` means the activation is quantised at run
    time, once per group of fp8 GEMMs sharing an input. On the unquantised
    checkpoint there is nothing to quantise — the sort of difference a single
    model-wide dtype cannot express.
    """
    bf16 = _spec()
    fp8 = replace(
        bf16, weight_dtype="fp8", expert_dtype="fp8",
        op_dtype_overrides=(("lm_head", "bf16"), ("moe_router", "fp32")),
    )
    batch = BatchConfig(batch=32, kv_cache_len=8192)
    assert not _ops(predict_glm_graph(bf16, batch=batch), "act_quant")
    # Two per block: one ahead of the attention GEMMs, one ahead of the FFN —
    # the two groups of fp8 GEMMs, with bf16 work in between.
    assert len(_ops(predict_glm_graph(fp8, batch=batch), "act_quant")) == 2 * fp8.n_layers


def test_plan_warns_that_a_speculative_rate_is_a_ceiling():
    """``tokens_per_step`` counts ``1 + D*alpha``; a verifier accepts a prefix.

    The convention is shared with every family and not this branch's to change, so
    the rate is printed as-is — but a number that is up to 1.8x optimistic at D=5
    must not leave the CLI unlabelled, or the design note's caveat protects only
    the readers who found the design note.
    """
    from gitm.planner.registry import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["glm-5.2-fp8", "--gpu", "H200", "--batch", "32", "--kv-len", "8192",
              "--tp", "8", "--ep", "8", "--spec-tokens", "5",
              "--acceptance-rate", "0.5"])
    out = buf.getvalue()
    assert "speculative step (D=5)" in out
    assert "1.78x optimistic" in out

    # A non-speculative step says nothing, because nothing is being approximated.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["glm-5.2-fp8", "--gpu", "H200", "--batch", "32", "--kv-len", "8192"])
    assert "speculative step" not in buf.getvalue()


def test_detect_family_routes_glm_before_sparse_moe():
    """Both families carry index_topk + n_routed_experts; model_type must win."""
    assert detect_family(GLM_CONFIG) == "glm_moe_dsa"
    assert is_glm_moe_dsa_config(GLM_CONFIG)


@pytest.mark.parametrize("entry", ["glm-5.2", "glm-5.2-fp8"])
def test_catalogue_entry_loads_and_predicts(entry):
    assert entry in available()
    spec = load_spec(entry)
    assert spec.n_layers == 78 and spec.n_full_indexer_layers == 21
    # Frozen and hashable: a schedule or an override that arrived as a list would
    # look fine until something tried to key on the spec.
    assert isinstance(spec.op_dtype_overrides, tuple)
    assert hash(spec)
    g, family = predict(entry, batch=BatchConfig(batch=1, kv_cache_len=4096))
    assert family == "glm_moe_dsa"
    assert g.total_pred_s > 0


def test_catalogue_schedules_cover_the_model_exactly():
    """A schedule one entry short does not fail — it falls through and may be right.

    The missing layers take the modulo fallback, which on GLM-5.2 happens to
    produce the correct count. A plausible total resting on evidence that is not
    there is the exact failure explicit schedules exist to prevent, so the
    catalogue path refuses it rather than accepting the luck.
    """
    published = ["full"] * 3 + (["shared"] * 3 + ["full"]) * 18 + ["shared"] * 3
    for entry in ("glm-5.2", "glm-5.2-fp8"):
        spec = load_spec(entry)
        assert len(spec.indexer_types) == spec.n_layers == 78
        assert list(spec.indexer_types) == published

    from gitm.planner.model_catalogue import CATALOGUE_DIR
    from gitm.planner.model_catalogue import load_spec as _load

    short = yaml.safe_load((CATALOGUE_DIR / "glm-5.2.yaml").read_text())
    short["spec"]["indexer_types"] = short["spec"]["indexer_types"][:-1]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(short, fh)
        path = fh.name
    try:
        with pytest.raises(ValueError, match="must cover the model exactly"):
            _load(path)
    finally:
        os.unlink(path)


def test_fp8_entry_inherits_the_schedule_it_shares():
    """Two precisions of one architecture, and only the dtypes are written twice.

    The FP8 entry ``extends`` the bf16 one. Copying a 78-entry schedule into both
    files is duplicated evidence that can drift apart silently — and it did:
    the schedule was one entry short for a while, in the file that had it twice.
    ``provenance`` is deliberately not inherited; each checkpoint is validated
    against its own published size.
    """
    bf16, fp8 = load_spec("glm-5.2"), load_spec("glm-5.2-fp8")
    assert fp8.indexer_types == bf16.indexer_types
    assert fp8.n_layers == bf16.n_layers and fp8.hidden == bf16.hidden
    assert (fp8.weight_dtype, fp8.kv_dtype) == ("fp8", "fp8")
    assert (bf16.weight_dtype, bf16.kv_dtype) == ("bf16", "bf16")

    entry = load_entry("glm-5.2-fp8")
    fields = {e["field"] for e in entry["provenance"]["estimated"]}
    assert "kv_dtype" in fields  # its own, not the base's


def test_fp8_entry_is_the_deployable_one():
    """Precision is what decides whether the model fits a node, so it is checked.

    bf16 needs ~11 H200s for weights alone; fp8 fits 8 with room for KV. The two
    entries exist to make that comparison, so a drift in either dtype is a real
    regression.
    """
    bf16, fp8 = load_spec("glm-5.2"), load_spec("glm-5.2-fp8")
    assert bf16.weight_dtype == "bf16" and fp8.weight_dtype == "fp8"
    per_gpu_h200 = 141e9
    assert model_weight_bytes(bf16) / per_gpu_h200 > 8
    assert model_weight_bytes(fp8) / per_gpu_h200 < 8


def test_tp_must_divide_heads():
    spec = _spec()
    with pytest.raises(ValueError, match="does not divide"):
        predict_glm_graph(spec, sharding=ShardingConfig(tp=7))


def test_missing_required_field_refuses_default():
    broken = {k: v for k, v in GLM_CONFIG.items() if k != "kv_lora_rank"}
    with pytest.raises(ValueError, match="kv_lora_rank"):
        spec_from_hf_config(broken)
