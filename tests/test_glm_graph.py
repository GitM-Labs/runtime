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

from dataclasses import replace

import pytest

from gitm.planner.glm_graph import (
    GlmMoeDsaModelSpec,
    core_read_entries,
    is_glm_moe_dsa_config,
    kv_entry_bytes,
    model_weight_bytes,
    predict_glm_graph,
    spec_from_hf_config,
)
from gitm.planner.model_catalogue import available, load_spec, predict
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
    assert len(_ops(g, "attn_score_value")) == spec.n_layers + spec.num_nextn_predict_layers


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
    assert len(_ops(g, "moe_router")) == spec.n_sparse_mlp_layers + spec.num_nextn_predict_layers


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
    n_blocks = spec.n_layers + spec.num_nextn_predict_layers
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
    assert len(_ops(g, "moe_all_to_all")) == (
        spec.n_sparse_mlp_layers + spec.num_nextn_predict_layers
    )
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
