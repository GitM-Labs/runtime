"""Catalogue entries: DeepSeek-V4-Pro-0813, DeepSeek-V4-Flash-0731, DeepSeek-R1-0528.

Each entry is checked against what was read from its pinned checkpoint (config,
index and every safetensors header): it loads, it predicts, its footprint lands
where the provenance says it does, and the structural facts it encodes (hash
layers, indexer placement, DSpark block count, no-indexer reduction) hold.

The V4 footprints are pinned on the dominant term (routed experts, exact) and
on a stated band for the total, because the remaining residual is a net of
known formula errors listed under ``unmodelled`` — not a fit.
"""

from __future__ import annotations

import pytest

from gitm.planner import glm_graph, moe_graph
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.roofline import BatchConfig, ShardingConfig, weight_bytes

V4_PRO = "deepseek-v4-pro-0813"
V4_FLASH = "deepseek-v4-flash-0731"
R1 = "deepseek-r1-0528"

# Sums of every header's data_offsets at the pinned revisions.
V4_PRO_PAYLOAD = 892_727_580_904  # == index total_size
V4_FLASH_PAYLOAD = 166_878_536_440  # == index total_size
R1_PAYLOAD = 688_574_839_360  # != index total_size (that is 2 x numel)
R1_INDEX_TOTAL_SIZE = 1_369_062_772_000

# Routed-expert bytes (weights + e8m0 scales, layers.* and mtp.*) from headers.
V4_PRO_ROUTED = 862_483_120_128
V4_FLASH_ROUTED = 157_437_394_944


def _routed_bytes(spec) -> float:
    layers = spec.n_layers + spec.num_nextn_predict_layers
    return (layers * spec.n_routed_experts * 3 * spec.hidden
            * spec.moe_intermediate_size * weight_bytes(spec.expert_dtype))


@pytest.mark.parametrize("model", [V4_PRO, V4_FLASH, R1])
def test_entry_loads_predicts_and_is_pinned(model):
    assert model in available()
    entry = load_entry(model)
    pin = entry["checkpoint"]
    assert pin["repository"] == entry["name"]
    assert len(pin["revision"]) == 40
    spec = load_spec(model)
    assert hash(spec)
    g, family = predict(model, batch=BatchConfig(batch=8, kv_cache_len=8192))
    assert family == entry["family"]
    assert g.total_pred_s > 0
    assert not g.has_unpriced_collectives


# ── DeepSeek-V4-Pro-0813 ─────────────────────────────────────────────────────


def test_v4_pro_shape():
    spec = load_spec(V4_PRO)
    assert load_entry(V4_PRO)["family"] == "sparse_moe"
    assert (spec.hidden, spec.n_layers, spec.n_heads) == (7168, 61, 128)
    assert (spec.n_routed_experts, spec.num_experts_per_tok) == (384, 6)
    assert (spec.q_lora_rank, spec.o_groups, spec.index_topk) == (1536, 16, 1024)
    assert (spec.expert_dtype, spec.weight_dtype) == ("fp4", "fp8")
    assert (spec.hc_width, spec.hc_sinkhorn_iters) == (4, 20)
    # 61 layers + 3 window-only DSpark slots.
    assert len(spec.compress_ratios) == 64
    assert spec.compress_ratios[61:] == (0, 0, 0)
    # No window-only layer in the main stack: 0, 1 and odd layers are HCA.
    kinds = [spec.attention_kind(i) for i in range(spec.n_layers)]
    assert "swa" not in kinds
    assert [i for i, k in enumerate(kinds) if k == "csa"] == list(range(2, 61, 2))
    assert moe_graph.kv_fixed_bytes_per_sequence(spec) == 0


def test_v4_pro_is_not_the_dataclass_default():
    from gitm.planner.roofline import SparseMoEModelSpec

    d, spec = SparseMoEModelSpec(), load_spec(V4_PRO)
    assert (d.hidden, d.n_layers, d.n_routed_experts) != (
        spec.hidden, spec.n_layers, spec.n_routed_experts)


@pytest.mark.parametrize("model,n_layers", [(V4_PRO, 61), (V4_FLASH, 43)])
def test_v4_hash_layers_skip_the_router_gemm(model, n_layers):
    spec = load_spec(model)
    assert spec.num_hash_layers == 3
    g, _ = predict(model, batch=BatchConfig(batch=4, kv_cache_len=4096))
    routed = sorted(n.layer for n in g.nodes if n.op == "moe_router")
    # Every layer (main + 3 DSpark blocks) except the hash-routed 0-2.
    assert routed == list(range(3, n_layers + 3))


@pytest.mark.parametrize("model,indexed", [(V4_PRO, 30), (V4_FLASH, 21)])
def test_v4_indexer_only_on_ratio4_layers(model, indexed):
    g, _ = predict(model, batch=BatchConfig(batch=4, kv_cache_len=65536))
    layers = sorted(n.layer for n in g.nodes if n.op == "attn_index_proj")
    assert len(layers) == indexed
    assert all(l % 2 == 0 and l >= 2 for l in layers)


@pytest.mark.parametrize("model", [V4_PRO, V4_FLASH])
def test_v4_dspark_blocks_and_taps(model):
    spec = load_spec(model)
    # Three DSpark blocks per inference/config.json and the mtp.0-2 weights,
    # overriding config.json's num_nextn_predict_layers 1.
    assert spec.num_nextn_predict_layers == 3
    # dspark_target_layer_ids are hidden-state taps, not extra weights.
    assert spec.dspark_layer_ids == ()
    g, _ = predict(model, batch=BatchConfig(batch=4, kv_cache_len=4096))
    assert not any(n.op == "dspark" for n in g.nodes)


@pytest.mark.parametrize("model,routed", [(V4_PRO, V4_PRO_ROUTED), (V4_FLASH, V4_FLASH_ROUTED)])
def test_v4_routed_expert_bytes_are_exact(model, routed):
    """The dominant term: fp4 at 0.53125 B/weight over 64 (46) MoE blocks."""
    assert _routed_bytes(load_spec(model)) == pytest.approx(routed, rel=1e-9)


def test_v4_pro_weight_bytes_vs_published():
    """879.48 GB predicted vs 892.73 GB published: -1.48%, within 2%."""
    pred = moe_graph.model_weight_bytes(load_spec(V4_PRO))
    delta = pred / V4_PRO_PAYLOAD - 1.0
    assert delta == pytest.approx(-0.0148, abs=0.0005)
    assert abs(delta) < 0.02


def test_v4_flash_weight_bytes_vs_published():
    """163.13 GB predicted vs 166.88 GB published: -2.25%, within 3%."""
    pred = moe_graph.model_weight_bytes(load_spec(V4_FLASH))
    delta = pred / V4_FLASH_PAYLOAD - 1.0
    assert delta == pytest.approx(-0.0225, abs=0.0005)
    assert abs(delta) < 0.03


def test_v4_pro_grouped_out_proj_is_known_low():
    """wo_a + wo_b are 11.81 GB on disk (fp8); the planner's formula gives 1/16th.

    Pinned so the day the grouped projection is fixed in moe_graph, this test
    (and the provenance) has to change with it.
    """
    s = load_spec(V4_PRO)
    layers = s.n_layers + s.num_nextn_predict_layers
    planner = layers * (s.n_heads * s.head_dim * s.o_lora_rank / s.o_groups
                        + s.o_lora_rank * s.hidden)
    real = layers * (s.n_heads * s.head_dim * s.o_lora_rank
                     + s.o_groups * s.o_lora_rank * s.hidden)
    assert real / planner == pytest.approx(16.0)


# ── DeepSeek-V4-Flash-0731 ───────────────────────────────────────────────────


def test_v4_flash_is_not_v41_flash():
    a, b = load_spec(V4_FLASH), load_spec("deepseek-v4.1-flash")
    assert (a.hidden, a.n_layers, a.n_routed_experts) == (4096, 43, 256)
    assert (b.hidden, b.n_layers, b.n_routed_experts) != (4096, 43, 256)
    assert load_entry(V4_FLASH)["name"] != load_entry("deepseek-v4.1-flash")["name"]


def test_v4_flash_backbone_equals_dataclass_defaults():
    """The SparseMoEModelSpec defaults are this model's backbone scalars."""
    from dataclasses import fields

    from gitm.planner.roofline import SparseMoEModelSpec

    d, spec = SparseMoEModelSpec(), load_spec(V4_FLASH)
    differ = {f.name for f in fields(d) if getattr(d, f.name) != getattr(spec, f.name)}
    assert differ == {
        "name", "compress_ratios", "num_nextn_predict_layers", "hc_width",
        "hc_sinkhorn_iters", "num_hash_layers", "dspark_markov_rank",
    }


def test_v4_flash_window_layers():
    spec = load_spec(V4_FLASH)
    assert len(spec.compress_ratios) == 46
    assert [spec.attention_kind(i) for i in (0, 1, 2, 3, 42)] == ["swa", "swa", "csa", "hca", "csa"]
    assert spec.compress_ratios[43:] == (0, 0, 0)


def test_v4_flash_plans_on_b200_without_fallback():
    g, _ = predict(
        V4_FLASH, hw=hardware_spec_for(peak_for_sku("B200")),
        batch=BatchConfig(batch=32, kv_cache_len=8192), sharding=ShardingConfig(tp=2),
    )
    assert not g.has_fallback_peaks and not g.has_unpriced_collectives
    routed = next(n for n in g.nodes if n.op == "moe_routed")
    assert routed.prediction.peak_dtype == "fp4"


# ── DeepSeek-R1-0528 ─────────────────────────────────────────────────────────


def test_r1_shape_and_no_indexer():
    spec = load_spec(R1)
    assert load_entry(R1)["family"] == "glm_moe_dsa"
    assert (spec.hidden, spec.n_layers, spec.n_heads, spec.kv_lora_rank) == (7168, 61, 128, 512)
    assert (spec.n_routed_experts, spec.num_experts_per_tok) == (256, 8)
    assert spec.first_k_dense_replace == 3 and spec.n_sparse_mlp_layers == 58
    assert spec.num_nextn_predict_layers == 1 and spec.mtp_separate_embedding_head
    assert spec.n_full_indexer_layers == 0
    assert spec.index_topk == 163840  # max_position_embeddings
    assert (spec.weight_dtype, spec.expert_dtype) == ("fp8", "fp8")
    g, _ = predict(R1, batch=BatchConfig(batch=16, kv_cache_len=8192))
    assert not any("index" in n.op for n in g.nodes)
    # KV is the MLA latent + RoPE key only, no index key term.
    assert glm_graph.kv_bytes_per_token(spec) == pytest.approx(
        (spec.n_layers) * spec.kv_entry_dim * 2.0)


def test_r1_attention_is_dense_in_context():
    from gitm.planner.glm_graph import core_qk_pairs, core_read_entries

    spec = load_spec(R1)
    d1 = BatchConfig(batch=1, kv_cache_len=4096)
    d2 = BatchConfig(batch=1, kv_cache_len=65536)
    assert core_qk_pairs(spec, d2) / core_qk_pairs(spec, d1) == pytest.approx(16.0)
    assert core_read_entries(spec, d2) / core_read_entries(spec, d1) == pytest.approx(16.0)


def test_r1_weight_bytes_match_stored_payload():
    """688,572,683,040 predicted vs 688,574,839,360 stored: -0.0003%."""
    pred = glm_graph.model_weight_bytes(load_spec(R1))
    assert abs(pred / R1_PAYLOAD - 1.0) < 1e-4
    # The index total_size is 2 x numel, not bytes; comparing against it would
    # read as a -50% miss.
    assert load_entry(R1)["checkpoint"]["index_total_size"] == R1_INDEX_TOTAL_SIZE
    assert pred / R1_INDEX_TOTAL_SIZE == pytest.approx(0.503, abs=0.001)
