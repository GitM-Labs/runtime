"""Catalogue entries: kimi-k2.7-code, qwen3.5-397b-a17b(-fp8), qwen3.8-flash-next.

Published sizes are each checkpoint's model.safetensors.index.json total_size,
cross-checked against the sum of every shard's safetensors header. The deltas
asserted here are the ones each YAML documents; a change that moves them should
also update the YAML's reasoning, not only these bounds.
"""

from __future__ import annotations

import pytest

from gitm.planner import glm_graph, hybrid_graph
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.roofline import BatchConfig, ShardingConfig

NEW = ("kimi-k2.7-code", "qwen3.5-397b-a17b", "qwen3.5-397b-a17b-fp8", "qwen3.8-flash-next")

KIMI_K27_TOTAL = 595_148_192_736
QWEN35_TOTAL = 806_795_875_168
QWEN35_FP8_TOTAL = 406_125_181_280
FLASH_NEXT_TOTAL = 359_999_963_128
FLASH_NEXT_NGRAM_TABLE = 128 * 2_500_012 * 160 * 2  # 102,400,491,520 B, from headers


def _h200():
    return hardware_spec_for(peak_for_sku("H200"))


def _delta(predicted: float, published: int) -> float:
    return predicted / published - 1.0


# ── structure common to all four ───────────────────────────────────────────


@pytest.mark.parametrize("model", NEW)
def test_entry_is_catalogued_with_full_provenance(model):
    assert model in available()
    entry = load_entry(model)
    prov = entry["provenance"]
    assert prov["verified"] and prov["estimated"] and prov["unmodelled"]
    assert any("weight bytes" in v["claim"] for v in prov["verified"])


@pytest.mark.parametrize("model", NEW)
def test_entry_predicts_a_graph_on_h200(model):
    g, family = predict(model, _h200(), BatchConfig(batch=4, kv_cache_len=1024),
                        ShardingConfig(tp=8))
    assert family == load_entry(model)["family"]
    assert g.nodes and all(n.prediction.t_pred_s >= 0 for n in g.nodes)


# ── kimi-k2.7-code ─────────────────────────────────────────────────────────


def test_kimi_k27_code_is_kimi_k25_verbatim():
    """Config diff against K2.5 is eos_token_id and two regex anchors; the
    weight map is the same 208,550 tensors. No spec override at all."""
    entry = load_entry("kimi-k2.7-code")
    assert entry["family"] == "glm_moe_dsa"
    assert entry["name"] == "moonshotai/Kimi-K2.7-Code"
    k27, k25 = load_spec("kimi-k2.7-code"), load_spec("kimi-k2.5")
    from dataclasses import fields

    differing = [f.name for f in fields(k27) if getattr(k27, f.name) != getattr(k25, f.name)]
    assert differing == ["name"]
    assert (k27.expert_dtype, k27.index_topk, k27.num_nextn_predict_layers) == ("int4", 262144, 0)


def test_kimi_k27_code_weight_bytes_match_published_checkpoint():
    """594.20 GB predicted vs 595.15 GB; the 0.94 GB vision tower is the gap."""
    d = _delta(glm_graph.model_weight_bytes(load_spec("kimi-k2.7-code")), KIMI_K27_TOTAL)
    assert -0.0025 < d < 0.0


# ── qwen3.5-397b-a17b ──────────────────────────────────────────────────────


def test_qwen35_shape_read_from_config_and_headers():
    s = load_spec("qwen3.5-397b-a17b")
    assert isinstance(s, hybrid_graph.HybridMoEModelSpec)
    assert (s.hidden, s.n_layers, s.vocab) == (4096, 60, 248320)
    assert (s.n_heads, s.num_kv_heads, s.head_dim, s.attn_output_gate) == (32, 2, 256, True)
    assert s.rope_dim == 64
    assert (s.linear_num_key_heads, s.linear_num_value_heads) == (16, 64)
    assert (s.num_experts, s.top_k, s.moe_intermediate_size,
            s.shared_expert_intermediate_size) == (512, 10, 1024, 1024)
    assert [i for i in range(60) if s.is_full_attention(i)] == list(range(3, 60, 4))
    assert s.n_linear_attention_layers == 45 and s.mtp_num_hidden_layers == 1
    # conv1d.weight header [12288, 1, 4]: q + k + v at key/key/value width.
    assert s.conv_channels == 12288 == 2 * s.linear_key_dim + s.linear_value_dim


def test_qwen35_linear_state_is_flat_and_dominated_by_the_fp32_recurrence():
    s = load_spec("qwen3.5-397b-a17b")
    assert s.recurrent_state_elems * 4 == 4_194_304
    assert s.linear_state_bytes_per_layer == 4_194_304 + 12288 * 3 * 2
    # 15 full-attention layers x 2 KV heads x (256 + 256) x bf16.
    assert hybrid_graph.kv_bytes_per_token(load_spec("qwen3.5-397b-a17b")) == 15 * 2 * 512 * 2


def test_qwen35_weight_bytes_against_published_checkpoint():
    """794.90 GB vs 806.80 GB, -1.47%: MTP (-13.19 GB) and vision (-0.91 GB)
    are omitted, the DeltaNet query rule over-counts +2.26 GB."""
    d = _delta(hybrid_graph.model_weight_bytes(load_spec("qwen3.5-397b-a17b")), QWEN35_TOTAL)
    assert -0.02 < d < -0.01


def test_qwen35_fp8_inherits_shape_and_overrides_precision():
    fp8, bf16 = load_spec("qwen3.5-397b-a17b-fp8"), load_spec("qwen3.5-397b-a17b")
    assert load_entry("qwen3.5-397b-a17b-fp8")["family"] == "hybrid"
    for f in ("hidden", "n_layers", "layer_types", "num_experts", "linear_conv_channels",
              "linear_num_value_heads", "mtp_num_hidden_layers"):
        assert getattr(fp8, f) == getattr(bf16, f)
    assert fp8.weight_dtype == "fp8"
    assert fp8.dtype_for("moe_router", fp8.weight_dtype) == "bf16"
    assert fp8.dtype_for("lm_head", fp8.weight_dtype) == "bf16"
    assert fp8.dtype_for("qkv_proj", fp8.weight_dtype) == "fp8"


def test_qwen35_fp8_weight_bytes_against_published_checkpoint():
    """397.55 GB vs 406.13 GB, -2.11%: MTP, vision, and bf16 embed/lm_head/router
    priced at fp8 by model_weight_bytes."""
    d = _delta(hybrid_graph.model_weight_bytes(load_spec("qwen3.5-397b-a17b-fp8")),
               QWEN35_FP8_TOTAL)
    assert -0.03 < d < -0.015


def test_qwen35_fp8_lm_head_and_router_compute_at_bf16():
    """The overrides reach the graph as compute dtype. Their weight bytes stay
    at fp8 width — a shared-code gap the entry lists under unmodelled."""
    g, _ = predict("qwen3.5-397b-a17b-fp8", _h200(), BatchConfig(batch=1, kv_cache_len=128),
                   ShardingConfig(tp=8))
    for op in ("lm_head", "moe_router"):
        nodes = [n for n in g.nodes if n.op == op]
        assert nodes and all(n.prediction.dtype == "bf16" for n in nodes), op
    qkv = [n for n in g.nodes if n.op == "qkv_proj"]
    assert qkv and all(n.prediction.dtype == "fp8" for n in qkv)


# ── qwen3.8-flash-next ─────────────────────────────────────────────────────


def test_flash_next_backbone_read_from_config_and_headers():
    s = load_spec("qwen3.8-flash-next")
    assert (s.hidden, s.n_layers, s.vocab) == (2560, 48, 248320)
    # q_proj header [12288, 2560] = 24 x 256 query + 24 x 256 sigmoid gate.
    assert s.q_dim == 6144 and s.attn_output_gate
    assert (s.linear_num_key_heads, s.linear_num_value_heads) == (16, 48)
    assert s.conv_channels == 10240 == 2 * s.linear_key_dim + s.linear_value_dim
    assert (s.num_experts, s.top_k, s.moe_intermediate_size) == (512, 10, 640)
    assert [i for i in range(48) if s.is_full_attention(i)] == list(range(3, 48, 4))
    assert s.mtp_layer_kind == "full_attention"


def test_flash_next_weight_bytes_miss_exactly_what_the_entry_says():
    """250.84 GB vs 360.00 GB, -30.3%. The n-gram table alone is 102.40 GB;
    against total_size less the table the backbone is within -2.6%."""
    p = hybrid_graph.model_weight_bytes(load_spec("qwen3.8-flash-next"))
    assert -0.31 < _delta(p, FLASH_NEXT_TOTAL) < -0.30
    assert -0.03 < _delta(p, FLASH_NEXT_TOTAL - FLASH_NEXT_NGRAM_TABLE) < -0.02


def test_flash_next_lists_every_new_mechanism_as_unmodelled():
    text = " ".join(load_entry("qwen3.8-flash-next")["provenance"]["unmodelled"]).lower()
    for piece in ("n-gram", "ple", "hyper-connection", "indexer", "mtp", "vision"):
        assert piece in text, piece
    assert "102.40 gb" in text


def test_flash_next_advertised_176b_is_total_less_vision_and_mtp():
    """179.999981459B params in the headers; less 0.449B vision and 2.607B MTP."""
    desc = load_entry("qwen3.8-flash-next")["description"]
    assert "176B" in desc and "360.00 GB" in desc
