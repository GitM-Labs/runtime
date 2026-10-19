"""Catalogue entry: meta-models/Muse-Spark-Open-816B-A42B (``muse-spark-open``).

The canonical repo is gated; the facts below were read from the ungated
unsloth/Muse-Spark-Open-816B-A42B mirror at revision
49cade7b0c6ed297f0434229ac67eea5f9ae1220 (config.json, index total_size and all
65 safetensors headers), cross-checked against the mlx-community 4-bit mirror's
text_config. They are embedded here so the test needs no network.
"""
from __future__ import annotations

import pytest

from gitm.planner.hybrid_graph import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    kv_bytes_per_token,
    kv_fixed_bytes_per_sequence,
    model_weight_bytes,
    predict_hybrid_graph,
)
from gitm.planner.model_catalogue import load_entry, load_spec
from gitm.planner.roofline import BatchConfig, ShardingConfig

ENTRY = "muse-spark-open"

# text_config, as read (unsloth @ 49cade7b; identical in mlx-community 4bit).
TEXT_CONFIG = {
    "hidden_size": 8192,
    "num_hidden_layers": 62,
    "vocab_size": 202048,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "sliding_window": 2048,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "num_shared_experts": 0,
    "moe_hidden_size": 4096,
    "moe_intermediate_size": 4096,
    "layer_types": (["sliding_attention", "full_attention",
                     "sliding_attention", "sliding_attention"] * 16)[:62],
    "layer_rope_theta": ([500000.0, 0, 500000.0, 500000.0] * 16)[:62],
    "mlp_layer_types": ["sparse"] * 62,
}

# Per-layer tensor shapes from the safetensors headers (uniform across layers).
HEADERS = {
    "experts.gate_up_proj": (256, 8192, 4096),
    "experts.down_proj": (256, 4096, 4096),
    "pre_expert_proj": (4096, 8192),
    "post_expert_proj": (8192, 4096),
    "q_proj": (8192, 8192),
    "k_proj": (1024, 8192),
    "v_proj": (1024, 8192),
    "o_proj": (8192, 8192),
    "attn_gate_proj": (8192, 8192),
    "router_f32": (256, 8192),
}
PUBLISHED_TOTAL_SIZE = 1_644_333_137_920
LATENT_PROJ_BYTES = 62 * 2 * 8192 * 4096 * 2          # 8.32 GB
VISION_AUDIO_BYTES = 3_705_279_488 + 83_886_080 + 67_108_864 + 235_680_768
ROUTER_F32_EXTRA = 62 * 256 * 8192 * 2                # stored F32, priced at 2 B
NORMS_AND_GATES_BYTES = (                             # bf16, per layer + final norm
    62 * (4 * 8192 + 2 * 4096) * 2 + 8192 * 2 + 62 * 256 * 4
)


@pytest.fixture(scope="module")
def spec():
    return load_spec(ENTRY)


def test_entry_identity_and_family():
    e = load_entry(ENTRY)
    assert e["name"] == "meta-models/Muse-Spark-Open-816B-A42B"
    assert e["family"] == "hybrid"
    prov = e["provenance"]
    assert prov["verified"] and prov["estimated"] and prov["open"] and prov["unmodelled"]
    # The fold must be flagged as estimated, not passed off as a config value.
    assert any(x["field"] == "moe_intermediate_size" for x in prov["estimated"])
    # The canonical config was not read directly — that must stay an open item.
    assert any("canonical" in q["question"] for q in prov["open"])


def test_shape_matches_config(spec):
    c = TEXT_CONFIG
    assert spec.hidden == c["hidden_size"]
    assert spec.n_layers == c["num_hidden_layers"]
    assert spec.vocab == c["vocab_size"]
    assert spec.n_heads == c["num_attention_heads"]
    assert spec.num_kv_heads == c["num_key_value_heads"]
    assert spec.head_dim == c["head_dim"]
    assert spec.sliding_window == c["sliding_window"]
    assert spec.num_experts == c["num_local_experts"]
    assert spec.num_experts_per_tok == c["num_experts_per_tok"]
    assert spec.shared_expert_intermediate_size == 0
    assert not spec.dense_layers
    assert spec.mtp_num_hidden_layers == 0
    assert tuple(spec.layer_types) == tuple(c["layer_types"])


def test_attention_shapes_match_headers(spec):
    assert spec.q_dim == HEADERS["q_proj"][0]
    assert spec.kv_dim == HEADERS["k_proj"][0] == HEADERS["v_proj"][0]
    assert spec.o_proj_in == HEADERS["o_proj"][1]
    # gate_proj is a q-width output gate.
    assert spec.attn_output_gate and HEADERS["attn_gate_proj"][0] == spec.q_dim


def test_schedule_full_layers_are_the_nope_layers(spec):
    full = [i for i in range(spec.n_layers) if spec.is_full_attention(i)]
    nope = [i for i, t in enumerate(TEXT_CONFIG["layer_rope_theta"]) if t == 0]
    assert full == nope == list(range(1, 62, 4))
    assert spec.n_full_attention_layers == 16
    assert spec.n_sliding_attention_layers == 46
    assert spec.n_linear_attention_layers == 0
    assert spec.layer_kind(0) == SLIDING_ATTENTION
    assert spec.layer_kind(61) == FULL_ATTENTION


def test_latent_expert_fold_is_parameter_exact(spec):
    """The experts run in a 4096 latent; the spec prices 3 x hidden x I per
    expert, so I=2048 must reproduce the header shapes exactly."""
    e, two_i, d_lat = HEADERS["experts.gate_up_proj"]
    _, d_lat2, i_lat = HEADERS["experts.down_proj"]
    assert d_lat == TEXT_CONFIG["moe_hidden_size"] == d_lat2
    assert two_i == 2 * TEXT_CONFIG["moe_intermediate_size"]
    real_per_expert = two_i * d_lat + d_lat2 * i_lat
    assert 3 * spec.hidden * spec.moe_intermediate_size == real_per_expert
    # Transcribing the config's 4096 would double the expert bank.
    assert 3 * spec.hidden * TEXT_CONFIG["moe_intermediate_size"] == 2 * real_per_expert
    # And there really is a projection into and out of the latent space.
    assert HEADERS["pre_expert_proj"] == (d_lat, spec.hidden)
    assert HEADERS["post_expert_proj"] == (spec.hidden, d_lat)


def test_parameter_count_reconciles_with_name(spec):
    experts = 62 * spec.num_experts * 3 * spec.hidden * spec.moe_intermediate_size
    attn = 62 * spec.hidden * (3 * spec.q_dim + 2 * spec.kv_dim)  # q, gate, o + k, v
    latent = LATENT_PROJ_BYTES // 2
    router = 62 * spec.num_experts * spec.hidden
    total_ex_embed = experts + attn + latent + router
    assert 816e9 <= total_ex_embed < 817e9                    # "816B"
    active = experts * spec.top_k // spec.num_experts + attn + latent + router
    assert 42e9 <= active < 43e9                              # "A42B"


def test_predicted_weight_bytes_vs_published(spec):
    w = model_weight_bytes(spec)
    delta = (w - PUBLISHED_TOTAL_SIZE) / PUBLISHED_TOTAL_SIZE
    assert -0.01 < delta < 0.0  # -0.77%
    # The residual is exactly the itemised unmodelled bytes.
    missing = LATENT_PROJ_BYTES + VISION_AUDIO_BYTES + ROUTER_F32_EXTRA + NORMS_AND_GATES_BYTES
    assert w + missing == pytest.approx(PUBLISHED_TOTAL_SIZE, rel=1e-6)


def test_kv_rates(spec):
    assert kv_bytes_per_token(spec) == 16 * 16 * 128 * 2
    assert kv_fixed_bytes_per_sequence(spec) == 46 * 2048 * 16 * 128 * 2


def test_router_is_priced_fp32(spec):
    assert spec.dtype_for("moe_router", spec.weight_dtype) == "fp32"


def test_decode_graph_builds_at_tp8(spec):
    g = predict_hybrid_graph(
        spec, batch=BatchConfig(batch=8, kv_cache_len=4096),
        sharding=ShardingConfig(tp=8),
    )
    ops = {n.op for n in g.nodes}
    assert {"qkv_proj", "attn_score_value", "attn_out_proj",
            "moe_router", "moe_routed", "lm_head"} <= ops
    assert "linattn_recurrent" not in ops
    assert "moe_shared" not in ops
    assert sum(1 for n in g.nodes if n.op == "moe_routed") == 62
