"""The ``dense`` catalogue family, its HF config reader, and Llama-3.3-70B.

The dense graph (``ModelSpec`` + ``predict_graph``) existed long before it was
reachable from the catalogue or ``gitm plan``. These tests pin the three ways in
— catalogue id, raw ``config.json``, Python API — and the guard that matters
most: ``detect_family`` answers ``"dense"`` as a *fallback*, so the dense reader
must refuse MoE / MLA / linear-attention configs instead of pricing them as a
plain transformer.
"""
from __future__ import annotations

import json

import pytest

from gitm.planner.graph import DenseConfigUnsupported
from gitm.planner.graph import spec_from_hf_config as dense_spec_from_hf_config
from gitm.planner.model_catalogue import load_entry, load_spec, predict
from gitm.planner.registry import (
    detect_family,
    main,
    predict_for_config,
    spec_from_hf_config,
)
from gitm.planner.roofline import BatchConfig, ModelSpec, ShardingConfig

LLAMA = "llama-3.3-70b-instruct"
#: model.safetensors.index.json total_size, unsloth/Llama-3.3-70B-Instruct
#: @99cd0d2c829e92a67c844f9144c2509632e5c87f (ungated mirror of meta-llama's).
LLAMA_TOTAL_SIZE = 141_107_412_992

#: config.json of that mirror, verbatim minus fields no reader looks at.
LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "torch_dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "vocab_size": 128256,
}


@pytest.fixture
def llama_config_path(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(LLAMA_CONFIG))
    return p


# ── catalogue family ────────────────────────────────────────────────────────


def test_llama_entry_declares_the_dense_family():
    entry = load_entry(LLAMA)
    assert entry["family"] == "dense"
    assert entry["name"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert {"verified", "estimated", "unmodelled"} <= set(entry["provenance"])


def test_llama_entry_builds_the_published_shape():
    spec = load_spec(LLAMA)
    assert isinstance(spec, ModelSpec)
    assert (spec.n_layers, spec.hidden, spec.n_heads, spec.num_kv_heads,
            spec.head_dim, spec.intermediate, spec.vocab) == (
        80, 8192, 64, 8, 128, 28672, 128256)
    assert spec.w_bytes == 2
    assert not spec.is_moe and not spec.is_hybrid_attention


def test_llama_weight_bytes_match_total_size_near_exactly():
    """A dense bf16 checkpoint has nothing to approximate: the only bytes the
    accounting omits are the RMSNorm weights, (2*80+1)*8192*2 B."""
    spec = load_spec(LLAMA)
    predicted = spec.total_params * spec.w_bytes
    norms = (2 * spec.n_layers + 1) * spec.hidden * 2
    assert predicted + norms == LLAMA_TOTAL_SIZE
    assert abs(predicted - LLAMA_TOTAL_SIZE) / LLAMA_TOTAL_SIZE < 1e-4


def test_dense_entry_predicts_one_decode_step():
    g, family = predict(LLAMA, batch=BatchConfig(batch=1, kv_cache_len=128))
    assert family == "dense"
    # 5 nodes per layer + lm_head.
    assert len(g.nodes) == 80 * 5 + 1
    assert {n.op for n in g.nodes} == {
        "qkv_proj", "attn_score_value", "attn_out_proj",
        "mlp_gate_up", "mlp_down", "lm_head"}
    assert g.total_pred_s > 0


def test_dense_family_rejects_unknown_spec_keys(tmp_path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("name: x\nfamily: dense\nspec:\n  hidden: 64\n  n_kv_heads: 8\n")
    with pytest.raises(ValueError, match="unknown spec field"):
        load_spec(bad)


@pytest.mark.parametrize("kw", [
    {"sharding": ShardingConfig(tp=2)},
    {"sharding": ShardingConfig(ep=2)},
    {"batch": BatchConfig(prefill_tokens=512)},
    {"batch": BatchConfig(speculative_tokens=2)},
])
def test_dense_refuses_what_it_does_not_model(kw):
    """Silently answering TP=2 with a TP=1 graph would read as a 2x slower rank."""
    with pytest.raises(ValueError):
        predict(LLAMA, **kw)


def test_dense_accepts_data_parallel():
    g1, _ = predict(LLAMA)
    g2, _ = predict(LLAMA, sharding=ShardingConfig(dp=4))
    assert g1.total_pred_s == g2.total_pred_s


# ── HF config reader ────────────────────────────────────────────────────────


def test_reader_builds_the_same_spec_as_the_catalogue_entry():
    spec = spec_from_hf_config(LLAMA_CONFIG, name="meta-llama/Llama-3.3-70B-Instruct")
    assert spec == load_spec(LLAMA)


def test_predict_for_config_on_a_llama_config():
    g, family = predict_for_config(LLAMA_CONFIG)
    assert family == "dense"
    assert len(g.nodes) == 401


@pytest.mark.parametrize("extra", [
    {"num_local_experts": 128, "num_experts_per_tok": 4},           # gpt-oss / Mixtral
    {"n_routed_experts": 256, "num_experts_per_tok": 8},             # DeepSeek-ish
    {"num_experts": 64, "moe_intermediate_size": 1024},              # Qwen-MoE
    {"q_lora_rank": 1536, "kv_lora_rank": 512},                      # MLA
    {"layer_types": ["sliding_attention", "full_attention"] * 40},   # mixed attention
    {"sliding_window": 4096},                                        # Mistral-v0.1
    {"quantization_config": {"quant_method": "fp8"}},
    {"torch_dtype": "float8_e4m3fn"},
])
def test_reader_refuses_configs_that_only_fall_through_to_dense(extra):
    cfg = {**LLAMA_CONFIG, **extra}
    with pytest.raises(DenseConfigUnsupported):
        dense_spec_from_hf_config(cfg)
    # Both exception bases, so existing callers' except clauses keep working.
    with pytest.raises(ValueError):
        predict_for_config(cfg)
    with pytest.raises(NotImplementedError):
        predict_for_config(cfg)


def test_reader_accepts_an_inactive_sliding_window():
    """Qwen2-style: sliding_window set, use_sliding_window false."""
    cfg = {**LLAMA_CONFIG, "sliding_window": 32768, "use_sliding_window": False}
    assert dense_spec_from_hf_config(cfg).n_layers == 80


@pytest.mark.parametrize("key", ["hidden_size", "vocab_size", "intermediate_size",
                                 "num_hidden_layers", "num_attention_heads"])
def test_reader_never_defaults_a_missing_shape(key):
    cfg = {k: v for k, v in LLAMA_CONFIG.items() if k != key}
    with pytest.raises(DenseConfigUnsupported, match=key):
        dense_spec_from_hf_config(cfg)


def test_reader_derives_head_dim_and_kv_heads_when_absent():
    cfg = {k: v for k, v in LLAMA_CONFIG.items()
           if k not in ("head_dim", "num_key_value_heads")}
    spec = dense_spec_from_hf_config(cfg)
    assert spec.head_dim == 8192 // 64
    assert spec.num_kv_heads == 64      # MHA when the config is silent


# ── gitm plan ───────────────────────────────────────────────────────────────


def test_plan_cli_on_the_catalogue_id(capsys):
    rc = main([LLAMA, "--gpu", "H200", "--batch", "8", "--kv-len", "4096", "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json.loads(out)
    assert result["family"] == "dense"
    assert result["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert len(result["nodes"]) == 401
    assert result["total_pred_s"] == sum(n["t_pred_s"] for n in result["nodes"])


def test_plan_cli_table_on_the_catalogue_id(capsys):
    assert main([LLAMA, "--gpu", "H200"]) == 0
    out = capsys.readouterr().out
    assert "[dense]" in out and "mlp_gate_up" in out


def test_plan_cli_on_a_raw_config_path(llama_config_path, capsys):
    rc = main([str(llama_config_path), "--gpu", "H200", "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    result = json.loads(out)
    assert result["family"] == "dense"
    assert result["provenance"] == "config.json (no provenance)"
    assert len(result["nodes"]) == 401


def test_plan_cli_config_and_catalogue_agree(llama_config_path, capsys):
    main([LLAMA, "--gpu", "H200", "--json"])
    by_id = json.loads(capsys.readouterr().out)
    main([str(llama_config_path), "--gpu", "H200", "--json"])
    by_cfg = json.loads(capsys.readouterr().out)
    assert by_id["total_pred_s"] == pytest.approx(by_cfg["total_pred_s"])


def test_plan_cli_refuses_tp_for_dense(capsys):
    assert main([LLAMA, "--gpu", "H200", "--tp", "2"]) == 2
    assert "unsharded" in capsys.readouterr().out


def test_plan_cli_refuses_an_moe_config_falling_through_to_dense(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({**LLAMA_CONFIG, "num_local_experts": 128,
                             "num_experts_per_tok": 4}))
    assert main([str(p), "--gpu", "H200"]) == 2
    assert "not a plain dense transformer" in capsys.readouterr().out


# ── detect_family did not move ──────────────────────────────────────────────

#: Every non-dense catalogue entry must still route to its own family from a
#: config shaped like its checkpoint. Minimal configs carrying the keys each
#: predicate keys on — the dense reader changed nothing in the detector, and
#: this proves the order still holds.
_ROUTING = [
    ({"model_type": "llama", **LLAMA_CONFIG}, "dense"),
    ({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
      "n_routed_experts": 256, "index_topk": 2048}, "glm_moe_dsa"),
    ({"n_routed_experts": 256, "num_experts_per_tok": 6, "index_topk": 512,
      "num_hidden_layers": 43}, "sparse_moe"),
]


@pytest.mark.parametrize("cfg,family", _ROUTING)
def test_detect_family_routes_known_shapes(cfg, family):
    assert detect_family(cfg) == family


def test_detect_family_routes_a_qwen_next_style_hybrid():
    cfg = {"num_experts": 256, "num_experts_per_tok": 8,
           "full_attention_interval": 4, "num_hidden_layers": 40}
    assert detect_family(cfg) == "hybrid"


def test_detect_family_routes_the_real_deepseek_v32_config():
    """A checked-in, full checkpoint config — not a minimal stand-in."""
    from pathlib import Path

    cfg = json.loads((Path(__file__).parent / "fixtures" /
                      "deepseek_v32_config.json").read_text())
    assert detect_family(cfg) == "glm_moe_dsa"
    assert load_entry("deepseek-v3.2")["family"] == "glm_moe_dsa"
