"""Tests for reading a live vLLM server's model shape (gitm/serve/model_config.py).

Filesystem-only and GPU-free: a fabricated ``config.json`` on disk plus a discover
``Target`` carrying a fabricated command line stand in for the running server, so the
config-resolution, aliasing, gate, and serving-override logic run on a dev box exactly
as on a pod. The behaviour under test is the one that matters for trust: a config the
planner cannot honestly predict is *refused with the missing keys named*, never
silently defaulted to DeepSeek-V4 numbers.
"""

from __future__ import annotations

import json

from gitm.planner.moe_graph import predict_moe_graph
from gitm.planner.roofline import HardwareSpec
from gitm.serve import discover
from gitm.serve import model_config as mc


def _write_config(root, cfg: dict, name: str = "config.json"):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(cfg))
    return root / name


def _deepseek_cfg(**over) -> dict:
    cfg = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 4,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "vocab_size": 129280,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "sliding_window": 128,
        "compress_ratios": [0, 0, 4, 128],
        "quantization_config": {"quant_method": "fp8"},
        "expert_dtype": "fp4",
        "torch_dtype": "bfloat16",
        "num_nextn_predict_layers": 1,
    }
    cfg.update(over)
    return cfg


def _target(cmdline: list[str], pid: int = 100) -> discover.Target:
    return discover.Target(pid=pid, cmdline=cmdline)


# --- model ref extraction ----------------------------------------------------


def test_model_ref_from_positional_and_flag_forms():
    assert mc.model_ref_from_cmdline(["vllm", "serve", "org/model", "--port", "8000"]) == "org/model"
    assert (
        mc.model_ref_from_cmdline(
            ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "org/model"]
        )
        == "org/model"
    )
    assert mc.model_ref_from_cmdline(["python", "-m", "x", "--model=org/model"]) == "org/model"
    assert mc.model_ref_from_cmdline([]) is None


# --- config path resolution --------------------------------------------------


def test_resolves_local_checkpoint_dir(tmp_path):
    _write_config(tmp_path / "ckpt", _deepseek_cfg())
    got = mc.resolve_config_path(str(tmp_path / "ckpt"))
    assert got == tmp_path / "ckpt" / "config.json"


def test_resolves_from_hf_cache_honouring_environ(tmp_path):
    # A relocated hub cache named by the target's own environment.
    hub = tmp_path / "custom_hub"
    snap = hub / "models--org--Model" / "snapshots" / "abc123"
    _write_config(snap, _deepseek_cfg())
    environ = {"HF_HUB_CACHE": str(hub)}
    got = mc.resolve_config_path("org/Model", environ)
    assert got == snap / "config.json"


def test_unresolvable_ref_returns_none(tmp_path):
    assert mc.resolve_config_path("org/Missing", {"HF_HUB_CACHE": str(tmp_path)}) is None
    assert mc.resolve_config_path(None) is None


# --- the gate ----------------------------------------------------------------


def test_deepseek_config_is_usable():
    assert mc.validate_moe_config(_deepseek_cfg()) == []


def test_sparse_candidate_with_partial_expert_shape_is_not_misread_as_dense():
    assert mc.is_sparse_moe_config({"n_routed_experts": 8})
    assert not mc.is_sparse_moe_config({"model_type": "llama"})


def test_mixtral_aliases_are_recognized_but_unsupported_shape_is_refused():
    mixtral = {
        "model_type": "mixtral",
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "intermediate_size": 14336,
    }
    missing = mc.validate_moe_config(mixtral)
    assert any("compress_ratios" in item for item in missing)
    assert any("hidden_size" in item for item in missing)
    norm = mc.normalize_moe_config(mixtral)
    assert norm["n_routed_experts"] == 8
    assert norm["moe_intermediate_size"] == 14336
    # The original dict is not mutated.
    assert "n_routed_experts" not in mixtral


def test_missing_topk_is_refused_and_names_the_key():
    cfg = {"num_local_experts": 8, "intermediate_size": 14336}
    missing = mc.validate_moe_config(cfg)
    assert any("experts per token" in m for m in missing)


def test_nonpositive_expert_shape_is_refused_instead_of_priced_as_dense():
    missing = mc.validate_moe_config(_deepseek_cfg(n_routed_experts=0))
    assert any("routed expert count" in item and "positive" in item for item in missing)


def test_short_compression_schedule_is_refused_instead_of_reusing_last_ratio():
    missing = mc.validate_moe_config(_deepseek_cfg(compress_ratios=[0, 4]))
    assert any("compress_ratios" in item and "4 layers" in item for item in missing)


def test_unpriceable_quant_method_is_refused_not_defaulted():
    cfg = _deepseek_cfg(quantization_config={"quant_method": "awq"})
    missing = mc.validate_moe_config(cfg)
    assert any("awq" in m for m in missing)


def test_final_spec_dtype_validation_covers_every_byte_contributor():
    from dataclasses import replace

    spec = mc.spec_from_hf_config(_deepseek_cfg())
    assert mc.validate_priceable_dtypes(spec) == []

    bad = replace(spec, kv_dtype="future_kv3", act_dtype="future_act3")
    missing = mc.validate_priceable_dtypes(bad)
    assert any("kv_dtype='future_kv3'" in item for item in missing)
    assert any("act_dtype='future_act3'" in item for item in missing)


# --- serving overrides -------------------------------------------------------


def test_serving_overrides_from_cmdline():
    cmd = [
        "vllm", "serve", "org/model",
        "--tensor-parallel-size", "8",
        "--enable-expert-parallel",
        "--kv-cache-dtype", "fp8_e4m3",
        "--max-model-len", "65536",
    ]
    ov = mc.serving_overrides_from_cmdline(cmd)
    assert ov["tp"] == 8 and ov["ep"] == 8 and ov["dp"] == 1
    assert ov["kv_dtype"] == "fp8"
    assert ov["max_model_len"] == 65536


def test_kv_cache_auto_follows_model_dtype_not_hardcoded_fp8():
    ov = mc.serving_overrides_from_cmdline(
        ["vllm", "serve", "m", "--dtype", "bfloat16", "--kv-cache-dtype", "auto"]
    )
    assert ov["act_dtype"] == "bf16"
    assert ov["kv_dtype"] == "bf16"


# --- end-to-end: live_moe_spec ----------------------------------------------


def test_live_moe_spec_applies_config_and_serving_facts(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write_config(ckpt, _deepseek_cfg())
    t = _target([
        "vllm", "serve", str(ckpt),
        "-tp", "8", "--enable-expert-parallel",
        "--kv-cache-dtype", "auto", "--dtype", "bfloat16",
        "--max-model-len", "32768",
    ])
    r = mc.live_moe_spec(t, environ={})
    assert isinstance(r, mc.LiveSpec)
    assert r.sharding.tp == 8 and r.sharding.ep == 8
    # --kv-cache-dtype auto + --dtype bfloat16 overrides the spec's hardcoded fp8.
    assert r.spec.kv_dtype == "bf16"
    assert r.batch.kv_cache_len == 32768
    assert r.spec.expert_dtype == "fp4"


def test_live_moe_spec_refuses_unpredictable_config_with_named_keys(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write_config(ckpt, {"model_type": "dense", "hidden_size": 4096})  # no MoE fields
    r = mc.live_moe_spec(_target(["vllm", "serve", str(ckpt)]), environ={})
    assert isinstance(r, mc.LiveSpecError)
    assert r.missing_keys
    assert "routed expert count" in r.render()


def test_unpriceable_command_line_dtype_is_refused_after_overrides(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write_config(ckpt, _deepseek_cfg())
    target = _target(["vllm", "serve", str(ckpt), "--kv-cache-dtype", "future_kv3"])

    r = mc.live_moe_spec(target, environ={})

    assert isinstance(r, mc.LiveSpecError)
    assert any("kv_dtype='future_kv3'" in item for item in r.missing_keys)


def test_missing_expert_dtype_refuses_dominant_term_prediction(tmp_path):
    ckpt = tmp_path / "ckpt"
    cfg = _deepseek_cfg()
    cfg.pop("expert_dtype")
    _write_config(ckpt, cfg)

    r = mc.live_moe_spec(_target(["vllm", "serve", str(ckpt)]), environ={})

    assert isinstance(r, mc.LiveSpecError)
    assert any("expert_dtype must be declared" in key for key in r.missing_keys)


def test_live_moe_spec_refuses_when_no_config_found(tmp_path):
    r = mc.live_moe_spec(
        _target(["vllm", "serve", "org/Missing"]),
        environ={"HF_HUB_CACHE": str(tmp_path)},
    )
    assert isinstance(r, mc.LiveSpecError)
    assert "no config.json" in r.reason


def test_live_moe_spec_refuses_when_no_model_ref():
    r = mc.live_moe_spec(_target(["vllm", "serve"]), environ={})
    assert isinstance(r, mc.LiveSpecError)
    assert "model argument" in r.reason


def test_resolved_spec_feeds_predict_moe_graph(tmp_path):
    ckpt = tmp_path / "ckpt"
    _write_config(ckpt, _deepseek_cfg())
    r = mc.live_moe_spec(_target(["vllm", "serve", str(ckpt), "-tp", "8"]), environ={})
    assert isinstance(r, mc.LiveSpec)
    g = predict_moe_graph(r.spec, HardwareSpec(), r.batch, r.sharding)
    assert g.total_pred_s > 0
    assert len(g.nodes) > 0
