"""DeepSeek V3.2 planner and live-serving wiring; CPU-only tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from gitm.planner import model_catalogue
from gitm.planner.glm_graph import model_weight_bytes
from gitm.planner.model_catalogue import load_spec, predict
from gitm.planner.registry import detect_family, spec_from_hf_config
from gitm.planner.roofline import BatchConfig, ShardingConfig
from gitm.serve import discover, model_config

CONFIG = json.loads((Path(__file__).parent / "fixtures/deepseek_v32_config.json").read_text())
REVISION = "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"

def test_catalogue_graph_and_raw_config_gate():
    spec = load_spec("deepseek-v3.2")
    assert (spec.n_layers, spec.n_full_indexer_layers, spec.n_sparse_mlp_layers) == (61, 61, 58)
    assert spec.stored_dtype_for("moe_router", spec.weight_dtype) == "bf16"
    assert spec.dtype_for("moe_router", spec.weight_dtype) == "fp32"
    assert detect_family(CONFIG) == "glm_moe_dsa"
    with pytest.raises(ValueError, match="pinned tensor index"):
        spec_from_hf_config(CONFIG)
    graph, family = predict("deepseek-v3.2", batch=BatchConfig(batch=1, kv_cache_len=4096))
    assert family == "glm_moe_dsa" and graph.total_pred_s > 0
    assert sum(n.op == "attn_index_proj" for n in graph.nodes) == 61
    assert len(graph.nodes) > 1000


def test_header_manifest_closes_exact_stored_bytes_but_formula_does_not():
    entry = model_catalogue.load_entry("deepseek-v3.2")
    manifest = json.loads((model_catalogue.CATALOGUE_DIR / entry["checkpoint"]["header_manifest"]).read_text())
    classes = manifest["classes"]
    assert sum(c["bytes"] for c in classes.values()) == 689_471_107_200
    assert sum(c["numel"] for c in classes.values()) * 2 == 1_370_793_842_752
    assert sum(c["tensors"] for c in classes.values()) == 92_425
    assert len(manifest["shard_headers"]) == 163
    # MTP tables now participate in the shared footprint path; its remaining
    # analytic residual is explicit instead of being called exact closure.
    residual = manifest["payload_bytes"] - model_weight_bytes(load_spec("deepseek-v3.2"))
    assert 0 < residual < 5e6


def test_parallelism_changes_per_rank_work_and_rejects_invalid_heads():
    spec = load_spec("deepseek-v3.2")
    one, _ = predict("deepseek-v3.2", sharding=ShardingConfig(tp=1))
    eight, _ = predict("deepseek-v3.2", sharding=ShardingConfig(tp=8, ep=8))
    assert model_weight_bytes(spec, ShardingConfig(tp=8, ep=8)) < model_weight_bytes(spec)
    assert any(n.op == "moe_all_to_all" for n in eight.nodes)
    assert not any(n.op == "moe_all_to_all" for n in one.nodes)
    prefill, _ = predict("deepseek-v3.2", batch=BatchConfig(batch=0, prefill_tokens=128),
                         sharding=ShardingConfig(tp=8, ep=8))
    assert prefill.total_pred_s != eight.total_pred_s
    with pytest.raises(ValueError, match="does not divide"):
        predict("deepseek-v3.2", sharding=ShardingConfig(tp=7))


@pytest.fixture
def synthetic_pin(tmp_path, monkeypatch):
    """Small, coherent index to exercise the verifier's negative paths in CI."""
    entry = model_catalogue.load_entry("deepseek-v3.2")
    weights = {}
    for i in range(61):
        weights[f"model.layers.{i}.self_attn.indexer.wk.weight"] = "shard.safetensors"
        if i < 3:
            weights[f"model.layers.{i}.mlp.gate_proj.weight"] = "shard.safetensors"
        else:
            weights[f"model.layers.{i}.mlp.experts.0.gate_proj.weight"] = "shard.safetensors"
    weights["model.layers.61.self_attn.indexer.wk.weight"] = "shard.safetensors"
    for name in ("embed_tokens.weight", "shared_head.head.weight", "eh_proj.weight"):
        weights[f"model.layers.61.{name}"] = "shard.safetensors"
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(CONFIG, sort_keys=True))
    index_path.write_text(json.dumps({"metadata": {"total_size": 2 * len(weights)},
                                      "weight_map": weights}, sort_keys=True))
    hashes = (hashlib.sha256(config_path.read_bytes()).hexdigest(),
              hashlib.sha256(index_path.read_bytes()).hexdigest())
    pin = entry["checkpoint"]
    pin.update(config_sha256=hashes[0], index_sha256=hashes[1],
               tensor_count=len(weights), shard_count=1,
               index_total_size=2 * len(weights), payload_bytes=2 * len(weights))
    manifest = {
        "revision": REVISION, "config_sha256": hashes[0], "index_sha256": hashes[1],
        "tensors": len(weights), "shards": 1, "index_total_size": 2 * len(weights),
        "payload_bytes": 2 * len(weights),
        "classes": {"weight/BF16": {"tensors": len(weights), "numel": len(weights),
                                     "bytes": 2 * len(weights)}},
        "components": {"synthetic": {"weight/BF16": {
            "tensors": len(weights), "numel": len(weights), "bytes": 2 * len(weights)}}},
        "shard_headers": [{"shard": "shard.safetensors", "header_sha256": "synthetic"}],
    }
    (tmp_path / pin["header_manifest"]).write_text(json.dumps(manifest))
    (tmp_path / "deepseek-v3.2.yaml").write_text(yaml.safe_dump(entry))
    monkeypatch.setattr(model_catalogue, "CATALOGUE_DIR", tmp_path)
    return config_path, index_path, entry


def test_live_production_path_uses_verified_catalogue_and_serving_kv(synthetic_pin):
    config_path, index_path, _ = synthetic_pin
    snapshot = config_path.parent / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / config_path.name).write_bytes(config_path.read_bytes())
    (snapshot / index_path.name).write_bytes(index_path.read_bytes())
    target = discover.Target(pid=123, cmdline=[
        "vllm", "serve", str(snapshot), "--kv-cache-dtype", "fp8",
        "--tensor-parallel-size", "8", "--enable-expert-parallel",
    ])
    live = model_config.live_moe_spec(target, environ={})
    assert isinstance(live, model_config.LiveSpec)
    assert live.family == "glm_moe_dsa" and live.spec.n_full_indexer_layers == 61
    assert live.spec.kv_dtype == "fp8" and (live.sharding.tp, live.sharding.ep) == (8, 8)
    assert live.checkpoint_evidence["status"] == "STATIC_VERIFIED_ENGINE_UNVERIFIED"
    (snapshot / index_path.name).write_text("{}")
    bad = model_config.live_moe_spec(target, environ={})
    assert isinstance(bad, model_config.LiveSpecError)
    assert "SHA256 mismatch" in bad.reason


def test_live_hub_revision_must_match_loaded_snapshot(synthetic_pin, tmp_path):
    config_path, index_path, _ = synthetic_pin
    hub = tmp_path / "hub"
    snapshots = hub / "models--deepseek-ai--DeepSeek-V3.2" / "snapshots"
    for revision in ("0" * 40, REVISION):
        snapshot = snapshots / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(config_path.read_bytes())
        (snapshot / index_path.name).write_bytes(index_path.read_bytes())
    model = "deepseek-ai/DeepSeek-V3.2"
    target = discover.Target(pid=123, cmdline=["vllm", "serve", model, "--revision", "0" * 40])
    bad = model_config.live_moe_spec(target, environ={}, cache_root=hub)
    assert isinstance(bad, model_config.LiveSpecError)
    assert "revision mismatch" in bad.reason
    target.cmdline = ["vllm", "serve", model, "--revision", REVISION]
    good = model_config.live_moe_spec(target, environ={}, cache_root=hub)
    assert isinstance(good, model_config.LiveSpec)
    target.cmdline = ["vllm", "serve", model]
    unknown = model_config.live_moe_spec(target, environ={}, cache_root=hub)
    assert isinstance(unknown, model_config.LiveSpecError)
    assert "explicit immutable --revision" in unknown.reason


def test_deepseek_architecture_cannot_fall_back_to_glm_defaults(synthetic_pin):
    config_path, index_path, _ = synthetic_pin
    cfg = json.loads(config_path.read_text())
    cfg["model_type"] = "other"
    cfg["architectures"] = ["DeepseekV32ForCausalLM"]
    with pytest.raises(ValueError, match="pinned tensor index"):
        spec_from_hf_config(cfg)
    snapshot = config_path.parent / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(cfg))
    (snapshot / index_path.name).write_bytes(index_path.read_bytes())
    target = discover.Target(pid=123, cmdline=["vllm", "serve", str(snapshot)])
    bad = model_config.live_moe_spec(target, environ={})
    assert isinstance(bad, model_config.LiveSpecError)
    assert "static verification failed" in bad.reason
