"""Static DeepSeek V3.2 checkpoint evidence; no engine or GPU assertions."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gitm.planner import model_catalogue
from gitm.planner.checkpoint_evidence import verify_deepseek_checkpoint

CONFIG = json.loads((Path(__file__).parent / "fixtures/deepseek_v32_config.json").read_text())
REVISION = "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"


def test_catalogue_loads_and_static_ledger_closes():
    spec = model_catalogue.load_spec("deepseek-v3.2")
    entry = model_catalogue.load_entry("deepseek-v3.2")
    pin = entry["checkpoint"]
    manifest = json.loads((model_catalogue.CATALOGUE_DIR / pin["header_manifest"]).read_text())
    assert spec.n_layers == 61 and spec.n_full_indexer_layers == 61
    assert spec.mtp_separate_embedding_head
    assert spec.stored_dtype_overrides
    assert pin["revision"] == manifest["revision"] == REVISION
    assert len(manifest["shard_headers"]) == pin["shard_count"] == 163
    assert sum(c["tensors"] for c in manifest["classes"].values()) == pin["tensor_count"] == 92425
    assert sum(c["bytes"] for c in manifest["classes"].values()) == pin["payload_bytes"] == 689471107200
    assert sum(c["bytes"] for group in manifest["components"].values()
               for c in group.values()) == pin["payload_bytes"]
    with pytest.raises(ValueError, match="planner integration pending"):
        model_catalogue.predict("deepseek-v3.2")


@pytest.fixture
def synthetic_pin(tmp_path, monkeypatch):
    """Coherent small index to test fail-closed verification without downloading weights."""
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
    pin = entry["checkpoint"]
    pin.update(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
               index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
               tensor_count=len(weights), shard_count=1,
               index_total_size=2 * len(weights), payload_bytes=2 * len(weights))
    manifest = {
        "revision": REVISION, "config_sha256": pin["config_sha256"],
        "index_sha256": pin["index_sha256"], "tensors": len(weights), "shards": 1,
        "index_total_size": 2 * len(weights), "payload_bytes": 2 * len(weights),
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


def test_pinned_identity_and_changed_revision_or_recipe_fail(synthetic_pin):
    config_path, index_path, entry = synthetic_pin
    result = verify_deepseek_checkpoint(config_path, index_path, revision=REVISION)
    assert result["status"] == "STATIC_VERIFIED_ENGINE_UNVERIFIED"
    assert "engine revision" in result["unverified"]
    with pytest.raises(ValueError, match="revision mismatch"):
        verify_deepseek_checkpoint(config_path, index_path, revision="0" * 40)
    config_path.write_text(config_path.read_text().replace('"hidden_size": 7168',
                                                       '"hidden_size": 7169'))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_deepseek_checkpoint(config_path, index_path, revision=REVISION)
    config_path.write_text(json.dumps(CONFIG, sort_keys=True))
    entry["spec"]["weight_dtype"] = "bf16"
    (model_catalogue.CATALOGUE_DIR / "deepseek-v3.2.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(ValueError, match="precision recipe"):
        verify_deepseek_checkpoint(config_path, index_path, revision=REVISION)


def test_changed_tensor_schedule_fails_even_with_updated_hash(synthetic_pin):
    config_path, index_path, entry = synthetic_pin
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.12.self_attn.indexer.wk.weight"]
    index_path.write_text(json.dumps(index, sort_keys=True))
    entry["checkpoint"]["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    entry["checkpoint"]["tensor_count"] = len(index["weight_map"])
    (model_catalogue.CATALOGUE_DIR / "deepseek-v3.2.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(ValueError, match="indexer schedule"):
        verify_deepseek_checkpoint(config_path, index_path, revision=REVISION)


def test_header_auditor_checks_offsets_against_shape(tmp_path):
    shard = tmp_path / "one.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    tensor = {"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}
    header = json.dumps(tensor).encode()
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    index.write_text(json.dumps({"metadata": {"total_size": 8},
                                 "weight_map": {"weight": shard.name}}))
    script = Path(__file__).parents[1] / "scripts/audit_safetensors_headers.py"
    cmd = [sys.executable, str(script), "--index", str(index), "--shards-dir", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["payload_bytes"] == 8
    tensor["weight"]["data_offsets"] = [0, 7]
    header = json.dumps(tensor).encode()
    shard.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    bad = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert bad.returncode != 0 and "offset bytes disagree" in bad.stderr
