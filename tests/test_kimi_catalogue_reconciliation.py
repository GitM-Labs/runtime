"""Kimi K2.5 / K2.6 catalogue entries reconciled against their pinned checkpoints.

The catalogue's `checkpoint` pins must be the same immutable revisions and file
hashes the inventory recorded, so every provenance claim cites a commit that
can be re-fetched. Engine and deployment facts are not covered here.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.glm_graph import model_weight_bytes
from gitm.planner.inventory import DEFAULT_INVENTORY
from gitm.planner.model_catalogue import load_entry, load_spec, predict
from gitm.planner.roofline import BatchConfig, ShardingConfig

ENTRIES = {"kimi-k2.5": "kimi-k2.5", "kimi-k2.6": "kimi-k2.6-nvfp4", "kimi-k2.6-int4": "kimi-k2.6"}


def _inventory():
    doc = yaml.safe_load(DEFAULT_INVENTORY.read_text())
    return {c["id"]: c for c in doc["checkpoints"]}


@pytest.mark.parametrize("entry,inv_id", ENTRIES.items())
def test_catalogue_pin_matches_inventory_record(entry, inv_id):
    pin = load_entry(entry)["checkpoint"]
    rec = _inventory()[inv_id]
    files = rec["artifacts"]["files"]
    assert pin["repository"] == rec["repository"] == load_entry(entry)["name"]
    assert pin["revision"] == rec["revision"]
    assert pin["config_sha256"] == files["config.json"]["sha256"]
    assert pin["index_sha256"] == files["model.safetensors.index.json"]["sha256"]
    shards = [k for k in files if k.endswith(".safetensors")]
    assert pin["shard_count"] == len(shards)
    if "hf_quant_config_sha256" in pin:
        assert pin["hf_quant_config_sha256"] == files["hf_quant_config.json"]["sha256"]


@pytest.mark.parametrize("entry", ["kimi-k2.6", "kimi-k2.6-int4"])
def test_k26_does_not_inherit_the_k25_pin(entry):
    assert load_entry(entry)["checkpoint"]["revision"] != load_entry("kimi-k2.5")["checkpoint"]["revision"]


def test_deployed_k26_is_the_k25_spec_unchanged():
    # moonshotai/Kimi-K2.6 @7eb5002f differs from K2.5 only in eos_token_id.
    k25 = load_spec("kimi-k2.5")
    assert load_spec("kimi-k2.6-int4") == replace(k25, name="moonshotai/Kimi-K2.6")


def test_the_served_model_id_resolves_to_the_int4_entry():
    # What vLLM on H200 reports, per Adit (2026-10-20); never the NVFP4 entry.
    assert load_entry("moonshotai/Kimi-K2.6")["checkpoint"]["revision"].startswith("7eb5002f")


@pytest.mark.parametrize("entry", ENTRIES)
def test_weight_bytes_residual_against_index_total_size(entry):
    """-0.16% for both: the unmodelled ~0.9 GB vision tower. The old -0.03% K2.5
    figure predates the dense-FFN double-count fix and must not come back."""
    total = load_entry(entry)["checkpoint"]["index_total_size"]
    residual = model_weight_bytes(load_spec(entry)) / total - 1.0
    assert -0.0025 < residual < -0.0010


DEPLOYED = ["kimi-k2.5", "kimi-k2.6-int4"]


@pytest.mark.parametrize("entry", DEPLOYED)
def test_deployment_agrees_with_checkpoint_and_spec(entry):
    """Engine facts sit in `deployment`, apart from `checkpoint` and `spec`; where
    they overlap the three must agree."""
    e, spec = load_entry(entry), load_spec(entry)
    dep = e["deployment"]
    assert dep["served_revision"] == e["checkpoint"]["revision"]
    assert dep["kv_cache_dtype_resolved"] == spec.kv_dtype == "bf16"
    assert dep["speculative_decoding"] == "none" and spec.num_nextn_predict_layers == 0
    assert dep["engine"] == "vllm" and (dep["tp"], dep["ep"]) == (8, 1)
    assert dep["gpus"] == dep["tp"] * dep["ep"]


def test_deployment_gaps_are_recorded():
    assert load_entry("kimi-k2.5")["deployment"]["status"] == "verified"
    assert load_entry("kimi-k2.5")["deployment"]["engine_version"] == "0.29.0"
    k26 = load_entry("kimi-k2.6-int4")["deployment"]
    assert k26["status"] == "partial" and k26["engine_version"] is None
    assert "engine_version" in k26["unverified"]


@pytest.mark.parametrize("entry", ["kimi-k2.6", "kimi-k2.7-code"])
def test_children_do_not_inherit_the_k25_pin_or_deployment(entry):
    # `extends` shares architecture only. kimi-k2.7-code sets neither block, so
    # it must come back with none, not K2.5's revision and MI355X run.
    e = load_entry(entry)
    assert e.get("deployment") is None
    assert (e.get("checkpoint") or {}).get("revision") != load_entry("kimi-k2.5")["checkpoint"]["revision"]


def test_k27_code_registry_note_claims_no_deployment():
    from gitm.planner.registry import _load

    assert "deployment verified" not in _load("kimi-k2.7-code")[2]


def test_registry_note_reports_the_deployment():
    from gitm.planner.registry import _load

    assert "deployment verified (gaps: none)" in _load("kimi-k2.5")[2]
    assert "deployment partial (gaps: engine_version" in _load("kimi-k2.6-int4")[2]
    assert "STATIC_CHECKPOINT_ONLY" in _load("kimi-k2.6")[2]


@pytest.mark.parametrize("entry", DEPLOYED)
def test_entry_predicts_at_its_deployment_shape(entry):
    dep = load_entry(entry)["deployment"]
    hw = hardware_spec_for(peak_for_sku(dep["gpu"]))
    graph, family = predict(entry, hw=hw, batch=BatchConfig(batch=32, kv_cache_len=8192),
                            sharding=ShardingConfig(tp=dep["tp"], ep=dep["ep"]))
    assert family == "glm_moe_dsa"
    assert graph.total_pred_s > 0


def test_undeployed_nvfp4_predicts_at_nvidia_recipe_shape():
    # No local deployment; TP8 on B300 is NVIDIA's published recipe shape.
    hw = hardware_spec_for(peak_for_sku("B300"))
    graph, family = predict("kimi-k2.6", hw=hw, batch=BatchConfig(batch=32, kv_cache_len=8192),
                            sharding=ShardingConfig(tp=8))
    assert family == "glm_moe_dsa"
    assert graph.total_pred_s > 0
