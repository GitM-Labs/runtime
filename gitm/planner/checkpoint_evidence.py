"""Pinned checkpoint checks before a DeepSeek V3.2 config becomes a planner spec.

The catalogue describes architecture; a config alone cannot establish the
indexer schedule or the stored tensor inventory. This module checks the pinned
config and tensor index against the catalogue's evidence record. The header
ledger is separately reproducible from local safetensors shards; no engine or
deployment behavior is inferred by these static checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_pinned_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{Path(path).name} SHA256 mismatch: {actual} != {expected_sha256}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _indexer_schedule(weight_map: dict[str, str], n_layers: int) -> tuple[str, ...]:
    return tuple(
        "full" if any(k.startswith(f"model.layers.{i}.self_attn.indexer.")
                      for k in weight_map) else "shared"
        for i in range(n_layers)
    )


def verify_deepseek_checkpoint(
    config_path: str | Path,
    index_path: str | Path,
    *,
    revision: str,
    entry_name: str = "deepseek-v3.2",
) -> dict[str, Any]:
    """Fail closed on a changed pin, shape, precision recipe, or tensor schedule.

    Returns a *static* evidence report, not an engine/deployment qualification.
    The safetensors header ledger is pinned by a machine-readable manifest; its
    shard header digests are auditable when the shards are available locally.
    """
    from gitm.planner.model_catalogue import CATALOGUE_DIR, load_entry, load_spec

    entry = load_entry(entry_name)
    pin = entry.get("checkpoint")
    if not isinstance(pin, dict):
        raise ValueError(f"{entry_name}: missing machine-readable checkpoint pin")
    if revision != pin["revision"]:
        raise ValueError(f"checkpoint revision mismatch: {revision} != {pin['revision']}")
    config = _read_pinned_json(config_path, pin["config_sha256"])
    index = _read_pinned_json(index_path, pin["index_sha256"])
    spec = load_spec(entry_name)

    expected_config = {
        "model_type": "deepseek_v32",
        "hidden_size": spec.hidden,
        "num_hidden_layers": spec.n_layers,
        "vocab_size": spec.vocab,
        "num_attention_heads": spec.n_heads,
        "q_lora_rank": spec.q_lora_rank,
        "kv_lora_rank": spec.kv_lora_rank,
        "qk_nope_head_dim": spec.qk_nope_head_dim,
        "qk_rope_head_dim": spec.qk_rope_head_dim,
        "v_head_dim": spec.v_head_dim,
        "index_n_heads": spec.index_n_heads,
        "index_head_dim": spec.index_head_dim,
        "index_topk": spec.index_topk,
        "n_routed_experts": spec.n_routed_experts,
        "n_shared_experts": spec.n_shared_experts,
        "num_experts_per_tok": spec.num_experts_per_tok,
        "moe_intermediate_size": spec.moe_intermediate_size,
        "intermediate_size": spec.intermediate_size,
        "first_k_dense_replace": spec.first_k_dense_replace,
        "num_nextn_predict_layers": spec.num_nextn_predict_layers,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"checkpoint config {key}: {config.get(key)!r} != {expected!r}")
    quant = config.get("quantization_config") or {}
    expected_quant = {
        "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0", "activation_scheme": "dynamic",
    }
    for key, expected in expected_quant.items():
        if quant.get(key) != expected:
            raise ValueError(f"checkpoint quantization {key}: {quant.get(key)!r} != {expected!r}")
    if (spec.weight_dtype, spec.expert_dtype, spec.act_dtype) != ("fp8", "fp8", "bf16"):
        raise ValueError("catalogue precision recipe differs from pinned checkpoint")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("tensor index has no weight_map")
    schedule = _indexer_schedule(weight_map, spec.n_layers)
    if schedule != tuple(spec.indexer_kind(i) for i in range(spec.n_layers)):
        raise ValueError("tensor index indexer schedule differs from catalogue")
    for i in range(spec.n_layers):
        prefix = f"model.layers.{i}.mlp."
        dense = f"{prefix}gate_proj.weight" in weight_map
        experts = any(k.startswith(prefix + "experts.") for k in weight_map)
        if dense != (not spec.is_sparse_mlp(i)) or experts != spec.is_sparse_mlp(i):
            raise ValueError(f"tensor index MLP schedule differs at layer {i}")
    mtp_has_indexer = any(k.startswith(f"model.layers.{spec.n_layers}.self_attn.indexer.")
                          for k in weight_map)
    if mtp_has_indexer != (not spec.index_share_for_mtp_iteration):
        raise ValueError("tensor index MTP indexer differs from catalogue")
    mtp_aux = (
        f"model.layers.{spec.n_layers}.embed_tokens.weight",
        f"model.layers.{spec.n_layers}.shared_head.head.weight",
        f"model.layers.{spec.n_layers}.eh_proj.weight",
    )
    if spec.mtp_separate_embedding_head and not all(k in weight_map for k in mtp_aux):
        raise ValueError("tensor index missing separate MTP embedding/head/fusion weights")
    if len(weight_map) != pin["tensor_count"] or len(set(weight_map.values())) != pin["shard_count"]:
        raise ValueError("tensor index count or shard count differs from pin")
    if index.get("metadata", {}).get("total_size") != pin["index_total_size"]:
        raise ValueError("tensor index metadata.total_size differs from pin")

    manifest_path = CATALOGUE_DIR / pin["header_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (manifest["config_sha256"] != pin["config_sha256"]
            or manifest["index_sha256"] != pin["index_sha256"]
            or manifest["revision"] != revision):
        raise ValueError("header manifest identity differs from checkpoint pin")
    if (manifest["tensors"] != len(weight_map)
            or manifest["shards"] != len(set(weight_map.values()))
            or sum(c["tensors"] for c in manifest["classes"].values()) != len(weight_map)
            or len(manifest["shard_headers"]) != manifest["shards"]):
        raise ValueError("header manifest tensor/shard counts do not close")
    if (sum(c["bytes"] for c in manifest["classes"].values())
            != manifest["payload_bytes"]):
        raise ValueError("stored-byte class ledger does not close")
    components = manifest.get("components")
    if not isinstance(components, dict) or (
        sum(c["bytes"] for group in components.values() for c in group.values())
        != manifest["payload_bytes"]
    ):
        raise ValueError("component stored-byte ledger does not close")
    if (sum(c["numel"] for c in manifest["classes"].values()) * 2
            != manifest["index_total_size"]):
        raise ValueError("index metadata does not match twice the tensor element count")
    if manifest["payload_bytes"] != pin["payload_bytes"]:
        raise ValueError("stored-byte payload differs from checkpoint pin")
    return {
        "status": "STATIC_VERIFIED_ENGINE_UNVERIFIED",
        "revision": revision,
        "config_sha256": pin["config_sha256"],
        "index_sha256": pin["index_sha256"],
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "stored_payload_bytes": manifest["payload_bytes"],
        "stored_classes": manifest["classes"],
        "stored_components": manifest["components"],
        "index_total_size": manifest["index_total_size"],
        "unverified": ["engine revision", "KV-cache dtype/layout", "MLA absorption",
                       "MTP execution", "expert padding", "deployment topology",
                       "traffic workload", "qualified trace"],
    }


def revision_from_snapshot(config_path: str | Path) -> str:
    """Require a Hugging Face snapshot directory; never guess the newest revision."""
    path = Path(config_path)
    if path.parent.parent.name != "snapshots" or len(path.parent.name) != 40:
        raise ValueError(
            "DeepSeek V3.2 needs a pinned snapshots/<40-character-revision>/config.json; "
            "a local or floating config has no verified checkpoint identity"
        )
    return path.parent.name
