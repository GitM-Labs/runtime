"""Validate a versioned workload contract without turning unknowns into defaults.

This is a static preflight. It does not claim that a launch happened, a trace
qualified, or an engine source tree matches an image. Those need their owners'
verification interfaces and a real run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED: dict[str, tuple[str, str]] = {
    "checkpoint.engagement_match": ("Collin", "checkpoint identity for target engagement"),
    "engine.source_repository": ("Collin", "engine source provenance"),
    "engine.source_revision": ("Collin", "engine-backed prediction"),
    "engine.image_digest": ("Collin", "reproducible launch"),
    "engine.launch_argv": ("Collin", "reproducible launch"),
    "engine.environment": ("Collin", "execution-changing defaults"),
    "engine.kv_cache_layout": ("Collin", "KV memory and traffic"),
    "engine.mla_backend": ("Collin", "attention execution regions"),
    "engine.expert_backend_padding": ("Collin", "expert traffic"),
    "hardware.sku": ("Adit", "per-rank memory and ceiling"),
    "hardware.topology": ("Adit", "per-rank collectives"),
    "parallelism.tp": ("Adit", "per-rank shapes"),
    "parallelism.ep": ("Adit", "expert placement and collectives"),
    "parallelism.dp": ("Adit", "world size"),
    "traffic.manifest_sha256": ("Collin", "replay workload identity"),
    "traffic.source_kind": ("Collin", "workload representativeness"),
    "traffic.replay_argv": ("Medha", "reproducible traffic"),
    "traffic.expected_input_tokens": ("Medha", "replay token reconciliation"),
    "traffic.expected_output_tokens": ("Medha", "replay token reconciliation"),
    "telemetry.collector_contract": ("Nathan", "qualified trace"),
    "telemetry.expected_ranks": ("Nathan", "multi-rank capture"),
    "telemetry.qualification_gate": ("Nathan", "admissible observation"),
    "planner.graph_sha256": ("Medha", "predicted graph identity"),
    "planner.ceiling_sha256": ("Medha", "predicted ceiling identity"),
    "commands.bring_up": ("Adit", "second-engineer reproduction"),
    "commands.capture": ("Nathan", "qualified capture"),
    "commands.tear_down": ("Adit", "second-engineer reproduction"),
    "privacy.storage_reference": ("Collin", "private customer artifact handling"),
}


def _field(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"contract missing required field {path}; use UNVERIFIED explicitly")
        value = value[part]
    return value


def validate_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Return blockers with owners and affected claims; refuse structural drift."""
    if data.get("schema") != "gitm.deepseek.deployment/v1":
        raise ValueError("unsupported deployment contract schema")
    from gitm.planner.model_catalogue import load_entry, load_spec

    pin = load_entry("deepseek-v3.2")["checkpoint"]
    ck = data.get("checkpoint") or {}
    for key in ("repository", "revision", "config_sha256", "index_sha256"):
        if ck.get(key) != pin[key]:
            raise ValueError(f"deployment checkpoint {key} differs from catalogue pin")
    if _field(data, "planner.catalogue_entry") != "deepseek-v3.2":
        raise ValueError("planner catalogue entry differs from checkpoint contract")

    blockers = []
    for path, (owner, claim) in REQUIRED.items():
        value = _field(data, path)
        if value == "UNVERIFIED":
            blockers.append({"field": path, "owner": owner, "blocks": claim})
        elif value is None or value == "" or value == [] or value == {}:
            raise ValueError(f"{path}: empty value must be UNVERIFIED, not a silent default")
    parallel = data["parallelism"]
    tp, ep, dp = (parallel[k] for k in ("tp", "ep", "dp"))
    known = [x for x in (tp, ep, dp) if x != "UNVERIFIED"]
    if any(type(x) is not int or x <= 0 for x in known):
        raise ValueError("TP/EP/DP must be positive integers or UNVERIFIED")
    if isinstance(tp, int) and load_spec("deepseek-v3.2").n_heads % tp:
        raise ValueError("TP does not divide DeepSeek attention heads")
    if isinstance(tp, int) and isinstance(ep, int) and ep > tp:
        raise ValueError("EP exceeds the declared TP group; check engine interpretation")
    if any(v == "UNVERIFIED" for v in (tp, ep, dp)):
        blockers.append({"field": "parallelism.shape", "owner": "Adit",
                         "blocks": "per-rank memory and collective ledger"})
    return {"contract_valid": True, "launch_contract_complete": not blockers,
            "blockers": blockers, "checkpoint_revision": ck["revision"]}


def load_contract(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("deployment contract must be a mapping")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_contract(load_contract(args.contract))
    except (OSError, ValueError) as exc:
        print(json.dumps({"contract_valid": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["launch_contract_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
