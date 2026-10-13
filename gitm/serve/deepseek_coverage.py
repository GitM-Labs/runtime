"""Generate DeepSeek onboarding coverage from evidence, tests, and contract state."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from gitm.serve.deployment_contract import load_contract, validate_contract

REQUIRED_TESTS = frozenset({
    "test_catalogue_graph_and_raw_config_gate",
    "test_header_manifest_closes_exact_stored_bytes_but_formula_does_not",
    "test_parallelism_changes_per_rank_work_and_rejects_invalid_heads",
    "test_static_verifier_accepts_identity_and_exposes_engine_unknowns",
    "test_live_production_path_uses_verified_catalogue_and_serving_kv",
    "test_live_hub_revision_must_match_loaded_snapshot",
    "test_deepseek_architecture_cannot_fall_back_to_glm_defaults",
    "test_changed_revision_dimensions_and_recipe_fail",
    "test_changed_tensor_schedule_fails_even_with_updated_hash",
    "test_deployment_contract_reports_blockers_and_rejects_drift",
    "test_deployment_contract_rejects_inconsistent_rank_groups",
    "test_coverage_requires_all_core_cases",
    "test_header_auditor_checks_offsets_against_shape",
})


def _test_state(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "UNVERIFIED", "No JUnit report supplied"
    root = ET.parse(path).getroot()
    cases = [c for c in root.iter("testcase") if "deepseek_v32" in c.get("classname", "")
             or "deepseek_v32" in c.get("name", "")]
    if not cases:
        return "UNVERIFIED", "No DeepSeek test cases in JUnit report"
    failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c for c in cases if c.find("skipped") is not None]
    present = {c.get("name", "") for c in cases}
    missing = REQUIRED_TESTS - present
    status = "FAIL" if failed or skipped else "PARTIAL" if missing else "PASS"
    return (status, f"{len(cases)} cases, {len(failed)} failed, "
            f"{len(skipped)} skipped, {len(missing)} required cases missing")


def coverage(contract_path: Path, *, config_path: Path | None = None,
             index_path: Path | None = None, junit_path: Path | None = None) -> dict:
    contract = load_contract(contract_path)
    preflight = validate_contract(contract)
    checkpoint = {"status": "UNVERIFIED", "detail": "Pinned config/index not supplied"}
    if (config_path is None) != (index_path is None):
        raise ValueError("supply both --config and --index to verify checkpoint")
    if config_path is not None:
        from gitm.planner.checkpoint_evidence import verify_deepseek_checkpoint

        report = verify_deepseek_checkpoint(config_path, index_path,
                                            revision=contract["checkpoint"]["revision"])
        checkpoint = {"status": "VERIFIED_STATIC", "detail": report["status"]}
    tests, test_detail = _test_state(junit_path)
    blockers = preflight["blockers"]
    return {
        "checkpoint": checkpoint,
        "planner_tests": {"status": tests, "detail": test_detail},
        "deployment_contract": {
            "status": "COMPLETE" if not blockers else "BLOCKED",
            "detail": f"{len(blockers)} unresolved fields", "blockers": blockers,
        },
        "qualified_trace": {"status": "UNVERIFIED",
                            "detail": "Nathan's gate result and matched run not supplied"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args(argv)
    try:
        result = coverage(args.contract, config_path=args.config,
                          index_path=args.index, junit_path=args.junit)
    except (OSError, ValueError, ET.ParseError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
