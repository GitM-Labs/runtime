"""Deployment-contract blockers and generated coverage for DeepSeek V3.2."""
from __future__ import annotations

from pathlib import Path

import pytest

from gitm.serve import deepseek_coverage
from gitm.serve.deployment_contract import REQUIRED, load_contract, validate_contract

REVISION = "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"

def test_deployment_contract_reports_blockers_and_rejects_drift():
    path = Path(__file__).parents[1] / "docs/deepseek-v3.2/deployment-contract.yaml"
    data = load_contract(path)
    result = validate_contract(data)
    assert result["contract_valid"] and not result["launch_contract_complete"]
    assert {b["field"] for b in result["blockers"]} >= {
        "engine.source_revision", "parallelism.shape", "telemetry.qualification_gate",
    }
    data["checkpoint"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="checkpoint revision"):
        validate_contract(data)
    data["checkpoint"]["revision"] = REVISION
    data["parallelism"]["tp"] = 7
    with pytest.raises(ValueError, match="does not divide"):
        validate_contract(data)


def test_mismatched_engagement_checkpoint_blocks_complete_contract():
    path = Path(__file__).parents[1] / "docs/deepseek-v3.2/deployment-contract.yaml"
    data = load_contract(path)
    for field in REQUIRED:
        section, key = field.split(".")
        data[section][key] = 1 if field.startswith("parallelism.") or field == "telemetry.expected_ranks" else "verified"
    data["checkpoint"]["engagement_match"] = False
    result = validate_contract(data)
    assert result["contract_valid"] and not result["launch_contract_complete"]
    assert any(b["field"] == "checkpoint.engagement_match" for b in result["blockers"])
    data["checkpoint"]["engagement_match"] = True
    assert validate_contract(data)["launch_contract_complete"]


def test_deployment_contract_rejects_inconsistent_rank_groups():
    path = Path(__file__).parents[1] / "docs/deepseek-v3.2/deployment-contract.yaml"
    data = load_contract(path)
    data["parallelism"].update(tp=8, ep=3, dp=2)
    with pytest.raises(ValueError, match="EP must divide"):
        validate_contract(data)
    data["parallelism"]["ep"] = 4
    data["telemetry"]["expected_ranks"] = 8
    with pytest.raises(ValueError, match=r"TP \* DP"):
        validate_contract(data)


def test_coverage_requires_all_core_cases(tmp_path):
    report = tmp_path / "partial.xml"
    report.write_text('<testsuite><testcase classname="tests.test_deepseek_planner" '
                      'name="test_catalogue_graph_and_raw_config_gate"/></testsuite>')
    status, detail = deepseek_coverage._test_state(report)
    assert status == "PARTIAL" and "required cases missing" in detail
    cases = "".join(f'<testcase classname="tests.test_deepseek_planner" name="{name}"/>'
                    for name in deepseek_coverage.REQUIRED_TESTS)
    report.write_text(f"<testsuite>{cases}</testsuite>")
    assert deepseek_coverage._test_state(report)[0] == "PASS"
