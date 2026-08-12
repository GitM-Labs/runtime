from __future__ import annotations

from scripts.compare_results import EXACT_PKGS, compare


def _report() -> dict:
    return {
        "schema": "gitm.verify_report/v1",
        "git_sha": "abc",
        "gitm_version": "1",
        "python": "3.12",
        "dataset_manifests": {"hft": "deadbeef"},
        "packages": {pkg: "1.0" for pkg in EXACT_PKGS},
        "git_dirty": False,
        "gpu": {"name": "H100"},
    }


def test_compare_refuses_reports_with_missing_identity_fields():
    mismatches, _ = compare({}, {})

    assert any("schema" in mismatch for mismatch in mismatches)
    assert any("packages.pydantic" in mismatch for mismatch in mismatches)


def test_compare_accepts_complete_matching_reports():
    assert compare(_report(), _report()) == ([], [])
