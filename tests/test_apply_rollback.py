"""Tests for the rollback-gated intervention apply path.

Covers the two forced-failure rollback cases the ticket requires (bad value,
mid-apply crash) plus the regression rollback and the happy path, against both
the in-memory and config-file applicators.
"""

from __future__ import annotations

import json

import pytest
import yaml

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import (
    ConfigFileApplicator,
    DictApplicator,
    apply_intervention,
    apply_intervention_from_file,
)


def _spec(**over) -> InterventionSpec:
    base = dict(
        name="block_size_16", summary="set block size", knob="block_size", value=16,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="https://docs.vllm.ai/en/latest/configuration/optimization.html",
    )
    base.update(over)
    return InterventionSpec.model_validate(base)


# --- happy path -------------------------------------------------------------


def test_apply_keeps_on_positive_delta():
    cfg = {"block_size": 8}
    app = DictApplicator(cfg, measure_fn=lambda s: +0.08)
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert res.measured_delta == pytest.approx(0.08)
    assert cfg["block_size"] == 16  # kept


def test_apply_only_when_no_measurement_keeps():
    cfg = {"block_size": 8}
    res = apply_intervention(_spec(), DictApplicator(cfg))  # measure -> None
    assert res.applied and not res.rolled_back
    assert res.measured_delta is None
    assert cfg["block_size"] == 16


# --- rollback case 1: bad value (apply raises) ------------------------------


def test_rollback_on_bad_value():
    cfg = {"block_size": 8}
    app = DictApplicator(cfg, measure_fn=lambda s: +0.1)
    res = apply_intervention(_spec(value=None), app)  # no value -> apply raises
    assert not res.applied and res.rolled_back
    assert "apply failed" in res.error
    assert cfg == {"block_size": 8}  # snapshot restored exactly


# --- rollback case 2: mid-apply / measurement crash -------------------------


def test_rollback_on_measure_crash():
    cfg = {"block_size": 8}

    def boom(_spec):
        raise RuntimeError("engine crashed mid-measurement")

    res = apply_intervention(_spec(), DictApplicator(cfg, measure_fn=boom))
    assert not res.applied and res.rolled_back
    assert "measure failed" in res.error
    assert cfg == {"block_size": 8}  # restored despite the apply having mutated it


# --- regression rollback ----------------------------------------------------


def test_rollback_on_regression():
    cfg = {"block_size": 8}
    app = DictApplicator(cfg, measure_fn=lambda s: -0.05)  # 5% slower
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)
    assert res.applied and res.rolled_back  # applied then reverted
    assert res.measured_delta == pytest.approx(-0.05)
    assert cfg == {"block_size": 8}  # restored


# --- safety audit trail on the apply seam -----------------------------------


def test_apply_audits_kept_change(tmp_path):
    from gitm.safety import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    cfg = {"block_size": 8}
    apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: +0.08), audit=log)

    events = [e.event for e in log.entries()]
    assert events == ["apply"]  # kept -> one apply, no revert
    assert log.entries()[0].detail == {"knob": "block_size", "value": 16}


def test_apply_audits_apply_then_revert_on_regression(tmp_path):
    from gitm.safety import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    cfg = {"block_size": 8}
    apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: -0.05), audit=log)

    # A regressing change is applied then reverted — both must be on the trail.
    assert [e.event for e in log.entries()] == ["apply", "revert"]


def test_apply_without_audit_writes_nothing(tmp_path):
    """No audit arg -> no trail (dry-run/no-op applies stay off the record)."""
    log_path = tmp_path / "audit.jsonl"
    apply_intervention(_spec(), DictApplicator({"block_size": 8}, measure_fn=lambda s: +0.1))
    assert not log_path.exists()


# --- config-file applicator + CLI helper ------------------------------------


def test_config_file_applicator_roundtrip(tmp_path):
    target = tmp_path / "engine.yaml"
    target.write_text(yaml.safe_dump({"block_size": 8, "dtype": "bf16"}))

    # keep
    app = ConfigFileApplicator(target, measure_fn=lambda s: +0.1)
    res = apply_intervention(_spec(), app)
    assert res.applied and not res.rolled_back
    assert yaml.safe_load(target.read_text())["block_size"] == 16

    # regression -> file restored byte-for-byte
    before = target.read_bytes()
    app2 = ConfigFileApplicator(target, measure_fn=lambda s: -0.2)
    res2 = apply_intervention(_spec(value=32), app2)
    assert res2.rolled_back
    assert target.read_bytes() == before


def test_apply_from_file_without_config_is_noop(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(json.loads(_spec().model_dump_json())))
    out = apply_intervention_from_file(spec_path)
    assert out["applied"] is False
    assert "no target config" in out["error"]


def test_apply_from_file_with_config(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(json.loads(_spec().model_dump_json())))
    target = tmp_path / "engine.yaml"
    target.write_text(yaml.safe_dump({"block_size": 8}))
    out = apply_intervention_from_file(spec_path, config=target)
    assert out["applied"] is True and out["rolled_back"] is False
    assert yaml.safe_load(target.read_text())["block_size"] == 16


# --- library now carries the 21 curated levers ------------------------------


def test_library_has_18_validated_levers_with_values():
    from gitm.kernels import load_library

    specs = load_library()
    assert len(specs) == 21
    for s in specs:
        assert s.value is not None, f"{s.name} missing value"
        assert s.expected_delta_lo <= s.expected_delta_mean <= s.expected_delta_hi
        assert s.source.startswith(("http://", "https://", "benchmarks/"))


# --- overhead measurement ---------------------------------------------------


def test_measure_overhead_runs_and_reports():
    from benchmarks.overhead.measure_overhead import measure_overhead

    result = measure_overhead(lambda: sum(range(10_000)), runs=3)
    assert result["runs"] == 3
    assert result["baseline_mean_s"] > 0
    assert "overhead_fraction" in result
