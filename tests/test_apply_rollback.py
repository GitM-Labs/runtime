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
    assert log.entries()[0].detail == {"knobs": {"block_size": 16}}


def test_apply_audits_apply_then_revert_on_regression(tmp_path):
    from gitm.safety import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    cfg = {"block_size": 8}
    apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: -0.05), audit=log)

    # A regressing change is applied then reverted — both must be on the trail.
    assert [e.event for e in log.entries()] == ["apply", "revert"]
    # The revert entry names which knob(s) were involved, not just the cause —
    # essential once a rollback can be a joint (multi-knob) candidate.
    assert log.entries()[1].detail["knobs"] == {"block_size": 16}


def test_apply_audits_knobs_on_apply_and_measure_failures(tmp_path):
    """The revert entry carries which knob(s) were attempted even when the
    failure happens before or during measurement, not just on a regression."""
    from gitm.safety import AuditLog

    class _RaisesOnApply:
        def snapshot(self):
            return {}

        def apply(self, spec):
            raise RuntimeError("bad value")

        def restore(self, snapshot):
            pass

        def measure(self, spec):
            return None

    class _RaisesOnMeasure(_RaisesOnApply):
        def apply(self, spec):
            pass

        def measure(self, spec):
            raise RuntimeError("measurement crashed")

    for applicator in (_RaisesOnApply(), _RaisesOnMeasure()):
        log = AuditLog(tmp_path / f"audit-{type(applicator).__name__}.jsonl")
        apply_intervention(_spec(), applicator, audit=log)
        assert log.entries()[-1].event == "revert"
        assert log.entries()[-1].detail["knobs"] == {"block_size": 16}


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


def test_library_has_validated_levers_with_values():
    from gitm.kernels import load_library

    specs = load_library()
    assert len(specs) == 28
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


# --- correctness gate: interrupts, hangs, and a gate lost in transit ---------


@pytest.fixture(autouse=True)
def _no_gate_outlives_its_test():
    """A gate cut off at its deadline blocks every later apply until it exits."""
    yield
    from gitm.optimizer import apply as apply_mod

    for t in list(apply_mod._timed_out_gates):
        t.join(5)
    assert not apply_mod._gates_still_running()


class _Trail:
    def __init__(self, raise_on: str | None = None):
        self.events: list[tuple[str, str]] = []
        self._raise_on = raise_on

    def record(self, event, name, cause, **kw):
        self.events.append((event, cause))
        if event == self._raise_on:
            raise KeyboardInterrupt


def _gated(gate, **safety) -> InterventionSpec:
    spec = _spec(safety={"requires_correctness_gate": True, **safety})
    return spec.model_copy(update={"correctness_gate": gate})


class _InterruptingApplicator(DictApplicator):
    """Mutates the config, then is interrupted before apply returns."""

    def apply(self, spec):
        super().apply(spec)
        raise KeyboardInterrupt


@pytest.mark.parametrize("where", ["apply", "measure", "gate"])
def test_interrupt_restores_then_propagates(where):
    cfg = {"block_size": 8}

    def interrupt(_spec):
        raise KeyboardInterrupt

    if where == "apply":
        app, spec = _InterruptingApplicator(cfg, measure_fn=lambda s: +0.1), _spec()
    elif where == "measure":
        app, spec = DictApplicator(cfg, measure_fn=interrupt), _spec()
    else:
        app = DictApplicator(cfg, measure_fn=lambda s: +0.1)
        spec = _spec().model_copy(update={"correctness_gate": interrupt})
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(spec, app)
    assert cfg == {"block_size": 8}


def test_a_gate_that_never_returns_times_out_and_restores():
    import threading
    import time

    cfg = {"block_size": 8}
    release = threading.Event()
    spec = _gated(lambda _s: release.wait(), correctness_gate_timeout_s=0.2)
    t0 = time.monotonic()
    try:
        res = apply_intervention(spec, DictApplicator(cfg, measure_fn=lambda s: +0.1))
    finally:
        release.set()
    assert time.monotonic() - t0 < 5
    assert res.applied and res.rolled_back
    assert "timed out" in res.error
    assert cfg == {"block_size": 8}


def test_a_spec_that_needs_a_gate_but_carries_none_is_not_applied():
    cfg = {"block_size": 8}
    spec = _spec(safety={"requires_correctness_gate": True})
    res = apply_intervention(spec, DictApplicator(cfg, measure_fn=lambda s: +0.1))
    assert not res.applied and not res.rolled_back
    assert "requires a correctness gate" in res.error
    assert cfg == {"block_size": 8}


def test_the_gate_requirement_survives_a_save_and_reload():
    spec = _gated(lambda _s: None)
    reloaded = InterventionSpec.model_validate(json.loads(spec.model_dump_json()))
    assert reloaded.correctness_gate is None
    assert reloaded.safety.requires_correctness_gate
    cfg = {"block_size": 8}
    res = apply_intervention(reloaded, DictApplicator(cfg, measure_fn=lambda s: +0.1))
    assert not res.applied
    assert cfg == {"block_size": 8}


def test_intervention_spec_has_a_json_schema():
    schema = InterventionSpec.model_json_schema()
    assert "correctness_gate" not in schema["properties"]
    safety = schema["$defs"]["SafetyGate"]["properties"]
    assert "requires_correctness_gate" in safety


def test_a_zero_gate_deadline_is_rejected():
    with pytest.raises(ValueError):
        _spec(safety={"correctness_gate_timeout_s": 0})


# --- the guard against main's restore-failure path ---------------------------


class _CountingRestore(DictApplicator):
    """Counts restores; optionally interrupted inside one, or failing every one."""

    def __init__(self, config, *, interrupt_in_restore=False, restore_raises=False, **kw):
        super().__init__(config, **kw)
        self.restores = 0
        self._interrupt = interrupt_in_restore
        self._raises = restore_raises

    def restore(self, snapshot):
        self.restores += 1
        if self._raises:
            raise RuntimeError("could not rebuild the baseline")
        super().restore(snapshot)
        if self._interrupt:
            raise KeyboardInterrupt


def test_an_interrupt_inside_a_restore_does_not_restore_twice():
    cfg = {"block_size": 8}
    trail = _Trail()
    app = _CountingRestore(cfg, interrupt_in_restore=True, measure_fn=lambda s: -0.5)
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(_spec(), app, audit=trail)
    assert app.restores == 1
    assert cfg == {"block_size": 8}
    assert trail.events[-1] == (
        "interrupted",
        "aborted by KeyboardInterrupt after a restore began; whether it finished is unknown")


def test_an_interrupt_whose_restore_fails_still_propagates():
    class Audit:
        events: list = []

        def record(self, event, name, cause, **kw):
            self.events.append((event, cause))

    def interrupt(_spec):
        raise KeyboardInterrupt

    audit = Audit()
    app = _CountingRestore({"block_size": 8}, restore_raises=True, measure_fn=interrupt)
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(_spec(), app, audit=audit)
    assert app.restores == 1
    assert ("restore_failed", "aborted by KeyboardInterrupt; restore failed: "
            "could not rebuild the baseline") in audit.events


def test_a_timed_out_gate_whose_restore_fails_is_reported_unrestored():
    import threading

    release = threading.Event()
    app = _CountingRestore({"block_size": 8}, restore_raises=True, measure_fn=lambda s: +0.1)
    try:
        res = apply_intervention(
            _gated(lambda _s: release.wait(), correctness_gate_timeout_s=0.2), app)
    finally:
        release.set()
    assert res.restore_failed and not res.kept
    assert "timed out" in res.error


def test_an_interrupt_after_the_keep_does_not_undo_it():
    class InterruptedCommit(DictApplicator):
        def commit(self):
            raise KeyboardInterrupt

    cfg = {"block_size": 8}
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(_spec(), InterruptedCommit(cfg, measure_fn=lambda s: +0.1))
    assert cfg == {"block_size": 16}


def test_a_timed_out_gate_blocks_the_next_candidate_until_it_exits():
    import threading

    from gitm.optimizer import apply as apply_mod

    release = threading.Event()
    first = apply_intervention(
        _gated(lambda _s: release.wait(), correctness_gate_timeout_s=0.2),
        DictApplicator({"block_size": 8}, measure_fn=lambda s: +0.1))
    assert "timed out" in first.error

    cfg = {"block_size": 8}
    blocked = apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: +0.1))
    assert not blocked.applied and "still running" in blocked.error
    assert cfg == {"block_size": 8}

    release.set()
    for t in list(apply_mod._timed_out_gates):
        t.join(5)
    after = apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: +0.1))
    assert after.kept and cfg == {"block_size": 16}


def test_a_real_ctrl_c_during_the_gate_holds_the_next_candidate():
    import _thread
    import threading

    release = threading.Event()

    def gate(_spec):
        # interrupt_main delivers SIGINT to the main thread, as a real Ctrl-C
        # does, so it surfaces from the join, not from this thread.
        _thread.interrupt_main()
        release.wait(10)

    cfg = {"block_size": 8}
    try:
        with pytest.raises(KeyboardInterrupt):
            apply_intervention(_gated(gate, correctness_gate_timeout_s=60),
                               DictApplicator(cfg, measure_fn=lambda s: +0.1))
        assert cfg == {"block_size": 8}
        assert any(t.name.startswith("correctness-gate") and t.is_alive()
                   for t in threading.enumerate())

        nxt = {"block_size": 8}
        blocked = apply_intervention(_spec(), DictApplicator(nxt, measure_fn=lambda s: +0.1))
        assert not blocked.applied and "still running" in blocked.error
        assert nxt == {"block_size": 8}
    finally:
        release.set()
        for t in threading.enumerate():
            if t.name.startswith("correctness-gate"):
                t.join(5)


def test_a_spec_missing_its_gate_is_refused_before_the_snapshot():
    class NoSnapshot(DictApplicator):
        def snapshot(self):
            raise AssertionError("snapshot taken for a spec that must be refused")

    spec = _spec(safety={"requires_correctness_gate": True})
    res = apply_intervention(spec, NoSnapshot({"block_size": 8}))
    assert not res.applied and "requires a correctness gate" in res.error


def test_an_interrupt_after_a_failed_restore_in_apply_does_not_retry_it():
    from gitm.optimizer.apply import RestoreFailed

    class RebuildFailed(_CountingRestore):
        def apply(self, spec):
            raise RestoreFailed("baseline rebuild failed inside apply")

    app = RebuildFailed({"block_size": 8})
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(_spec(), app, audit=_Trail(raise_on="restore_failed"))
    assert app.restores == 0


def test_an_infinite_gate_deadline_is_rejected():
    with pytest.raises(ValueError):
        _spec(safety={"correctness_gate_timeout_s": float("inf")})
