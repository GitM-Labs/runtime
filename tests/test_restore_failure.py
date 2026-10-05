"""A rollback that fails ends the candidates, not the run.

In serial restart mode a rollback is a baseline rebuild, and the rebuild fails
when the device memory has not come back. That exception used to escape
``apply_intervention`` and take the whole run with it: no report, no summary,
and every A/B already measured lost at the moment the run most needed to write
them down (known problem P1-4, traceback ``apply.py:94`` -> ``workloads.py:829``).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import gitm.scheduler.loop as loop
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import (
    ApplyResult,
    LiveEngineApplicator,
    RestoreFailed,
    apply_intervention,
)
from gitm.optimizer.degradation import ENGINE_LOST
from gitm.tracer.schema import KernelEvent, Trace


def _spec(knob: str = "block_size", value=16) -> InterventionSpec:
    return InterventionSpec.model_validate(
        dict(name=knob, summary=f"set {knob}", knob=knob, value=value,
             expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
             source="test"))


class _Engine:
    def __init__(self, tps: float):
        self._tps = tps
        self.activated = 0

    @staticmethod
    def gitm_activate_fn(engine):
        engine.activated += 1


def _tps_of(e):
    return e._tps


# --------------------------------------------------------------------------- #
# apply_intervention                                                           #
# --------------------------------------------------------------------------- #
class _RestoreRaises:
    """Measures a regression, then cannot put the baseline back."""

    def __init__(self, *, measure_raises: bool = False):
        self.restores = 0
        self.measure_raises = measure_raises

    def snapshot(self):
        return {}

    def apply(self, spec):
        pass

    def measure(self, spec):
        if self.measure_raises:
            raise RuntimeError("probe died")
        return -0.2

    def restore(self, snapshot):
        self.restores += 1
        raise RuntimeError("Free memory 0.0/287.98 GiB")


def test_a_failed_restore_is_returned_not_raised():
    app = _RestoreRaises()
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)

    assert res.restore_failed
    # Nothing was rolled back, so saying it was would claim a baseline is in place.
    assert not res.rolled_back
    # The measurement was taken before the restore and is still a measurement.
    assert res.measured_delta == pytest.approx(-0.2)
    assert "regression" in res.error and "Free memory" in res.error
    assert app.restores == 1


def test_a_failed_restore_after_a_failed_measure_is_returned_too():
    res = apply_intervention(_spec(), _RestoreRaises(measure_raises=True), min_keep_delta=0.0)
    assert res.restore_failed and not res.rolled_back
    assert "probe died" in res.error


def test_a_sound_restore_is_unchanged():
    from gitm.optimizer.apply import DictApplicator

    cfg = {"block_size": 8}
    res = apply_intervention(_spec(), DictApplicator(cfg, measure_fn=lambda s: -0.1),
                             min_keep_delta=0.0)
    assert res.rolled_back and not res.restore_failed
    assert cfg == {"block_size": 8}


def test_a_restore_failure_inside_apply_is_not_retried():
    """An applicator that already tried the rebuild inside apply() and failed
    raises RestoreFailed. Calling restore() again would retry the same rebuild."""

    class _Lost(_RestoreRaises):
        def apply(self, spec):
            raise RestoreFailed("baseline rebuild failed")

    app = _Lost()
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)
    assert res.restore_failed and not res.applied
    assert app.restores == 0


# --------------------------------------------------------------------------- #
# LiveEngineApplicator, serial                                                 #
# --------------------------------------------------------------------------- #
def _serial(baseline, *, restart_fn, baseline_restart_fn):
    return LiveEngineApplicator(
        baseline, throughput_fn=_tps_of, restart_fn=restart_fn,
        baseline_restart_fn=baseline_restart_fn, restart_mode="serial")


def test_serial_candidate_and_baseline_both_failing_rebuild_once():
    """The traceback from the MI355X run: the candidate cannot build, and then
    neither can the baseline. One rebuild attempt, and a result rather than a
    crash."""
    attempts = {"n": 0}

    def no_candidate(_old, _values):
        raise RuntimeError("candidate OOM")

    def no_baseline(_old):
        attempts["n"] += 1
        raise RuntimeError("Free memory 0.0/287.98 GiB")

    app = _serial(_Engine(100.0), restart_fn=no_candidate, baseline_restart_fn=no_baseline)
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)

    assert res.restore_failed and not res.rolled_back
    assert attempts["n"] == 1
    assert "Free memory" in res.error


def test_serial_rollback_failing_to_rebuild_is_returned():
    def no_baseline(_old):
        raise RuntimeError("Free memory 0.0/287.98 GiB")

    app = _serial(_Engine(100.0), restart_fn=lambda _o, _v: _Engine(50.0),
                  baseline_restart_fn=no_baseline)
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)

    assert res.restore_failed and not res.rolled_back
    assert res.measured_delta is not None  # the A/B itself was measured
    # The record is consumed, so a second restore() cannot retry the rebuild.
    app.restore({})


def test_serial_baseline_rebuilt_after_a_failed_candidate_is_activated():
    """The workload runner drives the last-activated engine. A rebuilt baseline
    that is only assigned leaves it on the engine that was just shut down."""
    restored = _Engine(100.0)

    def no_candidate(_old, _values):
        raise RuntimeError("candidate OOM")

    app = _serial(_Engine(100.0), restart_fn=no_candidate,
                  baseline_restart_fn=lambda _old: restored)
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)

    assert res.rolled_back and not res.restore_failed
    assert app.engine is restored and restored.activated == 1


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #
class _Runner:
    workload_id = "vllm-decode"

    def __call__(self) -> dict:
        return {"events": 1}


def _trace(run_id="r"):
    names = ["flash_fwd_kernel", "void cutlass_gemm"]
    events = [KernelEvent(name=n, start_ns=i * 1000, end_ns=i * 1000 + 900, stream_id=7,
                          device_id=0, correlation_id=i) for i, n in enumerate(names)]
    return Trace(workload_id="vllm-decode", fingerprint="f", run_id=run_id,
                 device_count=1, vendor="nvidia", captured_at_ns=0,
                 duration_ns=len(names) * 1000, events=events)


def test_the_run_stops_trying_candidates_and_still_reports(tmp_path, monkeypatch):
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        yield _trace(run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    applies: list[str] = []

    def lost(spec, applicator, **kw):
        applies.append(spec.name)
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error="apply failed; restore failed: Free memory 0.0",
                           restore_failed=True)

    monkeypatch.setattr(loop, "apply_intervention", lost)

    out = loop.run_loop(loop.LoopConfig(budget="30s", scratch=str(tmp_path),
                                        workload_runner=_Runner()))
    summary, run_dir = out["summary"], Path(out["run_dir"])

    assert len(applies) == 1, "a candidate was tried after the engine was lost"
    assert "restore failed" in summary["engine_lost"]
    assert summary["n_untried"] >= 1
    assert (run_dir / "report.md").exists()

    degradations = json.loads((run_dir / "degradations.json").read_text())
    stages = [d["stage"] for d in degradations["items"]]
    assert ENGINE_LOST in stages
    # Autoresearch is skipped, and says why rather than reading as "found nothing".
    ar = json.loads((run_dir / "autoresearch.json").read_text())
    assert ar["results"] == []
    assert any("engine lost" in d["reason"] for d in ar["degradations"])


def test_a_run_that_kept_its_engine_reports_none(tmp_path, monkeypatch):
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        yield _trace(run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    out = loop.run_loop(loop.LoopConfig(budget="30s", scratch=str(tmp_path),
                                        workload_runner=_Runner()))
    assert out["summary"]["engine_lost"] is None
    assert out["summary"]["n_untried"] == 0
