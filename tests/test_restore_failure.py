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


def test_a_failed_baseline_rebuild_leaves_no_engine():
    """The baseline was shut down before the failed rebuild. Keeping its handle
    hands the caller (via cfg.engine) a dead engine that looks live."""
    def no_candidate(_old, _values):
        raise RuntimeError("candidate OOM")

    def no_baseline(_old):
        raise RuntimeError("Free memory 0.0/287.98 GiB")

    app = _serial(_Engine(100.0), restart_fn=no_candidate, baseline_restart_fn=no_baseline)
    res = apply_intervention(_spec(), app, min_keep_delta=0.0)

    assert res.restore_failed
    assert app.engine is None


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


# --------------------------------------------------------------------------- #
# a candidate whose restore failed is not a kept candidate                     #
# --------------------------------------------------------------------------- #
def test_a_failed_restore_is_not_kept():
    """Every restore follows a rejection, so the gate had already said no.
    ``not rolled_back`` read as kept would turn a measured regression into a
    kept result."""
    lost = ApplyResult(True, rolled_back=False, measured_delta=-0.2,
                       error="regression; restore failed", restore_failed=True)
    assert not lost.kept
    assert ApplyResult(True, rolled_back=False, measured_delta=0.1).kept
    assert not ApplyResult(True, rolled_back=True, measured_delta=-0.1).kept


def test_the_export_does_not_record_a_failed_restore_as_kept():
    """History maps kept+significant to a win. A -20% whose rollback failed
    must reach it as a loss."""
    from gitm.optimizer.apply import EngineABResult
    from gitm.optimizer.verification_export import build_record

    ab = EngineABResult(knob="block_size", value=16, baseline_tps=100.0,
                        candidate_tps=80.0, speedup=0.8, kept=False, significant=True)
    lost = ApplyResult(True, rolled_back=False, measured_delta=-0.2,
                       error="regression; restore failed", restore_failed=True)
    assert build_record(_spec(), ab, lost).kept is False


def test_the_report_does_not_count_a_failed_restore_as_verified():
    from gitm.optimizer.report import Claim, _default_summary

    lost = Claim(summary="s", residual_invariant="kernel_time", residual_value=0.0,
                 causal_evidence="e", intervention_name="block_size",
                 predicted_delta=0.05, measured_delta=0.3, restore_failed=True)
    assert _default_summary([lost]).startswith("No claims verified")


def test_autoresearch_counts_what_it_never_reached(monkeypatch):
    """Only the pass knows what it had ranked, so it says how many it skipped."""
    import gitm.agents.autoresearch as ar
    from gitm.agents.policy import RankedCandidate

    specs = [_spec(f"knob_{i}", 1) for i in range(7)]
    # Survivors, gate rejections and a known loser from history, interleaved
    # after the one that loses the engine.
    rejected = {"knob_2", "knob_4"}
    measured_loser = "knob_6"
    monkeypatch.setattr(ar, "select_interventions", lambda *a, **kw: [
        RankedCandidate(
            spec=s,
            predicted_delta=-0.05 if s.name == measured_loser else 0.05,
            delta_source="measured" if s.name == measured_loser else "prior",
            rejected_reason="policy.skip_high_risk" if s.name in rejected else None)
        for s in specs])

    class _Proposer:
        def propose(self, cls, target_op=None):
            return specs

    calls = {"n": 0}

    def lost(spec, applicator, **kw):
        calls["n"] += 1
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error="restore failed", restore_failed=True)

    monkeypatch.setattr(ar, "apply_intervention", lost)
    run = ar.autoresearch(_trace(), applicator=object(), proposer=_Proposer())

    assert calls["n"] == 1
    assert run.n_untried == 3  # knob_1, knob_3, knob_5
    # The gate's rejections and the history verdict needed no engine, so they
    # are still recorded rather than counted untried.
    not_applied = {r.spec.name: r.rejected_reason for r in run.results if not r.applicable}
    assert set(not_applied) == {"knob_2", "knob_4", "knob_6"}
    assert not_applied["knob_6"].startswith("history:")
    assert len(run.results) == 4


def test_phase4_still_records_what_the_gate_rejected(tmp_path, monkeypatch):
    """Breaking on a lost engine dropped the gate's rejections queued behind it:
    they reached neither n_rejected nor n_untried."""
    from gitm.agents.policy import RankedCandidate

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        yield _trace(run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    queue = [("a", None), ("b", "policy.skip_high_risk"), ("c", None), ("d", "not_applicable: x")]
    monkeypatch.setattr(loop, "select_interventions", lambda *a, **kw: [
        RankedCandidate(spec=_spec(name, 1), predicted_delta=0.05, rejected_reason=why)
        for name, why in queue])
    monkeypatch.setattr(loop, "apply_intervention", lambda *a, **kw: ApplyResult(
        False, rolled_back=False, measured_delta=None, error="restore failed",
        restore_failed=True))

    out = loop.run_loop(loop.LoopConfig(budget="30s", scratch=str(tmp_path),
                                        workload_runner=_Runner()))
    assert out["summary"]["n_untried"] == 1   # c
    assert out["summary"]["n_rejected"] == 2  # b, d


def test_shutdown_does_not_repeat_a_working_gitm_shutdown_fn():
    """gitm_shutdown_fn already released the engine; a second generic shutdown
    redoes teardown and warns when it fails on the released engine."""
    import warnings

    calls = []

    class _E:
        def gitm_shutdown_fn(self, _e):
            calls.append("custom")

        def shutdown(self):
            calls.append("generic")
            raise RuntimeError("already shut down")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LiveEngineApplicator._shutdown(_E())
    assert calls == ["custom"]

    class _Broken(_E):
        def gitm_shutdown_fn(self, _e):
            raise RuntimeError("hook failed")

    calls.clear()
    with pytest.warns(RuntimeWarning):
        LiveEngineApplicator._shutdown(_Broken())
    assert calls == ["generic"]  # the fallback still runs when the hook fails


def test_a_release_that_fails_on_keep_does_not_undo_the_keep():
    """commit() raising (e.g. a release warning under -W error) must not turn a
    candidate the gate kept into a crash of the A/B."""
    class _App:
        engine = _Engine(100.0)

        def snapshot(self):
            return {}

        def apply(self, spec):
            pass

        def measure(self, spec):
            return 0.5

        def restore(self, snap):
            raise AssertionError("a keep must not roll back")

        def commit(self):
            raise RuntimeWarning("engine release step 'shutdown' failed")

    res = apply_intervention(_spec(), _App(), min_keep_delta=0.0)
    assert res.kept and "releasing the replaced engine failed" in res.error
