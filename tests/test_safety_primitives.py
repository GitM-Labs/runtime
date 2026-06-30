"""Safety primitives: fail-open guard, auto-revert, gated rollout."""

from __future__ import annotations

from pathlib import Path
import pytest

from gitm.safety import AuditLog, AutoRevert, FailOpenGuard, GatedRollout

# --------- fail-open -----------------------------------------------------------
def test_failopen_reverts_on_normal_exit():
    reverted = []
    with FailOpenGuard(install_signal_handlers=False) as g:
        g.register("knob", lambda: reverted.append("knob"))
    assert reverted == ["knob"]


def test_failopen_reverts_on_exception():
    reverted = []
    with pytest.raises(RuntimeError):
        with FailOpenGuard(install_signal_handlers=False) as g:
            g.register("a", lambda: reverted.append("a"))
            g.register("b", lambda: reverted.append("b"))
            raise RuntimeError("boom")
    assert reverted == ["b", "a"]  # LIFO


def test_failopen_disarm_keeps_change():
    reverted = []
    with FailOpenGuard(install_signal_handlers=False) as g:
        g.register("kept", lambda: reverted.append("kept"))
        g.disarm("kept")  # cleared the gate -> keep it
    assert reverted == []


def test_failopen_logs_reverts(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    with FailOpenGuard(audit=log, install_signal_handlers=False) as g:
        g.register("x", lambda: None, cause="test")
    assert [e.event for e in log.entries()] == ["revert"]


# --------- auto-revert ---------------------------------------------------------
def test_autorevert_warms_up_then_holds_within_tolerance():
    ar = AutoRevert(baseline=100.0, tolerance=0.05, window=3)
    assert not ar.observe(99).should_revert  # warming
    assert not ar.observe(98).should_revert  # warming
    d = ar.observe(99)  # mean 98.67 -> -1.3% within 5%
    assert not d.should_revert


def test_autorevert_fires_on_regression():
    ar = AutoRevert(baseline=100.0, tolerance=0.05, window=3)
    ar.observe(90)
    ar.observe(90)
    d = ar.observe(90)  # mean 90 -> -10% beyond 5%
    assert d.should_revert and d.relative_delta == pytest.approx(-0.1)


# --------- gated rollout -------------------------------------------------------
def test_rollout_shadow_then_manual_promote(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    r = GatedRollout(audit=log)
    r.stage("lever", knob="k", value=1, reason="headroom")
    assert not r.is_live("lever")  # shadow by default

    assert r.promote("lever", confirm=True)
    assert r.is_live("lever")
    events = [e.event for e in log.entries()]
    assert events == ["stage", "apply"]


def test_rollout_unconfirmed_promote_aborts():
    r = GatedRollout()
    r.stage("lever", knob="k", value=1)
    assert not r.promote("lever", confirm=False)
    assert not r.is_live("lever")
    assert r.shadow_changes() == []  # aborted, not lingering in shadow