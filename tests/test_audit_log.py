"""Audit log: append-only apply/revert trail with cause, read back in order."""

from __future__ import annotations

from pathlib import Path

from gitm.safety import AuditLog


def _counter():
    n = {"t": 0}

    def clock() -> int:
        n["t"] += 1
        return n["t"]

    return clock


def test_apply_then_revert_recorded_in_order(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl", clock=_counter())
    log.record_apply("max_num_batched_tokens=8192", knob="max_num_batched_tokens",
                     value=8192, cause="headroom: batch underfilled")
    log.record_revert("max_num_batched_tokens=8192", reason="TPOT regression",
                      cause="auto-revert: +6% latency over baseline window")

    entries = log.entries()
    assert [e.event for e in entries] == ["apply", "revert"]
    assert entries[0].ts_ns < entries[1].ts_ns
    assert entries[0].detail["value"] == 8192
    assert "regression" in entries[1].detail["reason"]
    assert entries[0].cause and entries[1].cause


def test_append_only_across_instances(tmp_path: Path):
    p = tmp_path / "audit.jsonl"
    AuditLog(p, clock=_counter()).record("apply", "x", "first")
    # A fresh instance must append, not truncate.
    AuditLog(p, clock=_counter()).record("revert", "x", "second")
    assert [e.event for e in AuditLog(p).entries()] == ["apply", "revert"]


def test_empty_log_reads_empty(tmp_path: Path):
    assert AuditLog(tmp_path / "none.jsonl").entries() == []
