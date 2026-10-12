"""A run that traced nothing says why, and never reads as clean (L-4).

The default workload decoded in 1.4 s, the tracer flushed nothing, and the run
reported NO DATA beside ``"clean": true``, with nothing to tell "the workload
was too short" from "the tracer never loaded". Those have opposite fixes.
"""

from __future__ import annotations

from pathlib import Path

import gitm.scheduler.loop as loop
from gitm.optimizer.degradation import TRACE_EMPTY
from gitm.tracer import injection


def test_no_tracer_configured(monkeypatch):
    monkeypatch.setattr(injection, "active_vendor", lambda: None)
    assert "no tracer was configured" in loop._why_no_kernels(1.0)


def test_configured_but_never_loaded(monkeypatch):
    monkeypatch.setattr(injection, "active_vendor", lambda: "nvidia")
    monkeypatch.setattr(injection, "shard_paths", lambda: [])
    assert "no process loaded it" in loop._why_no_kernels(30.0)


def test_loaded_but_the_workload_was_too_short(monkeypatch, tmp_path):
    monkeypatch.setattr(injection, "active_vendor", lambda: "nvidia")
    monkeypatch.setattr(injection, "shard_paths", lambda: [tmp_path / "t.jsonl.1"])
    why = loop._why_no_kernels(1.4)
    assert "decoded for only 1.4s" in why and "GITM_VLLM_PROMPTS" in why


def test_a_long_workload_with_shards_is_not_blamed_on_length(monkeypatch, tmp_path):
    monkeypatch.setattr(injection, "active_vendor", lambda: "nvidia")
    monkeypatch.setattr(injection, "shard_paths", lambda: [tmp_path / "t.jsonl.1"])
    assert loop._why_no_kernels(30.0) is None


def test_a_run_that_traced_nothing_is_never_clean(tmp_path: Path, without_vllm):
    from gitm import optimize

    result = optimize(workload="vllm-decode", budget="1s", scratch=str(tmp_path))
    summary = result["summary"]
    assert summary["status"] == "no_data"
    assert summary["degraded"] is True
    assert TRACE_EMPTY in summary["degradations"]["unreliable"]


def test_an_in_process_capture_is_not_blamed_on_missing_injection(monkeypatch):
    """Without injection settings a capture can still trace in-process; then the
    missing settings are not the cause and its own diagnostic stands."""
    monkeypatch.setattr(injection, "active_vendor", lambda: None)
    assert loop._why_no_kernels(30.0, "cupti") is None
    assert "no tracer was configured" in loop._why_no_kernels(30.0, "none")
