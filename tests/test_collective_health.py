"""Tests for the minimal pre-loop GPU AllReduce check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gitm.health.collective import HealthResult, detect_gpus, run_collective_health


def test_detects_nvidia(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4),
        version=SimpleNamespace(hip=None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert detect_gpus() == ("nvidia", 4)


def test_detects_amd(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 8),
        version=SimpleNamespace(hip="6.3"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert detect_gpus() == ("amd", 8)


def test_skips_when_fewer_than_two_gpus(monkeypatch):
    monkeypatch.setattr("gitm.health.collective.detect_gpus", lambda: ("none", 0))
    result = run_collective_health()
    assert result.ok
    assert result.status == "skipped"


def test_allreduce_runs_across_all_visible_gpus(monkeypatch):
    seen: dict[str, object] = {}

    class Proc:
        returncode = 0
        pid = 123

        def __init__(self, cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs

        def communicate(self, timeout):
            seen["timeout"] = timeout
            Path(seen["cmd"][-1]).write_text(json.dumps({"ok": True}))
            return "", ""

    monkeypatch.setattr("gitm.health.collective.detect_gpus", lambda: ("nvidia", 4))
    monkeypatch.setattr("gitm.health.collective.subprocess.Popen", Proc)

    result = run_collective_health(timeout_s=5)

    assert result == HealthResult("nvidia", 4, "pass", "AllReduce reached every visible GPU")
    assert "--nproc-per-node=4" in seen["cmd"]
    assert seen["timeout"] == 5


def test_wrong_result_fails(monkeypatch):
    class Proc:
        returncode = 0
        pid = 123

        def __init__(self, cmd, **_kwargs):
            self.result_path = Path(cmd[-1])

        def communicate(self, timeout):
            self.result_path.write_text(json.dumps({"ok": False}))
            return "", ""

    monkeypatch.setattr("gitm.health.collective.detect_gpus", lambda: ("amd", 2))
    monkeypatch.setattr("gitm.health.collective.subprocess.Popen", Proc)
    result = run_collective_health()
    assert not result.ok
    assert result.detail == "AllReduce returned incorrect values"


def test_timeout_kills_torchrun_process_group(monkeypatch):
    proc = Mock(pid=123, returncode=None)
    proc.communicate.side_effect = subprocess.TimeoutExpired("torchrun", 1)
    monkeypatch.setattr("gitm.health.collective.detect_gpus", lambda: ("nvidia", 2))
    monkeypatch.setattr("gitm.health.collective.subprocess.Popen", lambda *a, **k: proc)
    killpg = Mock()
    monkeypatch.setattr("gitm.health.collective.os.killpg", killpg)

    result = run_collective_health(timeout_s=1)

    assert not result.ok
    assert "timed out" in result.detail
    killpg.assert_called_once()
    proc.wait.assert_called_once()


def test_loop_aborts_before_factory_on_health_failure(tmp_path: Path, monkeypatch):
    from gitm.scheduler.loop import LoopConfig, run_loop

    monkeypatch.setattr(
        "gitm.health.run_collective_health",
        lambda: HealthResult("nvidia", 2, "fail", "unreachable GPU"),
    )
    monkeypatch.setattr(
        "gitm.scheduler.loop.get_factory",
        lambda _w: (_ for _ in ()).throw(AssertionError("factory must not run")),
    )

    out = run_loop(LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path)))

    assert out["summary"]["status"] == "no_data"
    assert "unreachable GPU" in out["summary"]["diagnostic"]
    health = Path(out["summary"]["report_path"]).parent / "collective_health.json"
    assert json.loads(health.read_text())["status"] == "fail"
