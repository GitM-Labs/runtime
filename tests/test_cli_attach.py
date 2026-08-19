"""gitm attach --job: argument wiring + user-space, fail-open plan."""

from __future__ import annotations

import json
from pathlib import Path

from gitm.cli import _parser, main
from gitm.deploy import attach_job
from gitm.tracer import injection


def _fake_proc(tmp_path: Path, pid: int, env: dict[str, str]) -> Path:
    """A minimal /proc/<pid> with an environ file classify() can read."""
    proc = tmp_path / "proc"
    d = proc / str(pid)
    d.mkdir(parents=True)
    (d / "environ").write_bytes(
        b"".join(f"{k}={v}\x00".encode() for k, v in env.items())
    )
    return proc


def test_parser_accepts_attach():
    args = _parser().parse_args(["attach", "--job", "abc", "--dry-run"])
    assert args.cmd == "attach"
    assert args.job == "abc"
    assert args.dry_run is True


def test_attach_no_target_when_unresolvable(monkeypatch):
    monkeypatch.delenv("GITM_ATTACH_PID", raising=False)
    plan = attach_job("job-1", dry_run=False)
    assert plan["status"] == "no_target"
    assert plan["mode"] == "user-space"


def test_attach_dry_run_plans_with_resolved_pid():
    plan = attach_job("job-2", pid=4321, dry_run=True)
    assert plan["status"] == "planned"
    assert plan["pid"] == 4321
    # fail-open invariant is part of the documented plan.
    assert any("fail-open" in step for step in plan["steps"])


def test_main_attach_returns_zero_on_plan(capsys):
    rc = main(["attach", "--job", "j", "--pid", "999", "--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "j" and out["status"] == "planned"


def test_parser_accepts_duration_and_out():
    args = _parser().parse_args(
        ["attach", "--job", "j", "--pid", "7", "--duration", "5", "--out", "/tmp/o"]
    )
    assert args.duration == 5.0 and args.out == "/tmp/o"


def test_attach_no_target_when_pid_dead(tmp_path):
    # A committed attach against a PID with no /proc entry is a miss, not a crash.
    proc = tmp_path / "proc"
    proc.mkdir()
    plan = attach_job("job", pid=4242, dry_run=False, proc=proc)
    assert plan["status"] == "no_target"


def test_attach_unsupported_when_not_under_collector(tmp_path):
    # Live, ours to read, but never launched under libgitm_inject.so: cannot be made
    # traceable in user space, so refuse with the restart remedy — not a fake attach.
    proc = _fake_proc(tmp_path, 100, {"PATH": "/usr/bin"})
    plan = attach_job("job", pid=100, dry_run=False, proc=proc)
    assert plan["status"] == "unsupported"
    assert injection.ENV_LIB in plan["reason"]  # remedy names the var to export
    assert plan["n_events"] is None


def test_attach_adopts_collector_and_opens_window(tmp_path):
    # Target is running under our collector: open a zero-length window and merge.
    # No GPU and no shards here, so the honest result is a clean 0-kernel capture.
    base = tmp_path / "collector" / "trace.jsonl"
    base.parent.mkdir(parents=True)
    proc = _fake_proc(
        tmp_path,
        200,
        {injection.ENV_LIB: "/lib/libgitm_inject.so", injection.ENV_OUT: str(base)},
    )
    out_dir = tmp_path / "out"
    plan = attach_job(
        "job", pid=200, dry_run=False, duration_s=0.0, out=out_dir, proc=proc
    )
    assert plan["status"] == "attached"
    assert plan["n_events"] == 0
    assert plan["out"] == str(out_dir)
    assert (out_dir / "trace.jsonl").exists()
    # Fail-open: the borrowed injection env is restored, the window is disarmed.
    import os

    assert injection.ENV_LIB not in os.environ
    assert not (base.parent / (base.name + ".arm")).exists()


def test_attach_busy_when_window_already_open(tmp_path):
    base = tmp_path / "collector" / "trace.jsonl"
    base.parent.mkdir(parents=True)
    (base.parent / (base.name + ".arm")).touch()  # another window already open
    proc = _fake_proc(
        tmp_path,
        300,
        {injection.ENV_LIB: "/lib/libgitm_inject.so", injection.ENV_OUT: str(base)},
    )
    plan = attach_job(
        "job", pid=300, dry_run=False, duration_s=0.0, out=tmp_path / "o", proc=proc
    )
    assert plan["status"] == "busy"
