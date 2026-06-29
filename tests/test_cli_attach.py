"""gitm attach --job: argument wiring + user-space, fail-open plan."""

from __future__ import annotations

import json

from gitm.cli import _parser, main
from gitm.deploy import attach_job


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
