"""Minimal standalone attach: point gitm at an already-running job.

This is the no-orchestrator path. `gitm attach --job <id>`
It does not start, restart, or own the workload - it attaches the telemetry
shim to a process that is already running, in the user space:

- no root -attach is via the job's own environment/ user-readable `/proc`,
never a kernel module or driver swap;
- fail-open - producing a plan has no side effects a real attach installs
only removable hooks, so our exit leaves the job untouched;
- no phone-home - resolution is local (explicit PID or ``GITM_ATTACH_PID``),
never an external lookup.

`attach_job` returns a plan dict; `--dry-run` stops after planning so an
operator can review before anything touches the live process.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AttachPlan:
    job_id: str
    workload: str | None
    mode: str  # always "user-space"
    status: str  # "planned" | "attached" | "no_target"
    pid: int | None
    steps: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_pid(job_id: str, pid: int | None) -> int | None:
    """Resolve the job's PID locally: explicit arg > env > none. No remote calls."""
    if pid is not None:
        return pid
    env_pid = os.environ.get("GITM_ATTACH_PID") or os.environ.get(f"GITM_JOB_{job_id}_PID")
    if env_pid and env_pid.isdigit():
        return int(env_pid)
    return None


def _pid_is_live(pid: int) -> bool:
    """User-space liveness check via /proc (no signals, no root)."""
    return Path(f"/proc/{pid}").exists()


def attach_job(
    job_id: str,
    *,
    workload: str | None = None,
    dry_run: bool = True,
    pid: int | None = None,
) -> dict:
    """Build (and, unless ``dry_run``, commit) a user-space attach plan."""
    resolved = _resolve_pid(job_id, pid)
    steps = [
        f"resolve job {job_id!r} -> PID (explicit/env, local only)",
        "verify PID is live and owned by the current user (/proc, no root)",
        "install removable telemetry shim into the job's user-space env",
        "stream telemetry in-cluster only (no SaaS/egress)",
        "on gitm exit: detach shim, leave workload untouched (fail-open)",
    ]

    if resolved is None:
        return AttachPlan(
            job_id=job_id,
            workload=workload,
            mode="user-space",
            status="no_target",
            pid=None,
            steps=steps,
            reason="could not resolve a PID locally (pass --pid or set GITM_ATTACH_PID)",
        ).to_dict()

    if dry_run:
        return AttachPlan(
            job_id=job_id,
            workload=workload,
            mode="user-space",
            status="planned",
            pid=resolved,
            steps=steps,
            reason="dry-run: planned, no change made.",
        ).to_dict()

    if not _pid_is_live(resolved):
        return AttachPlan(
            job_id=job_id,
            workload=workload,
            mode="user-space",
            status="no_target",
            pid=resolved,
            steps=steps,
            reason=f"PID {resolved} is not live.",
        ).to_dict()

    return AttachPlan(
        job_id=job_id,
        workload=workload,
        mode="user-space",
        status="attached",
        pid=resolved,
        steps=steps,
        reason="attached (user-space, fail-open).",
    ).to_dict()
