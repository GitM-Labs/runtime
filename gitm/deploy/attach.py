"""Minimal standalone attach: point gitm at an already-running job.

This is the no-orchestrator path. `gitm attach --job <id>`
It does not start, restart, or own the workload - it attaches the telemetry
shim to a process that is already running, in the user space:

- no root -attach is via the job's own environment/ user-readable `/proc`,
never a kernel module or driver swap;
- fail-open - an attach only arms a removable file marker on a collector the
job already loaded, so our exit leaves the job untouched;
- no phone-home - resolution is local (explicit PID or ``GITM_ATTACH_PID``),
never an external lookup.

The one thing this path cannot do is *inject* a collector into a live process:
the CUDA driver reads ``CUDA_INJECTION64_PATH`` once, at CUDA init, and never
again (see :mod:`gitm.serve.discover`). So "attach the telemetry shim" means
*adopting* the collector the job was already launched under and opening a
**bounded** window inside its lifetime - arm, watch for ``--duration``, disarm,
merge the per-process kernel shards. A job that was not started under the
collector is reported ``unsupported`` with the exact restart that would fix it,
never a false ``attached``.

``attach_job`` returns a plan dict. ``--dry-run`` stops after planning, before
anything touches the live process; without it the window is actually opened and
the merged trace is written under ``out``.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROC = Path("/proc")


@dataclass
class AttachPlan:
    job_id: str
    workload: str | None
    mode: str  # always "user-space"
    status: str  # "planned" | "attached" | "busy" | "unsupported" | "no_target"
    pid: int | None
    steps: list[str] = field(default_factory=list)
    reason: str = ""
    out: str | None = None
    n_events: int | None = None

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


def _pid_is_live(pid: int, proc: Path = PROC) -> bool:
    """User-space liveness check via /proc (no signals, no root)."""
    return (proc / str(pid)).exists()


_STEPS = [
    "resolve job -> PID (explicit/env, local only)",
    "verify PID is ours and running under the gitm collector (/proc/environ, no root)",
    "arm a bounded telemetry window on the job's collector (removable file marker)",
    "merge the window's per-process kernel shards; stream in-cluster only (no egress)",
    "disarm on exit: window closed, workload untouched (fail-open)",
]


def attach_job(
    job_id: str,
    *,
    workload: str | None = None,
    dry_run: bool = True,
    pid: int | None = None,
    duration_s: float = 30.0,
    out: str | Path | None = None,
    proc: Path = PROC,
) -> dict:
    """Build (and, unless ``dry_run``, commit) a user-space attach.

    When ``dry_run`` is false and the target is running under the gitm collector,
    a bounded ``duration_s`` window is opened on it and the merged trace is written
    under ``out`` (default ``$GITM_SCRATCH/traces/attach-<ts>/trace.jsonl``).
    """
    resolved = _resolve_pid(job_id, pid)

    def plan(status: str, reason: str, **extra) -> dict:
        return AttachPlan(
            job_id=job_id,
            workload=workload,
            mode="user-space",
            status=status,
            pid=resolved,
            steps=list(_STEPS),
            reason=reason,
            **extra,
        ).to_dict()

    if resolved is None:
        return plan(
            "no_target",
            "could not resolve a PID locally (pass --pid or set GITM_ATTACH_PID)",
        )

    if dry_run:
        return plan("planned", "dry-run: planned, no change made.")

    if not _pid_is_live(resolved, proc):
        return plan("no_target", f"PID {resolved} is not live.")

    # Reuse the injectability verdict the serve path already states plainly, rather
    # than re-parsing /proc/environ here: classify reads the target's environment,
    # confirms it is ours to read, and confirms it is being collected by *our*
    # library (not nsys, not nothing). Everything below trusts target.traceable.
    from gitm.serve.discover import classify

    target = classify(resolved, proc)
    if not target.traceable:
        # Not an error - the job simply cannot be attached in user space. Carry the
        # exact restart that would make it traceable instead of a bare refusal.
        return plan("unsupported", f"{target.reason}\n\n{target.remedy()}")

    return _open_window(target, job_id, workload, duration_s, out, plan)


def _open_window(target, job_id, workload, duration_s, out, plan) -> dict:
    """Arm a bounded window on the adopted collector, merge shards, disarm.

    Borrow-not-adopt the injection environment: point this process at the target's
    collector only for the duration of the window, and restore the prior values on
    every exit path so a failed attach never leaves us aimed at another run's shards.
    """
    from gitm._paths import traces_dir
    from gitm.tracer import injection
    from gitm.tracer.capture import capture

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out) if out else traces_dir() / f"attach-{stamp}"
    trace_path = out_dir / "trace.jsonl"

    prior_lib = os.environ.get(injection.ENV_LIB)
    prior_out = os.environ.get(injection.ENV_OUT)

    def restore() -> None:
        for key, val in ((injection.ENV_LIB, prior_lib), (injection.ENV_OUT, prior_out)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    try:
        os.environ[injection.ENV_LIB] = str(target.inject_lib)
        os.environ[injection.ENV_OUT] = str(target.trace_out)

        # Two armed windows on one collector is not a partial failure: both traces
        # end up holding both windows' kernels. Refuse rather than corrupt silently.
        if injection.arm_path().exists():
            return plan(
                "busy",
                f"another capture window is already open on this collector "
                f"({injection.arm_path()}) - wait for it to finish, or remove the "
                "marker if its owner died.",
                out=str(out_dir),
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        # capture() sees the borrowed env, takes the injected/merge path, and owns the
        # whole window in a finally: clear stale shards, bound the CUPTI clock, arm,
        # settle, disarm, merge. We only hold it open for the observation window.
        with capture(
            trace_path, workload_id="gitm-attach", fingerprint=workload or job_id
        ) as trace:
            time.sleep(duration_s)
        n_events = len(trace.events)
    finally:
        restore()

    return plan(
        "attached",
        f"attached (user-space, fail-open); captured {n_events} kernel record(s) over "
        f"{duration_s:.0f}s into {trace_path}.",
        out=str(out_dir),
        n_events=n_events,
    )
