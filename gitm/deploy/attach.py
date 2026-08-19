"""Standalone attach: open a bounded telemetry window on an already-running job.

`gitm attach --job <id>` does not own the workload. Since the CUDA driver reads
CUDA_INJECTION64_PATH only at CUDA init, a collector cannot be injected into a
live process; attach instead *adopts* a collector the job was already launched
under, arms a bounded window, and merges the per-process shards. User-space
(no root), fail-open, no phone-home. `--dry-run` stops after planning.
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
    mode: str
    status: str  # "planned" | "attached" | "busy" | "unsupported" | "no_target"
    pid: int | None
    steps: list[str] = field(default_factory=list)
    reason: str = ""
    out: str | None = None
    n_events: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_pid(job_id: str, pid: int | None) -> int | None:
    if pid is not None:
        return pid
    env_pid = os.environ.get("GITM_ATTACH_PID") or os.environ.get(f"GITM_JOB_{job_id}_PID")
    if env_pid and env_pid.isdigit():
        return int(env_pid)
    return None


def _pid_is_live(pid: int, proc: Path = PROC) -> bool:
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
    """Plan (dry-run) or open a bounded ``duration_s`` window on the job's collector."""
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

    from gitm.serve.discover import classify

    target = classify(resolved, proc)
    if not target.traceable:
        return plan("unsupported", f"{target.reason}\n\n{target.remedy()}")

    return _open_window(target, job_id, workload, duration_s, out, plan)


def _open_window(target, job_id, workload, duration_s, out, plan) -> dict:
    """Borrow the target's injection env for one window; restore it on every exit."""
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

        # Two windows on one collector corrupt both traces; refuse rather than arm.
        if injection.arm_path().exists():
            return plan(
                "busy",
                f"another capture window is already open on this collector "
                f"({injection.arm_path()}) - wait for it to finish, or remove the "
                "marker if its owner died.",
                out=str(out_dir),
            )

        out_dir.mkdir(parents=True, exist_ok=True)
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
