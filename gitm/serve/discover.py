"""Find the vLLM server to attach to, and decide whether it can be traced at all.

Everything here is ``/proc`` and the current user: no root, no ptrace, no phone-home,
matching the fail-open contract in :mod:`gitm.deploy.attach`. Resolution is local by
construction — a PID, a port, or a scan of this box's process table.

The constraint this module exists to state plainly: **the CUDA driver reads
``CUDA_INJECTION64_PATH`` exactly once, at CUDA init, and never looks again.** A vLLM
server that started without it cannot be made traceable while it runs — not by us,
not by nsys, not by anything short of restarting the process. So "attach" here means
*adopting* a server that was already launched under the injection library; what gitm
adds is finding it, proving it is ours, and bounding a window inside its lifetime.

That is worth saying up front rather than discovering it as an empty trace an hour
later, so :func:`classify` reports untraceable targets with the exact environment to
restart under, and the attach path refuses to arm a window it knows will be empty.

The one piece of good news: because the arm marker is a file the injected collector
stats on every buffer flush, the process that opens the window does not have to be
the process that launched the server. A long-lived server can be captured many times
by many short-lived gitm invocations.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gitm.tracer import injection

PROC = Path("/proc")

# What a vLLM OpenAI server looks like in a command line. The console script
# (``vllm serve``), the module form, and the api_server entry point all appear in
# the wild; matching one of them only would silently miss the others.
_VLLM_PATTERNS = (
    re.compile(r"(^|/)vllm$"),
    re.compile(r"vllm\.entrypoints\.openai\.api_server"),
    re.compile(r"vllm\.entrypoints\.api_server"),
)


def read_cmdline(pid: int, proc: Path = PROC) -> list[str]:
    """The process's argv, or ``[]`` if it is gone or not ours to read."""
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def read_environ(pid: int, proc: Path = PROC) -> dict[str, str] | None:
    """The process's environment, or ``None`` when it cannot be read.

    ``/proc/<pid>/environ`` is readable only by the process owner (and root), which is
    exactly the boundary we want: gitm attaches to *your* jobs without privilege, and
    declines somebody else's rather than asking for a way around it. ``None`` and
    ``{}`` mean different things — unreadable versus genuinely empty — so the caller
    can tell "not yours" from "started with no environment".
    """
    try:
        raw = (proc / str(pid) / "environ").read_bytes()
    except (PermissionError, OSError):
        return None
    env: dict[str, str] = {}
    for chunk in raw.decode("utf-8", "replace").split("\0"):
        if not chunk:
            continue
        k, sep, v = chunk.partition("=")
        if sep:
            env[k] = v
    return env


def is_vllm_server(cmdline: list[str]) -> bool:
    """True for a vLLM OpenAI server command line — and not for its own workers.

    ``vllm serve`` spawns ``VLLM::EngineCore`` and one worker per TP rank; those
    inherit the same environment and are traced by the same collector, but they do not
    hold the HTTP port and are not what an operator means by "the server".
    """
    if not cmdline:
        return False
    joined = " ".join(cmdline)
    if "EngineCore" in joined or "VLLM::" in joined:
        return False
    if any(p.search(cmdline[0]) for p in _VLLM_PATTERNS) and "serve" in cmdline[1:3]:
        return True
    return any(p.search(joined) for p in _VLLM_PATTERNS[1:])


def iter_pids(proc: Path = PROC) -> list[int]:
    try:
        return sorted(int(p.name) for p in proc.iterdir() if p.name.isdigit())
    except OSError:
        return []


def find_vllm_pids(proc: Path = PROC) -> list[int]:
    """Every vLLM OpenAI server on this box, oldest first."""
    return [pid for pid in iter_pids(proc) if is_vllm_server(read_cmdline(pid, proc))]


def _listening_inodes(port: int, proc: Path = PROC) -> set[int]:
    """Socket inodes bound to ``port`` in LISTEN state, from /proc/net/tcp{,6}.

    Parsed rather than shelled out to ``ss``/``lsof``: those are not installed on every
    inference image, and a missing binary should not be the reason a capture cannot
    find its server.
    """
    inodes: set[int] = set()
    for name in ("net/tcp", "net/tcp6"):
        try:
            lines = (proc / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            local, state, inode = fields[1], fields[3], fields[9]
            _, _, port_hex = local.rpartition(":")
            try:
                if int(port_hex, 16) != port or state != "0A":  # 0A == TCP_LISTEN
                    continue
                inodes.add(int(inode))
            except ValueError:
                continue
    return inodes


def pid_listening_on(port: int, proc: Path = PROC) -> int | None:
    """Which of our processes holds the listening socket on ``port``.

    Only processes owned by this user are visible (``/proc/<pid>/fd`` is private),
    which is the same boundary :func:`read_environ` enforces — a server we could not
    read the environment of could not be attached to anyway.
    """
    inodes = _listening_inodes(port, proc)
    if not inodes:
        return None
    wanted = {f"socket:[{i}]" for i in inodes}
    for pid in iter_pids(proc):
        fd_dir = proc / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in entries:
            try:
                if os.readlink(fd) in wanted:
                    return pid
            except OSError:
                continue
    return None


@dataclass
class Target:
    """A candidate server, and the verdict on whether gitm can trace it."""

    pid: int
    cmdline: list[str] = field(default_factory=list)
    inject_lib: str | None = None
    trace_out: str | None = None
    traceable: bool = False
    reason: str = ""
    base_url: str | None = None
    shard_pids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def remedy(self) -> str:
        """The exact restart that would make this server traceable.

        Printed on every untraceable target because there is no in-place fix: the
        injection path is read at CUDA init, so the only route from here to a trace is
        through a restart, and the operator should not have to go and read the tracer
        source to find out what to export.
        """
        lib = injection.lib_path()
        return (
            "This server was not started under the gitm collector, and the CUDA driver\n"
            "reads CUDA_INJECTION64_PATH only at CUDA init — it cannot be added to a\n"
            "live process. Restart it under the collector:\n\n"
            f"    export {injection.ENV_LIB}={lib}\n"
            f"    export {injection.ENV_OUT}=$GITM_SCRATCH/traces/<name>/trace.jsonl\n"
            "    vllm serve <model> ...\n\n"
            "or let gitm own the launch, which does the same thing and keeps the\n"
            "server up for later windows:\n\n"
            "    gitm capture serve --keep-server -- vllm serve <model> ..."
        )


def shard_pids_for(trace_out: str | os.PathLike) -> list[int]:
    """PIDs that have opened a shard under this trace base.

    Each CUDA process the collector was loaded into writes ``<trace_out>.<pid>``, so
    the shard names are a direct census of the traced cohort — frontend, EngineCore,
    and one per TP rank. An attach target whose cohort is just the frontend means the
    engine processes started before the injection variable was set, which produces a
    trace with no model kernels in it.
    """
    base = Path(trace_out)
    pids: list[int] = []
    try:
        candidates = list(base.parent.glob(base.name + ".*"))
    except OSError:
        return []
    for p in candidates:
        suffix = p.suffix.lstrip(".")
        if suffix.isdigit():
            pids.append(int(suffix))
    return sorted(pids)


def classify(pid: int, proc: Path = PROC, *, base_url: str | None = None) -> Target:
    """Decide whether ``pid`` can be traced, and say why not when it cannot."""
    cmdline = read_cmdline(pid, proc)
    target = Target(pid=pid, cmdline=cmdline, base_url=base_url)

    if not (proc / str(pid)).exists():
        target.reason = f"PID {pid} is not live."
        return target

    env = read_environ(pid, proc)
    if env is None:
        target.reason = (
            f"cannot read /proc/{pid}/environ — the process belongs to another user. "
            "gitm attaches user-space only and will not escalate."
        )
        return target

    lib = env.get(injection.ENV_LIB)
    out = env.get(injection.ENV_OUT)
    target.inject_lib = lib
    target.trace_out = out

    if not lib:
        target.reason = (
            f"PID {pid} started without {injection.ENV_LIB}: nothing is collecting "
            "inside it, and nothing can be made to."
        )
        return target
    if Path(lib).name != injection.LIB_NAME:
        # nsys sets this variable too. Arming a window against a profiler that is not
        # ours would produce no shards and look identical to a broken install.
        target.reason = (
            f"PID {pid} is being collected by another profiler "
            f"({injection.ENV_LIB}={lib}), not by {injection.LIB_NAME}. Two CUPTI "
            "activity collectors cannot share a process."
        )
        return target
    if not out:
        target.reason = (
            f"PID {pid} has {injection.ENV_LIB} set but no {injection.ENV_OUT}: the "
            "collector has nowhere to write and is inert."
        )
        return target

    target.shard_pids = shard_pids_for(out)
    target.traceable = True
    target.reason = (
        f"collecting into {out} ({len(target.shard_pids)} CUDA process(es) in the cohort)"
    )
    return target


def resolve_target(
    *,
    pid: int | None = None,
    port: int | None = None,
    proc: Path = PROC,
    base_url: str | None = None,
) -> tuple[Target | None, str]:
    """Resolve an attach target from the most specific hint available.

    Order: explicit PID, then the process holding the port, then a scan for a lone
    vLLM server. The scan deliberately refuses to choose when it finds several — on a
    box running two servers, guessing means capturing the wrong model and never
    knowing. Returns ``(target, message)``; ``target`` is ``None`` only when nothing
    could be resolved at all.
    """
    if pid is not None:
        return classify(pid, proc, base_url=base_url), f"explicit --pid {pid}"

    if port is not None:
        found = pid_listening_on(port, proc)
        if found is None:
            return None, (
                f"nothing owned by this user is listening on port {port}. Pass --pid, "
                "or check the server is on this box (tracing is same-host only)."
            )
        return classify(found, proc, base_url=base_url), f"port {port} -> PID {found}"

    candidates = find_vllm_pids(proc)
    if not candidates:
        return None, (
            "no vLLM server found in this box's process table. Start one with "
            "`gitm capture serve`, or pass --pid/--port."
        )
    if len(candidates) > 1:
        listing = ", ".join(str(p) for p in candidates)
        return None, (
            f"{len(candidates)} vLLM servers are running (PIDs {listing}); refusing to "
            "guess which one to trace. Pass --pid or --port."
        )
    return classify(candidates[0], proc, base_url=base_url), f"discovered PID {candidates[0]}"
