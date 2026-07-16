"""Cross-process trace collection via the CUDA driver's injection hook.
The in-process CUPTI shim can only see kernels launched by the interpreter that
imported it. vLLM's V1 engine runs the model in a separate ``EngineCore`` process,
so that shim captures nothing for a vLLM run — the trace comes back empty and the
pipeline reports "no-data". Disabling vLLM's multiprocessing would fix the symptom
and corrupt the measurement: ``EngineCore`` lives in its own process precisely to
keep the scheduler loop off the GIL that the frontend holds for detokenization, and
folding it back into the parent injects idle gaps that don't exist in production —
into exactly the stall/idle signal we are trying to measure.
So instead we let the CUDA driver load our collector into the child. Setting
``CUDA_INJECTION64_PATH`` makes the driver ``dlopen`` that library in every process
that initializes CUDA and call its ``InitializeInjection()`` before any kernel runs.
It is an ordinary environment variable, so it is inherited across fork/spawn. vLLM's
process model is untouched; it never knows we are there.
Each process writes ``$GITM_TRACE_OUT.<pid>`` (see ``cupti_inject.c``). This module
is the other half: it arms the window, merges the shards, and drops records outside
the window.
Both environment variables must be set **before the traced process starts CUDA**,
which for vLLM means before the engine is constructed — the driver reads
``CUDA_INJECTION64_PATH`` at CUDA init, long before ``capture()`` is entered. Export
them in the shell that launches the run; ``run_env()`` renders the exact pair.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gitm.tracer.schema import TraceEvent

ENV_LIB = "CUDA_INJECTION64_PATH"
ENV_OUT = "GITM_TRACE_OUT"
ENV_SETTLE = "GITM_TRACE_SETTLE_S"

LIB_NAME = "libgitm_inject.so"

# How long to wait, after the workload finishes, for in-flight CUPTI buffers in
# other processes to land on disk. We cannot reach into the child to force a
# flush, so the injected library flushes on a period (GITM_TRACE_FLUSH_MS, default
# 100ms) and we wait out one period plus slack before merging. Too short and the
# tail of the trace is silently missing.
DEFAULT_SETTLE_S = 0.5


def lib_path() -> Path:
    """Where the injection library is built, whether or not it exists yet."""
    from gitm.tracer import _cupti

    return Path(_cupti.__file__).resolve().parent / LIB_NAME


def active() -> bool:
    """True when this run is being collected by our injection library.
    Checks that the injection path actually points at ``libgitm_inject.so``: another
    profiler (nsys sets this variable too) means the trace is not ours to merge, and
    we must not silently claim its records or skip our own in-process collection.
    """
    lib = os.environ.get(ENV_LIB, "")
    return bool(lib) and Path(lib).name == LIB_NAME and bool(os.environ.get(ENV_OUT))


def run_env(out_path: str | Path) -> dict[str, str]:
    """The environment a traced run needs, ready to export."""
    return {ENV_LIB: str(lib_path()), ENV_OUT: str(Path(out_path).resolve())}


def _out_base() -> Path:
    return Path(os.environ[ENV_OUT])


def arm_path() -> Path:
    base = _out_base()
    return base.with_name(base.name + ".arm")


def shard_paths() -> list[Path]:
    """Every per-pid shard for this run, excluding the arm marker."""
    base = _out_base()
    return sorted(
        p
        for p in base.parent.glob(base.name + ".*")
        if p != arm_path() and p.suffix != ".arm"
    )


def arm() -> None:
    """Open the collection window. The injected library writes only while this exists."""
    arm_path().parent.mkdir(parents=True, exist_ok=True)
    arm_path().touch()


def disarm() -> None:
    arm_path().unlink(missing_ok=True)


def _shard_pid(path: Path) -> int | None:
    """The pid a shard belongs to, from its ``.<pid>`` suffix."""
    try:
        return int(path.suffix.lstrip("."))
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    """Signal 0 — probe for existence, portable, and never touches /proc.
    Errs toward "alive": a pid we can see but may not signal (PermissionError) is
    still a process, and deleting its shard is the failure this whole check exists to
    prevent. Guessing "dead" costs a silently empty trace; guessing "alive" costs a
    stale file.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def clear_shards() -> None:
    """Delete every shard, live or not. Only safe when nothing is collecting."""
    for p in shard_paths():
        p.unlink(missing_ok=True)


def clear_stale_shards() -> None:
    """Drop shards from dead processes, and ONLY from dead processes.
    A shard is created and held open by the injected library the moment its process
    initializes CUDA — which, for vLLM, is while the engine is being built, before
    ``capture()`` is ever entered. Unlinking it here would pull the file out from
    under a live EngineCore: its FILE* keeps writing happily into a deleted inode,
    nothing reaches disk, and the merge comes back empty with no error anywhere.
    (That is exactly what happened, and it looked identical to the injection hook not
    firing at all.)
    So: only remove shards whose owning process is gone. Records that a live process
    already wrote before the window opened are excluded by the CUPTI-timestamp filter
    in ``read_shards``, not by deleting them.
    """
    for p in shard_paths():
        pid = _shard_pid(p)
        if pid is None or not _pid_alive(pid):
            p.unlink(missing_ok=True)


def settle_seconds() -> float:
    raw = os.environ.get(ENV_SETTLE)
    try:
        return float(raw) if raw else DEFAULT_SETTLE_S
    except ValueError:
        return DEFAULT_SETTLE_S


def settle() -> None:
    time.sleep(settle_seconds())


def read_shards(start_ns: int | None = None, end_ns: int | None = None) -> list[TraceEvent]:
    """Merge every shard into one decoded, time-sorted event list.
    ``start_ns``/``end_ns`` bound the window in the CUPTI clock domain (see
    ``gitm_cupti_timestamp``), not wall-clock. Records outside it are dropped: the
    injected library is loaded for the process's entire lifetime, so without this
    filter a vLLM trace would be dominated by weight loading, ``torch.compile`` and
    CUDA-graph capture — around 80 seconds of it — and kernel-time coverage would be
    meaningless.
    A malformed trailing line is expected and ignored: a process killed mid-write
    leaves a partial record, and losing the last kernel of a shard is a better
    outcome than failing the whole run.
    """
    from gitm.tracer._cupti_decode import decode_records

    records: list[dict] = []
    for shard in shard_paths():
        try:
            text = shard.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from a killed process
            if not isinstance(rec, dict):
                continue
            ts = rec.get("start_ns")
            if not isinstance(ts, int):
                continue
            if start_ns is not None and ts < start_ns:
                continue
            if end_ns is not None and ts > end_ns:
                continue
            records.append(rec)

    return decode_records(records)


def cupti_now() -> int | None:
    """Read the CUPTI clock, the time base the activity records use.
    Safe while the injection library owns collection: reading the clock does not
    register activity callbacks, so it cannot fight the injected collector for the
    process's single callback registration.
    """
    from gitm.tracer._cupti import load_shim

    shim = load_shim()
    if shim is None or not hasattr(shim, "timestamp"):
        return None
    try:
        ts = int(shim.timestamp())
    except Exception:
        return None
    return ts or None