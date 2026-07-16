"""Trace capture context manager.
    with gitm.tracer.capture(out_path) as trace:
        run_workload()
    # trace.events populated; JSONL written to out_path
Two collection modes:
* **in-process** (default) — the compiled CUPTI shim collects kernels launched by
  this interpreter. Wired up behind ``_backend()``.
* **injected** — when ``CUDA_INJECTION64_PATH`` points at ``libgitm_inject.so``, the
  CUDA driver has already loaded that collector into *every* CUDA process,
  including children we don't control. We then take no part in collection: we arm
  the window, wait for in-flight buffers, and merge the per-pid shards. This is the
  only mode that sees vLLM's kernels, which run in a child ``EngineCore`` process.
  See :mod:`gitm.tracer.injection`.
The two must never both collect. CUPTI allows one activity-callback registration per
process, so calling ``backend.start()`` while the injected library holds the
registration would silently clobber it.
When no backend is available (dev box without GPU), capture is a no-op that still
writes a well-formed empty trace — useful for plumbing tests.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gitm.tracer import injection
from gitm.tracer.schema import Trace


@contextmanager
def capture(
    out_path: str | Path,
    *,
    workload_id: str = "unknown",
    fingerprint: str = "unknown",
    run_id: str | None = None,
) -> Iterator[Trace]:
    """Capture a CUPTI/rocprof trace into ``out_path`` as JSONL.
    The yielded ``Trace`` is updated in-place as events arrive; the file is
    flushed on context exit. Capture overhead target: <10% of workload runtime today,
    tightening to <5% on the roadmap.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    injected = injection.active()
    backend = None if injected else _backend()
    started_ns = time.time_ns()

    trace = Trace(
        workload_id=workload_id,
        fingerprint=fingerprint,
        run_id=run_id or uuid.uuid4().hex,
        device_count=_device_count(backend, injected),
        vendor="nvidia" if injected else (backend.vendor if backend else "none"),
        captured_at_ns=started_ns,
        duration_ns=0,
    )

    # Bounds of the collection window, in the CUPTI clock domain — NOT wall-clock.
    # Only meaningful under injection, where the collector runs for the whole
    # lifetime of every CUDA process and we have to carve our window back out.
    window_start: int | None = None

    if injected:
        # Dead processes' shards only — a live EngineCore is already holding its shard
        # open (it opened it at CUDA init, during the engine build, before we got
        # here), and unlinking it would silently redirect every kernel record into a
        # deleted inode.
        injection.clear_stale_shards()
        window_start = injection.cupti_now()
        if window_start is None:
            warnings.warn(
                "injected capture cannot read the CUPTI clock, so the window can't "
                "be bounded: the trace will include everything the traced processes "
                "did, including model load and CUDA-graph capture. Build the shim "
                "(python -m gitm.tracer._cupti.build) to fix.",
                stacklevel=2,
            )
        injection.arm()
    elif backend is not None:
        # Enabling collection can fail at runtime (e.g. CUPTI returns
        # NOT_COMPATIBLE on a driver/CUPTI version skew). Degrade to a well-formed
        # no-op trace rather than taking down the whole run — the tracer is
        # best-effort instrumentation, not critical to the workload.
        try:
            backend.start()
        except Exception as exc:
            warnings.warn(f"trace capture disabled: backend.start() failed: {exc}", stacklevel=2)
            backend = None
            trace.vendor = "none"
            trace.device_count = 0
    try:
        yield trace
    finally:
        ended_ns = time.time_ns()
        if injected:
            window_end = injection.cupti_now()
            # Stay armed while other processes' in-flight CUPTI buffers land — we
            # can't reach into a child to force a flush, so we wait out its flush
            # period. Disarming first would drop the tail of the trace.
            try:
                injection.settle()
                injection.disarm()
                trace.events = injection.read_shards(window_start, window_end)
            except Exception as exc:
                warnings.warn(f"trace capture incomplete: shard merge failed: {exc}", stacklevel=2)
        elif backend is not None:
            try:
                trace.events = backend.stop()
            except Exception as exc:
                warnings.warn(f"trace capture incomplete: backend.stop() failed: {exc}", stacklevel=2)
        trace.duration_ns = ended_ns - started_ns
        _write_jsonl(out_path, trace)
def write_trace_jsonl(path: str | Path, trace: Trace) -> None:
    """Write a trace to JSONL — header line, then one event per line.
    The canonical on-disk trace format, shared by ``capture()`` and the
    deviation-only trace writer so the two never drift. Creates the parent dir.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        header = trace.model_dump(exclude={"events"})
        fh.write(json.dumps({"_header": header}))
        fh.write("\n")
        for ev in trace.events:
            fh.write(ev.model_dump_json())
            fh.write("\n")
# Back-compat internal alias.
_write_jsonl = write_trace_jsonl


def _device_count(backend, injected: bool) -> int:
    """Device count for the trace header.
    Under injection we never construct a backend — collection belongs to the driver-
    loaded library — but the compiled shim is still importable and counting devices
    doesn't touch CUPTI's activity callbacks, so it stays safe to ask.
    """
    if injected:
        from gitm.tracer._cupti import load_shim

        shim = load_shim()
        if shim is None:
            return 0
        try:
            return int(shim.device_count())
        except Exception:
            return 0
    return backend.device_count() if backend else 0


def _backend():
    """Return the active CUPTI/rocprof backend, or ``None`` if unavailable.
    Tries the CUPTI backend (real, via the compiled shim — see
    :mod:`gitm.tracer.cupti`). When the shim isn't built (CPU-only host, or a
    GPU box where ``python -m gitm.tracer._cupti.build`` hasn't run),
    construction raises and we return ``None`` so capture is a well-formed
    no-op and the rest of the pipeline runs without a GPU.
    """
    try:
        from gitm.tracer.cupti import CuptiBackend  # noqa: F401
    except Exception:
        return None
    try:
        return CuptiBackend()
    except Exception:
        return None
