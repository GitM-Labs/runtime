"""Trace capture context manager.

    with gitm.tracer.capture(out_path) as trace:
        run_workload()
    # trace.events populated; JSONL written to out_path

CUPTI / rocprof backends are wired up behind ``_backend()``. When neither is
available (dev box without GPU), capture is a no-op that still writes a
well-formed empty trace — useful for plumbing tests.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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

    backend = _backend()
    started_ns = time.time_ns()

    trace = Trace(
        workload_id=workload_id,
        fingerprint=fingerprint,
        run_id=run_id or uuid.uuid4().hex,
        device_count=backend.device_count() if backend else 0,
        vendor=backend.vendor if backend else "none",
        captured_at_ns=started_ns,
        duration_ns=0,
    )

    # Enabling collection can fail at runtime (e.g. CUPTI returns
    # NOT_COMPATIBLE on a driver/CUPTI version skew). Degrade to a well-formed
    # no-op trace rather than taking down the whole run — the tracer is
    # best-effort instrumentation, not critical to the workload.
    if backend is not None:
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
        if backend is not None:
            try:
                trace.events = backend.stop()
            except Exception as exc:
                warnings.warn(f"trace capture incomplete: backend.stop() failed: {exc}", stacklevel=2)
        trace.duration_ns = ended_ns - started_ns
        _write_jsonl(out_path, trace)


def _write_jsonl(path: Path, trace: Trace) -> None:
    """Stream trace events to JSONL — header line, then one event per line."""
    with path.open("w", encoding="utf-8") as fh:
        header = trace.model_dump(exclude={"events"})
        fh.write(json.dumps({"_header": header}))
        fh.write("\n")
        for ev in trace.events:
            fh.write(ev.model_dump_json())
            fh.write("\n")


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
