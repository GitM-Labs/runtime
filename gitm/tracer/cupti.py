"""CUPTI backend — per-kernel activity ingestion.

Wraps the native CUPTI shim (:mod:`gitm.tracer._cupti`) behind the tracer
backend interface that :func:`gitm.tracer.capture.capture` expects. On
``start()`` CUPTI activity collection is enabled; on ``stop()`` the buffers are
flushed and the raw record dicts are decoded into validated
:class:`~gitm.tracer.schema.TraceEvent` objects by :mod:`gitm.tracer._cupti_decode`.

Construction raises ``RuntimeError`` if the compiled shim is absent (CPU-only
host, or a GPU box where ``python -m gitm.tracer._cupti.build`` hasn't run), so
``capture`` falls back to a well-formed no-op trace rather than failing. The
heavy lifting and all unsafe struct handling live in the C shim; everything in
this module is ordinary Python.
"""

from __future__ import annotations

from gitm.tracer._cupti import load_shim
from gitm.tracer._cupti_decode import decode_records
from gitm.tracer.schema import TraceEvent


class CuptiBackend:
    vendor = "nvidia"

    def __init__(self) -> None:
        self._shim = load_shim()
        if self._shim is None:
            raise RuntimeError(
                "CUPTI shim not built. On a GPU box run "
                "`python -m gitm.tracer._cupti.build` (needs the CUDA toolkit + "
                "CUPTI). On CPU-only hosts the tracer degrades to a no-op."
            )

    def device_count(self) -> int:
        return int(self._shim.device_count())

    def start(self) -> None:
        self._shim.start()

    def stop(self) -> list[TraceEvent]:
        records = self._shim.stop()
        # After stop(), whose flush delivers the last buffers. A shim built before
        # it counted, or a CUPTI that would not say, leaves the count unknown.
        dropped = int(self._shim.dropped()) if hasattr(self._shim, "dropped") else -1
        self.dropped_records = dropped if dropped >= 0 else None
        return decode_records(records)
