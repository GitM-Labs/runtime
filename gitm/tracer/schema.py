"""Event-telemetry schema — per-kernel records.

Distinct from ``gitm.telemetry`` (which is summary state at 1 Hz). This is the
trace structure: every kernel launch, every memcpy, every sync, with
nanosecond timestamps and stream IDs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _TraceEventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    start_ns: int
    end_ns: int
    stream_id: int
    device_id: int
    correlation_id: int | None = None


class KernelEvent(_TraceEventBase):
    kind: Literal["kernel"] = "kernel"
    name: str
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_mem_bytes: int = 0
    registers_per_thread: int = 0
    bytes_read: int | None = None  # filled by attribution layer when available
    bytes_written: int | None = None
    # Exact op/layer identity recovered from an enclosing NVTX range (see
    # gitm.tracer.nvtx_correlate), when the capture had range instrumentation.
    # None means no range was found — callers fall back to name-based
    # classify_op(). Never guessed or backfilled; only set by correlation.
    range_op: str | None = None
    range_layer: int | None = None


class MemcpyEvent(_TraceEventBase):
    kind: Literal["memcpy"] = "memcpy"
    bytes: int
    src: Literal["host", "device", "unified"]
    dst: Literal["host", "device", "unified"]


class SyncEvent(_TraceEventBase):
    kind: Literal["sync"] = "sync"
    sync_kind: Literal["stream", "event", "device"]


TraceEvent = KernelEvent | MemcpyEvent | SyncEvent


class Trace(BaseModel):
    """A captured event-telemetry trace.

    Workload fingerprint and labels travel with the trace so downstream
    components don't need to thread context separately.
    """

    model_config = ConfigDict(extra="forbid")

    workload_id: str
    fingerprint: str
    run_id: str
    device_count: int
    vendor: str  # "nvidia" | "amd"
    captured_at_ns: int
    duration_ns: int
    events: list[TraceEvent] = Field(default_factory=list)
    # Provenance of the event plane. Default "cupti" preserves every existing
    # capture/test construction; importers set nsys-import / torch-import.
    source: Literal["cupti", "rocprof", "nsys-import", "torch-import", "none"] = "cupti"

    def kernels(self) -> list[KernelEvent]:
        # Duck-type on ``kind`` so importers can store slotted lightweight
        # events at multi-million scale without full pydantic instances.
        return [e for e in self.events if getattr(e, "kind", None) == "kernel"]  # type: ignore[return-value]

    def by_stream(self) -> dict[int, list[TraceEvent]]:
        out: dict[int, list[TraceEvent]] = {}
        for e in self.events:
            out.setdefault(e.stream_id, []).append(e)
        return out
