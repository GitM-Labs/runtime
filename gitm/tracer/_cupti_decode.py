"""Decode CUPTI activity records into GITM trace events.

The native shim (``gitm/tracer/_cupti/cupti_shim.c``) does the unsafe work —
buffer management, walking records with ``cuptiActivityGetNextRecord``, reading
fields off the real CUPTI structs (layout resolved by the compiler against the
installed ``cupti_activity.h``, never hand-guessed). It hands Python a flat list
of plain dicts. *This* module turns those dicts into validated
:class:`~gitm.tracer.schema.KernelEvent` / ``MemcpyEvent`` / ``SyncEvent``.

Keeping the boundary at dicts means all the interpretation logic — enum
mappings, field folding, schema validation — is pure Python and fully unit
tested without a GPU. The C side only copies primitives.

Dict contract (the shim emits exactly these shapes):

    kernel  {kind:"kernel", name, start_ns, end_ns, device_id, context_id,
             stream_id, correlation_id, grid:[x,y,z], block:[x,y,z],
             static_shared_mem, dynamic_shared_mem, registers_per_thread}
    memcpy  {kind:"memcpy", copy_kind:int, bytes, start_ns, end_ns, device_id,
             context_id, stream_id, correlation_id}
    sync    {kind:"sync", sync_type:int, start_ns, end_ns, device_id,
             context_id, stream_id, correlation_id}
"""

from __future__ import annotations

from typing import Literal

from gitm.tracer.nvtx_correlate import correlate_kernels_to_ranges
from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, TraceEvent

Endpoint = Literal["host", "device", "unified"]

# CUpti_ActivityMemcpyKind -> (src, dst). Arrays live in device memory; managed
# transfers are reported by the *_managed copy kinds, mapped to "unified".
# Values from cupti_activity.h (ABI-stable across CUPTI versions).
_COPY_KIND: dict[int, tuple[Endpoint, Endpoint]] = {
    0: ("device", "device"),   # UNKNOWN — safe default
    1: ("host", "device"),     # HTOD
    2: ("device", "host"),     # DTOH
    3: ("host", "device"),     # HTOA  (array == device memory)
    4: ("device", "host"),     # ATOH
    5: ("device", "device"),   # ATOA
    6: ("device", "device"),   # ATOD
    7: ("device", "device"),   # DTOA
    8: ("device", "device"),   # DTOD
    9: ("host", "host"),       # HTOH
    10: ("device", "device"),  # PTOP (peer device-to-device)
}

# CUpti_ActivitySynchronizationType -> schema sync_kind.
_SYNC_KIND: dict[int, Literal["stream", "event", "device"]] = {
    0: "device",   # UNKNOWN — safe default
    1: "event",    # EVENT_SYNCHRONIZE
    2: "stream",   # STREAM_WAIT_EVENT
    3: "stream",   # STREAM_SYNCHRONIZE
    4: "device",   # CONTEXT_SYNCHRONIZE
}


def decode_kernel(d: dict) -> KernelEvent:
    grid = d.get("grid", [1, 1, 1])
    block = d.get("block", [1, 1, 1])
    return KernelEvent(
        start_ns=int(d["start_ns"]),
        end_ns=int(d["end_ns"]),
        stream_id=int(d["stream_id"]),
        device_id=int(d["device_id"]),
        correlation_id=_opt_int(d.get("correlation_id")),
        name=d.get("name") or "<anonymous>",
        grid_x=int(grid[0]), grid_y=int(grid[1]), grid_z=int(grid[2]),
        block_x=int(block[0]), block_y=int(block[1]), block_z=int(block[2]),
        shared_mem_bytes=int(d.get("static_shared_mem", 0)) + int(d.get("dynamic_shared_mem", 0)),
        registers_per_thread=int(d.get("registers_per_thread", 0)),
        range_op=d.get("range_op"),
        range_layer=_opt_int(d.get("range_layer")),
    )


def decode_memcpy(d: dict) -> MemcpyEvent:
    src, dst = _COPY_KIND.get(int(d.get("copy_kind", 0)), ("device", "device"))
    return MemcpyEvent(
        start_ns=int(d["start_ns"]),
        end_ns=int(d["end_ns"]),
        stream_id=int(d["stream_id"]),
        device_id=int(d["device_id"]),
        correlation_id=_opt_int(d.get("correlation_id")),
        bytes=int(d["bytes"]),
        src=src,
        dst=dst,
    )


def decode_sync(d: dict) -> SyncEvent:
    return SyncEvent(
        start_ns=int(d["start_ns"]),
        end_ns=int(d["end_ns"]),
        stream_id=int(d.get("stream_id", 0)),
        device_id=int(d.get("device_id", 0)),
        correlation_id=_opt_int(d.get("correlation_id")),
        sync_kind=_SYNC_KIND.get(int(d.get("sync_type", 0)), "device"),
    )


_DECODERS = {"kernel": decode_kernel, "memcpy": decode_memcpy, "sync": decode_sync}


def decode_record(d: dict) -> TraceEvent | None:
    """Decode one record dict, or ``None`` for kinds GITM doesn't model."""
    fn = _DECODERS.get(d.get("kind"))
    return fn(d) if fn else None


def decode_records(records: list[dict]) -> list[TraceEvent]:
    """Decode a shim record batch, dropping unmodeled kinds, sorted by start.

    Kernel dicts are first run through :func:`correlate_kernels_to_ranges` so
    ``range_op``/``range_layer`` are populated whenever the capture carries
    NVTX range instrumentation (``runtime``/``marker`` records) — a no-op
    today since the shim doesn't emit those kinds yet, and harmless on any
    trace that doesn't have them (``decode_kernel`` just sees ``None``).
    ``runtime``/``marker`` records themselves are consumed by correlation and
    never reach ``_DECODERS`` — GITM doesn't model them as events.

    Sorting by ``start_ns`` gives a stable timeline regardless of the order
    CUPTI flushed buffers (concurrent kernels on multiple streams interleave).
    """
    enriched_kernels = iter(correlate_kernels_to_ranges(records))
    events: list[TraceEvent] = []
    for d in records:
        if d.get("kind") == "kernel":
            events.append(decode_kernel(next(enriched_kernels)))
        elif (ev := decode_record(d)) is not None:
            events.append(ev)
    events.sort(key=lambda e: e.start_ns)
    return events


def _opt_int(v) -> int | None:
    return None if v is None else int(v)
