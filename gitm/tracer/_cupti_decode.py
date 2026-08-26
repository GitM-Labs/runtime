"""Decode CUPTI activity records into GITM trace events.

The native shim (``gitm/tracer/_cupti/cupti_shim.c``) does the unsafe work —
buffer management, walking records with ``cuptiActivityGetNextRecord``, reading
fields off the real CUPTI structs (layout resolved by the compiler against the
installed ``cupti_activity.h``, never hand-guessed). It hands Python a flat list
of plain dicts. This module turns those dicts into validated
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

import re
import warnings
from typing import Literal

from gitm.distributed.correlate import correlate_kernels_to_ranges
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


def _kernel_dims(value: object, *, what: str) -> tuple[int, int, int]:
    """Return a 3-tuple of launch dimensions, warning on missing or malformed
    input. A truncated CUPTI record (e.g. ``[256, 1]``) must degrade to a named
    fallback rather than raising IndexError deep in construction."""
    if value is None:
        warnings.warn(
            f"CUPTI kernel {what} dimensions unavailable; using 1x1x1",
            RuntimeWarning,
            stacklevel=3,
        )
        return (1, 1, 1)
    try:
        x, y, z = (int(value[0]), int(value[1]), int(value[2]))  # type: ignore[index]
    except (TypeError, IndexError, ValueError):
        warnings.warn(
            f"CUPTI kernel {what} dimensions malformed ({value!r}); using 1x1x1",
            RuntimeWarning,
            stacklevel=3,
        )
        return (1, 1, 1)
    return (x, y, z)


def decode_kernel(d: dict) -> KernelEvent:
    grid = _kernel_dims(d.get("grid"), what="grid")
    block = _kernel_dims(d.get("block"), what="block")
    name = d.get("name")
    if not name:
        warnings.warn(
            "CUPTI kernel name unavailable; using <anonymous>",
            RuntimeWarning,
            stacklevel=2,
        )
    return KernelEvent(
        start_ns=int(d["start_ns"]),
        end_ns=int(d["end_ns"]),
        stream_id=int(d["stream_id"]),
        device_id=int(d["device_id"]),
        correlation_id=_opt_int(d.get("correlation_id")),
        name=name or "<anonymous>",
        grid_x=grid[0], grid_y=grid[1], grid_z=grid[2],
        block_x=block[0], block_y=block[1], block_z=block[2],
        shared_mem_bytes=int(d.get("static_shared_mem", 0)) + int(d.get("dynamic_shared_mem", 0)),
        registers_per_thread=int(d.get("registers_per_thread", 0)),
        range_op=d.get("range_op"),
        range_layer=_opt_int(d.get("range_layer")),
    )


def decode_memcpy(d: dict) -> MemcpyEvent:
    raw_kind = d.get("copy_kind")
    kind = int(raw_kind) if raw_kind is not None else 0
    if kind not in _COPY_KIND or kind == 0:
        warnings.warn(
            f"unknown CUPTI memcpy kind {kind}; using device↔device endpoint fallback",
            RuntimeWarning,
            stacklevel=2,
        )
    src, dst = _COPY_KIND.get(kind, ("device", "device"))
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
    raw_type = d.get("sync_type")
    sync_type = int(raw_type) if raw_type is not None else 0
    if sync_type not in _SYNC_KIND or sync_type == 0:
        warnings.warn(
            f"unknown CUPTI synchronization type {sync_type}; using device-sync fallback",
            RuntimeWarning,
            stacklevel=2,
        )
    return SyncEvent(
        start_ns=int(d["start_ns"]),
        end_ns=int(d["end_ns"]),
        stream_id=int(d.get("stream_id", 0)),
        device_id=int(d.get("device_id", 0)),
        correlation_id=_opt_int(d.get("correlation_id")),
        sync_kind=_SYNC_KIND.get(sync_type, "device"),
    )


_DECODERS = {"kernel": decode_kernel, "memcpy": decode_memcpy, "sync": decode_sync}


def decode_record(d: dict) -> TraceEvent | None:
    """Decode one record dict, or ``None`` for kinds GITM doesn't model."""
    fn = _DECODERS.get(d.get("kind"))
    return fn(d) if fn else None


#: ``marker_flags`` values, mirroring GITM_MARKER_START/END in cupti_core.h.
MARKER_START, MARKER_END = 0, 1

#: vLLM's ``--enable-layerwise-nvtx-tracing`` names each range with the repr of a
_VLLM_MODULE_RE = re.compile(r"'Module':\s*'([^']*)'")


def normalize_range_name(name: str) -> str:
    """Rewrite a collector range name into the ``L{layer}/{op}`` form.

    Passes through anything already in that form, so ranges pushed by
    :func:`gitm.tracer.vllm_stats.instrument_model`. vLLM's own
    instrumentation is translated through the same module-path table those hooks
    use, which is what lets either source feed one correlation path.
    """
    if not name or not name.startswith("{"):
        return name
    m = _VLLM_MODULE_RE.search(name)
    if not m:
        return name

    from gitm.tracer.vllm_stats import op_for_module, range_name

    path = m.group(1)
    mapped = op_for_module(path)
    if mapped is not None:
        return range_name(*mapped)

    leaf = path.rsplit(".", 1)[-1] or path
    layer = _LAYER_IN_PATH.search(path)
    if layer is None:
        return leaf
    # The block itself: its leaf is the index, so `layers.1` would read `L1/1`.
    # Name it for what it is — the range enclosing everything in that block.
    if leaf == layer.group(1):
        leaf = "layer"
    return f"L{layer.group(1)}/{leaf}"


_LAYER_IN_PATH = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


def pair_markers(records: list[dict]) -> list[dict]:
    """Join NVTX push/pop halves into whole ranges.

    ``CUpti_ActivityMarker2`` carries a single ``timestamp``, so a range arrives
    as two records sharing a ``marker_id`` — and the header is explicit that the
    name "will be NULL for an end marker". The C side stays stateless and emits
    both halves; this is where they become the
    ``{kind, name, start_ns, end_ns, thread_id}`` range that
    :mod:`gitm.distributed.correlate` documents as its input.

    Non-marker records pass through untouched and in order. Unpaired halves are
    dropped rather than repaired: a start without an end has no window, and
    inventing one — closing it at the capture's end, say — would produce a range
    that appears to contain every launch after it and would silently claim them
    all. A capture killed mid-step leaves exactly that, so this is the common
    case, not a corner one.
    """
    starts: dict[tuple, dict] = {}
    out: list[dict] = []
    for r in records:
        if r.get("kind") != "marker":
            out.append(r)
            continue
        # marker_id is unique per process, but pair within a thread anyway: an
        # id reused across threads would otherwise splice two ranges into one
        # spanning window.
        key = (r.get("thread_id"), r.get("marker_id"))
        if r.get("marker_flags") == MARKER_END:
            start = starts.pop(key, None)
            if start is None:
                continue  # end without a start: nothing to bound
            out.append({
                "kind": "marker",
                "name": normalize_range_name(start.get("name") or ""),
                "start_ns": int(start["timestamp_ns"]),
                "end_ns": int(r["timestamp_ns"]),
                "thread_id": r.get("thread_id"),
            })
        else:
            starts[key] = r
    return out


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
    correlated = correlate_kernels_to_ranges(pair_markers(records))
    enriched_kernels = iter(correlated)
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
