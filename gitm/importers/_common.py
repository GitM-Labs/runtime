"""Shared helpers for profiler importers."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace, TraceEvent

# Above this event count, skip O(n) exact-row dedupe (dominant cost on 5M-event
# imports; identical-row rate on real profiler dumps is near zero).
_DEDUPE_MAX_EVENTS = 100_000


@dataclass
class ImportStats:
    """Bookkeeping for one import (warnings, drops, device counts)."""

    source_path: str
    format: str
    sku: str | None = None
    captured_at_source: str = "mtime"  # "metadata" | "mtime"
    deduped: int = 0
    dropped_invalid: int = 0
    total_raw_events: int = 0
    per_device_kernel_counts: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    device_name: str | None = None


class ImportError(Exception):
    """Per-file import failure with a customer-readable message."""


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp-file + rename; never leave a partial artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def event_key(ev: Any) -> tuple:
    """Exact-identity key for dedupe (identical rows collapse)."""
    kind = getattr(ev, "kind", None)
    base = (kind, ev.start_ns, ev.end_ns, ev.stream_id, ev.device_id, ev.correlation_id)
    if kind == "kernel":
        return base + (
            ev.name,
            ev.grid_x,
            ev.grid_y,
            ev.grid_z,
            ev.block_x,
            ev.block_y,
            ev.block_z,
            ev.shared_mem_bytes,
            ev.registers_per_thread,
        )
    if kind == "memcpy":
        return base + (ev.bytes, ev.src, ev.dst)
    if kind == "sync":
        return base + (ev.sync_kind,)
    return base


def normalize_and_clean(
    events: list[TraceEvent],
    *,
    strict: bool = False,
) -> tuple[list[TraceEvent], int, int]:
    """Dedupe identical rows, drop end<start, re-base timestamps so min start = 0.

    Returns ``(cleaned_events, n_deduped, n_dropped_invalid)``.
    Large imports skip exact-row dedupe unless ``strict`` (see ``_DEDUPE_MAX_EVENTS``).
    """
    if not events:
        return [], 0, 0

    deduped = 0
    if strict or len(events) <= _DEDUPE_MAX_EVENTS:
        seen: set[tuple] = set()
        unique: list[TraceEvent] = []
        for ev in events:
            k = event_key(ev)
            if k in seen:
                deduped += 1
                continue
            seen.add(k)
            unique.append(ev)
        if deduped:
            warnings.warn(
                f"deduped {deduped} exactly-identical event row(s)",
                stacklevel=2,
            )
    else:
        unique = events

    valid: list[TraceEvent] = []
    dropped = 0
    for ev in unique:
        if ev.end_ns < ev.start_ns:
            dropped += 1
            continue
        valid.append(ev)

    total = max(len(unique), 1)
    drop_frac = dropped / total
    if dropped:
        msg = f"dropped {dropped} event(s) with end < start ({drop_frac:.1%} of unique)"
        if strict and drop_frac > 0.01:
            raise ImportError(msg + " — fatal under --strict (>1% dropped)")
        if drop_frac > 0.01:
            warnings.warn(msg + " — >1% dropped", stacklevel=2)
        else:
            warnings.warn(msg, stacklevel=2)

    if not valid:
        return [], deduped, dropped

    t0 = min(ev.start_ns for ev in valid)
    if t0 == 0:
        return valid, deduped, dropped

    # In-place re-base — avoid pydantic model_copy (doubles peak RSS).
    cleaned: list[TraceEvent] = []
    for ev in valid:
        start = ev.start_ns - t0
        end = ev.end_ns - t0
        if start < 0 or end < 0:
            dropped += 1
            continue
        try:
            object.__setattr__(ev, "start_ns", start)
            object.__setattr__(ev, "end_ns", end)
            cleaned.append(ev)
        except (AttributeError, TypeError):
            cleaned.append(ev.model_copy(update={"start_ns": start, "end_ns": end}))

    return cleaned, deduped, dropped


def device_count_from_events(events: Iterable[TraceEvent], meta_count: int | None = None) -> int:
    if meta_count is not None and meta_count > 0:
        return meta_count
    max_id = -1
    for ev in events:
        if ev.device_id > max_id:
            max_id = ev.device_id
    return 1 + max_id if max_id >= 0 else 1


def filter_device(events: list[TraceEvent], device: int) -> list[TraceEvent]:
    return [e for e in events if e.device_id == device]


def per_device_kernel_counts(events: Iterable[Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for ev in events:
        if getattr(ev, "kind", None) == "kernel":
            counts[ev.device_id] = counts.get(ev.device_id, 0) + 1
    return counts


@dataclass(slots=True)
class LightKernel:
    """Slotted kernel event for multi-million imports (duck-types as KernelEvent)."""

    name: str
    start_ns: int
    end_ns: int
    stream_id: int = 0
    device_id: int = 0
    correlation_id: int | None = None
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_mem_bytes: int = 0
    registers_per_thread: int = 0
    bytes_read: int | None = None
    bytes_written: int | None = None
    kind: str = "kernel"


@dataclass(slots=True)
class LightMemcpy:
    start_ns: int
    end_ns: int
    stream_id: int
    device_id: int
    bytes: int
    src: str
    dst: str
    correlation_id: int | None = None
    kind: str = "memcpy"


@dataclass(slots=True)
class LightSync:
    start_ns: int
    end_ns: int
    stream_id: int
    device_id: int
    sync_kind: str
    correlation_id: int | None = None
    kind: str = "sync"


def make_kernel_fast(
    *,
    name: str,
    start_ns: int,
    end_ns: int,
    stream_id: int = 0,
    device_id: int = 0,
    correlation_id: int | None = None,
    grid_x: int = 1,
    grid_y: int = 1,
    grid_z: int = 1,
    block_x: int = 1,
    block_y: int = 1,
    block_z: int = 1,
    shared_mem_bytes: int = 0,
    registers_per_thread: int = 0,
    strict: bool = False,
) -> KernelEvent | LightKernel:
    """Build a kernel event; slotted light form unless ``strict``."""
    if strict:
        return KernelEvent(
            kind="kernel",
            name=name,
            start_ns=start_ns,
            end_ns=end_ns,
            stream_id=stream_id,
            device_id=device_id,
            correlation_id=correlation_id,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
            block_x=block_x,
            block_y=block_y,
            block_z=block_z,
            shared_mem_bytes=shared_mem_bytes,
            registers_per_thread=registers_per_thread,
            bytes_read=None,
            bytes_written=None,
        )
    return LightKernel(
        name=sys.intern(name),
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        correlation_id=correlation_id,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        block_x=block_x,
        block_y=block_y,
        block_z=block_z,
        shared_mem_bytes=shared_mem_bytes,
        registers_per_thread=registers_per_thread,
    )


def make_memcpy_fast(
    *,
    start_ns: int,
    end_ns: int,
    stream_id: int,
    device_id: int,
    correlation_id: int | None,
    nbytes: int,
    src: str,
    dst: str,
    strict: bool = False,
) -> MemcpyEvent | LightMemcpy:
    if strict:
        return MemcpyEvent(
            kind="memcpy",
            start_ns=start_ns,
            end_ns=end_ns,
            stream_id=stream_id,
            device_id=device_id,
            correlation_id=correlation_id,
            bytes=nbytes,
            src=src,  # type: ignore[arg-type]
            dst=dst,  # type: ignore[arg-type]
        )
    return LightMemcpy(
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        correlation_id=correlation_id,
        bytes=nbytes,
        src=src,
        dst=dst,
    )


def make_sync_fast(
    *,
    start_ns: int,
    end_ns: int,
    stream_id: int,
    device_id: int,
    correlation_id: int | None,
    sync_kind: str,
    strict: bool = False,
) -> SyncEvent | LightSync:
    if strict:
        return SyncEvent(
            kind="sync",
            start_ns=start_ns,
            end_ns=end_ns,
            stream_id=stream_id,
            device_id=device_id,
            correlation_id=correlation_id,
            sync_kind=sync_kind,  # type: ignore[arg-type]
        )
    return LightSync(
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        correlation_id=correlation_id,
        sync_kind=sync_kind,
    )


def finish_trace(
    *,
    events: list[TraceEvent],
    workload_id: str,
    source: str,
    vendor: str = "nvidia",
    device_count: int,
    captured_at_ns: int,
    run_id: str | None = None,
    strict: bool = False,
) -> tuple[Trace, ImportStats]:
    """Normalize events, fingerprint, and build a Trace.

    Full pydantic round-trip validation runs only when ``strict=True`` (customer
    dumps of millions of events cannot afford ``model_dump`` of the whole list).
    """
    import uuid

    from gitm.optimizer.qualification import fingerprint

    cleaned, deduped, dropped = normalize_and_clean(events, strict=strict)
    duration = 0
    if cleaned:
        duration = max(e.end_ns for e in cleaned) - min(e.start_ns for e in cleaned)

    rid = run_id or f"import-{uuid.uuid4().hex}"
    if strict:
        trace = Trace(
            workload_id=workload_id,
            fingerprint="pending",
            run_id=rid,
            device_count=device_count,
            vendor=vendor,
            captured_at_ns=captured_at_ns,
            duration_ns=duration,
            events=cleaned,
            source=source,  # type: ignore[arg-type]
        )
        fp = fingerprint(trace)
        trace = trace.model_copy(update={"fingerprint": fp})
        Trace.model_validate(trace.model_dump())
    else:
        # Placeholder fingerprint, then setattr — never model_dump the event list.
        trace = Trace.model_construct(
            workload_id=workload_id,
            fingerprint="pending",
            run_id=rid,
            device_count=device_count,
            vendor=vendor,
            captured_at_ns=captured_at_ns,
            duration_ns=duration,
            events=cleaned,
            source=source,
        )
        fp = fingerprint(trace)
        object.__setattr__(trace, "fingerprint", fp)
        # Optional sample validation only for pydantic events (light events skip).
        for sample in cleaned[:3]:
            if hasattr(sample, "model_dump") and hasattr(type(sample), "model_validate"):
                type(sample).model_validate(sample.model_dump())

    stats = ImportStats(
        source_path="",
        format=source,
        deduped=deduped,
        dropped_invalid=dropped,
        total_raw_events=len(events),
        per_device_kernel_counts=per_device_kernel_counts(cleaned),
    )
    return trace, stats


def file_mtime_ns(path: Path) -> int:
    st = path.stat()
    return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))


def as_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
