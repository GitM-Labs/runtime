"""Nsight Systems importer (``.nsys-rep`` and pre-exported ``.sqlite``).

Supported export schemas: nsys 2024.x and 2025.x. Other versions hard-error.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gitm.importers._common import (
    ImportError,
    ImportStats,
    as_int,
    device_count_from_events,
    file_mtime_ns,
    filter_device,
    finish_trace,
    per_device_kernel_counts,
)
from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace, TraceEvent

# ---------------------------------------------------------------------------
# CUPTI enum maps — integers differ across CUPTI versions; one table per kind.
# Values annotated with the CUPTI header name from cupti_activity.h.
# ---------------------------------------------------------------------------

# CUpti_ActivityMemoryKind
_MEMORY_KIND: dict[int, str] = {
    0: "device",  # CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN — default host-side-unknown → device
    1: "host",  # CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE
    2: "host",  # CUPTI_ACTIVITY_MEMORY_KIND_PINNED
    3: "device",  # CUPTI_ACTIVITY_MEMORY_KIND_DEVICE
    4: "device",  # CUPTI_ACTIVITY_MEMORY_KIND_ARRAY
    5: "unified",  # CUPTI_ACTIVITY_MEMORY_KIND_MANAGED
    6: "device",  # CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC
    7: "unified",  # CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC
}

# CUpti_ActivityMemcpyKind (used when srcKind/dstKind absent; copyKind alone)
_COPY_KIND_TO_ENDPOINTS: dict[int, tuple[str, str]] = {
    # CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN
    0: ("device", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_HTOD
    1: ("host", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_DTOH
    2: ("device", "host"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_HTOA
    3: ("host", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_ATOH
    4: ("device", "host"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_ATOA
    5: ("device", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_ATOD
    6: ("device", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_DTOA
    7: ("device", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_DTOD
    8: ("device", "device"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_HTOH
    9: ("host", "host"),
    # CUPTI_ACTIVITY_MEMCPY_KIND_PTOP (peer)
    10: ("device", "device"),
}

# CUpti_ActivitySynchronizationType
_SYNC_TYPE: dict[int, str] = {
    0: "stream",  # CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN
    1: "event",  # CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE
    2: "stream",  # CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT
    3: "stream",  # CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE
    4: "device",  # CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE
}

_SUPPORTED_MAJOR = frozenset({2024, 2025})

_KERNEL_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
)


def _export_nsys_rep(path: Path) -> Path:
    """Shell out to ``nsys export --type sqlite``. Never bundle nsys."""
    nsys = shutil.which("nsys")
    if nsys is None:
        raise ImportError(
            f"nsys not on PATH; cannot convert {path.name}. "
            "Ask the customer to export first:\n"
            f"  nsys export --type sqlite --output {path.stem}.sqlite {path.name}"
        )
    out = Path(tempfile.mkdtemp(prefix="gitm-nsys-")) / f"{path.stem}.sqlite"
    # argv list only — never a shell string (path may contain spaces/metachars).
    cmd = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "--output",
        str(out),
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
    except OSError as exc:
        raise ImportError(f"failed to run nsys export: {exc}") from exc
    if proc.returncode != 0 or not out.exists():
        raise ImportError(
            f"nsys export failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}\n"
            f"Manual export: nsys export --type sqlite --output {path.stem}.sqlite {path.name}"
        )
    return out


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def _detect_version(conn: sqlite3.Connection) -> str:
    """Return version string like '2024.5'; hard-error on unsupported major."""
    names = _table_names(conn)
    # Probe both spellings of the metadata table.
    for table in ("MetaData", "METADATA", "metadata"):
        if table not in names:
            continue
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        except sqlite3.Error:
            continue
        # Common shapes: (name, value) or (key, value) or (Name, Value)
        key_col = next((c for c in cols if c.lower() in ("name", "key", "id")), None)
        val_col = next((c for c in cols if c.lower() in ("value", "val")), None)
        if not key_col or not val_col:
            continue
        rows = conn.execute(f"SELECT {key_col}, {val_col} FROM {table}").fetchall()
        kv = {str(k).lower(): str(v) for k, v in rows if k is not None}
        for key in (
            "exportversion",
            "export_version",
            "nsysversion",
            "nsys_version",
            "version",
            "contentschemaversion",
            "contentschema_version",
        ):
            if key in kv:
                return _parse_version(kv[key])
        # Some exports store "NVIDIA Nsight Systems version 2024.5.1"
        for v in kv.values():
            if "2024" in v or "2025" in v or "2023" in v:
                return _parse_version(v)
    # Fixture / minimal DBs may stamp a dedicated table.
    if "GITM_NSYS_META" in names:
        row = conn.execute("SELECT version FROM GITM_NSYS_META LIMIT 1").fetchone()
        if row:
            return _parse_version(str(row[0]))
    raise ImportError(
        "could not detect nsys export version from metadata tables "
        f"(present: {sorted(names)}). Supported: 2024.x and 2025.x. "
        "Re-export with a supported nsys version."
    )


def _parse_version(raw: str) -> str:
    import re

    m = re.search(r"(20\d{2})\.(\d+)", raw)
    if not m:
        # bare year
        m2 = re.search(r"(20\d{2})", raw)
        if not m2:
            raise ImportError(
                f"unrecognized nsys version string {raw!r}; "
                "supported: 2024.x and 2025.x"
            )
        major = int(m2.group(1))
        if major not in _SUPPORTED_MAJOR:
            raise ImportError(
                f"unsupported nsys export version {major} (from {raw!r}); "
                "supported: 2024.x and 2025.x — re-export with a supported nsys"
            )
        return f"{major}.0"
    major, minor = int(m.group(1)), m.group(2)
    if major not in _SUPPORTED_MAJOR:
        raise ImportError(
            f"unsupported nsys export version {major}.{minor}; "
            "supported: 2024.x and 2025.x — re-export with a supported nsys"
        )
    return f"{major}.{minor}"


def _require_tables(conn: sqlite3.Connection) -> str:
    """Return the kernel table name; error listing missing required tables."""
    names = _table_names(conn)
    missing: list[str] = []
    kernel_table = next((t for t in _KERNEL_TABLES if t in names), None)
    if kernel_table is None:
        missing.append("CUPTI_ACTIVITY_KIND_KERNEL (or CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)")
    if "StringIds" not in names:
        missing.append("StringIds")
    if missing:
        raise ImportError(
            "nsys sqlite missing required table(s): " + ", ".join(missing)
        )
    assert kernel_table is not None
    return kernel_table


def _string_map(conn: sqlite3.Connection) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in conn.execute("SELECT id, value FROM StringIds"):
        sid, val = row[0], row[1]
        if sid is not None and val is not None:
            out[int(sid)] = str(val)
    return out


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _pick(row: sqlite3.Row, *names: str, default: Any = None) -> Any:
    keys = row.keys()
    for n in names:
        if n in keys:
            return row[n]
    # case-insensitive
    lower = {k.lower(): k for k in keys}
    for n in names:
        if n.lower() in lower:
            return row[lower[n.lower()]]
    return default


def _resolve_name(row: sqlite3.Row, strings: dict[int, str]) -> str:
    demangled_id = _pick(row, "demangledName", "demangledNameId")
    short_id = _pick(row, "shortName", "shortNameId")
    # Already-resolved text columns (some export modes inline strings).
    for key in ("demangledName", "shortName", "name"):
        val = _pick(row, key)
        if isinstance(val, str) and val and not val.isdigit():
            # Prefer non-numeric text; id columns are ints though.
            if key in ("demangledName", "shortName") and isinstance(
                _pick(row, key), int
            ):
                continue
            # If the column is the id int stored oddly, skip.
    # Primary: demangled via StringIds.
    for cand in (demangled_id, short_id):
        if cand is None:
            continue
        if isinstance(cand, str) and cand.strip() and not cand.strip().isdigit():
            return cand.strip()
        try:
            sid = int(cand)
        except (TypeError, ValueError):
            continue
        if sid in strings and strings[sid]:
            return strings[sid]
    # Never emit bare numeric ids.
    return "unknown_kernel"


def _map_memory_kind(val: Any, *, strict: bool) -> str:
    if val is None:
        return "device"
    try:
        iv = int(val)
    except (TypeError, ValueError):
        if strict:
            raise ImportError(f"unknown memory kind {val!r}") from None
        return "device"
    if iv in _MEMORY_KIND:
        return _MEMORY_KIND[iv]
    if strict:
        raise ImportError(f"unknown CUPTI memory kind enum {iv}")
    return "device"


def _map_sync_type(val: Any, *, strict: bool) -> str:
    if val is None:
        return "stream"
    try:
        iv = int(val)
    except (TypeError, ValueError):
        s = str(val).lower()
        if "event" in s:
            return "event"
        if "context" in s or "device" in s:
            return "device"
        if "stream" in s:
            return "stream"
        if strict:
            raise ImportError(f"unknown sync type {val!r}") from None
        return "stream"
    if iv in _SYNC_TYPE:
        return _SYNC_TYPE[iv]
    if strict:
        raise ImportError(f"unknown CUPTI sync type enum {iv}")
    return "stream"


def _iter_kernels(
    conn: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
    *,
    device_id: int | None = None,
    strict: bool = False,
) -> Iterator[KernelEvent]:
    from gitm.importers._common import make_kernel_fast

    conn.row_factory = sqlite3.Row
    if device_id is None:
        cur = conn.execute(f"SELECT * FROM {table}")
    else:
        # Prefer column filter so sqlite streams only one device.
        cur = conn.execute(
            f"SELECT * FROM {table} WHERE deviceId = ? OR deviceId IS NULL",
            (device_id,),
        )
    for row in cur:  # cursor iteration — never fetchall()
        start = as_int(_pick(row, "start", "startNs", "timestamp"))
        end = as_int(_pick(row, "end", "endNs"), default=start)
        static_sm = as_int(_pick(row, "staticSharedMemory", "staticSharedMem"), 0)
        dyn_sm = as_int(_pick(row, "dynamicSharedMemory", "dynamicSharedMem"), 0)
        corr = _pick(row, "correlationId", "correlation")
        dev = as_int(_pick(row, "deviceId", "device"), 0)
        if device_id is not None and dev != device_id:
            continue
        yield make_kernel_fast(
            name=_resolve_name(row, strings),
            start_ns=start,
            end_ns=end,
            stream_id=as_int(_pick(row, "streamId", "stream"), 0),
            device_id=dev,
            correlation_id=as_int(corr) if corr is not None else None,
            grid_x=as_int(_pick(row, "gridX"), 1),
            grid_y=as_int(_pick(row, "gridY"), 1),
            grid_z=as_int(_pick(row, "gridZ"), 1),
            block_x=as_int(_pick(row, "blockX"), 1),
            block_y=as_int(_pick(row, "blockY"), 1),
            block_z=as_int(_pick(row, "blockZ"), 1),
            shared_mem_bytes=static_sm + dyn_sm,
            registers_per_thread=as_int(_pick(row, "registersPerThread", "registers"), 0),
            strict=strict,
        )


def _iter_memcpys(
    conn: sqlite3.Connection,
    *,
    strict: bool,
    device_id: int | None = None,
) -> Iterator[MemcpyEvent]:
    from gitm.importers._common import make_memcpy_fast

    names = _table_names(conn)
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if table not in names:
        return
    conn.row_factory = sqlite3.Row
    if device_id is None:
        cur = conn.execute(f"SELECT * FROM {table}")
    else:
        cur = conn.execute(f"SELECT * FROM {table} WHERE deviceId = ?", (device_id,))
    for row in cur:
        start = as_int(_pick(row, "start", "startNs"))
        end = as_int(_pick(row, "end", "endNs"), default=start)
        src_k = _pick(row, "srcKind", "sourceKind")
        dst_k = _pick(row, "dstKind", "destinationKind")
        copy_k = _pick(row, "copyKind", "memcpyKind")
        if src_k is not None or dst_k is not None:
            src = _map_memory_kind(src_k, strict=strict)
            dst = _map_memory_kind(dst_k, strict=strict)
        elif copy_k is not None:
            try:
                iv = int(copy_k)
            except (TypeError, ValueError):
                if strict:
                    raise ImportError(f"unknown copyKind {copy_k!r}") from None
                src, dst = "device", "device"
            else:
                if iv in _COPY_KIND_TO_ENDPOINTS:
                    src, dst = _COPY_KIND_TO_ENDPOINTS[iv]
                elif strict:
                    raise ImportError(f"unknown CUPTI copyKind enum {iv}")
                else:
                    src, dst = "device", "device"
        else:
            src, dst = "device", "device"
        corr = _pick(row, "correlationId", "correlation")
        dev = as_int(_pick(row, "deviceId", "device"), 0)
        if device_id is not None and dev != device_id:
            continue
        yield make_memcpy_fast(
            start_ns=start,
            end_ns=end,
            stream_id=as_int(_pick(row, "streamId", "stream"), 0),
            device_id=dev,
            correlation_id=as_int(corr) if corr is not None else None,
            nbytes=as_int(_pick(row, "bytes", "size"), 0),
            src=src,
            dst=dst,
            strict=strict,
        )


def _iter_syncs(
    conn: sqlite3.Connection,
    *,
    strict: bool,
    device_id: int | None = None,
) -> Iterator[SyncEvent]:
    from gitm.importers._common import make_sync_fast

    names = _table_names(conn)
    table = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    if table not in names:
        return
    conn.row_factory = sqlite3.Row
    if device_id is None:
        cur = conn.execute(f"SELECT * FROM {table}")
    else:
        cur = conn.execute(f"SELECT * FROM {table} WHERE deviceId = ?", (device_id,))
    for row in cur:
        start = as_int(_pick(row, "start", "startNs"))
        end = as_int(_pick(row, "end", "endNs"), default=start)
        corr = _pick(row, "correlationId", "correlation")
        dev = as_int(_pick(row, "deviceId", "device"), 0)
        if device_id is not None and dev != device_id:
            continue
        yield make_sync_fast(
            start_ns=start,
            end_ns=end,
            stream_id=as_int(_pick(row, "streamId", "stream"), 0),
            device_id=dev,
            correlation_id=as_int(corr) if corr is not None else None,
            sync_kind=_map_sync_type(_pick(row, "syncType", "typeOfSync", "type"), strict=strict),
            strict=strict,
        )


def _device_meta(conn: sqlite3.Connection) -> tuple[int | None, str | None, int | None]:
    """(device_count, device_name, session_start_ns) from TARGET_INFO_* if present."""
    names = _table_names(conn)
    count: int | None = None
    name: str | None = None
    session_start: int | None = None
    for table in ("TARGET_INFO_GPU", "TARGET_INFO_CUDA_DEVICE"):
        if table not in names:
            continue
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            count = len(rows)
            row = rows[0]
            name = _pick(row, "name", "deviceName", "modelName")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if name is not None:
                name = str(name)
    # Session start from MetaData if present.
    for table in ("MetaData", "METADATA", "GITM_NSYS_META"):
        if table not in names:
            continue
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        except sqlite3.Error:
            continue
        if "sessionStart" in cols or "session_start_ns" in cols:
            col = "sessionStart" if "sessionStart" in cols else "session_start_ns"
            row = conn.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchone()
            if row and row[0] is not None:
                session_start = as_int(row[0])
        if "deviceName" in cols and name is None:
            row = conn.execute(f"SELECT deviceName FROM {table} LIMIT 1").fetchone()
            if row and row[0]:
                name = str(row[0])
        if "device_count" in cols and count is None:
            row = conn.execute(f"SELECT device_count FROM {table} LIMIT 1").fetchone()
            if row and row[0] is not None:
                count = as_int(row[0])
    return count, name, session_start


def import_nsys(
    path: str | Path,
    *,
    workload_id: str | None = None,
    device: int | None = None,
    strict: bool = False,
    run_id: str | None = None,
    sku: str | None = None,
) -> tuple[list[Trace], ImportStats]:
    """Import an nsys-rep or nsys-exported sqlite into one Trace per device.

    ``device`` is an optional filter (keep only that device). Default is all
    devices found in the file.
    """
    path = Path(path)
    if not path.is_file():
        raise ImportError(f"not a file: {path}")

    sqlite_path = path
    tmp_export: Path | None = None
    # Detect whether we need export (non-sqlite).
    with path.open("rb") as fh:
        magic = fh.read(16)
    is_sqlite = magic.startswith(b"SQLite format 3")
    if not is_sqlite:
        tmp_export = _export_nsys_rep(path)
        sqlite_path = tmp_export

    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ImportError(f"cannot open nsys sqlite {sqlite_path}: {exc}") from exc

    try:
        version = _detect_version(conn)
        kernel_table = _require_tables(conn)
        strings = _string_map(conn)
        meta_count, device_name, session_start = _device_meta(conn)

        # Pass 1: device ids via SQL aggregate (streams; never fetchall of events).
        all_counts: dict[int, int] = {}
        for row in conn.execute(
            f"SELECT deviceId, COUNT(*) FROM {kernel_table} GROUP BY deviceId"
        ):
            all_counts[as_int(row[0], 0)] = as_int(row[1], 0)
        if not all_counts:
            raise ImportError(f"{path.name}: no CUPTI events found")

        device_ids = sorted(all_counts.keys())
        if device is not None:
            if device not in device_ids:
                raise ImportError(
                    f"--device {device} not present; per-device kernel counts: {all_counts}"
                )
            device_ids = [device]

        wl = workload_id or path.stem
        captured_at = session_start if session_start is not None else file_mtime_ns(path)
        captured_src = "metadata" if session_start is not None else "mtime"
        import uuid

        rid = run_id or f"import-{uuid.uuid4().hex}"
        dcount = meta_count or (max(all_counts.keys()) + 1)
        traces: list[Trace] = []
        total_events = 0
        # Pass 2: one device at a time — peak RAM ≈ max(per-device), not sum.
        for dev in device_ids:
            dev_events: list[TraceEvent] = []
            dev_events.extend(
                _iter_kernels(
                    conn, kernel_table, strings, device_id=dev, strict=strict
                )
            )
            dev_events.extend(_iter_memcpys(conn, strict=strict, device_id=dev))
            dev_events.extend(_iter_syncs(conn, strict=strict, device_id=dev))
            if not dev_events:
                continue
            total_events += len(dev_events)
            trace, _st = finish_trace(
                events=dev_events,
                workload_id=wl,
                source="nsys-import",
                vendor="nvidia",
                device_count=dcount,
                captured_at_ns=captured_at,
                run_id=rid,
                strict=strict,
            )
            traces.append(trace)
            del dev_events

        if not traces:
            raise ImportError(f"{path.name}: no events left after device filter")

        stats = ImportStats(
            source_path=str(path),
            format=f"nsys-import/{version}",
            sku=sku or device_name,
            device_name=device_name,
            captured_at_source=captured_src,
            per_device_kernel_counts=all_counts,
            total_raw_events=total_events,
        )
        if len(device_ids) > 1:
            stats.warnings.append(
                f"multi-GPU input: analyzing devices {device_ids}; "
                f"per-device kernel counts: {all_counts}"
            )
        elif device is not None:
            stats.warnings.append(f"device filter: keeping device {device} only")
        return traces, stats
    finally:
        conn.close()
        if tmp_export is not None:
            try:
                tmp_export.unlink(missing_ok=True)
                tmp_export.parent.rmdir()
            except OSError:
                pass
