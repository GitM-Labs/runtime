"""PyTorch profiler / chrome-trace JSON importer.

Accepts ``.json`` and ``.json.gz`` written by ``torch.profiler`` or legacy
``chrome_trace`` export. Detected by top-level ``traceEvents`` (object form)
or a top-level list of chrome events (array form).
"""

from __future__ import annotations

import gzip
import json
import re
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
from gitm.tracer.schema import KernelEvent, MemcpyEvent, Trace, TraceEvent

# Max decompressed size for .json.gz (customer dumps can be multi-GB).
DEFAULT_MAX_DECOMPRESSED_BYTES = 20 * 1024 ** 3  # 20 GiB

# Complete events only.
_PHASE_COMPLETE = "X"

# Kernel categories across torch versions.
_KERNEL_CATS = frozenset({"kernel", "Kernel", "gpu_op", "gpu_memcpy", "Memcpy"})
_MEMCPY_CATS = frozenset({"gpu_memcpy", "Memcpy", "memcpy", "gpu_memset"})  # memset skipped later
_SKIP_CATS = frozenset({
    "cpu_op", "cpu_instant_event", "user_annotation", "python_function",
    "gpu_memset", "Memset", "memset",
    # kineto chrome-trace non-GPU categories (must not become kernels)
    "Operator", "Runtime", "Trace", "async", "gpu_user_annotation",
    "CudaRuntime", "cuda_runtime",
})

# Arg key candidate lists (torch version drift).
_STREAM_KEYS = ("stream", "stream id", "Stream", "streamId", "correlation stream")
_DEVICE_KEYS = ("device", "device id", "Device", "deviceId", "gpu", "device_id")
_CORR_KEYS = (
    "correlation",
    "correlation id",
    "CorrelationId",
    "External id",
    "External Id",
    "external id",  # kineto chrome export
    "External Id",
)
_GRID_KEYS = ("grid", "gridX", "grid dimensions", "grid dim", "Grid")
_BLOCK_KEYS = ("block", "blockX", "block dimensions", "block dim", "blockDim", "Block")
_SHARED_KEYS = (
    "shared memory",
    "shared_memory",
    "static shared memory",
    "sharedMemory",
    "Shared Memory",
)
_REGS_KEYS = (
    "registers per thread",
    "registers_per_thread",
    "registers",
    "reg/thread",
    "Registers Per Thread",
)
_BYTES_KEYS = ("bytes", "Bytes", "size", "Memory Size", "memory size", "bytecount")


def _arg_get(args: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if not args:
        return None
    # Exact then case-insensitive.
    for k in keys:
        if k in args:
            return args[k]
    lower = {str(k).lower(): v for k, v in args.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _as_dims(val: Any) -> tuple[int, int, int]:
    if val is None:
        return (1, 1, 1)
    if isinstance(val, (list, tuple)) and len(val) >= 1:
        x = as_int(val[0], 1) if len(val) > 0 else 1
        y = as_int(val[1], 1) if len(val) > 1 else 1
        z = as_int(val[2], 1) if len(val) > 2 else 1
        return (max(x, 1), max(y, 1), max(z, 1))
    if isinstance(val, dict):
        return (
            max(as_int(val.get("x", val.get("X", 1)), 1), 1),
            max(as_int(val.get("y", val.get("Y", 1)), 1), 1),
            max(as_int(val.get("z", val.get("Z", 1)), 1), 1),
        )
    # Single int → x only.
    try:
        return (max(int(val), 1), 1, 1)
    except (TypeError, ValueError):
        return (1, 1, 1)


def _is_device_side_gpu_op(args: dict[str, Any] | None) -> bool:
    """For cat=gpu_op, only keep device-side kernels (not CPU wrappers)."""
    if not args:
        return True
    # Torch sometimes sets "Device Type" / "device_type".
    for key in ("Device Type", "device_type", "device type"):
        if key in args:
            v = str(args[key]).lower()
            if "cpu" in v:
                return False
            if "cuda" in v or "gpu" in v:
                return True
    # Presence of stream/device id implies GPU.
    if _arg_get(args, _STREAM_KEYS) is not None:
        return True
    if _arg_get(args, _DEVICE_KEYS) is not None:
        return True
    return True


def _classify(cat: str | None, name: str, args: dict[str, Any] | None) -> str | None:
    """Return 'kernel' | 'memcpy' | None (skip)."""
    c = cat or ""
    if c in _SKIP_CATS or c.lower() in {"memset", "gpu_memset"}:
        return None
    if c in _MEMCPY_CATS or c.lower() in {"gpu_memcpy", "memcpy"}:
        # Distinguish memset by name.
        if "memset" in (name or "").lower():
            return None
        return "memcpy"
    if c in {"kernel", "Kernel"}:
        return "kernel"
    if c == "gpu_op":
        if "memcpy" in (name or "").lower() or "copy" in (name or "").lower() and "kernel" not in (
            name or ""
        ).lower():
            # Some exports label copies as gpu_op.
            if "memset" in (name or "").lower():
                return None
            # Prefer memcpy only when args carry bytes and name looks like a copy.
            if "memcpy" in (name or "").lower():
                return "memcpy"
        if not _is_device_side_gpu_op(args):
            return None
        return "kernel"
    # Fallback: name-based for sparse exports.
    n = (name or "").lower()
    if "memset" in n:
        return None
    if "memcpy" in n:
        return "memcpy"
    return None



def _memcpy_event(
    *,
    start_ns: int,
    end_ns: int,
    stream: int,
    device: int,
    corr: int | None,
    nbytes: int,
    src: str,
    dst: str,
    strict: bool = False,
) -> MemcpyEvent:
    from gitm.importers._common import make_memcpy_fast

    return make_memcpy_fast(
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream,
        device_id=device,
        correlation_id=corr,
        nbytes=nbytes,
        src=src,
        dst=dst,
        strict=strict,
    )


def event_from_chrome(
    obj: dict[str, Any],
    *,
    strict: bool = False,
) -> TraceEvent | None:
    """Public mapping of one chrome-trace event dict → TraceEvent or None."""
    from gitm.importers._common import make_kernel_fast

    if obj.get("ph") != _PHASE_COMPLETE:
        return None
    cat = obj.get("cat")
    name = str(obj.get("name") or "unknown")
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    kind = _classify(str(cat) if cat is not None else None, name, args)
    if kind is None:
        if cat and str(cat).lower() in {"cuda", "gpu"}:
            kind = "kernel"
        else:
            return None

    try:
        ts = float(obj.get("ts", 0.0))
        dur = float(obj.get("dur", 0.0))
    except (TypeError, ValueError):
        return None
    start_ns = int(ts * 1000.0)
    end_ns = int((ts + dur) * 1000.0)

    stream = as_int(_arg_get(args, _STREAM_KEYS), 0)
    device = as_int(_arg_get(args, _DEVICE_KEYS), 0)
    corr_raw = _arg_get(args, _CORR_KEYS)
    corr = as_int(corr_raw) if corr_raw is not None else None

    if kind == "memcpy":
        nbytes = as_int(_arg_get(args, _BYTES_KEYS), 0)
        src, dst = "host", "device"
        nlow = name.lower()
        if "dtoh" in nlow or "device_to_host" in nlow:
            src, dst = "device", "host"
        elif "dtod" in nlow or "device_to_device" in nlow:
            src, dst = "device", "device"
        elif "htoh" in nlow:
            src, dst = "host", "host"
        return _memcpy_event(
            start_ns=start_ns,
            end_ns=end_ns,
            stream=stream,
            device=device,
            corr=corr,
            nbytes=nbytes,
            src=src,
            dst=dst,
            strict=strict,
        )

    grid = _as_dims(_arg_get(args, _GRID_KEYS))
    if _arg_get(args, _GRID_KEYS) is None:
        gx = _arg_get(args, ("gridX", "grid_x"))
        if gx is not None:
            grid = (
                as_int(gx, 1),
                as_int(_arg_get(args, ("gridY", "grid_y")), 1),
                as_int(_arg_get(args, ("gridZ", "grid_z")), 1),
            )
    block = _as_dims(_arg_get(args, _BLOCK_KEYS))
    if _arg_get(args, _BLOCK_KEYS) is None:
        bx = _arg_get(args, ("blockX", "block_x"))
        if bx is not None:
            block = (
                as_int(bx, 1),
                as_int(_arg_get(args, ("blockY", "block_y")), 1),
                as_int(_arg_get(args, ("blockZ", "block_z")), 1),
            )

    return make_kernel_fast(
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream,
        device_id=device,
        correlation_id=corr,
        grid_x=grid[0],
        grid_y=grid[1],
        grid_z=grid[2],
        block_x=block[0],
        block_y=block[1],
        block_z=block[2],
        shared_mem_bytes=as_int(_arg_get(args, _SHARED_KEYS), 0),
        registers_per_thread=as_int(_arg_get(args, _REGS_KEYS), 0),
        strict=strict,
    )


def _iter_json_array_objects(text_iter: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a streaming ``[ obj, obj, ... ]`` body."""
    buf = ""
    in_array = False
    depth = 0
    in_string = False
    escape = False
    obj_start = -1

    for chunk in text_iter:
        buf += chunk
        i = 0
        while i < len(buf):
            ch = buf[i]
            if not in_array:
                if ch == "[":
                    in_array = True
                i += 1
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
                i += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    blob = buf[obj_start : i + 1]
                    try:
                        yield json.loads(blob)
                    except json.JSONDecodeError:
                        pass
                    # Drop consumed prefix.
                    buf = buf[i + 1 :]
                    i = 0
                    obj_start = -1
                    continue
                i += 1
                continue
            if ch == "]" and depth == 0:
                return
            i += 1
        # Keep only a trailing partial object in the buffer.
        if obj_start > 0:
            buf = buf[obj_start:]
            obj_start = 0
        elif obj_start < 0 and len(buf) > 1_000_000:
            # No open object and huge buffer of whitespace/commas — trim.
            buf = buf[-16:]


def _open_text_chunks(
    path: Path,
    *,
    gzipped: bool,
    chunk_size: int = 1 << 20,
    max_decompressed_bytes: int | None = None,
) -> Iterator[str]:
    """Yield text chunks; enforce a decompressed-byte ceiling for gzip inputs."""
    limit = max_decompressed_bytes
    total = 0
    if gzipped:
        opener = gzip.open
    else:
        opener = open  # type: ignore[assignment]
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[arg-type]
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            total += len(chunk.encode("utf-8", errors="replace"))
            if limit is not None and total > limit:
                raise ImportError(
                    f"{path.name}: decompressed size exceeds limit of "
                    f"{limit} bytes ({limit / (1024 ** 3):.1f} GiB). "
                    f"Re-export a shorter window or raise the limit via "
                    f"max_decompressed_bytes."
                )
            yield chunk


def _load_json_capped(
    path: Path,
    *,
    gzipped: bool,
    max_decompressed_bytes: int,
) -> Any:
    """json.load with a decompressed-size cap for gzip."""
    if not gzipped:
        with path.open("rt", encoding="utf-8") as fh:
            return json.load(fh)
    # Stream into memory with a hard ceiling so a zip-bomb cannot OOM us.
    buf = bytearray()
    with gzip.open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > max_decompressed_bytes:
                raise ImportError(
                    f"{path.name}: decompressed size exceeds limit of "
                    f"{max_decompressed_bytes} bytes "
                    f"({max_decompressed_bytes / (1024 ** 3):.1f} GiB). "
                    f"Re-export a shorter window or raise the limit."
                )
    return json.loads(buf.decode("utf-8"))


def _iter_chrome_event_dicts(
    path: Path,
    *,
    gzipped: bool,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> Iterator[dict[str, Any]]:
    """Yield raw chrome-trace event dicts without loading the whole file.

    Strategy (in order):
      1. Compact one-JSON-object-per-line inside ``traceEvents`` (bench + many
         kineto exports) — line-oriented ``json.loads`` per line.
      2. Fallback brace-stream for pretty-printed multi-line objects.
    Never uses a whole-file ``json.load`` of multi-million-event dumps.
    """
    opener: Any = gzip.open if gzipped else open
    total = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[arg-type]
        # Seek to the start of the events array.
        head = fh.read(65536)
        total += len(head.encode("utf-8", errors="replace"))
        m = re.search(r'"traceEvents"\s*:\s*\[', head)
        if m:
            rest = head[m.end() :]
            array_form = False
        elif head.lstrip().startswith("["):
            rest = head.lstrip()[1:]
            array_form = True
        else:
            # Maybe key is later — read more until found or give up.
            buf = head
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    raise ImportError(f"{path.name}: no 'traceEvents' array found")
                total += len(chunk.encode("utf-8", errors="replace"))
                if total > max_decompressed_bytes:
                    raise ImportError(
                        f"{path.name}: decompressed size exceeds limit of "
                        f"{max_decompressed_bytes} bytes"
                    )
                buf += chunk
                m = re.search(r'"traceEvents"\s*:\s*\[', buf)
                if m:
                    rest = buf[m.end() :]
                    array_form = False
                    break
                if len(buf) > 50_000_000:
                    raise ImportError(f"{path.name}: no 'traceEvents' array found")

        # Line-oriented fast path for compact dumps (one object per line).
        # Process residual + remaining lines.
        def _handle_line(line: str) -> dict[str, Any] | None:
            s = line.strip()
            if not s:
                return None
            if s[0] == "]":
                return None  # end marker handled by caller
            if s[-1] == ",":
                s = s[:-1]
            if not s.startswith("{"):
                return None
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None

        # Prefer line mode for compact one-object-per-line dumps.
        # If ``rest`` ends mid-line (head cut), carry the incomplete prefix into
        # the first line from the file so we never drop a boundary event.
        carry = ""
        nl = rest.rfind("\n")
        if nl >= 0:
            complete, carry = rest[: nl + 1], rest[nl + 1 :]
            for line in complete.splitlines(keepends=True):
                if line.strip().startswith("]"):
                    return
                obj = _handle_line(line)
                if obj is not None:
                    yield obj
        else:
            carry = rest
        for line in fh:
            total += len(line.encode("utf-8", errors="replace"))
            if total > max_decompressed_bytes:
                raise ImportError(
                    f"{path.name}: decompressed size exceeds limit of "
                    f"{max_decompressed_bytes} bytes"
                )
            if carry:
                line = carry + line
                carry = ""
            if line.strip().startswith("]"):
                return
            obj = _handle_line(line)
            if obj is not None:
                yield obj
        _ = array_form  # reserved


def _device_id_from_chrome_obj(obj: dict[str, Any]) -> int | None:
    """Return GPU device id for a chrome event, or None if not a GPU event."""
    if obj.get("ph") != _PHASE_COMPLETE:
        return None
    cat = obj.get("cat")
    name = str(obj.get("name") or "")
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    kind = _classify(str(cat) if cat is not None else None, name, args)
    if kind is None and not (cat and str(cat).lower() in {"cuda", "gpu"}):
        return None
    return as_int(_arg_get(args, _DEVICE_KEYS), 0)


def _scan_chrome_devices(
    path: Path,
    *,
    gzipped: bool,
    max_decompressed_bytes: int,
) -> tuple[dict[int, int], str | None, int]:
    """Pass 1: device→kernel counts + optional SKU hint. O(file) time, O(1) RAM."""
    counts: dict[int, int] = {}
    sku: str | None = None
    n = 0
    for obj in _iter_chrome_event_dicts(
        path, gzipped=gzipped, max_decompressed_bytes=max_decompressed_bytes
    ):
        n += 1
        if sku is None and obj.get("name") in ("process_name", "thread_name"):
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            nm = args.get("name") if args else None
            if isinstance(nm, str) and any(
                x in nm for x in ("NVIDIA", "A100", "H100", "L40", "L4", "V100", "Tesla")
            ):
                sku = nm
        dev = _device_id_from_chrome_obj(obj)
        if dev is None:
            continue
        # Count only kernels (not memcpy) for the kernel-count table.
        cat = obj.get("cat")
        name = str(obj.get("name") or "")
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        kind = _classify(str(cat) if cat is not None else None, name, args)
        if kind == "kernel" or (kind is None and cat and str(cat).lower() in {"cuda", "gpu"}):
            counts[dev] = counts.get(dev, 0) + 1
        elif kind == "memcpy":
            counts.setdefault(dev, counts.get(dev, 0))
    return counts, sku, n


def _load_device_events(
    path: Path,
    *,
    gzipped: bool,
    max_decompressed_bytes: int,
    device_id: int,
    strict: bool,
) -> list[TraceEvent]:
    """Pass 2: materialize TraceEvents for a single device only."""
    out: list[TraceEvent] = []
    for obj in _iter_chrome_event_dicts(
        path, gzipped=gzipped, max_decompressed_bytes=max_decompressed_bytes
    ):
        dev = _device_id_from_chrome_obj(obj)
        if dev is None or dev != device_id:
            continue
        ev = event_from_chrome(obj, strict=strict)
        if ev is not None:
            out.append(ev)
    return out


def _device_name_from_meta(meta: dict[str, Any]) -> str | None:
    for key in ("deviceProperties", "computeProperties", "device_properties"):
        props = meta.get(key)
        if isinstance(props, list) and props:
            props = props[0]
        if isinstance(props, dict):
            for nk in ("name", "deviceName", "device_name", "gpu_name"):
                if props.get(nk):
                    return str(props[nk])
    return None


def _workload_stem(path: Path) -> str:
    stem = path.name
    for suffix in (
        ".pt.trace.json.gz",
        ".trace.json.gz",
        ".json.gz",
        ".pt.trace.json",
        ".trace.json",
        ".json",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


# Files below this size use a full json.load (pretty-printed fixtures, small
# customer dumps). Larger compact dumps use the line-stream multi-pass path.
_FULL_LOAD_MAX_BYTES = 80 * 1024 * 1024  # 80 MiB


def _import_torch_from_event_dicts(
    raw_events: list[dict[str, Any]],
    *,
    path: Path,
    workload_id: str | None,
    device: int | None,
    strict: bool,
    run_id: str | None,
    sku: str | None,
    root_meta: dict[str, Any],
) -> tuple[list[Trace], ImportStats]:
    """Shared finish path once raw chrome event dicts are in hand."""
    events: list[TraceEvent] = []
    for obj in raw_events:
        ev = event_from_chrome(obj, strict=strict)
        if ev is not None:
            events.append(ev)
    if not events:
        raise ImportError(
            f"{path.name}: no complete GPU kernel/memcpy events found in traceEvents"
        )
    all_counts = per_device_kernel_counts(events)
    device_ids = sorted(all_counts.keys()) if all_counts else sorted({e.device_id for e in events})
    if device is not None:
        if device not in device_ids:
            raise ImportError(
                f"--device {device} not present; per-device kernel counts: {all_counts}"
            )
        device_ids = [device]
    device_name = _device_name_from_meta(root_meta)
    if device_name is None:
        for obj in raw_events:
            if obj.get("name") in ("process_name", "thread_name"):
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
                nm = args.get("name") if args else None
                if isinstance(nm, str) and any(
                    x in nm for x in ("NVIDIA", "A100", "H100", "L40", "L4", "V100", "Tesla")
                ):
                    device_name = nm
                    break
    wl = workload_id or _workload_stem(path)
    captured_at = file_mtime_ns(path)
    import uuid

    rid = run_id or f"import-{uuid.uuid4().hex}"
    dcount = device_count_from_events(events)
    traces: list[Trace] = []
    for dev in device_ids:
        dev_events = filter_device(events, dev)
        if not dev_events:
            continue
        trace, _st = finish_trace(
            events=dev_events,
            workload_id=wl,
            source="torch-import",
            vendor="nvidia",
            device_count=dcount,
            captured_at_ns=captured_at,
            run_id=rid,
            strict=strict,
        )
        traces.append(trace)
    if not traces:
        raise ImportError(f"{path.name}: no events left after device filter")
    stats = ImportStats(
        source_path=str(path),
        format="torch-import",
        sku=sku or device_name,
        device_name=device_name,
        captured_at_source="mtime",
        per_device_kernel_counts=all_counts,
        total_raw_events=len(events),
    )
    if len(device_ids) > 1:
        stats.warnings.append(
            f"multi-GPU input: analyzing devices {device_ids}; "
            f"per-device kernel counts: {all_counts}"
        )
    elif device is not None:
        stats.warnings.append(f"device filter: keeping device {device} only")
    return traces, stats


def import_torch_trace(
    path: str | Path,
    *,
    workload_id: str | None = None,
    device: int | None = None,
    strict: bool = False,
    run_id: str | None = None,
    sku: str | None = None,
    gzipped: bool | None = None,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> tuple[list[Trace], ImportStats]:
    """Import chrome-trace JSON into one Trace per device.

    Small/pretty-printed files: full ``json.load`` (correct for multi-line
    objects). Large compact dumps: multi-pass line-stream so peak RSS stays
    near one device's events rather than the whole file DOM + all devices.
    """
    path = Path(path)
    if not path.is_file():
        raise ImportError(f"not a file: {path}")

    if gzipped is None:
        with path.open("rb") as fh:
            gzipped = fh.read(2) == b"\x1f\x8b"

    size = path.stat().st_size
    use_full_load = gzipped or size <= _FULL_LOAD_MAX_BYTES

    if use_full_load:
        try:
            data = _load_json_capped(
                path, gzipped=bool(gzipped), max_decompressed_bytes=max_decompressed_bytes
            )
        except json.JSONDecodeError as exc:
            raise ImportError(f"corrupt JSON in {path.name}: {exc}") from exc
        except OSError as exc:
            raise ImportError(f"failed to read {path.name}: {exc}") from exc
        if isinstance(data, list):
            raw_events = [e for e in data if isinstance(e, dict)]
            root_meta: dict[str, Any] = {}
        elif isinstance(data, dict):
            te = data.get("traceEvents")
            if not isinstance(te, list):
                raise ImportError(
                    f"{path.name}: JSON object missing top-level 'traceEvents' list"
                )
            raw_events = [e for e in te if isinstance(e, dict)]
            root_meta = {k: v for k, v in data.items() if k != "traceEvents"}
        else:
            raise ImportError(f"{path.name}: unsupported JSON root type {type(data).__name__}")
        if not raw_events:
            raise ImportError(f"{path.name}: no traceEvents found")
        return _import_torch_from_event_dicts(
            raw_events,
            path=path,
            workload_id=workload_id,
            device=device,
            strict=strict,
            run_id=run_id,
            sku=sku,
            root_meta=root_meta,
        )

    # Large compact dump — single-pass line stream into per-device Light* buckets.
    # Slotted events keep 5M-event peak RSS well under 4 GiB (see evidence/perf.json).
    import uuid
    from collections import defaultdict

    buckets: dict[int, list[TraceEvent]] = defaultdict(list)
    all_counts: dict[int, int] = defaultdict(int)
    scanned_sku: str | None = None
    n_raw = 0
    try:
        for obj in _iter_chrome_event_dicts(
            path, gzipped=False, max_decompressed_bytes=max_decompressed_bytes
        ):
            n_raw += 1
            if scanned_sku is None and obj.get("name") in ("process_name", "thread_name"):
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
                nm = args.get("name") if args else None
                if isinstance(nm, str) and any(
                    x in nm
                    for x in ("NVIDIA", "A100", "H100", "L40", "L4", "V100", "Tesla")
                ):
                    scanned_sku = nm
            if device is not None:
                dev_hint = _device_id_from_chrome_obj(obj)
                if dev_hint is not None and dev_hint != device:
                    continue
            ev = event_from_chrome(obj, strict=strict)
            if ev is None:
                continue
            buckets[ev.device_id].append(ev)
            if getattr(ev, "kind", None) == "kernel":
                all_counts[ev.device_id] += 1
    except json.JSONDecodeError as exc:
        raise ImportError(f"corrupt JSON in {path.name}: {exc}") from exc
    except OSError as exc:
        raise ImportError(f"failed to read {path.name}: {exc}") from exc

    if n_raw == 0:
        raise ImportError(f"{path.name}: no traceEvents found")
    if not buckets:
        raise ImportError(
            f"{path.name}: no complete GPU kernel/memcpy events found in traceEvents"
        )

    device_ids = sorted(buckets.keys())
    if device is not None:
        if device not in device_ids:
            raise ImportError(
                f"--device {device} not present; per-device kernel counts: {dict(all_counts)}"
            )
        device_ids = [device]

    wl = workload_id or _workload_stem(path)
    captured_at = file_mtime_ns(path)
    rid = run_id or f"import-{uuid.uuid4().hex}"
    dcount = max(device_ids) + 1
    traces: list[Trace] = []
    total_events = 0
    for dev in device_ids:
        dev_events = buckets.pop(dev)
        total_events += len(dev_events)
        trace, _st = finish_trace(
            events=dev_events,
            workload_id=wl,
            source="torch-import",
            vendor="nvidia",
            device_count=dcount,
            captured_at_ns=captured_at,
            run_id=rid,
            strict=strict,
        )
        traces.append(trace)

    if not traces:
        raise ImportError(f"{path.name}: no events left after device filter")

    stats = ImportStats(
        source_path=str(path),
        format="torch-import",
        sku=sku or scanned_sku,
        device_name=scanned_sku,
        captured_at_source="mtime",
        per_device_kernel_counts=dict(all_counts),
        total_raw_events=total_events,
    )
    if len(device_ids) > 1:
        stats.warnings.append(
            f"multi-GPU input: analyzing devices {device_ids}; "
            f"per-device kernel counts: {dict(all_counts)}"
        )
    elif device is not None:
        stats.warnings.append(f"device filter: keeping device {device} only")
    return traces, stats
