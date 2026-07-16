"""Format sniffing for customer profiler dumps.

Never relies on file extension alone — magic bytes, sqlite headers, and
top-level JSON keys decide the format.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DetectedFormat(str, Enum):
    NSYS_REP = "nsys-rep"
    NSYS_SQLITE = "nsys-sqlite"
    TORCH_JSON = "torch-json"
    TORCH_JSON_GZ = "torch-json-gz"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectResult:
    format: DetectedFormat
    reason: str


# SQLite database header: "SQLite format 3\000"
_SQLITE_MAGIC = b"SQLite format 3\x00"
# Nsight Systems .nsys-rep container signatures seen in the wild.
_NSYS_MAGICS = (
    b"NSYS",
    b"QDSTRM",
    b"NVTX",
)


def _looks_like_nsys_sqlite(conn: sqlite3.Connection) -> bool:
    """True if the sqlite DB has nsys-style CUPTI activity tables."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    names = {row[0] for row in cur.fetchall()}
    has_kernel = any(
        n in names
        for n in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        )
    )
    has_strings = "StringIds" in names
    return has_kernel and has_strings


def _probe_sqlite(path: Path) -> DetectResult:
    """Open a file that *looks* like sqlite; never raise on truncated/corrupt DBs."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return DetectResult(DetectedFormat.UNKNOWN, f"sqlite open failed: {exc}")
    try:
        try:
            is_nsys = _looks_like_nsys_sqlite(conn)
        except sqlite3.Error as exc:
            # Header matched but body is truncated/corrupt — customer dump, not a crash.
            return DetectResult(
                DetectedFormat.UNKNOWN, f"sqlite header present but not a database: {exc}"
            )
        if is_nsys:
            return DetectResult(
                DetectedFormat.NSYS_SQLITE,
                "sqlite header + CUPTI kernel table + StringIds",
            )
        return DetectResult(
            DetectedFormat.UNKNOWN,
            "sqlite database without nsys CUPTI tables",
        )
    finally:
        conn.close()


def _probe_json_bytes(raw: bytes) -> bool:
    """True if bytes look like a chrome/torch trace (traceEvents present)."""
    sample = raw.lstrip()[:65536]
    if not sample:
        return False
    if sample[0:1] not in (b"{", b"["):
        return False
    if b"traceEvents" in sample:
        return True
    # Array form: top-level list of chrome events.
    if sample[0:1] == b"[":
        try:
            if len(raw) < 2_000_000:
                data = json.loads(raw)
                if isinstance(data, list) and data:
                    first = data[0]
                    return isinstance(first, dict) and (
                        "ph" in first or "cat" in first or "name" in first
                    )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return False
    return False


def detect_format(path: str | Path) -> DetectResult:
    """Sniff the format of ``path``. Extension is never the sole signal."""
    path = Path(path)
    if not path.is_file():
        return DetectResult(DetectedFormat.UNKNOWN, "not a regular file")

    try:
        size = path.stat().st_size
        if size == 0:
            return DetectResult(DetectedFormat.UNKNOWN, "empty file")

        with path.open("rb") as fh:
            prefix = fh.read(64)

        # Gzip?
        if prefix[:2] == b"\x1f\x8b":
            try:
                with gzip.open(path, "rb") as gzh:
                    gprefix = gzh.read(65536)
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                # Truncated or bogus gzip — common in half-uploaded customer dumps.
                return DetectResult(
                    DetectedFormat.UNKNOWN, f"gzip decompress failed: {exc}"
                )
            if _probe_json_bytes(gprefix) or b"traceEvents" in gprefix:
                return DetectResult(
                    DetectedFormat.TORCH_JSON_GZ,
                    "gzip + chrome/torch traceEvents",
                )
            return DetectResult(DetectedFormat.UNKNOWN, "gzip without traceEvents")

        # SQLite?
        if prefix.startswith(_SQLITE_MAGIC):
            return _probe_sqlite(path)

        # Nsight .nsys-rep proprietary container.
        for magic in _NSYS_MAGICS:
            if magic in prefix[:32]:
                return DetectResult(
                    DetectedFormat.NSYS_REP,
                    f"nsys container magic {magic!r}",
                )

        # Some .nsys-rep files are ZIP-based packages.
        if prefix[:2] == b"PK":
            try:
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()
                if any("sqlite" in n.lower() or n.endswith(".db") for n in names):
                    return DetectResult(
                        DetectedFormat.NSYS_REP,
                        "zip container with embedded sqlite (nsys-rep)",
                    )
            except (zipfile.BadZipFile, OSError):
                pass

        # JSON chrome/torch?
        if prefix.lstrip()[:1] in (b"{", b"["):
            with path.open("rb") as fh:
                sample = fh.read(min(size, 256_000))
            if _probe_json_bytes(sample) or b"traceEvents" in sample:
                return DetectResult(
                    DetectedFormat.TORCH_JSON,
                    "JSON with traceEvents / chrome event shape",
                )

    except OSError as exc:
        return DetectResult(DetectedFormat.UNKNOWN, f"read failed: {exc}")

    # Extension is a soft last-resort hint only for .nsys-rep when magic failed
    # (some nsys versions use opaque headers). Still require non-empty binary.
    if path.suffix.lower() == ".nsys-rep" and path.stat().st_size > 64:
        return DetectResult(
            DetectedFormat.NSYS_REP,
            "opaque binary with .nsys-rep extension (magic unrecognized)",
        )

    return DetectResult(DetectedFormat.UNKNOWN, "no recognized profiler signature")


def is_recognized(path: str | Path) -> bool:
    return detect_format(path).format != DetectedFormat.UNKNOWN
