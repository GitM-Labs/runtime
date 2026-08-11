"""Newline-delimited JSON sink — default for local dev."""

from __future__ import annotations

import io
import threading
from pathlib import Path

from gitm.telemetry.schema import Sample


class JsonlSink:
    """Append-only JSONL writer. Thread-safe across collector threads."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: io.TextIOBase = self._path.open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, sample: Sample) -> None:
        line = sample.model_dump_json()
        with self._lock:
            self._fh.write(line)
            self._fh.write("\n")

    def close(self) -> None:
        with self._lock:
            # Let the collector record and warn on finalization failures. Swallowing
            # one here makes a truncated telemetry artifact look complete.
            self._fh.flush()
            self._fh.close()
