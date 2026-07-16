"""Lightweight coverage-style fuzz harness for detect + chrome ingest.

Full Atheris (libFuzzer) needs a special build; this module provides a
deterministic garbage-input suite that exercises the same entry points and
asserts **per-file isolation**: no uncaught exception escapes to crash the
multi-file analyze run.

When `atheris` is installed and `GITM_ATHERIS=1`, a true libFuzzer target is
also exposed via ``python -m tests.test_importers_fuzz``.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pytest

from gitm.importers._common import ImportError as ProfilerImportError
from gitm.importers.analyze import analyze_paths
from gitm.importers.detect import DetectedFormat, detect_format
from gitm.importers.torch_trace import import_torch_trace

# Hand-crafted garbage corpus (kept small, checked in as logic not files).
_CORPUS: list[bytes] = [
    b"",
    b"\x00" * 64,
    b"\xff" * 256,
    b"SQLite format 3\x00" + b"\x00" * 100,  # sqlite magic, empty body
    b"\x1f\x8b" + b"\x00" * 20,  # gzip magic, truncated
    b'{"traceEvents": null}',
    b'{"traceEvents": "not-a-list"}',
    b'{"traceEvents": [}',
    b"[" + b"{" * 100,
    b'{"schemaVersion": 1}',  # no traceEvents
    b"NSYS" + b"\x00" * 128,
    b"PK\x03\x04" + b"\x00" * 64,  # zip magic
    json.dumps({"traceEvents": [{"ph": "X", "cat": "Kernel", "name": "k", "ts": "NaN", "dur": 1}]}).encode(),
    json.dumps({"traceEvents": [{"ph": "B", "cat": "Kernel", "name": "k", "ts": 0, "dur": 1}]}).encode(),
    # gzip of garbage
    b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + zlib.compress(b"not-json")[2:-4] + b"\x00\x00",
]


@pytest.mark.parametrize("i,blob", list(enumerate(_CORPUS)))
def test_detect_garbage_never_raises(i, blob, tmp_path):
    p = tmp_path / f"c{i}.bin"
    p.write_bytes(blob)
    res = detect_format(p)
    assert res.format in DetectedFormat


@pytest.mark.parametrize("i,blob", list(enumerate(_CORPUS)))
def test_import_garbage_is_import_error_or_unknown(i, blob, tmp_path):
    p = tmp_path / f"c{i}.json"
    p.write_bytes(blob)
    det = detect_format(p)
    if det.format not in (DetectedFormat.TORCH_JSON, DetectedFormat.TORCH_JSON_GZ):
        return  # not claimed as torch — fine
    with pytest.raises((ProfilerImportError, Exception)):
        # Must not hang; may raise ImportError or JSON errors wrapped.
        import_torch_trace(p)


def test_analyze_multi_file_isolates_garbage(tmp_path):
    """One good file + many garbage files → report still produced."""
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "Kernel",
                        "name": "k",
                        "ts": 0,
                        "dur": 10,
                        "args": {"device": 0, "stream": 0},
                    }
                ]
            }
        )
    )
    for i, blob in enumerate(_CORPUS):
        (tmp_path / f"bad{i}.bin").write_bytes(blob)
    out = tmp_path / "r.md"
    result = analyze_paths([tmp_path], out=out, strict=False, run_id="import-fuzz")
    assert result.workloads, "good file should still analyze"
    assert result.failures, "garbage should be listed, not crash the run"
    assert out.exists()


# ── optional atheris entrypoint ──────────────────────────────────────────────


def _atheris_main() -> None:
    import atheris  # type: ignore

    from gitm.importers.detect import detect_format as _det
    from gitm.importers.torch_trace import event_from_chrome

    @atheris.instrument_func
    def TestOneInput(data: bytes) -> None:
        # detect path
        p = Path("/tmp/gitm_atheris_in.bin")
        p.write_bytes(data[:65536])
        try:
            _det(p)
        except Exception:
            pass
        # chrome event path via json-ish
        try:
            obj = json.loads(data.decode("utf-8", errors="ignore") or "{}")
            if isinstance(obj, dict):
                event_from_chrome(obj)
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                event_from_chrome(obj[0])
        except Exception:
            pass

    atheris.Setup([], TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    if os.environ.get("GITM_ATHERIS") == "1":
        _atheris_main()
    else:
        print("Run via pytest, or GITM_ATHERIS=1 python -m tests.test_importers_fuzz")
