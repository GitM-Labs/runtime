"""Property-based tests for importer field mapping and normalization.

Uses hypothesis to generate edge-case chrome events / timestamps instead of
only hand-written cases. These are intended to catch hollow green tests.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gitm.importers._common import (
    LightKernel,
    LightMemcpy,
    normalize_and_clean,
)
from gitm.importers.detect import DetectedFormat, detect_format
from gitm.importers.node_rollup import is_comm_kernel
from gitm.importers.torch_trace import event_from_chrome
from gitm.tracer.schema import KernelEvent, MemcpyEvent, Trace, TraceEvent

# ── µs → ns conversion ───────────────────────────────────────────────────────


@given(
    ts=st.floats(min_value=0.0, max_value=1e15, allow_nan=False, allow_infinity=False),
    dur=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_prop_us_to_ns_monotonic(ts: float, dur: float):
    ev = event_from_chrome(
        {
            "ph": "X",
            "cat": "Kernel",
            "name": "k",
            "ts": ts,
            "dur": dur,
            "args": {"device": 0, "stream": 0},
        }
    )
    # Default import path uses LightKernel for scale; --strict uses KernelEvent.
    assert isinstance(ev, (KernelEvent, LightKernel))
    assert ev.start_ns == int(ts * 1000.0)
    assert ev.end_ns == int((ts + dur) * 1000.0)
    assert ev.end_ns >= ev.start_ns or dur == 0.0


# ── null / missing args fall back without crashing ───────────────────────────


@given(
    cat=st.sampled_from(["Kernel", "kernel", "gpu_op", "Memcpy", "gpu_memcpy"]),
    name=st.text(min_size=1, max_size=40).filter(lambda s: s.strip() != ""),
    has_grid=st.booleans(),
    has_stream=st.booleans(),
    has_device=st.booleans(),
)
@settings(max_examples=150, deadline=None)
def test_prop_missing_args_never_crash(cat, name, has_grid, has_stream, has_device):
    args: dict = {}
    if has_stream:
        args["stream"] = 3
    if has_device:
        args["device"] = 0
    if has_grid:
        args["grid"] = [2, 1, 1]
        args["block"] = [32, 1, 1]
    # memset names must skip for Memcpy-ish cats
    assume("memset" not in name.lower())
    ev = event_from_chrome(
        {"ph": "X", "cat": cat, "name": name, "ts": 1.0, "dur": 2.0, "args": args}
    )
    # May be None for weird name/cat combos; must not raise.
    if ev is not None:
        assert isinstance(ev, (KernelEvent, MemcpyEvent, LightKernel, LightMemcpy))
        assert getattr(ev, "kind", None) in ("kernel", "memcpy")
        assert ev.end_ns >= ev.start_ns
        # Light events are not pydantic; only validate the full Trace path
        # when the object supports model_dump (strict/full KernelEvent path).
        if hasattr(ev, "model_dump"):
            Trace.model_validate(
                {
                    "workload_id": "w",
                    "fingerprint": "f",
                    "run_id": "r",
                    "device_count": 1,
                    "vendor": "nvidia",
                    "captured_at_ns": 0,
                    "duration_ns": max(ev.end_ns - ev.start_ns, 1),
                    "events": [ev.model_dump()],
                    "source": "torch-import",
                }
            )


# ── normalize: end < start dropped; t0 rebased ───────────────────────────────


@given(
    starts=st.lists(st.integers(min_value=0, max_value=10_000_000), min_size=1, max_size=20),
    offsets=st.lists(st.integers(min_value=-100, max_value=10_000), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_prop_normalize_nonneg_and_ordered(starts, offsets):
    n = min(len(starts), len(offsets))
    events: list[TraceEvent] = []
    for i in range(n):
        s = starts[i]
        e = s + offsets[i]
        events.append(
            KernelEvent(
                kind="kernel",
                name=f"k{i}",
                start_ns=s,
                end_ns=e,
                stream_id=0,
                device_id=0,
            )
        )
    cleaned, _deduped, _dropped = normalize_and_clean(events, strict=False)
    for ev in cleaned:
        assert ev.start_ns >= 0
        assert ev.end_ns >= ev.start_ns
    if cleaned:
        assert min(ev.start_ns for ev in cleaned) == 0


# ── detect never crashes on random bytes ─────────────────────────────────────


@given(data=st.binary(min_size=0, max_size=4096))
@settings(max_examples=100, deadline=None)
def test_prop_detect_garbage_bytes(data: bytes):
    import hashlib
    import tempfile

    name = hashlib.sha1(data[:64] if data else b"empty").hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / f"gitm_hyp_{name}.bin"
    path.write_bytes(data)
    try:
        res = detect_format(path)
        assert isinstance(res.format, DetectedFormat)
        assert isinstance(res.reason, str)
    finally:
        path.unlink(missing_ok=True)


# ── comm classifier is pure and case-insensitive for known patterns ──────────


@given(
    prefix=st.sampled_from(["", "void ", "nccl::"]),
    core=st.sampled_from(
        ["ncclAllReduce", "AllReduce", "all_gather", "ReduceScatter", "Broadcast", "SendRecv"]
    ),
    suffix=st.text(alphabet="abcXYZ_012", max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_prop_comm_patterns_match(prefix, core, suffix):
    assert is_comm_kernel(prefix + core + suffix)


@given(name=st.sampled_from(["cutlass_gemm", "ampere_fp16", "elementwise", "memcpy_kernel"]))
def test_prop_non_comm_not_matched(name):
    assert not is_comm_kernel(name)
