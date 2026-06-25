"""Trace schema JSON round-trip — example-based coverage of the
serialization contract.

A trace is captured as JSONL: header on line 1, one event per subsequent
line. Replay, `gitm.bench`, and any future inspection tooling all parse it
back. If serialization and deserialization disagree even on a single field,
silent corruption propagates. These tests cover each event type in the
discriminated union plus a few real-world cases (unicode workload IDs,
nanosecond-precision timestamps, mixed-event traces).

Property-based coverage with `hypothesis` would be a natural next step but is
deliberately deferred — `hypothesis` is not yet a project dev dependency and
adding it is out of scope for this test-only change.
"""

from __future__ import annotations

from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace

from .conftest import make_kernel, make_memcpy, make_sync, make_trace

# ── per-event-type round trip ─────────────────────────────────────────────────


def test_kernel_event_json_round_trip():
    ev = make_kernel(
        name="ampere_sgemm_128x64_nn",
        start_ns=1_700_000_000_123_000_000,
        end_ns=1_700_000_000_123_001_500,
        stream_id=7,
        device_id=0,
        grid=(128, 1, 1),
        block=(64, 2, 1),
        shared_mem_bytes=4096,
        registers_per_thread=64,
        correlation_id=42,
        bytes_read=1 << 24,
        bytes_written=1 << 16,
    )
    blob = ev.model_dump_json()
    back = KernelEvent.model_validate_json(blob)
    assert back == ev


def test_memcpy_event_json_round_trip():
    ev = make_memcpy(
        bytes=1 << 26,
        src="device",
        dst="host",
        start_ns=10,
        end_ns=2_000_000,
        stream_id=3,
        device_id=1,
        correlation_id=99,
    )
    blob = ev.model_dump_json()
    back = MemcpyEvent.model_validate_json(blob)
    assert back == ev


def test_sync_event_json_round_trip():
    ev = make_sync(
        sync_kind="device",
        start_ns=500,
        end_ns=900,
        stream_id=0,
        device_id=0,
        correlation_id=1,
    )
    blob = ev.model_dump_json()
    back = SyncEvent.model_validate_json(blob)
    assert back == ev


# ── whole-Trace round trip (mixed event union) ────────────────────────────────


def test_trace_round_trip_with_mixed_events():
    """Round-trip a Trace whose ``events`` field is a list of mixed
    KernelEvent / MemcpyEvent / SyncEvent. The discriminated union must
    resolve by the ``kind`` field on the way back in."""
    t = make_trace(
        events=[
            make_kernel(name="qkv_proj", start_ns=1_000, end_ns=2_000),
            make_memcpy(bytes=64, src="host", dst="device", start_ns=2_500, end_ns=2_600),
            make_sync(sync_kind="stream", start_ns=3_000, end_ns=3_100),
            make_kernel(name="attn_score_value", start_ns=4_000, end_ns=4_900),
        ],
        workload_id="vllm-decode",
        vendor="nvidia",
        duration_ns=5_000,
    )
    blob = t.model_dump_json()
    back = Trace.model_validate_json(blob)
    assert back == t
    # Sanity: the typed filter still works after round-trip.
    assert len(back.kernels()) == 2


# ── unicode and very large nanosecond timestamps ──────────────────────────────


def test_trace_round_trip_unicode_workload_id():
    """Unicode in workload_id and kernel name must survive UTF-8 encoding."""
    t = make_trace(
        events=[make_kernel(name="融合カーネル_v1", start_ns=0, end_ns=1)],
        workload_id="vLLM-디코드-本番",
        fingerprint="nvidia:abc123",
        vendor="nvidia",
    )
    blob = t.model_dump_json()
    back = Trace.model_validate_json(blob)
    assert back == t


def test_trace_round_trip_large_nanosecond_timestamps():
    """Timestamps near the int64 ceiling must round-trip without precision
    loss (i.e. must not get float-cast somewhere)."""
    big = 2**62  # well above the float-safe-integer limit of 2**53
    t = make_trace(
        events=[make_kernel(name="late_kernel", start_ns=big, end_ns=big + 100)],
        captured_at_ns=big,
        duration_ns=10**18,
    )
    blob = t.model_dump_json()
    back = Trace.model_validate_json(blob)
    assert back == t
    assert back.events[0].start_ns == big  # explicit — no precision loss


# ── empty traces and schema strictness ────────────────────────────────────────


def test_trace_round_trip_empty_events():
    t = make_trace(events=[], duration_ns=0)
    blob = t.model_dump_json()
    back = Trace.model_validate_json(blob)
    assert back == t
    assert back.events == []


def test_kernel_event_extra_field_rejected():
    """Pydantic's ``extra='forbid'`` must reject unknown fields — silent
    schema drift would otherwise let bad data into the trace pipeline."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KernelEvent.model_validate(
            {
                "kind": "kernel",
                "name": "x",
                "start_ns": 0,
                "end_ns": 1,
                "stream_id": 0,
                "device_id": 0,
                "bogus_new_field": "not allowed",
            }
        )


def test_trace_union_dispatches_by_kind_field():
    """Across the wire, an event dict tagged ``kind='memcpy'`` must
    deserialize as MemcpyEvent — not as the first member of the union."""
    blob = (
        '{"workload_id":"w","fingerprint":"f","run_id":"r","device_count":1,'
        '"vendor":"nvidia","captured_at_ns":0,"duration_ns":100,"events":['
        '{"kind":"memcpy","bytes":8,"src":"host","dst":"device",'
        '"start_ns":1,"end_ns":2,"stream_id":0,"device_id":0,"correlation_id":null}'
        "]}"
    )
    back = Trace.model_validate_json(blob)
    assert len(back.events) == 1
    assert isinstance(back.events[0], MemcpyEvent)
    assert back.events[0].bytes == 8
