"""correlate_kernels_to_ranges: kernel -> runtime (by correlation_id) -> NVTX
range (by host-timestamp containment on the runtime record, never the
kernel's own device-clock window). See docs/kernel_identity.md.
"""

from __future__ import annotations

from gitm.distributed.correlate import correlate_kernels_to_ranges, parse_range_name


def _kernel(name: str, corr_id: int, dev_start: int, dev_end: int) -> dict:
    return {
        "kind": "kernel", "name": name, "correlation_id": corr_id,
        "start_ns": dev_start, "end_ns": dev_end,
    }


def _runtime(corr_id: int, thread_id: int, host_start: int, host_end: int) -> dict:
    return {
        "kind": "runtime", "correlation_id": corr_id, "thread_id": thread_id,
        "start_ns": host_start, "end_ns": host_end,
    }


def _marker(name: str, thread_id: int, start: int, end: int) -> dict:
    return {"kind": "marker", "name": name, "thread_id": thread_id, "start_ns": start, "end_ns": end}


def test_parse_range_name():
    assert parse_range_name("L3/qkv_proj") == ("qkv_proj", 3)
    assert parse_range_name("L0/attn_out_proj") == ("attn_out_proj", 0)
    assert parse_range_name("lm_head") == ("lm_head", None)


def test_full_chain_correlates_by_correlation_id_and_host_containment():
    # Device-clock kernel window (100..900) is *outside* the host-clock range
    # window (1000..1100) on purpose -- async execution. Containment must go
    # through the runtime record's host window (1000..1050), not the kernel's.
    records = [
        _marker("L2/qkv_proj", thread_id=1, start=1000, end=1100),
        _runtime(corr_id=7, thread_id=1, host_start=1000, host_end=1050),
        _kernel("ampere_fp16_s16816gemm_128x128", corr_id=7, dev_start=100, dev_end=900),
    ]
    out = correlate_kernels_to_ranges(records)
    assert len(out) == 1
    assert out[0]["range_op"] == "qkv_proj"
    assert out[0]["range_layer"] == 2
    # original kernel fields preserved
    assert out[0]["name"] == "ampere_fp16_s16816gemm_128x128"


def test_no_runtime_record_leaves_range_unset():
    records = [_kernel("mystery_kernel", corr_id=1, dev_start=0, dev_end=10)]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None
    assert out[0]["range_layer"] is None


def test_runtime_outside_every_marker_leaves_range_unset():
    records = [
        _marker("L0/qkv_proj", thread_id=1, start=0, end=100),
        _runtime(corr_id=1, thread_id=1, host_start=200, host_end=250),  # after the range closed
        _kernel("k", corr_id=1, dev_start=200, dev_end=300),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None


def test_different_thread_marker_is_not_matched():
    records = [
        _marker("L0/qkv_proj", thread_id=2, start=0, end=1000),  # different thread
        _runtime(corr_id=1, thread_id=1, host_start=100, host_end=150),
        _kernel("k", corr_id=1, dev_start=100, dev_end=150),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None


def test_nested_ranges_pick_innermost():
    records = [
        _marker("L1/mlp_gate_up", thread_id=1, start=0, end=1000),   # outer
        _marker("silu_and_mul", thread_id=1, start=100, end=200),     # inner, no layer prefix
        _runtime(corr_id=1, thread_id=1, host_start=120, host_end=150),
        _kernel("k", corr_id=1, dev_start=120, dev_end=150),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] == "silu_and_mul"
    assert out[0]["range_layer"] is None


def test_multiple_kernels_preserve_order_and_independent_matches():
    records = [
        _marker("L0/qkv_proj", thread_id=1, start=0, end=100),
        _runtime(corr_id=1, thread_id=1, host_start=10, host_end=20),
        _kernel("first", corr_id=1, dev_start=10, dev_end=20),
        _marker("L0/attn_out_proj", thread_id=1, start=100, end=200),
        _runtime(corr_id=2, thread_id=1, host_start=110, host_end=120),
        _kernel("second", corr_id=2, dev_start=110, dev_end=120),
    ]
    out = correlate_kernels_to_ranges(records)
    assert [o["name"] for o in out] == ["first", "second"]
    assert out[0]["range_op"] == "qkv_proj"
    assert out[1]["range_op"] == "attn_out_proj"


# ── NVTX marker pairing ─────────────────────────────────────────────────────
#
# CUpti_ActivityMarker2 carries one timestamp, so a range arrives as two records
# sharing a marker_id, and the header states the name is NULL on the end. The C
# side stays stateless and emits both halves; pair_markers is where they become
# the {name, start_ns, end_ns, thread_id} range correlate.py documents.


def _start(mid, name, t, thread=7):
    return {"kind": "marker", "name": name, "timestamp_ns": t,
            "marker_id": mid, "marker_flags": 0, "thread_id": thread}


def _end(mid, t, thread=7):
    return {"kind": "marker", "name": "", "timestamp_ns": t,
            "marker_id": mid, "marker_flags": 1, "thread_id": thread}


def test_a_push_and_pop_become_one_range():
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([_start(1, "L0/qkv_proj", 100), _end(1, 120)])
    assert got == [{"kind": "marker", "name": "L0/qkv_proj",
                    "start_ns": 100, "end_ns": 120, "thread_id": 7}]


def test_the_name_comes_from_the_start_half():
    """The end record's name is NULL by CUPTI's contract. Taking the name from
    whichever half arrived last would produce anonymous ranges."""
    from gitm.tracer._cupti_decode import pair_markers

    (range_,) = pair_markers([_start(1, "L3/moe_routed", 10), _end(1, 20)])
    assert range_["name"] == "L3/moe_routed"


def test_an_unclosed_range_is_dropped_not_extended():
    """A capture killed mid-step leaves a start with no end — the common case,
    not a corner one. Closing it at the capture's end would produce a range that
    appears to contain every launch after it and would silently claim them all."""
    from gitm.tracer._cupti_decode import pair_markers

    assert pair_markers([_start(1, "L0/a", 10)]) == []


def test_an_unopened_end_is_dropped():
    from gitm.tracer._cupti_decode import pair_markers

    assert pair_markers([_end(1, 10)]) == []


def test_ranges_pair_within_a_thread_not_across_them():
    """A marker id reused on another thread would otherwise splice two ranges
    into one window spanning both."""
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([
        _start(1, "L0/a", 10, thread=1), _start(1, "L0/b", 12, thread=2),
        _end(1, 20, thread=2), _end(1, 30, thread=1),
    ])
    by_thread = {r["thread_id"]: (r["name"], r["start_ns"], r["end_ns"]) for r in got}
    assert by_thread == {1: ("L0/a", 10, 30), 2: ("L0/b", 12, 20)}


def test_nested_ranges_survive_pairing():
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([
        _start(1, "L0/layer", 0), _start(2, "L0/qkv_proj", 10),
        _end(2, 20), _end(1, 100),
    ])
    assert {(r["name"], r["start_ns"], r["end_ns"]) for r in got} == {
        ("L0/layer", 0, 100), ("L0/qkv_proj", 10, 20),
    }


def test_non_marker_records_pass_through_in_order():
    from gitm.tracer._cupti_decode import pair_markers

    k = {"kind": "kernel", "name": "x", "start_ns": 1, "end_ns": 2}
    rt = {"kind": "runtime", "start_ns": 1, "end_ns": 2, "correlation_id": 9,
          "thread_id": 7}
    assert pair_markers([k, rt]) == [k, rt]


def test_an_anonymous_gemm_resolves_to_layer_and_op_through_the_full_chain():
    """The acceptance criterion for the whole NVTX path.

    `nvjet_sm90_tst_*` is cuBLAS's JIT GEMM family; its name carries no
    projection, so classify_op declines it and always will. Correlation recovers
    the identity that name matching cannot.
    """
    from gitm.optimizer.deviation import classify_op
    from gitm.tracer._cupti_decode import decode_records

    name = "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT"
    assert classify_op(name) is None  # unattributable by name, by construction

    events = decode_records([
        _start(1, "L0/qkv_proj", 100),
        {"kind": "runtime", "start_ns": 105, "end_ns": 108,
         "correlation_id": 42, "thread_id": 7},
        _end(1, 120),
        {"kind": "kernel", "name": name, "start_ns": 200, "end_ns": 900,
         "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": 42},
    ])
    (kernel,) = [e for e in events if e.kind == "kernel"]
    assert (kernel.range_op, kernel.range_layer) == ("qkv_proj", 0)
