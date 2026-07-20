"""correlate_kernels_to_ranges: kernel -> runtime (by correlation_id) -> NVTX
range (by host-timestamp containment on the runtime record, never the
kernel's own device-clock window). See docs/kernel_identity.md.
"""

from __future__ import annotations

from gitm.tracer.nvtx_correlate import correlate_kernels_to_ranges, parse_range_name


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
