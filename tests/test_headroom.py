"""Blind headroom: ceiling distance, stall-class split, already-optimized flag."""

from __future__ import annotations

import pytest

from gitm.optimizer.headroom import build_headroom, render_headroom_md
from gitm.optimizer.metrics import HardwarePeak, compute_metrics
from gitm.tracer.schema import KernelEvent, MemcpyEvent, Trace

US = 1000
PEAK = HardwarePeak(name="H100", peak_flops=1e14, peak_bw_bytes_s=1e10)


def _trace(wall_us: int) -> Trace:
    return Trace(
        workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
        vendor="nvidia", captured_at_ns=0, duration_ns=wall_us * US,
        events=[
            KernelEvent(start_ns=0, end_ns=50 * US, stream_id=0, device_id=0, name="k0"),
            KernelEvent(start_ns=100 * US, end_ns=150 * US, stream_id=0, device_id=0, name="k1"),
            MemcpyEvent(start_ns=60 * US, end_ns=65 * US, stream_id=0, device_id=0,
                        bytes=1_000_000, src="host", dst="device"),
        ],
    )


def test_ceiling_distance_and_gap_split():
    trace = _trace(wall_us=200)  # 200us wall, 100us busy
    metrics = compute_metrics(trace, PEAK)
    r = build_headroom(trace, predicted_floor_s=100e-6, metrics=metrics, workload="vllm-decode")

    # (200 - 100)/200 = 0.5 recoverable
    assert r.ceiling_distance == pytest.approx(0.5)
    assert not r.already_optimized
    # gap classes sum to the ceiling distance
    assert sum(r.gap_by_stall_class.values()) == pytest.approx(0.5, abs=1e-3)
    # idle side now comes from the real stall breakdown, not one lumped "idle_stall".
    assert set(r.gap_by_stall_class) == {
        "sync_wait", "transfer_bound", "launch_latency", "idle",
        "memory_bound", "compute_bound",
    }


def test_already_optimized_flag_when_near_floor():
    trace = _trace(wall_us=200)
    metrics = compute_metrics(trace, PEAK)
    # floor within 5% of observed -> already optimized, nothing to bill
    r = build_headroom(trace, predicted_floor_s=195e-6, metrics=metrics, workload="vllm-decode")
    assert r.already_optimized
    assert "already-optimized" in render_headroom_md(r)


def test_render_contains_key_lines():
    trace = _trace(wall_us=200)
    metrics = compute_metrics(trace, PEAK)
    r = build_headroom(trace, predicted_floor_s=100e-6, metrics=metrics,
                       workload="vllm-decode", sku="NVIDIA H100")
    md = render_headroom_md(r)
    assert "Blind headroom — vllm-decode on NVIDIA H100" in md
    assert "Ceiling distance" in md
