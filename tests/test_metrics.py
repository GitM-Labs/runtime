"""HFU/MFU/MBU and busy/stall fractions from a synthetic trace"""

from __future__ import annotations

import pytest
from gitm.optimizer.metrics import HardwarePeak, compute_metrics
from gitm.tracer.schema import KernelEvent, MemcpyEvent, Trace

US = 1000 # ns per microsecond
PEAK = HardwarePeak(name="TEST", peak_flops=1e14, peak_bw_bytes_s=1e10)


def _trace() -> Trace: 
    # Two 50us kernels, non-overlapping; one 1MB memcpy; 200us wall.

    return Trace(
        workload_id="vLLM-decode",
        fingerprint="fp",
        run_id="r",
        device_count=1,
        vendor="nVIDIA"
        captured_at_ns=0,
        duration_ns=200 * US,
        events=[
            KernelEvent(start_ns=0, end_ns=50 * US, stream_id=0, device_id=0, name="gemm_a"),
            KernelEvent(start_ns=100 * US, end_ns=150 * US, stream_id=0, device_id=0, name="gemm_b"),
            MemcpyEvent(
                start_ns=60 * US, end_ns=65 * US, stream_id=0, device_id=0,
                bytes=1_000_000, src="host", dst="device",
            ),
        ],
    )


def test_busy_and_stall_fraction():
    m = compute_metrics(_trace(), PEAK)
    assert m.n_kernels == 2
    assert m.busy_fraction == pytest.approx(0.5)  # 100us busy / 200us wall
    assert m.stall_fraction == pytest.approx(0.5)


def test_hfu_mfu_with_flops_model():
    m = compute_metrics(
        _trace(), PEAK, flops_model=lambda k: 1e9, recompute_fraction=0.2
    )
    # 2e9 FLOPs over 200us = 1e13 FLOP/s; / 1e14 peak = 0.1 HFU.
    assert m.hfu == pytest.approx(0.1)
    assert m.mfu == pytest.approx(0.08)  # HFU * (1 - 0.2)


def test_mbu_from_memcpy():
    m = compute_metrics(_trace(), PEAK)
    # 1e6 bytes over 200us = 5e9 B/s; / 1e10 peak = 0.5 MBU.
    assert m.mbu == pytest.approx(0.5)


def test_hfu_none_without_flops_model():
    m = compute_metrics(_trace(), PEAK)
    assert m.hfu is None and m.mfu is None


def test_rejects_bad_recompute_fraction():
    with pytest.raises(ValueError):
        compute_metrics(_trace(), PEAK, recompute_fraction=1.0)






