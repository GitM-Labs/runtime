"""Tight unit tests for node rollup interval math — targets mutmut survivors."""

from __future__ import annotations

import pytest

from gitm.importers.node_rollup import (
    SKEW_THRESHOLD,
    _subtract_overlap,
    build_node_rollup,
    device_comm_stats,
    is_comm_kernel,
)
from gitm.tracer.schema import KernelEvent, Trace


def _k(name: str, start: int, end: int, device: int = 0, stream: int = 0) -> KernelEvent:
    return KernelEvent(
        kind="kernel",
        name=name,
        start_ns=start,
        end_ns=end,
        stream_id=stream,
        device_id=device,
    )


def _trace(events: list[KernelEvent], duration: int | None = None) -> Trace:
    if duration is None:
        duration = max((e.end_ns for e in events), default=0)
    return Trace(
        workload_id="w",
        fingerprint="f",
        run_id="r",
        device_count=1,
        vendor="nvidia",
        captured_at_ns=0,
        duration_ns=duration,
        events=list(events),
        source="torch-import",
    )


# ── _subtract_overlap ────────────────────────────────────────────────────────


def test_subtract_no_mask_returns_full_base():
    assert _subtract_overlap([(0, 100), (200, 250)], []) == 150


def test_subtract_empty_base_is_zero():
    assert _subtract_overlap([], [(0, 100)]) == 0


def test_subtract_full_cover():
    assert _subtract_overlap([(0, 100)], [(0, 100)]) == 0
    assert _subtract_overlap([(0, 100)], [(-10, 200)]) == 0


def test_subtract_partial_overlap_middle():
    # base [0,100], mask [25,75] → exposed 25+25=50
    assert _subtract_overlap([(0, 100)], [(25, 75)]) == 50


def test_subtract_partial_overlap_edges():
    # base [0,100], mask [0,30] and [80,100] → exposed 50
    assert _subtract_overlap([(0, 100)], [(0, 30), (80, 100)]) == 50


def test_subtract_disjoint_mask_no_effect():
    assert _subtract_overlap([(0, 50)], [(100, 200)]) == 50


def test_subtract_multiple_base_intervals():
    # [0,10]+[20,30] = 20; mask [5,25] covers 5 of first + 5 of second → exposed 10
    assert _subtract_overlap([(0, 10), (20, 30)], [(5, 25)]) == 10


# ── exposed comm end-to-end ──────────────────────────────────────────────────


def test_device_comm_fully_exposed():
    # one nccl kernel, no overlapping compute
    tr = _trace([_k("ncclDevKernel_AllReduce", 0, 100)], duration=200)
    cs = device_comm_stats(tr)
    assert cs.comm_ns == 100
    assert cs.exposed_comm_ns == 100
    assert cs.comm_share_of_busy == pytest.approx(1.0)
    assert cs.exposed_comm_share_of_wall == pytest.approx(0.5)


def test_device_comm_fully_overlapped():
    # nccl [0,100] fully covered by compute [0,100] on another stream
    tr = _trace(
        [
            _k("ncclAllReduce", 0, 100, stream=0),
            _k("cutlass_gemm", 0, 100, stream=1),
        ],
        duration=100,
    )
    cs = device_comm_stats(tr)
    assert cs.comm_ns == 100
    assert cs.exposed_comm_ns == 0
    assert cs.exposed_comm_share_of_wall == pytest.approx(0.0)


def test_device_comm_half_exposed():
    # nccl [0,100], compute [50,100] → exposed [0,50]
    tr = _trace(
        [
            _k("ncclAllReduce", 0, 100, stream=0),
            _k("cutlass_gemm", 50, 100, stream=1),
        ],
        duration=100,
    )
    cs = device_comm_stats(tr)
    assert cs.exposed_comm_ns == 50


def test_device_comm_no_comm_kernels():
    tr = _trace([_k("cutlass_gemm", 0, 80)], duration=100)
    cs = device_comm_stats(tr)
    assert cs.comm_ns == 0
    assert cs.exposed_comm_ns == 0
    assert cs.comm_share_of_busy == 0.0


# ── rollup skew / collective flags ───────────────────────────────────────────


def test_rollup_skew_and_straggler_flag():
    # device 0 busy 0.9, device 1 busy 0.5 → skew 0.4 > 0.05
    t0 = _trace([_k("gemm", 0, 90, device=0)], duration=100)
    t1 = _trace([_k("gemm", 0, 50, device=1)], duration=100)
    r = build_node_rollup(
        [(t0, 0.9, 0.1), (t1, 0.5, 0.5)],
        multi_device_file=True,
    )
    assert r.skew == pytest.approx(0.4)
    assert r.has_straggler is True
    assert r.skew > SKEW_THRESHOLD


def test_rollup_no_straggler_when_balanced():
    t0 = _trace([_k("gemm", 0, 80, device=0)], duration=100)
    t1 = _trace([_k("gemm", 0, 82, device=1)], duration=100)
    r = build_node_rollup(
        [(t0, 0.80, 0.2), (t1, 0.82, 0.18)],
        multi_device_file=True,
    )
    assert r.skew == pytest.approx(0.02)
    assert r.has_straggler is False


def test_rollup_weighted_ceiling():
    # equal walls → mean; unequal walls → weighted
    t0 = _trace([_k("gemm", 0, 50, device=0)], duration=100)
    t1 = _trace([_k("gemm", 0, 50, device=1)], duration=300)
    r = build_node_rollup(
        [(t0, 0.5, 0.10), (t1, 0.5, 0.40)],
        multi_device_file=True,
    )
    # (0.10*100 + 0.40*300) / 400 = (10+120)/400 = 0.325
    assert r.node_ceiling_distance == pytest.approx(0.325)


def test_rollup_comm_inconclusive():
    t0 = _trace([_k("gemm", 0, 50, device=0)], duration=100)
    t1 = _trace([_k("gemm", 0, 50, device=1)], duration=100)
    r = build_node_rollup([(t0, 0.5, 0.5), (t1, 0.5, 0.5)], multi_device_file=True)
    assert r.has_collective is False
    assert r.comm_inconclusive is True


def test_rollup_has_collective_when_nccl_present():
    t0 = _trace(
        [_k("ncclAllReduce", 0, 20, device=0), _k("gemm", 20, 50, device=0)],
        duration=100,
    )
    t1 = _trace([_k("gemm", 0, 50, device=1)], duration=100)
    r = build_node_rollup([(t0, 0.5, 0.5), (t1, 0.5, 0.5)], multi_device_file=True)
    assert r.has_collective is True
    assert r.comm_inconclusive is False


def test_is_comm_case_insensitive_substrings():
    assert is_comm_kernel("NCCLKERNEL_FOO")
    assert is_comm_kernel("my_AllGather_op")
    assert not is_comm_kernel("")
    assert not is_comm_kernel("attention_kernel")
