"""Collective (NCCL) time as a ranked causal signal."""

from __future__ import annotations

from pathlib import Path

from gitm.importers.torch_trace import import_torch_trace
from gitm.optimizer.collective_signal import collective_causes, worst_device_comm
from gitm.tracer.schema import KernelEvent, Trace

FIXTURES = Path(__file__).parent / "fixtures" / "importers"


def _trace(events: list[KernelEvent], duration_ns: int) -> Trace:
    return Trace(
        workload_id="test",
        fingerprint="fp",
        run_id="r",
        device_count=1,
        vendor="nvidia",
        captured_at_ns=0,
        duration_ns=duration_ns,
        events=events,
    )


def _kernel(name: str, start: int, end: int, *, device: int = 0, stream: int = 0) -> KernelEvent:
    return KernelEvent(
        name=name, start_ns=start, end_ns=end, stream_id=stream, device_id=device
    )


# --------------------------------------------------------------------------- #
# exposed vs overlapped — the distinction the signal exists to make           #
# --------------------------------------------------------------------------- #
def test_exposed_collective_raises_a_cause():
    # AllReduce runs alone: nothing overlaps it, so it is pure exposed cost.
    trace = _trace(
        [
            _kernel("gemm", 0, 1_000_000),
            _kernel("ncclDevKernel_AllReduce_Sum_f32", 1_000_000, 3_000_000),
        ],
        duration_ns=3_000_000,
    )
    causes = collective_causes(worst_device_comm(trace))
    assert any(c.signal == "exposed_collective" for c in causes)


def test_fully_overlapped_collective_raises_no_exposed_cause():
    # Same 2ms AllReduce, but compute runs concurrently on another stream the
    # whole time. That comm is hidden behind useful work and costs nothing —
    # this is the case that proves the overlap math is actually doing something.
    trace = _trace(
        [
            _kernel("gemm", 0, 3_000_000, stream=0),
            _kernel("ncclDevKernel_AllReduce_Sum_f32", 1_000_000, 3_000_000, stream=1),
        ],
        duration_ns=3_000_000,
    )
    stats = worst_device_comm(trace)
    assert stats.comm_ns > 0  # the comm did happen
    assert stats.exposed_comm_ns == 0  # ...but none of it was exposed
    assert not any(c.signal == "exposed_collective" for c in collective_causes(stats))


def test_no_collective_kernels_yields_no_causes():
    trace = _trace([_kernel("gemm", 0, 1_000_000)], duration_ns=1_000_000)
    assert collective_causes(worst_device_comm(trace)) == []


def test_empty_trace_and_none_are_safe():
    assert worst_device_comm(_trace([], duration_ns=0)) is None
    assert collective_causes(None) == []


def test_severity_is_bounded():
    # An entirely-collective trace must saturate at 1.0, not exceed it.
    trace = _trace(
        [_kernel("ncclDevKernel_AllReduce_Sum_f32", 0, 1_000_000)], duration_ns=1_000_000
    )
    causes = collective_causes(worst_device_comm(trace))
    assert causes and all(0.0 <= c.severity <= 1.0 for c in causes)


def test_causes_name_topology_knobs():
    trace = _trace(
        [_kernel("ncclDevKernel_AllReduce_Sum_f32", 0, 1_000_000)], duration_ns=1_000_000
    )
    for c in collective_causes(worst_device_comm(trace)):
        assert "tensor_parallel_size" in c.motivates_knobs


# --------------------------------------------------------------------------- #
# multi-device: a live capture holds every GPU's kernels in one trace          #
# --------------------------------------------------------------------------- #
def test_cross_device_compute_does_not_mask_exposed_comm():
    # dev0 communicates while dev1 computes. Treating this as one flat timeline
    # would let dev1's gemm "cover" dev0's AllReduce and report zero exposed
    # comm. Splitting per device keeps the overlap question inside a GPU.
    trace = _trace(
        [
            _kernel("ncclDevKernel_AllReduce_Sum_f32", 0, 2_000_000, device=0),
            _kernel("gemm", 0, 2_000_000, device=1),
        ],
        duration_ns=2_000_000,
    )
    stats = worst_device_comm(trace)
    assert stats.device_id == 0
    assert stats.exposed_comm_ns == 2_000_000


def test_nccl_fixture_produces_a_collective_cause():
    # The real 4xA100 NCCL fixture, merged back into a single trace the way a
    # live CUPTI capture of a multi-GPU run would present it.
    traces, _ = import_torch_trace(FIXTURES / "synthetic_4xA100_nccl.json", run_id="coll")
    merged_events = [e for t in traces for e in t.events]
    merged = traces[0].model_copy(
        update={
            "events": merged_events,
            "duration_ns": max(t.duration_ns for t in traces),
            "device_count": len(traces),
        }
    )
    stats = worst_device_comm(merged)
    assert stats is not None and stats.comm_ns > 0
    assert collective_causes(stats)
