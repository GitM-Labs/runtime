"""Edge fp16 intervention: the apply→prove A/B and its correctness gate.

GPU-free — the A/B is driven by an injected fake run_mode, exactly the seam the
real OpenPCDet runner plugs into on the pod. Mirrors tests/test_hft_intervention
in spirit: prove the gate keeps a faster-and-equivalent candidate and rolls back
a divergent one.
"""

from __future__ import annotations

import time

import pytest

from gitm.benchmarks.edge.optimize import (
    DetectionDivergenceError,
    EdgeFp16Applicator,
    detections_equivalent,
    edge_intervention_spec,
    optimize_edge,
)


def _fake_run_mode(*, fp16_count: int, fp32_count: int = 5,
                   fp16_sleep: float = 0.001, fp32_sleep: float = 0.012):
    """fp32 is the slow baseline; fp16 is faster. fp16_count controls whether the
    candidate's detections stay equivalent (== fp32_count) or diverge."""
    def run_mode(mode: str) -> dict:
        if mode == "fp16":
            time.sleep(fp16_sleep)
            n = fp16_count
        else:
            time.sleep(fp32_sleep)
            n = fp32_count
        return {"n_frames": 8, "n_detections": n, "scores": [0.9] * n}
    return run_mode


def test_detections_equivalent_gate():
    base = {"n_detections": 3, "scores": [0.9, 0.8, 0.7]}
    assert detections_equivalent(base, {"n_detections": 3, "scores": [0.91, 0.79, 0.71]})
    # count mismatch → not equivalent
    assert not detections_equivalent(base, {"n_detections": 2, "scores": [0.9, 0.8]})
    # score drift beyond tolerance → not equivalent
    assert not detections_equivalent(base, {"n_detections": 3, "scores": [0.5, 0.8, 0.7]})


def test_spec_applies_to_edge_workloads():
    spec = edge_intervention_spec()
    assert spec.name == "edge_fp16_autocast"
    assert set(spec.applicability.workloads) >= {"edge", "kitti", "nuscenes"}


def test_optimize_edge_keeps_faster_equivalent_candidate():
    rm = _fake_run_mode(fp16_count=5)  # equivalent to fp32 + faster
    r = optimize_edge(rm, reps=1)
    assert r.identical
    assert r.speedup > 1.0
    assert r.kept == "candidate"
    assert "faster" in r.verdict


def test_applicator_measure_returns_positive_delta_when_equivalent():
    app = EdgeFp16Applicator(_fake_run_mode(fp16_count=5), reps=1)
    snap = app.snapshot()
    app.apply(app.spec)
    delta = app.measure(app.spec)
    assert delta > 0
    assert app.last_result is not None and app.last_result.identical
    app.restore(snap)
    assert app.active == "fp32"


def test_applicator_rolls_back_on_detection_divergence():
    # fp16 drops a detection → gate must refuse the speedup
    app = EdgeFp16Applicator(_fake_run_mode(fp16_count=4), reps=1)
    with pytest.raises(DetectionDivergenceError):
        app.measure(app.spec)


def _fake_batch_run_mode(*, batched_count: int, serial_count: int = 5,
                         batched_sleep: float = 0.002, serial_sleep: float = 0.012):
    """serial is the slow baseline; batched is faster (fewer launches). batched_count
    controls whether batched detections stay equivalent (== serial_count) or diverge."""
    def run_mode(mode: str) -> dict:
        if mode == "batched":
            time.sleep(batched_sleep)
            n = batched_count
        else:
            time.sleep(serial_sleep)
            n = serial_count
        return {"n_frames": 8, "n_detections": n, "scores": [0.9] * n}
    return run_mode


def test_batching_spec_applies_to_edge_workloads():
    from gitm.benchmarks.edge.optimize import edge_batching_spec

    spec = edge_batching_spec()
    assert spec.name == "edge_frame_batching"
    assert set(spec.applicability.workloads) >= {"edge", "kitti", "nuscenes"}


def test_batching_applicator_keeps_faster_equivalent():
    from gitm.benchmarks.edge.optimize import EdgeBatchingApplicator

    app = EdgeBatchingApplicator(_fake_batch_run_mode(batched_count=5), reps=1)
    delta = app.measure(app.spec)
    assert delta > 0
    assert app.last_result is not None and app.last_result.identical
    assert app.last_result.kept == "candidate"


def test_batching_applicator_rolls_back_on_divergence():
    from gitm.benchmarks.edge.optimize import DetectionDivergenceError, EdgeBatchingApplicator

    app = EdgeBatchingApplicator(_fake_batch_run_mode(batched_count=4), reps=1)
    with pytest.raises(DetectionDivergenceError):
        app.measure(app.spec)


def test_batch_run_mode_chunks_non_divisible_length():
    """The serial and batched modes must cover every frame, including a trailing
    partial chunk (len=7, batch=4 → chunks of 4 + 3)."""
    from gitm.workloads import _make_edge_batch_run_mode

    class _Det(dict):
        pass

    class _Res:
        def __init__(self, n):
            self.n_detections = n
            self.detections = [{"score": 0.9} for _ in range(n)]

    class _FakeUnit:
        def run(self, it):
            return _Res(2)

        def run_batch(self, items):  # one det per frame fewer than serial would be a bug
            return [_Res(2) for _ in items]

    items = list(range(7))
    rm = _make_edge_batch_run_mode(_FakeUnit(), items, batch_size=4)
    serial = rm("serial")
    batched = rm("batched")
    assert serial["n_frames"] == 7 and batched["n_frames"] == 7
    # every frame covered in both modes → 7 frames × 2 dets
    assert serial["n_detections"] == 14
    assert batched["n_detections"] == 14


def test_attach_skips_applicator_when_no_frames():
    """No frames → no A/B → no applicator, so the loop falls back to
    measurement-only instead of a noise verdict over an empty comparison."""
    from gitm.workloads import _attach_edge_applicator

    def run() -> dict:
        return {}

    _attach_edge_applicator(run, object(), [])
    assert not hasattr(run, "applicator")


def test_applicator_runs_through_the_apply_gate():
    """End-to-end through the real rollback gate: equivalent+faster is kept."""
    from gitm.optimizer.apply import apply_intervention

    app = EdgeFp16Applicator(_fake_run_mode(fp16_count=5), reps=1)
    res = apply_intervention(app.spec, app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert res.measured_delta is not None and res.measured_delta > 0

    # divergent candidate → gate rolls back, no speedup kept
    app2 = EdgeFp16Applicator(_fake_run_mode(fp16_count=4), reps=1)
    res2 = apply_intervention(app2.spec, app2, min_keep_delta=0.0)
    assert res2.rolled_back
