"""Edge interventions (fp16 and batching): the apply→prove A/B and its per-frame
correctness gate.

GPU-free. The A/B is driven by an injected fake run_mode, exactly the seam the
real OpenPCDet runner plugs into on the pod. The gate matches detections per
frame by class and 3D center distance, tolerating a small amount of churn (the
rounding-level flips fp16 / batching cause) while still catching a real
divergence.
"""

from __future__ import annotations

import time

import pytest

from gitm.benchmarks.edge.optimize import (
    DetectionDivergenceError,
    EdgeBatchingApplicator,
    EdgeFp16Applicator,
    detections_equivalent,
    edge_batching_spec,
    edge_intervention_spec,
    optimize_edge,
)


def _frames(n_frames: int = 8, n_det: int = 5, *, shift: float = 0.0, drop: int = 0) -> dict:
    """Per-frame summary: each frame has the same boxes, one car per integer x.
    ``shift`` moves every box (to break the center match); ``drop`` removes boxes
    (to simulate detections flipping out)."""
    out = []
    for _ in range(n_frames):
        out.append([
            {"name": "car", "score": 0.9, "center": (float(i) + shift, 0.0, 0.0)}
            for i in range(max(0, n_det - drop))
        ])
    return {"n_frames": n_frames, "frames": out}


def _fake_run_mode(*, equivalent: bool = True, fast_sleep: float = 0.001,
                   slow_sleep: float = 0.012):
    """Baseline (slow) vs candidate (fast). Candidate is the fp16/batched mode.
    When not equivalent, the candidate's boxes are shifted far enough that none
    match by center, which the gate must reject."""
    base = _frames()
    cand = _frames() if equivalent else _frames(shift=10.0)

    def run_mode(mode: str) -> dict:
        if mode in ("fp16", "batched"):       # candidate
            time.sleep(fast_sleep)
            return cand
        time.sleep(slow_sleep)                # baseline
        return base
    return run_mode


def test_detections_equivalent_matches_per_frame():
    base = _frames(n_frames=2, n_det=3)
    # identical boxes → equivalent
    assert detections_equivalent(base, _frames(n_frames=2, n_det=3))
    # one borderline detection flipping out per frame is tolerated
    assert detections_equivalent(base, _frames(n_frames=2, n_det=3, drop=1))
    # every box moved beyond the center tolerance → real divergence, rejected
    assert not detections_equivalent(base, _frames(n_frames=2, n_det=3, shift=10.0))
    # all detections gone → rejected
    assert not detections_equivalent(base, _frames(n_frames=2, n_det=0))
    # frame count mismatch → rejected
    assert not detections_equivalent(base, _frames(n_frames=1, n_det=3))


def test_specs_apply_to_edge_workloads():
    for spec in (edge_intervention_spec(), edge_batching_spec()):
        assert set(spec.applicability.workloads) >= {"edge", "kitti", "nuscenes"}
    assert edge_intervention_spec().name == "edge_fp16_autocast"
    assert edge_batching_spec().name == "edge_frame_batching"


def test_optimize_edge_keeps_faster_equivalent_candidate():
    r = optimize_edge(_fake_run_mode(equivalent=True), reps=1)
    assert r.identical and r.speedup > 1.0 and r.kept == "candidate"
    assert "faster" in r.verdict


def test_fp16_applicator_keeps_when_equivalent_and_rolls_back_when_not():
    keep = EdgeFp16Applicator(_fake_run_mode(equivalent=True), reps=1)
    snap = keep.snapshot()
    keep.apply(keep.spec)
    assert keep.measure(keep.spec) > 0
    assert keep.last_result is not None and keep.last_result.identical
    keep.restore(snap)
    assert keep.active == "fp32"

    drift = EdgeFp16Applicator(_fake_run_mode(equivalent=False), reps=1)
    with pytest.raises(DetectionDivergenceError):
        drift.measure(drift.spec)


def test_batching_applicator_keeps_when_equivalent_and_rolls_back_when_not():
    keep = EdgeBatchingApplicator(_fake_run_mode(equivalent=True), reps=1)
    assert keep.measure(keep.spec) > 0
    assert keep.last_result is not None and keep.last_result.kept == "candidate"

    drift = EdgeBatchingApplicator(_fake_run_mode(equivalent=False), reps=1)
    with pytest.raises(DetectionDivergenceError):
        drift.measure(drift.spec)


def test_batch_run_mode_chunks_non_divisible_length():
    """serial and batched modes must cover every frame, including a trailing
    partial chunk (len=7, batch=4 → chunks of 4 + 3)."""
    from gitm.workloads import _make_edge_batch_run_mode

    class _Res:
        def __init__(self, n):
            self.n_detections = n
            self.detections = [
                {"name": "car", "score": 0.9, "box3d": [float(i), 0, 0, 1, 1, 1, 0]}
                for i in range(n)
            ]

    class _FakeUnit:
        def run(self, it):
            return _Res(2)

        def run_batch(self, items):
            return [_Res(2) for _ in items]

    items = list(range(7))
    rm = _make_edge_batch_run_mode(_FakeUnit(), items, batch_size=4)
    serial, batched = rm("serial"), rm("batched")
    assert serial["n_frames"] == 7 and batched["n_frames"] == 7
    # every frame covered in both modes → 7 frames × 2 dets
    assert sum(len(f) for f in serial["frames"]) == 14
    assert sum(len(f) for f in batched["frames"]) == 14


def test_attach_skips_applicator_when_no_frames():
    """No frames → no A/B → no applicator, so the loop falls back to
    measurement-only instead of a noise verdict over an empty comparison."""
    from gitm.workloads import _attach_edge_applicator

    def run() -> dict:
        return {}

    _attach_edge_applicator(run, object(), [])
    assert not hasattr(run, "applicator")


def test_applicator_runs_through_the_apply_gate():
    """End-to-end through the real rollback gate: equivalent+faster is kept,
    divergent is rolled back."""
    from gitm.optimizer.apply import apply_intervention

    keep = EdgeBatchingApplicator(_fake_run_mode(equivalent=True), reps=1)
    res = apply_intervention(keep.spec, keep, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert res.measured_delta is not None and res.measured_delta > 0

    drift = EdgeBatchingApplicator(_fake_run_mode(equivalent=False), reps=1)
    assert apply_intervention(drift.spec, drift, min_keep_delta=0.0).rolled_back
