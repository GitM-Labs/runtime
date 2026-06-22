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
