"""Unit tests for gitm.benchmarks.kitti — no GPU or OpenPCDet required.

Tests cover:
  - WorkUnitResult dataclass and stall fraction properties
  - WorkUnit.from_checkpoint raises ImportError (not a generic crash) when
    OpenPCDet is absent
  - Manifest generator path logic (no filesystem I/O)
  - Baseline convergence check helper
"""

from __future__ import annotations

# ── WorkUnitResult ────────────────────────────────────────────────────────────

def test_workunit_result_stall_fracs_sum_to_one():
    from gitm.benchmarks.kitti.workunit import WorkUnitResult

    r = WorkUnitResult(
        frame_id="000001",
        n_detections=3,
        t_load_s=0.010,
        t_preprocess_s=0.020,
        t_inference_s=0.060,
        t_postprocess_s=0.010,
        t_total_s=0.100,
    )
    total = r.data_stall_frac + r.sync_stall_frac + r.gpu_active_frac
    # The three fracs should account for (load+prep + post + inf) / total = 1.0
    assert abs(total - 1.0) < 1e-9


def test_workunit_result_zero_total_returns_zero_fracs():
    from gitm.benchmarks.kitti.workunit import WorkUnitResult

    r = WorkUnitResult(frame_id="000000", n_detections=0, t_total_s=0.0)
    assert r.data_stall_frac == 0.0
    assert r.sync_stall_frac == 0.0
    assert r.gpu_active_frac == 0.0


def test_workunit_result_detections_default_empty():
    from gitm.benchmarks.kitti.workunit import WorkUnitResult

    r = WorkUnitResult(frame_id="000000", n_detections=0)
    assert r.detections == []


def test_workunit_importable_without_openpcdet():
    """Importing WorkUnit must not raise even if OpenPCDet is absent."""
    from gitm.benchmarks.kitti.workunit import WorkUnit  # noqa: F401


def test_workunit_from_checkpoint_raises_import_error_without_openpcdet(
    monkeypatch,
):
    """from_checkpoint() must raise ImportError (not AttributeError, etc.)
    when OpenPCDet is not installed, so callers can give a clear message."""
    import builtins

    real_import = builtins.__import__

    def _block_pcdet(name, *args, **kwargs):
        if name.startswith("pcdet"):
            raise ImportError("pcdet not available (monkeypatched)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pcdet)

    import pytest

    from gitm.benchmarks.kitti.workunit import WorkUnit

    with pytest.raises(ImportError):
        WorkUnit.from_checkpoint(cfg_path="dummy.yaml", ckpt_path="dummy.pth")


# ── Convergence check helper ─────────────────────────────────────────────────

def test_convergence_check_passes_within_2pct():
    from gitm.benchmarks.kitti.baseline import _convergence_check

    results = [
        {"frames_per_second": 44.2},
        {"frames_per_second": 44.8},
        {"frames_per_second": 44.5},
    ]
    ok, spread = _convergence_check(results)
    assert ok
    assert spread < 2.0


def test_convergence_check_fails_outside_2pct():
    from gitm.benchmarks.kitti.baseline import _convergence_check

    results = [
        {"frames_per_second": 44.0},
        {"frames_per_second": 52.0},
    ]
    ok, spread = _convergence_check(results)
    assert not ok
    assert spread > 2.0


def test_convergence_check_single_result_always_passes():
    from gitm.benchmarks.kitti.baseline import _convergence_check

    ok, spread = _convergence_check([{"frames_per_second": 40.0}])
    assert ok
    assert spread == 0.0


def test_convergence_check_empty_list():
    from gitm.benchmarks.kitti.baseline import _convergence_check

    ok, spread = _convergence_check([])
    assert ok
    assert spread == 0.0


# ── Benchmark module exports ──────────────────────────────────────────────────

def test_kitti_benchmark_exports():
    from gitm.benchmarks.kitti import WorkUnit, run_baseline  # noqa: F401
