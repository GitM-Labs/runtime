"""Self-bootstrap: the two-command UX (pip install -> gitm run) requires that a
first run stage its own data and build its own shim, without manual steps.

These cover the data side end-to-end (runs on the pandas fallback, no GPU) and
the auto-build *gating* (the actual compile needs a GPU box, so we only assert
it's correctly skipped/disabled here).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_ensure_hft_data_generates_then_reuses(tmp_path: Path, monkeypatch):
    """Missing data is generated once; a second call reuses it (no regen)."""
    from gitm.workloads import _ensure_hft_data

    monkeypatch.setenv("GITM_BENCH_EVENTS", "2000")
    stage = tmp_path / "stage"

    _ensure_hft_data(stage, 42)
    seed_dir = stage / "hft_smoke_seed42"
    shards = sorted(seed_dir.glob("part-*.parquet"))
    assert shards, "smoke data should have been generated"

    mtimes = {p.name: p.stat().st_mtime_ns for p in shards}
    _ensure_hft_data(stage, 42)  # data exists now → must not regenerate
    after = {p.name: p.stat().st_mtime_ns for p in seed_dir.glob("part-*.parquet")}
    assert after == mtimes, "existing data must not be regenerated"


def test_ensure_hft_data_respects_autogen_disable(tmp_path: Path, monkeypatch):
    from gitm.workloads import _ensure_hft_data

    monkeypatch.setenv("GITM_BENCH_AUTOGEN", "0")
    with pytest.raises(FileNotFoundError):
        _ensure_hft_data(tmp_path / "empty", 42)


def test_hft_factory_autostages_and_runs(tmp_path: Path, monkeypatch):
    """The registered hft factory stages data and returns a runnable pipeline
    (pandas fallback path — exercises wiring without a GPU)."""
    monkeypatch.setenv("GITM_BENCH_STAGE", str(tmp_path / "stage"))
    monkeypatch.setenv("GITM_BENCH_SEED", "42")
    monkeypatch.setenv("GITM_BENCH_EVENTS", "2000")

    from gitm.workloads import get_factory

    runner = get_factory("hft-lob")(None)  # cfg is unused by this factory
    summary = runner()
    assert summary["events"] > 0


def test_autobuild_skipped_without_gpu(monkeypatch):
    import gitm.tracer._cupti as c

    monkeypatch.setattr(c, "_BUILD_ATTEMPTED", False)
    monkeypatch.setattr(c.shutil, "which", lambda _name: None)  # no nvidia-smi
    assert c._maybe_autobuild() is False


def test_autobuild_disabled_by_env(monkeypatch):
    import gitm.tracer._cupti as c

    monkeypatch.setattr(c, "_BUILD_ATTEMPTED", False)
    monkeypatch.setenv("GITM_AUTOBUILD_CUPTI", "0")
    assert c._maybe_autobuild() is False


def test_generate_importable_from_package():
    """Generator must ship in the wheel so a pip install can auto-stage."""
    from gitm.benchmarks.hft import generate

    assert generate.generate and generate.GenConfig
