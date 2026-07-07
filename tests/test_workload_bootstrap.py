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


def test_edge_workloads_registered():
    """kitti and nuscenes must be resolvable workload ids (README advertises
    them); edge stays as the nuScenes alias for back-compat."""
    from gitm.workloads import get_factory, registered

    names = registered()
    for name in ("kitti", "nuscenes", "edge"):
        assert name in names, f"{name} not registered"
        assert get_factory(name) is not None


def test_resolve_model_fails_loud_on_missing_path(tmp_path: Path, monkeypatch):
    """A missing cfg/ckpt yields an actionable FileNotFoundError naming the env
    var and the resolved path — not a KeyError or an opaque OpenPCDet error."""
    from gitm.workloads import _resolve_model

    monkeypatch.setenv("GITM_EDGE_CFG", str(tmp_path / "nope.yaml"))
    monkeypatch.setenv("GITM_EDGE_CKPT", str(tmp_path / "nope.pth"))
    with pytest.raises(FileNotFoundError, match="GITM_EDGE_CFG"):
        _resolve_model("kitti")


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


def test_vllm_factory_returns_runner_with_live_engine_hooks(monkeypatch):
    """The vLLM factory must return the decode runner, not just build the engine."""
    import sys
    import types

    import gitm.workloads as workloads

    class FakeSamplingParams:
        def __init__(self, *, max_tokens: int, temperature: float):
            self.max_tokens = max_tokens
            self.temperature = temperature

    class FakeLLM:
        instances = []

        def __init__(self, *, model: str, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.calls = 0
            FakeLLM.instances.append(self)

        def generate(self, prompts, params):
            self.calls += 1
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(token_ids=list(range(params.max_tokens)))]
                )
                for _ in prompts
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setattr(workloads, "sync_device", lambda: None)
    monkeypatch.setenv("GITM_VLLM_MODEL", "fake/model")
    monkeypatch.setenv("GITM_VLLM_PROMPTS", "2")
    monkeypatch.setenv("GITM_VLLM_MAX_TOKENS", "4")
    monkeypatch.setenv("GITM_VLLM_GPU_MEM", "0.45")

    runner = workloads.get_factory("vllm-decode")(None)

    assert callable(runner)
    assert runner.engine is FakeLLM.instances[0]
    assert callable(runner.engine.gitm_throughput_fn)
    assert callable(runner.engine.gitm_restart_fn)

    summary = runner()
    assert summary["generated_tokens"] == 8
    assert FakeLLM.instances[0].calls == 1
    assert runner.engine.gitm_throughput_fn(runner.engine) > 0

    restarted = runner.engine.gitm_restart_fn(runner.engine, "swap_space", 2)
    assert restarted.kwargs["gpu_memory_utilization"] == 0.45
    assert restarted.kwargs["swap_space"] == 2
