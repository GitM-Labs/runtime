"""Workload runner registry — maps a workload id to an in-process GPU driver.

A *runner* is a zero-arg callable that launches the workload's CUDA work and
returns a small summary dict. ``run_loop`` calls the registered runner *inside*
the tracer's capture window, so ``gitm run --workload <id>`` produces a real
per-kernel trace instead of capturing an empty no-op block.

Factories import their heavy/optional dependencies (cuDF, torch) lazily, so
importing this module is cheap and a CPU-only box degrades cleanly: a factory
that can't build a runner raises, ``run_loop`` catches it, and the empty-trace
guard reports *no-data* rather than fabricating results.

Built-in workloads read their data location from the environment (the same
convention the standalone driver uses):

    GITM_BENCH_STAGE      staged dataset dir (default /workspace/hft/staging/hft)
    GITM_BENCH_SEED       dataset seed (default 42)
    GITM_BENCH_MAX_EVENTS cap events processed (default: all)
    GITM_EDGE_CFG/CKPT    edge: OpenPCDet model yaml + checkpoint .pth
    GITM_EDGE_DATA_ROOT   edge: nuScenes data root
    GITM_EDGE_FRAMES      edge: frames to run (default 500)

For full control (telemetry sinks, streaming, per-run reports) use the
``gitm-run-workload`` driver directly; this registry is the minimal path that
makes the autonomous loop observe a real workload.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gitm.scheduler.loop import LoopConfig

# A runner launches GPU work and returns a summary dict (e.g. {"events": N}).
WorkloadRunner = Callable[[], dict[str, Any]]
# A factory builds a runner from the loop config (loads data, resolves seeds…).
WorkloadFactory = Callable[["LoopConfig"], WorkloadRunner]

_REGISTRY: dict[str, WorkloadFactory] = {}


def register(*names: str) -> Callable[[WorkloadFactory], WorkloadFactory]:
    """Register a factory under one or more workload ids."""

    def deco(fn: WorkloadFactory) -> WorkloadFactory:
        for n in names:
            _REGISTRY[n] = fn
        return fn

    return deco


def get_factory(name: str | None) -> WorkloadFactory | None:
    return _REGISTRY.get(name) if name else None


def registered() -> list[str]:
    return sorted(_REGISTRY)


# --- built-in workloads ------------------------------------------------------


@register("hft", "hft-lob")
def _hft_factory(cfg: LoopConfig) -> WorkloadRunner:
    """cuDF/CuPy LOB-replay pipeline. Data is loaded here (outside capture); the
    returned runner runs only the pipeline so the trace is the compute, not the
    Parquet decode. If no dataset is staged, a small smoke dataset is generated
    once (so ``pip install`` + ``gitm run`` works with no manual data step)."""
    from gitm.benchmarks.hft.harness import load_events, run_pipeline, select_backend

    stage = Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/hft/staging/hft"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))
    max_events_env = os.environ.get("GITM_BENCH_MAX_EVENTS")
    max_events = int(max_events_env) if max_events_env else None

    _ensure_hft_data(stage, seed)

    _kind, dflib, _xp = select_backend()
    df = load_events(stage, seed, dflib, max_events=max_events)

    def run() -> dict[str, Any]:
        return run_pipeline(df, dflib)

    return run


def _ensure_hft_data(stage: Path, seed: int) -> None:
    """Ensure a staged HFT dataset exists for ``seed`` under ``stage``.

    Checks for existing shards first and returns immediately if found — staged
    data (real or previously generated) is never regenerated. Otherwise, unless
    disabled with ``GITM_BENCH_AUTOGEN=0``, generates a smoke dataset
    (``GITM_BENCH_EVENTS`` events, default 200k) into ``hft_smoke_seed<seed>/``.
    """
    from gitm.benchmarks.hft.harness import _seed_dir

    try:
        _seed_dir(stage, seed)  # raises FileNotFoundError if nothing is staged
        return
    except FileNotFoundError:
        pass

    if os.environ.get("GITM_BENCH_AUTOGEN", "1") == "0":
        raise FileNotFoundError(
            f"no staged HFT data for seed {seed} under {stage} and autogen is "
            "disabled (GITM_BENCH_AUTOGEN=0). Stage a dataset or set "
            "GITM_BENCH_STAGE to one."
        )

    from gitm.benchmarks.hft.generate import GenConfig, generate

    events = int(os.environ.get("GITM_BENCH_EVENTS", "200000"))
    out = stage / f"hft_smoke_seed{seed}"
    generate(
        GenConfig(events=events, seed=seed, events_per_file=min(events, 100_000)),
        out,
    )


@register("edge")
def _edge_factory(cfg: LoopConfig) -> WorkloadRunner:
    """nuScenes CenterPoint-PointPillar. Warmup (context init, cudnn autotune)
    runs here, outside the capture window; the runner replays the frames."""
    import random

    import torch

    from gitm.benchmarks.edge.workunit import NuScenesWorkUnit

    cfg_path = Path(os.environ["GITM_EDGE_CFG"])
    ckpt_path = Path(os.environ["GITM_EDGE_CKPT"])
    data_root = os.environ.get("GITM_EDGE_DATA_ROOT", "/workspace/edge/OpenPCDet/data/nuscenes")
    n_frames = int(os.environ.get("GITM_EDGE_FRAMES", "500"))
    max_sweeps = int(os.environ.get("GITM_EDGE_MAX_SWEEPS", "10"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))

    unit = NuScenesWorkUnit.from_checkpoint(
        cfg_path=cfg_path, ckpt_path=ckpt_path, data_root=data_root, max_sweeps=max_sweeps
    )
    rng = random.Random(seed)
    indices = list(range(len(unit)))
    rng.shuffle(indices)
    run_indices = indices[:n_frames]

    for idx in run_indices[:10]:  # warmup outside capture
        unit.run(idx)
    torch.cuda.synchronize()

    def run() -> dict[str, Any]:
        total_dets = 0
        for idx in run_indices:
            total_dets += unit.run(idx).n_detections
        return {"frames": len(run_indices), "detections": total_dets}

    return run


def sync_device() -> None:
    """Block until queued GPU work completes, so all kernels land in the trace
    before capture stops. Best-effort — a no-op without CuPy/torch."""
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
        return
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
