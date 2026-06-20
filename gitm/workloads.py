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
    from gitm.benchmarks.hft.optimize import HftFewerScansApplicator, HftStreamingApplicator

    stage = Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/hft/staging/hft"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))
    max_events_env = os.environ.get("GITM_BENCH_MAX_EVENTS")
    max_events = int(max_events_env) if max_events_env else None
    stream = os.environ.get("GITM_BENCH_STREAM", "0") == "1"
    shards_per_batch = int(os.environ.get("GITM_BENCH_SHARDS_PER_BATCH", "30"))
    max_shards_env = os.environ.get("GITM_BENCH_MAX_SHARDS")
    max_shards = int(max_shards_env) if max_shards_env else None

    _ensure_hft_data(stage, seed)
    _kind, dflib, _xp = select_backend()

    # Streaming: process the sharded dataset batch-by-batch so a set too big for
    # one frame still runs end-to-end. The observe runner and the apply+prove A/B
    # each iterate their own fresh batch generator.
    if stream:
        make_batches = _hft_batches_factory(stage, seed, dflib, shards_per_batch, max_shards)

        def run() -> dict[str, Any]:
            total_events = 0
            total_vwap = 0
            for df in make_batches():
                s = run_pipeline(df, dflib)
                total_events += s["events"]
                total_vwap += s["vwap_buckets"]
            return {"events": total_events, "vwap_buckets": total_vwap}

        run.applicator = HftStreamingApplicator(make_batches, dflib, sync=sync_device)
        return run

    df = load_events(stage, seed, dflib, max_events=max_events)

    def run() -> dict[str, Any]:
        return run_pipeline(df, dflib)

    # Carry the rollback-gated intervention prover on the runner so the loop can
    # apply+prove the fewer-scan top-of-book on this exact frame. The A/B runs on
    # the active backend (cuDF on GPU, pandas on a laptop), so the speedup is a
    # real measurement, not a prediction.
    run.applicator = HftFewerScansApplicator(df, dflib, sync=sync_device)
    return run


def _free_gpu_pool() -> None:
    """Release cached GPU blocks so streaming stays memory-bounded. Best-effort."""
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _hft_batches_factory(stage: Path, seed: int, dflib, shards_per_batch: int,
                         max_shards: int | None):
    """Return a zero-arg callable yielding a *fresh* batch generator each call.

    Each batch is ``shards_per_batch`` parquet shards read into one device frame,
    freed before the next batch — so a 1B-event set never holds more than one
    batch resident. A fresh generator per call lets the observe pass and the A/B
    iterate the dataset independently (generators are one-shot)."""
    from gitm.benchmarks.hft.harness import _seed_dir

    def make():
        shards = sorted(_seed_dir(stage, seed).glob("part-*.parquet"))
        if max_shards is not None:
            shards = shards[:max_shards]
        if not shards:
            raise FileNotFoundError(f"no parquet shards for seed {seed} under {stage}")
        for i in range(0, len(shards), shards_per_batch):
            batch = shards[i : i + shards_per_batch]
            df = dflib.read_parquet(batch if len(batch) > 1 else batch[0])
            try:
                yield df
            finally:
                del df
                _free_gpu_pool()

    return make


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


@register("openfold", "alphafold", "af2")
def _openfold_factory(cfg: LoopConfig) -> WorkloadRunner:
    """AlphaFold2 inference via OpenFold (the biotech benchmark), wired for the loop.

    Builds the real OpenFold runner and runs warmup folds *outside* the capture
    window; the returned runner folds ``GITM_BENCH_PROTEINS`` proteins (length
    <= ``GITM_BENCH_MAX_LEN`` that have precomputed MSAs under ``$STAGE/msas``),
    so the trace is the Evoformer + structure-module compute — not MSA load or
    first-call autotune.

    **Inference only** — MSAs must be precomputed; this never runs mmseqs2, so
    the GPU does only the work the GPU is for. No intervention library applies to
    AF2 yet, so the loop emits an honest *measurement* report (the real kernels),
    the AF2 analog of the hft/edge measurement run — it does not fabricate a
    speedup. Weights come from ``OPENFOLD_WEIGHTS``; GPU-only, degrades to a
    no-data report where OpenFold/torch/data are absent.

    Env:
        GITM_BENCH_STAGE     staged dir with proteins_50k.fasta + msas/
        GITM_BENCH_SEED      inference seed (default 42)
        GITM_BENCH_PROTEINS  proteins to fold under capture (default 8)
        GITM_BENCH_MAX_LEN   max residue length (default 384)
        GITM_BENCH_WARMUP    untimed warmup folds before capture (default 2)
    """
    import statistics

    from benchmarks.biotech.fetch import read_fasta
    from benchmarks.biotech.harness import _msa_path, load_openfold_runner

    stage = Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/biotech/staging/biotech"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))
    n_proteins = int(os.environ.get("GITM_BENCH_PROTEINS", "8"))
    max_len = int(os.environ.get("GITM_BENCH_MAX_LEN", "384"))
    warmup = int(os.environ.get("GITM_BENCH_WARMUP", "2"))

    fasta = stage / "proteins_50k.fasta"
    if not fasta.exists():
        raise FileNotFoundError(
            f"missing {fasta} — stage the biotech dataset (proteins_50k.fasta + msas/) "
            "or point GITM_BENCH_STAGE at one."
        )
    # Take the first n proteins that are both short enough AND have precomputed
    # MSAs — robust whether the stage is the full 50k set or a smoke subset (we
    # never compute MSAs here). Early-exit so we don't stat all 50k records, and
    # cache the resolved MSA path so it's looked up once, not again per fold.
    # read_fasta yields records in file order, so the selection is deterministic.
    proteins: list[tuple[Any, Path]] = []
    for r in read_fasta(fasta):
        if len(r.seq) <= max_len and (p := _msa_path(stage, r)) is not None:
            proteins.append((r, p))
            if len(proteins) >= n_proteins:
                break
    if not proteins:
        raise FileNotFoundError(
            f"no proteins (len<={max_len}) with precomputed MSAs under {stage}/msas"
        )

    runner = load_openfold_runner(seed)

    # Warmup outside the capture window: first-call kernel autotune + allocator
    # growth must not pollute the trace.
    for r, msa in proteins[: min(warmup, len(proteins))]:
        runner.predict(r, msa)
    sync_device()

    def run() -> dict[str, Any]:
        plddts: list[float] = []
        for r, msa in proteins:
            out = runner.predict(r, msa)
            if "plddt" in out:
                plddts.append(float(out["plddt"]))
        if not plddts:
            # Every fold omitted plDDT — a silent instrumentation failure (e.g. a
            # runner version mismatch). Surface it rather than report None as ok.
            print("WARNING: no plDDT returned by any fold — runner output contract "
                  "may have changed; structures still folded under the trace.")
        return {
            "structures": len(proteins),
            "median_plddt": statistics.median(plddts) if plddts else None,
        }

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
