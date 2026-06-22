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


# Per-workload OpenPCDet config + checkpoint defaults under the standard pod
# layout (/workspace/edge/...). GITM_EDGE_CFG / GITM_EDGE_CKPT override either.
_EDGE_MODEL_DEFAULTS = {
    "kitti": (
        "/workspace/edge/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
        "/workspace/edge/data/checkpoints/kitti/pointpillar_7728.pth",
    ),
    "nuscenes": (
        "/workspace/edge/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",
        "/workspace/edge/OpenPCDet/checkpoints/cbgs_pp_centerpoint_nds6070.pth",
    ),
}


def _resolve_model(workload: str) -> tuple[Path, Path]:
    """Resolve (cfg, ckpt) for an edge workload from env overrides + defaults.

    Fails loud with the resolved path if either is missing, so a wrong default
    or a forgotten env var yields an actionable message instead of a KeyError or
    an opaque error deep inside OpenPCDet.
    """
    if workload not in _EDGE_MODEL_DEFAULTS:
        raise KeyError(
            f"no edge model defaults for {workload!r}; "
            f"known: {sorted(_EDGE_MODEL_DEFAULTS)}"
        )
    default_cfg, default_ckpt = _EDGE_MODEL_DEFAULTS[workload]
    cfg_path = Path(os.environ.get("GITM_EDGE_CFG", default_cfg))
    ckpt_path = Path(os.environ.get("GITM_EDGE_CKPT", default_ckpt))
    for env_name, p in (("GITM_EDGE_CFG", cfg_path), ("GITM_EDGE_CKPT", ckpt_path)):
        if not p.exists():
            raise FileNotFoundError(
                f"{workload} workload: {env_name} resolves to {p}, which does not "
                f"exist. Point {env_name} at the OpenPCDet path on this box."
            )
    return cfg_path, ckpt_path


def _positive_int_env(name: str, default: int) -> int:
    """Parse a positive-int env var with a clear error (vs a bare ValueError)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if val < 1:
        raise ValueError(f"{name} must be >= 1, got {val}")
    return val


def _make_edge_run_mode(unit: Any, items: list) -> Callable[[str], dict[str, Any]]:
    """Build the fp32/fp16 ``run_mode`` closure for the edge fp16 A/B.

    Runs ``items`` through ``unit.run`` in fp32, or under fp16 autocast when
    called with ``"fp16"``, and returns a detection summary the A/B gate compares
    (count + sorted confidence scores). Same model + weights either way — only
    the autocast context differs — so the only legitimate difference is fp16
    rounding, which the gate tolerates within ``score_atol``.
    """
    import contextlib

    import torch

    def run_mode(mode: str) -> dict[str, Any]:
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if mode == "fp16"
            else contextlib.nullcontext()
        )
        # unit.run() already syncs per frame (it does .cpu() on the outputs), and
        # optimize_edge's _timed syncs after each rep — so no extra sync here. The
        # gate (detections_equivalent) sorts, so scores stay in collection order.
        n_det = 0
        scores: list[float] = []
        with torch.no_grad(), ctx:
            for it in items:
                res = unit.run(it)
                n_det += res.n_detections
                scores.extend(float(d["score"]) for d in res.detections)
        return {"n_frames": len(items), "n_detections": n_det, "scores": scores}

    return run_mode


def _make_edge_batch_run_mode(
    unit: Any, items: list, batch_size: int
) -> Callable[[str], dict[str, Any]]:
    """Build the serial/batched ``run_mode`` closure for the edge batching A/B.

    ``"serial"`` runs ``items`` one at a time through ``unit.run``; ``"batched"``
    runs them ``batch_size`` at a time through ``unit.run_batch`` (one forward per
    chunk). Same model + weights; batching only changes how many frames share a
    launch, so per-frame detections are equivalent in eval mode and the gate
    tolerates float rounding within ``score_atol``.
    """

    def run_mode(mode: str) -> dict[str, Any]:
        n_det = 0
        scores: list[float] = []
        if mode == "batched":
            for i in range(0, len(items), batch_size):
                for res in unit.run_batch(items[i : i + batch_size]):
                    n_det += res.n_detections
                    scores.extend(float(d["score"]) for d in res.detections)
        else:  # serial baseline
            for it in items:
                res = unit.run(it)
                n_det += res.n_detections
                scores.extend(float(d["score"]) for d in res.detections)
        return {"n_frames": len(items), "n_detections": n_det, "scores": scores}

    return run_mode


def _attach_edge_applicator(run: WorkloadRunner, unit: Any, items: list) -> None:
    """Attach the edge intervention applicator so the loop runs apply→prove.

    The lever defaults to **batching** (GITM_EDGE_INTERVENTION=batching) — the
    right one for the launch-bound edge profile; set GITM_EDGE_INTERVENTION=fp16
    for the precision lever instead. The A/B is capped to a small frame count
    (GITM_EDGE_AB_FRAMES, default 30) so the rollback-gated measure is fast
    regardless of the observe-phase frame count. With no frames there is no A/B,
    so no applicator is attached and the loop falls back to measurement-only
    rather than a noise verdict over an empty comparison.
    """
    if not items:
        return

    ab_n = _positive_int_env("GITM_EDGE_AB_FRAMES", 30)
    ab_items = items[:ab_n]
    lever = os.environ.get("GITM_EDGE_INTERVENTION", "batching").lower()

    if lever == "fp16":
        from gitm.benchmarks.edge.optimize import EdgeFp16Applicator

        run.applicator = EdgeFp16Applicator(
            _make_edge_run_mode(unit, ab_items), sync=sync_device
        )
    elif lever == "batching":  # default — targets the launch-bound bottleneck
        from gitm.benchmarks.edge.optimize import EdgeBatchingApplicator

        batch_size = _positive_int_env("GITM_EDGE_BATCH_SIZE", 4)
        run.applicator = EdgeBatchingApplicator(
            _make_edge_batch_run_mode(unit, ab_items, batch_size),
            batch_size=batch_size,
            sync=sync_device,
        )
    else:
        raise ValueError(
            f"GITM_EDGE_INTERVENTION must be 'batching' or 'fp16', got {lever!r}"
        )


@register("edge", "nuscenes")
def _nuscenes_factory(cfg: LoopConfig) -> WorkloadRunner:
    """nuScenes CenterPoint-PointPillar. Warmup (context init, cudnn autotune)
    runs here, outside the capture window; the runner replays the frames. Carries
    the fp16 applicator so the loop runs the full apply→prove."""
    import random
    import time

    import torch

    from gitm.benchmarks.edge.workunit import NuScenesWorkUnit

    cfg_path, ckpt_path = _resolve_model("nuscenes")
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
        t0 = time.perf_counter()
        for idx in run_indices:
            total_dets += unit.run(idx).n_detections
        elapsed = max(time.perf_counter() - t0, 1e-9)
        return {
            "frames": len(run_indices),
            "detections": total_dets,
            "elapsed_s": elapsed,
            "fps": len(run_indices) / elapsed,
        }

    _attach_edge_applicator(run, unit, run_indices)
    return run


@register("kitti")
def _kitti_factory(cfg: LoopConfig) -> WorkloadRunner:
    """KITTI PointPillars. Unlike nuScenes, the KITTI WorkUnit is iterated by
    .bin file path (no dataset index), so frames come from the canonical velodyne
    enumerator. Carries the fp16 applicator for the full apply→prove loop."""
    import random
    import time

    import torch

    from gitm.benchmarks.kitti.baseline import _load_frame_paths
    from gitm.benchmarks.kitti.workunit import WorkUnit

    cfg_path, ckpt_path = _resolve_model("kitti")
    data_root = Path(
        os.environ.get("GITM_EDGE_DATA_ROOT")
        or os.environ.get("GITM_DATA_ROOT", "/workspace/edge")
    )
    n_frames = int(os.environ.get("GITM_EDGE_FRAMES", "500"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))

    unit = WorkUnit.from_checkpoint(cfg_path=cfg_path, ckpt_path=ckpt_path)
    all_paths = _load_frame_paths(data_root)  # sorted velodyne *.bin; fails loud
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    run_paths = all_paths[:n_frames]

    for p in run_paths[:10]:  # warmup outside capture
        unit.run(p)
    torch.cuda.synchronize()

    def run() -> dict[str, Any]:
        total_dets = 0
        t0 = time.perf_counter()
        for p in run_paths:
            total_dets += unit.run(p).n_detections
        elapsed = max(time.perf_counter() - t0, 1e-9)
        return {
            "frames": len(run_paths),
            "detections": total_dets,
            "elapsed_s": elapsed,
            "fps": len(run_paths) / elapsed,
        }

    _attach_edge_applicator(run, unit, run_paths)
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
    from benchmarks.biotech.optimize import AF2Bf16Applicator

    stage = Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/biotech/staging/biotech"))
    seed = int(os.environ.get("GITM_BENCH_SEED", "42"))
    n_proteins = int(os.environ.get("GITM_BENCH_PROTEINS", "8"))
    max_len = int(os.environ.get("GITM_BENCH_MAX_LEN", "384"))
    warmup = int(os.environ.get("GITM_BENCH_WARMUP", "2"))
    plddt_tol = float(os.environ.get("GITM_BENCH_PLDDT_TOL", "1.5"))

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

    # Carry the rollback-gated bf16 prover so the loop can apply+prove the
    # intervention on the same proteins it observed. measure() re-runs the
    # fp32-vs-bf16 A/B and gates on plDDT-equivalence — a real measured speedup.
    run.applicator = AF2Bf16Applicator(
        stage, seed, n_proteins=n_proteins, max_len=max_len, warmup=warmup, plddt_tol=plddt_tol
    )
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
