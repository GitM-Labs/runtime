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
import socket
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


def _free_port() -> int:
    """An OS-assigned free TCP port. Used to give a restart-A/B candidate engine a
    distinct distributed init port so its ``tcp://…:PORT`` bind doesn't collide
    with the still-alive baseline engine (a two-engines-one-process hazard on V1).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


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


def _frame_dets(res: Any) -> list[dict[str, Any]]:
    """One frame's detections in the gate's shape: name + score + 3D center.

    The gate matches boxes per frame by class and center distance, so it needs
    the center (first three of the 7-number box) and the class name per box, not
    just the score.
    """
    return [
        {"name": d["name"], "score": float(d["score"]), "center": tuple(d["box3d"][:3])}
        for d in res.detections
    ]


def _make_edge_run_mode(unit: Any, items: list) -> Callable[[str], dict[str, Any]]:
    """Build the fp32/fp16 ``run_mode`` closure for the edge fp16 A/B.

    Runs ``items`` through ``unit.run`` in fp32, or under fp16 autocast when
    called with ``"fp16"``, and returns a per-frame detection summary the gate
    compares. Same model + weights either way; only the autocast context differs.
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
        # optimize_edge's _timed syncs after each rep — so no extra sync here.
        frames: list[list[dict[str, Any]]] = []
        with torch.no_grad(), ctx:
            for it in items:
                frames.append(_frame_dets(unit.run(it)))
        return {"n_frames": len(items), "frames": frames}

    return run_mode


def _make_edge_batch_run_mode(
    unit: Any, items: list, batch_size: int
) -> Callable[[str], dict[str, Any]]:
    """Build the serial/batched ``run_mode`` closure for the edge batching A/B.

    ``"serial"`` runs ``items`` one at a time through ``unit.run``; ``"batched"``
    runs them ``batch_size`` at a time through ``unit.run_batch`` (one forward per
    chunk). Same model + weights; batching only changes how many frames share a
    launch, so per-frame detections are equivalent in eval mode. Returns the
    per-frame detection summary the gate compares (frames stay in item order).
    """

    def run_mode(mode: str) -> dict[str, Any]:
        frames: list[list[dict[str, Any]]] = []
        if mode == "batched":
            for i in range(0, len(items), batch_size):
                for res in unit.run_batch(items[i : i + batch_size]):
                    frames.append(_frame_dets(res))
        else:  # serial baseline
            for it in items:
                frames.append(_frame_dets(unit.run(it)))
        return {"n_frames": len(items), "frames": frames}

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

@register("vllm-decode")
def _vllm_decode_factory(cfg: LoopConfig) -> WorkloadRunner:
    """Launch a vLLM decode job inside the tracer capture window.

    The heavy engine build (weight load, CUDA graph capture) happens here in the
    factory, outside the capture window. The returned runner only issues the
    decode ``llm.generate``, so the trace is the decode compute, not the model load.

    Reads its config from the environment:
        GITM_VLLM_MODEL       HF model id (default facebook/opt-125m)
        GITM_VLLM_PROMPTS     number of prompts to decode (default 64)
        GITM_VLLM_MAX_TOKENS  tokens to generate per prompt (default 128)
        GITM_VLLM_SYNTHETIC   "1" -> CPU-only decode stand-in instead of vLLM
                              (exercises the wire/registry path with no GPU or
                              vLLM; produces no GPU kernels)
        GITM_VLLM_ENFORCE_EAGER "1" -> build with enforce_eager (no CUDA graphs).
                              Set it when the CUPTI tracer records no kernels
                              because decode runs via CUDA-graph replay that the
                              platform's CUPTI doesn't attribute; also skips
                              torch.compile + graph capture, so engine builds
                              (and restart-A/B candidates) are much faster.

    On a box without vLLM/GPU the import raises; ``run_loop`` catches it and the
    empty-trace guard reports "no-data" rather than fabricating a result.
    """
    model = os.environ.get("GITM_VLLM_MODEL", "facebook/opt-125m")
    n_prompts = int(os.environ.get("GITM_VLLM_PROMPTS", "64"))
    max_tokens = int(os.environ.get("GITM_VLLM_MAX_TOKENS", "128"))

    if os.environ.get("GITM_VLLM_SYNTHETIC") == "1":
        return _vllm_synthetic_runner(n_prompts, max_tokens)

    import time

    from vllm import LLM, SamplingParams

    # Optional GPU-memory cap (GITM_VLLM_GPU_MEM, e.g. 0.45). Structural restart
    # A/Bs inherit this cap; in parallel mode baseline+candidate must both fit,
    # while GITM_RESTART_MODE=serial releases the baseline before candidate build.
    _gpu_mem = os.environ.get("GITM_VLLM_GPU_MEM")
    _base_kwargs: dict[str, Any] = {}
    if _gpu_mem is not None:
        _base_kwargs["gpu_memory_utilization"] = float(_gpu_mem)
    # Disable CUDA graphs so CUPTI captures decode kernels on platforms where
    # graph-replayed kernels aren't attributed (and speed up every engine build).
    # Inherited by restart candidates (kwargs = dict(_base_kwargs)) for a fair A/B.
    if os.environ.get("GITM_VLLM_ENFORCE_EAGER") == "1":
        _base_kwargs["enforce_eager"] = True

    engine_ref: dict[str, Any] = {}

    def _shutdown_engine(engine: Any) -> None:
        if engine_ref.get("engine") is engine:
            engine_ref["engine"] = None
        try:
            if getattr(run, "engine", None) is engine:
                run.engine = None
        except NameError:
            pass

        for path in (
            "shutdown",
            "llm_engine.shutdown",
            "llm_engine.engine_core.shutdown",
            "llm_engine.engine_core.engine_core.shutdown",
            "llm_engine.model_executor.shutdown",
        ):
            obj: Any = engine
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if callable(obj):
                try:
                    obj()
                except Exception:
                    pass
                break

        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        for attr in ("llm_engine", "engine"):
            try:
                delattr(engine, attr)
            except Exception:
                pass

        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _activate_engine(engine: Any) -> None:
        engine_ref["engine"] = engine
        try:
            run.engine = engine
        except NameError:
            pass

    def _build_engine(kwargs: dict[str, Any]) -> Any:
        engine = LLM(model=model, **kwargs)
        engine.gitm_llm_kwargs = dict(kwargs)
        engine.gitm_shutdown_fn = _shutdown_engine
        engine.gitm_activate_fn = _activate_engine
        return engine

    llm = _build_engine(dict(_base_kwargs))
    _activate_engine(llm)
    prompts = [f"Benchmark decode prompt {i}." for i in range(n_prompts)]
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    def run() -> dict[str, Any]:
        active = engine_ref.get("engine")
        if active is None:
            raise RuntimeError("vLLM engine is not active")
        outputs = active.generate(prompts, params)
        produced = sum(len(o.outputs[0].token_ids) for o in outputs)
        sync_device()
        return {"prompts": len(prompts), "generated_tokens": produced, "model": model}

    def _throughput(eng: Any) -> float:
        """Decode-throughput probe (tokens/sec) for the Phase-4 A/B.

        Runs on WHATEVER engine it is handed — the original for hot-swap knobs, or
        a restarted engine for structural knobs — so a rebuilt engine is measured
        on itself, not on the original. (The loop's default probe re-runs the
        original runner, which cannot measure a restarted engine.)
        """
        t0 = time.perf_counter()
        outs = eng.generate(prompts, params)
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        sync_device()
        return toks / max(time.perf_counter() - t0, 1e-9)

    def _restart(_old_engine: Any, knob_values: dict[str, Any]) -> Any:
        """Rebuild a fresh vLLM engine with one or more structural knobs changed.

        The restart-apply path for structural levers that can't be hot-swapped on a
        running engine — most importantly ``kv_cache_dtype=fp8`` and
        ``quantization``, the real throughput/memory levers for decode. Structural
        knob names match the vLLM ``LLM`` kwargs (``kv_cache_dtype``,
        ``gpu_memory_utilization``, ``swap_space``, ``block_size``,
        ``quantization``, …), so the change is a kwargs update — one entry for a
        single-knob candidate, more for a joint one (e.g. a prerequisite flag
        turned on together with the knob it gates). Returns the new engine;
        ``LiveEngineApplicator`` swaps it in for the A/B and tears it down on
        restore. Raises on an unsupported knob/value (no fp8 support on this SKU,
        missing quantized checkpoint) → the candidate is rolled back cleanly,
        never a silent no-op.
        """
        kwargs = dict(getattr(_old_engine, "gitm_llm_kwargs", _base_kwargs))
        kwargs.update(knob_values)
        # Give each restarted engine a fresh distributed port so V1 init does not
        # collide with any prior in-process engine state.
        os.environ["VLLM_PORT"] = str(_free_port())
        try:
            return _build_engine(kwargs)
        except Exception as exc:
            # Diagnose the spawn/entrypoint failure specifically — it presents as a
            # child dying with FileNotFoundError on '<stdin>' or '-c'. Once CUDA is
            # initialized in the parent, vLLM must build this second engine with
            # 'spawn', and spawn re-imports __main__ in the child — which only works
            # if __main__ is an importable file guarded by `if __name__ ==
            # "__main__"`. gitm keeps the parent CUDA-free (see run()/_throughput) so
            # fork is used instead, but a caller that initializes CUDA in the parent
            # reintroduces the spawn path.
            knobs = ", ".join(knob_values)
            text = str(exc)
            if "<stdin>" in text or "spawn" in text.lower() or "run_path" in text:
                raise RuntimeError(
                    f"restart candidate for {knobs} could not build its second "
                    f"engine under the 'spawn' start method: the entrypoint is not "
                    f"spawn-safe. Run gitm from an importable script guarded by "
                    f"`if __name__ == \"__main__\":` (not `python -c` or a stdin "
                    f"heredoc), or avoid initializing CUDA in the parent process. "
                    f"Root cause: {exc}"
                ) from exc
            raise RuntimeError(
                f"restart candidate for {knobs} failed to build: {exc}"
            ) from exc

    def _baseline_restart(old_engine: Any) -> Any:
        kwargs = dict(getattr(old_engine, "gitm_llm_kwargs", _base_kwargs))
        return _build_engine(kwargs)

    # Expose the live engine + its A/B hooks so the loop can (a) sample scheduler
    # stats and (b) run the Phase-4 decode-throughput A/B on it. ``run.engine`` is
    # picked up as ``cfg.engine``; the loop reads ``gitm_throughput_fn`` /
    # ``gitm_restart_fn`` off the engine (gitm.scheduler.loop Phase 4). The restart
    # hook is what lets structural knobs (fp8 KV cache, quantization) be *measured*
    # via an engine rebuild instead of rejected; ``gitm_baseline_restart_fn``
    # rebuilds the baseline for ``restart_mode="serial"``. Mirrors the
    # ``.applicator`` convention the hft/edge/openfold factories use.
    run.engine = llm
    run.workload_id = "vllm-decode"
    llm.gitm_throughput_fn = _throughput
    llm.gitm_restart_fn = _restart
    llm.gitm_baseline_restart_fn = _baseline_restart
    return run

def _vllm_synthetic_runner(n_prompts: int, max_tokens: int) -> WorkloadRunner:
    """A CPU-only stand-in for the decode loop (no vLLM, no GPU).

    Exercises the registry → runner → capture path so the end-to-end wire can be
    tested without GPU hardware. It does real CPU work (small matmuls per "token")
    so the runner isn't an instant no-op, but emits no GPU kernels — the loop's
    empty-trace guard then reports no-data, which is the honest outcome here.
    """
    import numpy as np

    def run() -> dict[str, Any]:
        steps = 0
        a = np.ones((32, 32), dtype=np.float32)
        for _ in range(max_tokens):
            # A real (small) matmul per "token" so the runner isn't an instant
            # no-op; the product is discarded — we only want CPU cycles, and `a`
            # stays bounded (all-ones) rather than overflowing across iterations.
            _ = a @ a
            steps += 1
        return {"prompts": n_prompts, "decode_steps": steps, "synthetic": True}

    return run

def set_decode_run_defaults() -> dict[str, str]:
    """Populate the env a traced, KV-pressured vllm-decode run needs.
    So ``python scripts/fp8_ab.py`` Just Works with zero manual exports. Every value
    uses ``setdefault``, so anything you already exported wins — override any single
    knob without re-listing the rest.
    MUST be called before the engine is built (before CUDA init): the driver reads
    ``CUDA_INJECTION64_PATH`` at CUDA init, which is when the child EngineCore comes
    up, and the child inherits this process's environment. Returns the resolved
    values for logging.
    The defaults encode the two things this experiment gets wrong by accident:
    ``GPU_MEM=0.45`` so the baseline and the restart candidate both fit at once, and
    512x2048 so the ~151k-token KV cache is ~8x oversubscribed — without that,
    fp8 measures pure noise.
    """
    from gitm.tracer.injection import lib_path

    defaults = {
        "CUDA_INJECTION64_PATH": str(lib_path()),
        "GITM_TRACE_OUT": "/root/.cache/gitm/traces/vllm.jsonl",
        "GITM_VLLM_MODEL": "NousResearch/Meta-Llama-3-8B",
        "GITM_VLLM_GPU_MEM": "0.45",
        "GITM_VLLM_PROMPTS": "512",
        "GITM_VLLM_MAX_TOKENS": "2048",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    Path(os.environ["GITM_TRACE_OUT"]).parent.mkdir(parents=True, exist_ok=True)
    return {k: os.environ[k] for k in defaults}

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
