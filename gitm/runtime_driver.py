"""Run a benchmark workload *inside* the GITM runtime and measure all the detail.

Unlike `make run-<seed>` (which runs the harness as a subprocess and records only
its one-line throughput contract), this driver runs the workload's CUDA work
**in-process** so the runtime can actually observe it:

  * event telemetry  — CUPTI per-kernel trace (gitm.tracer.capture)
  * state telemetry   — 1 Hz NVML GPU samples (gitm.telemetry.Collector, best-effort)
  * deviation monitor — per-kernel residual vs that kernel's median duration,
                        checked against the 3 invariants (multi-basis filter)
  * causal attribution— Granger on the residual subgraph
  * provenance report — markdown summary of everything measured

Currently wired for the HFT cuDF/CuPy workload (the only fully-real harness).
Emits: <outdir>/<wl>_trace.jsonl, _telemetry.jsonl, _measure.json, _report.md.

    gitm-run-workload --workload hft --seed 42 \
        --stage /workspace/hft/staging/hft --max-events 150000000 \
        --outdir /workspace/hft/runs

(installed as the ``gitm-run-workload`` console command; the old
``python scripts/run_under_runtime.py ...`` invocation still works via a shim.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from contextlib import closing
from pathlib import Path

from gitm._timing import require_positive_duration


def _sync():
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception as exc:
        warnings.warn(
            f"CuPy device synchronization unavailable; trace completeness is not "
            f"guaranteed ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )


def _load_hft(stage: Path, seed: int, max_events: int | None):
    from gitm.benchmarks.hft.harness import _gpu_name, load_events, run_pipeline, select_backend
    from gitm.benchmarks.hft.optimize import run_pipeline_fast, verify_equivalent

    kind, dflib, _xp = select_backend()
    gpu_name, device_count = _gpu_name(kind)
    df = load_events(stage, seed, dflib, max_events=max_events)
    n = int(len(df))

    # Realize the gain on the actual run: use the verified-faster pipeline when
    # it is byte-identical on THIS backend/data (one-time check, outside the
    # timed/captured window); otherwise fall back to baseline — never silently
    # wrong. So a normal `gitm-run-workload --workload hft` runs the optimized
    # path, not just the `--optimize` A/B.
    pipeline = run_pipeline
    if verify_equivalent(df, dflib):
        pipeline = run_pipeline_fast
        print("optimization verified identical on this backend → running fewer-scan pipeline")
    else:
        print("optimization NOT identical on this backend → running baseline pipeline")

    def work() -> dict:
        return pipeline(df, dflib)

    return work, n, kind, gpu_name, device_count


def _free_gpu_pool():
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        warnings.warn(
            f"GPU memory-pool cleanup unavailable: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _stream_hft(stage: Path, seed: int, shards_per_batch: int, max_shards: int | None):
    """Stream the full sharded dataset through the pipeline, batch by batch.

    Reads ``shards_per_batch`` parquet shards at a time into one device frame,
    runs the pipeline, accumulates, and frees the GPU pool before the next
    batch — so a 1B-event dataset (200 shards) is processed end-to-end without
    ever materialising more than one batch on the 80GB GPU.
    """
    import pyarrow.parquet as pq

    from gitm.benchmarks.hft.harness import _gpu_name, _seed_dir, run_pipeline, select_backend

    kind, dflib, _xp = select_backend()
    gpu_name, device_count = _gpu_name(kind)
    shards = sorted(_seed_dir(stage, seed).glob("part-*.parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no parquet shards for seed {seed} under {stage}")
    n = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in shards)
    batches = [shards[i : i + shards_per_batch] for i in range(0, len(shards), shards_per_batch)]

    def work() -> dict:
        total_events = 0
        total_vwap = 0
        for bi, batch in enumerate(batches):
            df = dflib.read_parquet(batch if len(batch) > 1 else batch[0])
            s = run_pipeline(df, dflib)
            total_events += s["events"]
            total_vwap += s["vwap_buckets"]
            del df
            _free_gpu_pool()
            print(f"  batch {bi + 1}/{len(batches)}: +{s['events']:,} events "
                  f"(running {total_events:,})", flush=True)
        return {"events": total_events, "vwap_buckets": total_vwap}

    return work, n, kind, gpu_name, device_count, len(shards), len(batches)



def _load_edge(cfg_path: Path, ckpt_path: Path, n_frames: int, seed: int,
               data_root: str, max_sweeps: int):
    """nuScenes CenterPoint-PointPillar via gitm.benchmarks.edge (10-sweep keyframes)."""
    import random

    import torch

    from gitm.benchmarks.edge.workunit import NuScenesWorkUnit

    unit = NuScenesWorkUnit.from_checkpoint(
        cfg_path=cfg_path, ckpt_path=ckpt_path,
        data_root=data_root, max_sweeps=max_sweeps,
    )
    rng = random.Random(seed)
    indices = list(range(len(unit)))
    rng.shuffle(indices)
    run_indices = indices[:n_frames]

    gpu_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()

    # Warmup outside the capture window (context init, cudnn autotune).
    for idx in run_indices[:10]:
        unit.run(idx)
    torch.cuda.synchronize()

    def work() -> dict:
        total_dets = 0
        for i, idx in enumerate(run_indices):
            r = unit.run(idx)
            total_dets += r.n_detections
            if i % 100 == 0:
                print(f"  frame {i}/{len(run_indices)}", flush=True)
        return {"frames": len(run_indices), "detections": total_dets}

    return work, len(run_indices), "torch", gpu_name, device_count


def _stream_ab_batches(stage: Path, seed: int, dflib, shards_per_batch: int,
                       max_shards: int | None):
    """Yield (n_events_total, n_batches, batch_generator) for a streaming A/B.

    Mirrors :func:`_stream_hft`'s sharding, but the generator yields each batch's
    *device frame* (not a summary) so :func:`optimize_hft_streaming` can run both
    pipelines on it. The frame is freed after the consumer finishes the batch.
    """
    from gitm.benchmarks.hft.harness import _seed_dir

    shards = sorted(_seed_dir(stage, seed).glob("part-*.parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no parquet shards for seed {seed} under {stage}")

    import pyarrow.parquet as pq

    n = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in shards)
    batches = [shards[i : i + shards_per_batch] for i in range(0, len(shards), shards_per_batch)]

    def gen():
        for batch in batches:
            df = dflib.read_parquet(batch if len(batch) > 1 else batch[0])
            try:
                yield df
            finally:
                del df
                _free_gpu_pool()

    return n, len(batches), gen()


def _optimize_out_path(args, n_events: int) -> Path:
    """Per-size output filename so a smaller run never clobbers a larger one.

    The old fixed ``hft_seed{seed}_optimize.json`` meant a 200M run and a 400M
    run overwrote each other; encode the event count (and stream mode) instead.
    """
    tag = f"{n_events}ev" + ("_stream" if args.stream else "")
    return args.outdir / f"hft_seed{args.seed}_optimize_{tag}.json"


def _optimize_hft_cmd(args) -> int:
    """Verified baseline-vs-optimized A/B for HFT: measure → apply → prove.

    Single-frame by default; ``--stream`` runs the A/B batch-by-batch so the full
    (e.g. 1B-event) dataset is verified without fitting both pipelines in memory.
    """
    from gitm.benchmarks.hft.harness import _gpu_name, load_events, select_backend
    from gitm.benchmarks.hft.optimize import optimize_hft, optimize_hft_streaming

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/hft/staging/hft"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    kind, dflib, _xp = select_backend()
    gpu_name, device_count = _gpu_name(kind)

    if args.stream:
        n, n_batches, batch_gen = _stream_ab_batches(
            stage, args.seed, dflib, args.shards_per_batch, args.max_shards
        )
        print(f"optimize hft (streaming): {n:,} events on {gpu_name} ({kind}) in "
              f"{n_batches} batches of {args.shards_per_batch} shards — baseline vs candidate")

        def _progress(i, info):
            print(f"  batch {i}/{n_batches}: +{info['events']:,} events "
                  f"(identical so far: {info['identical']})", flush=True)

        # closing() guarantees the batch generator's finally (del df +
        # _free_gpu_pool) runs even if the A/B raises mid-stream, instead of
        # waiting on GC to reclaim the device frame.
        with closing(batch_gen):
            r = optimize_hft_streaming(batch_gen, dflib, sync=_sync, on_batch=_progress)
    else:
        df = load_events(stage, args.seed, dflib, max_events=args.max_events)
        n = int(len(df))
        print(f"optimize hft: {n:,} events on {gpu_name} ({kind}) — measuring baseline vs candidate")
        r = optimize_hft(df, dflib, reps=3, sync=_sync)

    # Filename/JSON reflect events *actually processed* (from the A/B result), not
    # the pre-scan metadata count, so a shard changed between scan and run can't
    # mislabel the output.
    n = int(r.baseline_summary.get("events", n))

    print(f"  baseline : {r.baseline_eps:,.0f} events/s")
    print(f"  candidate: {r.candidate_eps:,.0f} events/s  ({r.speedup:.2f}x)")
    print(f"  identical output: {r.identical}")
    print(f"  VERDICT: {r.verdict}")

    out = _optimize_out_path(args, n)
    out.write_text(json.dumps({
        "gpu_name": gpu_name,
        "device_count": device_count,
        "backend": kind,
        "events": n,
        "streamed": bool(args.stream),
        "baseline_events_per_second": r.baseline_eps,
        "candidate_events_per_second": r.candidate_eps,
        "speedup": r.speedup,
        "identical_output": r.identical,
        "kept": r.kept,
        "verdict": r.verdict,
    }, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a workload under the GITM runtime.")
    ap.add_argument("--workload", default="hft", choices=["hft", "edge"])
    ap.add_argument("--cfg", type=Path, default=None, help="edge: OpenPCDet model yaml")
    ap.add_argument("--ckpt", type=Path, default=None, help="edge: checkpoint .pth")
    ap.add_argument("--frames", type=int, default=500, help="edge: frames to trace")
    ap.add_argument("--data-root", default="/workspace/edge/OpenPCDet/data/nuscenes")
    ap.add_argument("--max-sweeps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", type=Path, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--stream", action="store_true",
                    help="Stream the sharded dataset in batches (for 1B-scale that won't fit one frame).")
    ap.add_argument("--shards-per-batch", type=int, default=30)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("/workspace/hft/runs"))
    ap.add_argument("--optimize", action="store_true",
                    help="hft: run the verified baseline-vs-optimized A/B and report the speedup.")
    args = ap.parse_args(argv)

    if args.optimize:
        if args.workload != "hft":
            raise SystemExit("--optimize currently supports --workload hft")
        return _optimize_hft_cmd(args)

    from gitm import __version__
    from gitm.optimizer.measure import measure_trace, measurement_claims
    from gitm.optimizer.report import Provenance, write_report
    from gitm.tracer import capture

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "/workspace/hft/staging/hft"))
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.workload == "edge":
        if not (args.cfg and args.ckpt):
            raise SystemExit("--workload edge requires --cfg and --ckpt")
        work, n, kind, gpu_name, device_count = _load_edge(
            args.cfg, args.ckpt, args.frames, args.seed,
            args.data_root, args.max_sweeps,
        )
    elif args.stream:
        work, n, kind, gpu_name, device_count, n_shards, n_batches = _stream_hft(
            stage, args.seed, args.shards_per_batch, args.max_shards
        )
        print(f"streaming {n_shards} shards in {n_batches} batches of {args.shards_per_batch}")
    else:
        work, n, kind, gpu_name, device_count = _load_hft(stage, args.seed, args.max_events)

    wl = f"{args.workload}-seed{args.seed}-{n}ev"
    print(f"workload={wl} backend={kind} gpu={gpu_name} x{device_count}")
    if args.workload == "hft":
        print(f"loaded {n:,} events from {stage}")

    trace_path = args.outdir / f"{args.workload}_seed{args.seed}_trace.jsonl"
    tele_path = args.outdir / f"{args.workload}_seed{args.seed}_telemetry.jsonl"

    # State telemetry remains fail-open, but every degradation is carried into
    # the measurement artifact and report.
    tele = None
    telemetry_diagnostics: list[str] = []
    try:
        from gitm.telemetry import Collector, CollectorConfig
        from gitm.telemetry.sinks import build_sink

        tele = Collector(CollectorConfig(interval_s=0.25, sinks=[build_sink(f"jsonl:{tele_path}")]))
    except Exception as exc:
        diagnostic = f"telemetry unavailable: {type(exc).__name__}: {exc}"
        telemetry_diagnostics.append(diagnostic)
        print(f"WARN: {diagnostic}")

    started_ns = time.time_ns()
    if tele:
        tele.start()
    _sync()
    with capture(trace_path, workload_id=wl) as tr:
        t0 = time.perf_counter()
        summary = work()
        _sync()
        elapsed = require_positive_duration(
            time.perf_counter() - t0, context=f"{args.workload} runtime driver"
        )
    if tele:
        tele.stop()
        telemetry_diagnostics.extend(tele.diagnostics)
    ended_ns = time.time_ns()

    units = summary.get("events", summary.get("frames", 0))
    events_per_second = units / elapsed
    if args.workload == "hft":
        print(
            f"events/sec = {events_per_second:,.0f}  "
            f"({summary['events']:,} events in {elapsed:.3f}s, {summary['vwap_buckets']:,} vwap buckets)"
        )
    else:
        print(
            f"frames/sec = {events_per_second:,.2f}  "
            f"({units:,} frames in {elapsed:.3f}s, {summary.get('detections', 0):,} detections)"
        )

    # --- runtime: residuals -> invariants -> attribution --------------------
    kernels = [e for e in tr.events if e.kind == "kernel"]
    memcpys = [e for e in tr.events if e.kind == "memcpy"]
    print(f"captured {len(kernels):,} kernel events, {len(memcpys):,} memcpy events")

    measure = {
        "workload_id": wl,
        "backend": kind,
        "gpu_name": gpu_name,
        "device_count": device_count,
        "events": units,
        "elapsed_s": elapsed,
        "events_per_second": events_per_second,
        **({"vwap_buckets": summary["vwap_buckets"]} if args.workload == "hft"
           else {"detections": summary.get("detections")}),
        "n_kernels": len(kernels),
        "n_memcpy": len(memcpys),
        "trace_path": str(trace_path),
    }

    measured = measure_trace(tr)
    measured.diagnostics.extend(telemetry_diagnostics)
    violations = measured.violations
    top_hyps = measured.top_hypotheses
    sc = measured.serialized_fraction
    print(
        f"serialized_concurrency_fraction = {sc:.3f}  |  "
        f"violations multi-basis={len(violations)}"
    )
    print(f"attribution families (>= 16 samples): {measured.families}")
    print(
        "top Granger hypotheses:",
        [(h.cause_op, h.effect_op, round(h.p_value, 4)) for h in top_hyps] or "none",
    )
    for diagnostic in measured.diagnostics:
        print(f"WARN: {diagnostic}")

    measure["serialized_concurrency_fraction"] = sc
    measure["n_violations"] = len(violations)
    measure["n_invalid_duration"] = measured.n_invalid_duration
    measure["diagnostics"] = measured.diagnostics
    measure["top_hypotheses"] = [
        {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value} for h in top_hyps
    ]

    (args.outdir / f"{args.workload}_seed{args.seed}_measure.json").write_text(
        json.dumps(measure, indent=2) + "\n"
    )

    # --- provenance report ---------------------------------------------------
    # Reuse the canonical helper rather than re-implement it (it resolves the SHA
    # from the cwd, falling back to "unknown" off-repo — fine for provenance).
    from gitm.optimizer.report import _git_sha

    prov = Provenance(
        workload_id=wl,
        fingerprint=f"{gpu_name}/{kind}/{n}ev",
        run_id=tr.run_id,
        git_sha=_git_sha(),
        gitm_version=__version__,
        started_at_ns=started_ns,
        ended_at_ns=ended_ns,
        trace_path=str(trace_path),
    )
    claims = measurement_claims(measured)
    kernel_coverage_available = measured.n_kernels > measured.n_invalid_duration
    if args.workload == "hft":
        run_summary = (
            f"HFT cuDF/CuPy on {gpu_name}: {events_per_second:,.0f} events/s over {n:,} events; "
            f"{len(kernels):,} kernels captured, {len(violations)} invariant deviation(s), "
            f"serialized-concurrency={sc:.3f}. Measurement run — no interventions applied."
        )
    else:
        run_summary = (
            f"nuScenes CenterPoint-PointPillar (10-sweep) on {gpu_name}: "
            f"{events_per_second:,.2f} frames/s over {n:,} frames; "
            f"{len(kernels):,} kernels captured, {len(violations)} invariant deviation(s), "
            f"serialized-concurrency={sc:.3f}. Measurement run — no interventions applied."
        )
    report_md = write_report(
        claims,
        prov,
        qualification_diagnostic=(
            "NO DATA — tracer captured no positive-duration GPU kernels; "
            "workload throughput was measured, "
            "but runtime kernel details were not."
            if not kernel_coverage_available
            else "Measurement-only run: runtime observed the workload; "
            "no intervention library applied."
        ),
        runtime_diagnostics=measured.diagnostics,
        summary=run_summary,
    )
    report_path = args.outdir / f"{args.workload}_seed{args.seed}_report.md"
    report_path.write_text(report_md)
    print(f"\nwrote: {trace_path}\n       {tele_path}\n       {report_path}\n       "
          f"{args.outdir / f'{args.workload}_seed{args.seed}_measure.json'}")
    if not kernel_coverage_available:
        print(
            "FAIL: workload ran, but no positive-duration GPU kernels were captured; "
            "runtime details unavailable."
        )
        return 3
    print("PASS: workload ran under the runtime; kernel details measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
