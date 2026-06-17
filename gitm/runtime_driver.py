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
import re
import time
from pathlib import Path


def _sync():
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


# Mangled CUDA kernel names → a small set of stable "families" so the residual
# series feeding Granger are well-populated (a variable = a kernel *type*, not
# every template instantiation). Noise tokens are dropped; the first distinctive
# identifier after the library prefix names the family.
_NOISE = {
    "detail", "kernel", "void", "const", "unsigned", "int", "long", "float",
    "double", "global", "device", "functor", "impl", "internal", "type",
    "types", "common", "native", "operator", "policy", "dispatch", "agent",
}


def _kernel_family(name: str) -> str:
    lib = ("cub" if "cub" in name else "cudf" if "cudf" in name
           else "thrust" if "thrust" in name else "k")
    toks = [t for t in re.findall(r"[a-z][a-z_]{3,}", name) if t not in _NOISE]
    fn = toks[0] if toks else "anon"
    return f"{lib}.{fn}"


def _load_hft(stage: Path, seed: int, max_events: int | None):
    from gitm.benchmarks.hft.harness import _gpu_name, load_events, run_pipeline, select_backend

    kind, dflib, _xp = select_backend()
    gpu_name, device_count = _gpu_name(kind)
    df = load_events(stage, seed, dflib, max_events=max_events)
    n = int(len(df))

    def work() -> dict:
        return run_pipeline(df, dflib)

    return work, n, kind, gpu_name, device_count


def _free_gpu_pool():
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


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
    args = ap.parse_args(argv)

    import numpy as np

    from gitm import __version__
    from gitm.optimizer.attribution import attribute
    from gitm.optimizer.monitor import (
        KernelResidual,
        Residuals,
        _serialized_fraction,
        check_invariants,
    )
    from gitm.optimizer.report import Claim, Provenance, write_report
    from gitm.planner.graph import predict_graph
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

    # State telemetry — best-effort; never block the run on it.
    tele = None
    try:
        from gitm.telemetry import Collector, CollectorConfig
        from gitm.telemetry.sinks import build_sink

        tele = Collector(CollectorConfig(interval_s=0.25, sinks=[build_sink(f"jsonl:{tele_path}")]))
    except Exception as exc:
        print(f"telemetry disabled (best-effort): {exc}")

    started_ns = time.time_ns()
    if tele:
        tele.start()
    _sync()
    with capture(trace_path, workload_id=wl) as tr:
        t0 = time.perf_counter()
        summary = work()
        _sync()
        elapsed = max(time.perf_counter() - t0, 1e-9)
    if tele:
        tele.stop()
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

    violations = []
    top_hyps: list = []
    sc = 0.0
    if kernels:
        sc = _serialized_fraction(kernels)
        by_name: dict[str, list[int]] = {}
        for k in kernels:
            by_name.setdefault(k.name, []).append(k.end_ns - k.start_ns)
        med = {nm: float(np.median(v)) for nm, v in by_name.items()}
        res = Residuals()
        res.serialized_concurrency_fraction = sc
        for k in kernels:
            m = med[k.name] or 1.0
            res.per_kernel.append(
                KernelResidual(op=k.name[:40], layer=None, r_kt=((k.end_ns - k.start_ns) - m) / m, r_mt=None)
            )
        v_mb = check_invariants(res, multi_basis=True)
        v_raw = check_invariants(res, multi_basis=False)
        violations = v_mb
        print(
            f"serialized_concurrency_fraction = {sc:.3f}  |  "
            f"violations multi-basis={len(v_mb)} raw={len(v_raw)} "
            f"(filter dropped {len(v_raw) - len(v_mb)})"
        )
        # Attribution: group kernels into families (see _kernel_family) and keep
        # only families with enough samples, so Granger's per-op residual series
        # are well-formed. attribute() truncates all series to the shortest, so a
        # sparse op would otherwise collapse every pair to too few points.
        from collections import Counter

        fam_of = {nm: _kernel_family(nm) for nm in by_name}
        fam_counts = Counter(fam_of[k.name] for k in kernels)
        MIN_ATTR = 16
        res_attr = Residuals()
        res_attr.serialized_concurrency_fraction = sc
        for k in kernels:
            fam = fam_of[k.name]
            if fam_counts[fam] < MIN_ATTR:
                continue
            m = med[k.name] or 1.0
            res_attr.per_kernel.append(
                KernelResidual(op=fam, layer=None, r_kt=((k.end_ns - k.start_ns) - m) / m, r_mt=None)
            )
        fams = sorted({kr.op for kr in res_attr.per_kernel})
        print(f"attribution families (>= {MIN_ATTR} samples): {fams}")
        ranked = attribute(res_attr, predict_graph())
        top_hyps = ranked.top(5)
        print(
            "top Granger hypotheses:",
            [(h.cause_op, h.effect_op, round(h.p_value, 4)) for h in top_hyps] or "none",
        )

    measure["serialized_concurrency_fraction"] = sc
    measure["n_violations"] = len(violations)
    measure["top_hypotheses"] = [
        {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value} for h in top_hyps
    ]

    (args.outdir / f"{args.workload}_seed{args.seed}_measure.json").write_text(
        json.dumps(measure, indent=2) + "\n"
    )

    # --- provenance report ---------------------------------------------------
    def _git_sha() -> str:
        import subprocess

        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return "unknown"

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
    claims: list[Claim] = []
    for v in violations[:5]:
        ev = (
            f"top hypothesis: {top_hyps[0].cause_op[:30]} -> {top_hyps[0].effect_op[:30]} "
            f"(p={top_hyps[0].p_value:.3g})"
            if top_hyps
            else "no ranked hypothesis"
        )
        claims.append(
            Claim(
                summary=f"{v.invariant} deviation on {v.node_op}",
                residual_invariant=v.invariant,
                residual_value=float(v.residual),
                causal_evidence=ev,
                intervention_name="(none — measurement run)",
                predicted_delta=0.0,
                measured_delta=None,
            )
        )
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
        qualification_diagnostic="Measurement-only run: runtime observed the workload; "
        "no intervention library applied.",
        summary=run_summary,
    )
    report_path = args.outdir / f"{args.workload}_seed{args.seed}_report.md"
    report_path.write_text(report_md)
    print(f"\nwrote: {trace_path}\n       {tele_path}\n       {report_path}\n       "
          f"{args.outdir / f'{args.workload}_seed{args.seed}_measure.json'}")
    print("PASS: workload ran under the runtime; all details measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
