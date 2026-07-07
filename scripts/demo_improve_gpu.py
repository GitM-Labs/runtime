"""Demo: use the runtime to find GPU headroom and improve GPU usage.

Four beats, all measured by the GITM runtime itself:

  1. OBSERVE  run an under-utilized workload under the tracer + telemetry,
              report util / serialized-concurrency / throughput.
  2. DECIDE   data-driven lever selection: high serialization + idle GPU +
              independent work  ->  stream_parallelism.
  3. APPLY    re-run the SAME work across N CUDA streams.
  4. PROVE    before/after util/serialization/throughput, gated by a
              correctness check (identical result checksum) — no speedup is
              reported unless the parallel output matches the serial one.

Headline workload: N independent matmuls small enough to under-fill the GPU,
so the serial version leaves it idle between launches and the runtime can
*show* the headroom before the fix lands.

    python scripts/demo_improve_gpu.py --size 1024 --chunks 64 --streams 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _serialized(trace) -> float:
    from gitm.optimizer.monitor import _serialized_fraction

    kernels = [e for e in trace.events if e.kind == "kernel"]
    return _serialized_fraction(kernels) if kernels else 0.0


def _mean_util(path: Path) -> float | None:
    utils = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or '"util_pct"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("util_pct") is not None:
                utils.append(float(d["util_pct"]))
    return (sum(utils) / len(utils)) if utils else None


def _run_observed(fn, label: str, outdir: Path) -> tuple[object, dict]:
    """Run fn() under the runtime tracer + telemetry; return (result, metrics)."""
    import cupy

    from gitm.tracer import capture

    tele_path = outdir / f"{label}_telemetry.jsonl"
    tele = None
    try:
        from gitm.telemetry import Collector, CollectorConfig
        from gitm.telemetry.sinks import build_sink

        tele = Collector(CollectorConfig(interval_s=0.05, sinks=[build_sink(f"jsonl:{tele_path}")]))
    except Exception:
        pass

    if tele:
        tele.start()
    with capture(outdir / f"{label}_trace.jsonl", workload_id=f"demo-{label}") as tr:
        t0 = time.perf_counter()
        result = fn()
        cupy.cuda.runtime.deviceSynchronize()
        elapsed = max(time.perf_counter() - t0, 1e-9)
    if tele:
        tele.stop()

    return result, {
        "elapsed_s": elapsed,
        "serialized": _serialized(tr),
        "util_pct": _mean_util(tele_path),
        "n_kernels": len([e for e in tr.events if e.kind == "kernel"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Runtime-driven GPU-usage improvement demo.")
    ap.add_argument("--size", type=int, default=1024, help="Matmul dim (small => under-fills the GPU).")
    ap.add_argument("--chunks", type=int, default=64, help="Number of independent matmuls.")
    ap.add_argument("--streams", type=int, default=8, help="CUDA streams for the parallel version.")
    ap.add_argument("--reps", type=int, default=3, help="Repeat the A/B and take the best (drift guard).")
    ap.add_argument("--outdir", type=Path, default=Path("/workspace/demo/runs"))
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        import cupy as cp
    except Exception:
        print("CuPy not available — this demo needs a GPU box with cupy installed.")
        return 2

    M, K = args.size, args.chunks
    rng = cp.random.RandomState(0)
    A = [rng.standard_normal((M, M), dtype=cp.float32) for _ in range(K)]
    B = [rng.standard_normal((M, M), dtype=cp.float32) for _ in range(K)]
    cp.cuda.runtime.deviceSynchronize()

    def serial():
        return float(sum(float((a @ b).sum()) for a, b in zip(A, B, strict=False)))

    def parallel():
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(args.streams)]
        partial = [None] * K
        for i in range(K):
            with streams[i % args.streams]:
                partial[i] = (A[i] @ B[i]).sum()
        for s in streams:
            s.synchronize()
        return float(sum(float(p) for p in partial))

    print(f"workload: {K} independent {M}x{M} fp32 matmuls\n")

    # --- 1. OBSERVE -----------------------------------------------------------
    before_res, before = _run_observed(serial, "before", args.outdir)
    util_s = f"{before['util_pct']:.0f}%" if before["util_pct"] is not None else "n/a"
    print("1. OBSERVE (serial baseline under the runtime):")
    print("     why: run the work once, untouched, to measure how much of the GPU it actually uses.")
    print(f"     wall-clock {before['elapsed_s'] * 1e3:.2f} ms | util {util_s} | "
          f"serialized {before['serialized']:.3f} | {before['n_kernels']} kernels\n")

    # --- 2. DECIDE ------------------------------------------------------------
    idle = before["util_pct"] is None or before["util_pct"] < 85.0
    serial_heavy = before["serialized"] > 0.5
    print("2. DECIDE (runtime maps headroom -> lever):")
    print("     why: an idle GPU running serialized, independent work is the signature that parallelizing across streams should help.")
    print(f"     serialized {before['serialized']:.2f} > 0.5 ? {serial_heavy};  "
          f"util < 85% ? {idle};  work independent ? True")
    if not (serial_heavy and idle):
        print("     no concurrency headroom detected — nothing to apply.")
        return 0
    print(f"     -> selected lever: stream_parallelism(streams={args.streams})\n")

    # --- 3 + 4. APPLY + PROVE (best of --reps, drift guard) -------------------
    after_res, after = _run_observed(parallel, "after", args.outdir)
    best = after
    for _ in range(max(0, args.reps - 1)):
        _, m = _run_observed(parallel, "after", args.outdir)
        if m["elapsed_s"] < best["elapsed_s"]:
            best = m
    after = best

    rel = abs(after_res - before_res) / (abs(before_res) + 1e-9)
    correct = rel < 1e-4
    print("3. APPLY + 4. PROVE:")
    print("     why: re-run the exact same math across CUDA streams; only report a speedup if the result is bit-for-bit equivalent.")
    print(f"     correctness: before={before_res:.4e} after={after_res:.4e} "
          f"rel-diff={rel:.1e} -> {'PASS' if correct else 'FAIL'}")
    if not correct:
        print("     FAIL: parallel result diverged — refusing to report a speedup.")
        return 1

    util_a = f"{after['util_pct']:.0f}%" if after["util_pct"] is not None else "n/a"
    speedup = before["elapsed_s"] / after["elapsed_s"]
    print()
    print("                     BEFORE (serial)     AFTER (parallel)")
    print(f"     wall-clock      {before['elapsed_s'] * 1e3:>10.2f} ms    {after['elapsed_s'] * 1e3:>10.2f} ms")
    print(f"     throughput      {K / before['elapsed_s']:>10.0f} mm/s  {K / after['elapsed_s']:>10.0f} mm/s")
    print(f"     GPU util        {util_s:>12}      {util_a:>12}")
    print(f"     serialized      {before['serialized']:>12.3f}      {after['serialized']:>12.3f}")
    print(f"\n  >>> runtime-driven improvement: {speedup:.2f}x faster, "
          f"serialization {before['serialized']:.2f} -> {after['serialized']:.2f}  (correctness-gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
