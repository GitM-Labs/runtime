"""Measure trace-capture overhead — GITM-017 (W1 target <10%, W2 target <5%).

Runs a workload ``runs`` times with instrumentation and ``runs`` times without,
and reports the mean wall-clock overhead the capture path adds. The workload is
pluggable so this measures the real decode loop on a GPU box; the default is a
synthetic CPU compute loop so the methodology is runnable anywhere (on a
CPU-only host capture is a no-op, so the measured overhead is ~0 — that is the
floor of the method, not the GPU number).

    python -m benchmarks.skeleton.measure_overhead --runs 3 --steps 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from gitm.tracer import capture


def synthetic_workload(steps: int = 100) -> None:
    """A inexpensive, deterministic stand-in for a decode loop (CPU-only)."""
    acc = 0.0
    for i in range(steps):
        # a little arithmetic per "decode step" so timing is non-trivial
        for j in range(1000):
            acc += (i * j) ** 0.5
    return None


def _time(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def measure_overhead(
    workload: Callable[[], None],
    *,
    runs: int = 3,
    trace_dir: Path | None = None,
) -> dict:
    """Return mean baseline/instrumented times and the overhead fraction.

    Each instrumented run wraps ``workload`` in :func:`gitm.tracer.capture`;
    each baseline run does not. Interleaving would be better on noisy hosts, but
    the means over ``runs`` repetitions are what the target is checked against.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    baseline = [_time(workload) for _ in range(runs)]

    def instrumented_once() -> None:
        ctx = (
            capture(trace_dir / "overhead.jsonl", workload_id="overhead")
            if trace_dir is not None
            else nullcontext()
        )
        with ctx:
            workload()

    instrumented = [_time(instrumented_once) for _ in range(runs)]

    base_mean = statistics.fmean(baseline)
    inst_mean = statistics.fmean(instrumented)
    overhead = (inst_mean - base_mean) / base_mean if base_mean else 0.0
    return {
        "runs": runs,
        "baseline_mean_s": base_mean,
        "instrumented_mean_s": inst_mean,
        "overhead_fraction": overhead,
        "baseline_s": baseline,
        "instrumented_s": instrumented,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure trace-capture overhead.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--trace-dir", type=Path, default=None,
                   help="If set, instrumented runs write a real trace here.")
    args = p.parse_args(argv)

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    result = measure_overhead(
        lambda: synthetic_workload(args.steps), runs=args.runs, trace_dir=args.trace_dir
    )
    print(f"baseline    mean: {result['baseline_mean_s'] * 1e3:.2f} ms")
    print(f"instrumented mean: {result['instrumented_mean_s'] * 1e3:.2f} ms")
    print(f"overhead: {result['overhead_fraction'] * 100:.2f}%  "
          f"(W1 target <10%, W2 target <5%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
