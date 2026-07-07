"""Headroom report for the HFT execution path (Task A).

observe -> decide: take the captured kernel times from the seed-42 nsys
profile, treat each candidate intervention in hft_library.yaml as a
counterfactual, and rank them by gitm.optimizer.replay.predict_delta() --
the SAME predictor the optimizer loop uses. Output is a ranked list of >=3
candidate interventions, each with a predicted delta (fractional + events/sec)
tied to the kernels (residual) it covers.

This is the "observe -> decide" half. The "prove" half (apply the #1 lever,
checksum-gate it, measure real before/after) is Task B and needs the pod.

HONESTY: predict_delta = coverage x expected_delta_mean. coverage is real
(measured kernel times). expected_delta_mean is a literature estimate from each
spec's cited source, NOT measured on our workload (see hft_library.yaml header).
So the ranking is defensible but the magnitudes are estimates until Task B.

Run:
    python3 headroom_kernel_rank.py
    python3 headroom_kernel_rank.py --baseline-eps 20700000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.replay import _applies, predict_delta
from gitm.tracer.schema import KernelEvent, Trace

# Real per-kernel times (ns) from the seed-42, 25M-event nsys profile.
# Source: benchmarks/hft/results.md "Top GPU Kernels". These are the aggregate
# kernel sums; the per-shard trace (Task 3 tooling) refines them.
OBSERVED_KERNELS_NS = {
    "zstd::decompression_kernel": int(252.076091 * 1e6),
    "cudf::io::parquet::detail::decode_page_data_generic": int(139.100450 * 1e6),
    "cub::detail::merge_sort::DeviceMergeSortMergeKernel": int(26.555531 * 1e6),
    "cub::detail::merge_sort::DeviceMergeSortBlockSortKernel": int(17.371249 * 1e6),
}

# Default baseline throughput for the events/sec delta (seed-42 sequential 25M).
DEFAULT_BASELINE_EPS = 20_700_000


def build_trace_from_observed() -> Trace:
    """Construct a Trace from the observed kernel sums (serialized on stream 0)."""
    events: list[KernelEvent] = []
    t = 0
    for name, dur in OBSERVED_KERNELS_NS.items():
        events.append(
            KernelEvent(
                kind="kernel", start_ns=t, end_ns=t + dur,
                stream_id=0, device_id=0, name=name,
            )
        )
        t += dur
    return Trace(
        workload_id="hft-lob-replay", fingerprint="seed42-25M", run_id="headroom",
        device_count=1, vendor="nvidia", captured_at_ns=0, duration_ns=t, events=events,
    )


def covered_kernels(spec: InterventionSpec, trace: Trace) -> list[str]:
    return [k.name for k in trace.kernels() if _applies(spec, k.name)]


def main() -> None:
    ap = argparse.ArgumentParser(description="HFT headroom / intervention ranking")
    ap.add_argument(
        "--library", type=Path,
        default=Path(__file__).parent / "hft_library.yaml",
        help="HFT intervention library YAML",
    )
    ap.add_argument(
        "--baseline-eps", type=float, default=DEFAULT_BASELINE_EPS,
        help="baseline events/sec to scale the predicted delta into events/sec",
    )
    args = ap.parse_args()

    trace = build_trace_from_observed()
    total_ms = trace.duration_ns / 1e6

    with open(args.library) as fh:
        raw = yaml.safe_load(fh) or {}
    specs = [InterventionSpec.model_validate(e) for e in raw.get("interventions", [])]

    rows = []
    for spec in specs:
        delta = predict_delta(trace, spec)  # fractional wall-clock improvement
        cov = covered_kernels(spec, trace)
        cov_ns = sum(
            (k.end_ns - k.start_ns) for k in trace.kernels() if k.name in cov
        )
        rows.append({
            "name": spec.name,
            "delta_frac": delta,
            "delta_eps": delta * args.baseline_eps,
            "coverage_pct": 100 * cov_ns / trace.duration_ns,
            "lo": spec.expected_delta_lo,
            "hi": spec.expected_delta_hi,
            "n_kernels": len(cov),
        })

    rows.sort(key=lambda r: r["delta_frac"], reverse=True)

    print(f"HFT headroom report  (trace total = {total_ms:.1f} ms, "
          f"baseline = {args.baseline_eps/1e6:.1f}M events/sec)")
    print("=" * 78)
    print(f"{'rank':<5}{'intervention':<32}{'cover%':>7}{'pred Δ':>9}"
          f"{'Δ events/sec':>14}")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['name']:<32}{r['coverage_pct']:>6.1f}%"
              f"{r['delta_frac']*100:>8.1f}%{r['delta_eps']:>14,.0f}")
    print("-" * 78)

    top = rows[0]
    print(f"\n#1: {top['name']}")
    print(f"    covers {top['coverage_pct']:.1f}% of GPU kernel time "
          f"({top['n_kernels']} kernel type(s))")
    print(f"    predicted speedup: {top['delta_frac']*100:.1f}% "
          f"(source range {top['lo']*100:.0f}-{top['hi']*100:.0f}%)")
    print(f"    predicted gain: +{top['delta_eps']:,.0f} events/sec "
          f"-> {(args.baseline_eps + top['delta_eps'])/1e6:.1f}M events/sec")
    print()
    print("NOTE: predicted deltas use literature expected_delta_mean values")
    print("(cited per spec), scaled by MEASURED kernel coverage. Real deltas")
    print("come from Task B: apply #1 via gitm/optimizer/apply.py behind a")
    print("derived-metric checksum gate, measure before/after on the pod.")


if __name__ == "__main__":
    main()
