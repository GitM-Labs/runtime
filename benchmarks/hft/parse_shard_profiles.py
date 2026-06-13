"""Parse 5 per-shard nsys profiles -> residual series -> Granger (Task 3).

Reads the cuda_gpu_kern_sum CSVs produced by profile_per_shard.sh, maps the
three HFT kernels (zstd / parquet decode / merge sort) to predicted-graph
ops, computes per-shard r_kt residuals against hft_graph, then feeds the
series into gitm.optimizer.attribution.attribute() for the real Granger
F-test.

With 5 shards we get 5 samples/op, which clears attribute()'s max_lag+2=4
minimum. The output ranks which stage Granger-causes the others' slowdown.

Run on the pod after profile_per_shard.sh:
    python3 parse_shard_profiles.py /workspace/shard_profiles

HONESTY NOTE: residuals are computed against hft_graph's predicted times,
which still depend on HFTDatasetSpec's byte estimates. Run
measure_compression.py first and patch the real ratio in, or the residual
*magnitudes* will be off (the Granger *ranking* is more robust to this since
it's about relative timing across shards, not absolute residual size).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Map substrings in the nsys kernel "Name" column to our predicted-graph ops.
KERNEL_OP_MAP = {
    "zstd": "zstd_decompress",
    "decode_page_data": "parquet_decode",
    "DeviceMergeSortMergeKernel": "merge_sort",
    "DeviceMergeSortBlockSortKernel": "merge_sort",
}


def kernel_to_op(name: str) -> str | None:
    for needle, op in KERNEL_OP_MAP.items():
        if needle in name:
            return op
    return None


def parse_one_csv(csv_path: Path) -> dict[str, float]:
    """Sum kernel time (ns) per op for a single shard's profile."""
    per_op: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # nsys cuda_gpu_kern_sum columns: "Time (%)","Total Time (ns)",...,"Name"
        time_col = next(
            (c for c in (reader.fieldnames or []) if "Total Time" in c), None
        )
        name_col = "Name" if "Name" in (reader.fieldnames or []) else None
        if not time_col or not name_col:
            return per_op
        for row in reader:
            op = kernel_to_op(row[name_col])
            if op is None:
                continue
            try:
                t_ns = float(row[time_col].replace(",", ""))
            except (ValueError, KeyError):
                continue
            per_op[op] = per_op.get(op, 0.0) + t_ns
    return per_op


def main(profile_dir: str) -> None:
    d = Path(profile_dir)
    csvs = sorted(d.glob("shard_*_cuda_gpu_kern_sum.csv")) or sorted(
        d.glob("shard_*.csv")
    )
    if not csvs:
        print(f"no per-shard CSVs in {d}. Run profile_per_shard.sh first.")
        return

    # Build observed kernel-time series (ms) per op, one entry per shard.
    obs_series: dict[str, list[float]] = {}
    for csv_path in csvs:
        per_op = parse_one_csv(csv_path)
        for op, t_ns in per_op.items():
            obs_series.setdefault(op, []).append(t_ns / 1e6)  # ns -> ms

    print(f"parsed {len(csvs)} shard profiles")
    for op, series in obs_series.items():
        print(f"  {op:<18} {len(series)} samples  {[round(x, 1) for x in series]}")

    # Build residuals vs predicted graph, scaled to 5M-event shards.
    try:
        from gitm.planner.hft_graph import HFTDatasetSpec, predict_hft_graph
        from gitm.optimizer.attribution import attribute
        from gitm.optimizer.monitor import KernelResidual, Residuals
    except Exception as e:  # noqa: BLE001
        print(f"\ncould not import gitm modules ({e}).")
        print("Run this from the repo root with the env that has pydantic/statsmodels.")
        return

    # Predicted graph sized for ONE shard (5M events), not 25M.
    shard_spec = HFTDatasetSpec(name="hft_shard_5M", n_events=5_000_000)
    g = predict_hft_graph(dataset=shard_spec)
    pred_ms = {n.op: n.prediction.t_pred_s * 1e3 for n in g.nodes}

    res = Residuals()
    n_shards = min((len(v) for v in obs_series.values()), default=0)
    for shard_i in range(n_shards):
        for op, series in obs_series.items():
            t_obs = series[shard_i]
            t_pred = pred_ms.get(op, 1e-9)
            r_kt = (t_obs - t_pred) / max(t_pred, 1e-12)
            res.per_kernel.append(KernelResidual(op=op, layer=None, r_kt=r_kt, r_mt=None))

    print(f"\nbuilt {len(res.per_kernel)} residual points "
          f"({n_shards} shards x {len(obs_series)} ops)")

    ranked = attribute(res, g, max_lag=2)
    if not ranked.hypotheses:
        print("\nattribute() returned no hypotheses.")
        print("Likely <4 samples/op (need more shards) or statsmodels missing.")
        return

    print("\nGranger causality ranking (lower p = stronger causal signal):")
    print(f"{'cause':<18} -> {'effect':<18} {'p_value':>10}  {'direction'}")
    print("-" * 64)
    for h in ranked.top(10):
        print(f"{h.cause_op:<18} -> {h.effect_op:<18} {h.p_value:>10.4f}  {h.direction}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/shard_profiles")
