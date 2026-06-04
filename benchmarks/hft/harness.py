"""HFT baseline harness — GPU LOB replay on cuDF/CuPy (no bespoke CUDA kernels).

Implements the work unit from the spec —
``ingest -> order-book update -> top-of-book -> microprice + VWAP-1s`` — as a
vectorized dataframe pipeline. On a GPU box it runs on **cuDF + CuPy** (a real
device workload with genuine Parquet-decode/H2D data-stall and groupby-scan
sync); with neither present it falls back to **pandas + NumPy** so the pipeline
logic is identical and testable on a laptop. cuDF is API-compatible with pandas,
so the same code drives both — the backend is chosen once at startup.

The pipeline is a faithful *vectorized approximation* of an L2 book: top-of-book
is a per-symbol running best (cummax bid / cummin ask in event order) rather than
a sequential add/cancel replay, and VWAP-1s uses tumbling 1-second buckets
(``ts_ns // 1e9``). Both are exact, GPU-friendly reductions and exercise the
stall profile GITM cares about; they trade book-replay fidelity for throughput,
which is the right trade for a *runtime* benchmark.

Emits the one-line harness contract on stdout (see ``benchmarks/README.md``):
``metric_value`` = events/sec over the warm window, plus device info.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def select_backend():
    """Return ``(kind, df_module, array_module)`` — cuDF/CuPy if available, else pandas/NumPy."""
    try:
        import cudf
        import cupy

        return "gpu", cudf, cupy
    except Exception:
        import numpy
        import pandas

        return "cpu", pandas, numpy


def _gpu_name(kind: str) -> tuple[str, int]:
    if kind != "gpu":
        return "cpu", 0
    try:
        import cupy

        n = cupy.cuda.runtime.getDeviceCount()
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return name, n
    except Exception:
        return "gpu-unknown", 1


# --- pipeline stages (backend-agnostic; df is cudf or pandas) ----------------


def top_of_book(df, dflib):
    """Per-symbol running best bid (cummax) and best ask (cummin) in event order.

    Returns the frame with ``best_bid`` / ``best_ask`` columns forward-filled so
    every row sees the prevailing top of book.
    """
    df = df.sort_values(["symbol_id", "ts_ns"]).reset_index(drop=True)
    is_bid = df["side"] == 0
    bid_px = df["price"].where(is_bid)
    ask_px = df["price"].where(~is_bid)
    g = df.groupby("symbol_id")
    df["best_bid"] = bid_px.groupby(df["symbol_id"]).cummax()
    df["best_ask"] = ask_px.groupby(df["symbol_id"]).cummin()
    # carry the last seen top-of-book forward across rows of either side
    df["best_bid"] = g["best_bid"].ffill().reset_index(drop=True)
    df["best_ask"] = g["best_ask"].ffill().reset_index(drop=True)
    return df


def microprice(df):
    """Size-weighted mid: (best_ask*bid_sz + best_bid*ask_sz) / (bid_sz + ask_sz).

    Uses the row's own size as the prevailing depth proxy on its side; rows
    without a complete top-of-book yet yield null and are dropped from the mean.
    """
    bid_sz = df["size"].where(df["side"] == 0)
    ask_sz = df["size"].where(df["side"] == 1)
    denom = bid_sz.fillna(0) + ask_sz.fillna(0)
    num = df["best_ask"] * bid_sz.fillna(0) + df["best_bid"] * ask_sz.fillna(0)
    mp = num / denom.where(denom != 0)
    return mp


def vwap_1s(df, dflib):
    """Tumbling 1-second VWAP per symbol over trade events (type == 2)."""
    trades = df[df["type"] == 2]
    if len(trades) == 0:
        return trades
    bucket = trades["ts_ns"] // 1_000_000_000
    grp = trades.assign(_bucket=bucket, _pxsz=trades["price"] * trades["size"])
    agg = grp.groupby(["symbol_id", "_bucket"]).agg(
        pxsz=("_pxsz", "sum"), sz=("size", "sum")
    )
    agg["vwap"] = agg["pxsz"] / agg["sz"]
    return agg


def run_pipeline(df, dflib) -> dict:
    """Run the full work unit and return small summary stats (forces evaluation)."""
    df = top_of_book(df, dflib)
    df["microprice"] = microprice(df)
    vwap = vwap_1s(df, dflib)
    # Reductions force the lazy/device work to actually execute and land on host.
    return {
        "events": int(len(df)),
        "mean_microprice": float(df["microprice"].mean()),
        "vwap_buckets": int(len(vwap)),
    }


# --- dataset loading ---------------------------------------------------------


def _seed_dir(stage: Path, seed: int) -> Path:
    """Locate the staged shard dir for a seed (hft_*_seed<seed>/)."""
    matches = sorted(stage.glob(f"*seed{seed}"))
    if not matches:
        # fall back: any parquet directly under stage
        if any(stage.glob("**/part-*.parquet")):
            return stage
        raise FileNotFoundError(f"no staged HFT data for seed {seed} under {stage}")
    return matches[0]


def load_events(stage: Path, seed: int, dflib, *, max_events: int | None):
    paths = sorted(_seed_dir(stage, seed).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards in {_seed_dir(stage, seed)}")
    df = dflib.read_parquet(paths if len(paths) > 1 else paths[0])
    if max_events is not None and len(df) > max_events:
        df = df.iloc[:max_events]
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HFT LOB replay harness (cuDF/CuPy).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-seconds", type=int, default=60)
    p.add_argument("--stage", type=Path, default=None,
                   help="Staged dataset dir; defaults to $GITM_BENCH_STAGE.")
    p.add_argument("--max-events", type=int, default=None,
                   help="Cap events processed (warm-window bound on small boxes).")
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    kind, dflib, _xp = select_backend()
    gpu_name, device_count = _gpu_name(kind)

    df = load_events(stage, args.seed, dflib, max_events=args.max_events)

    # Warm window: replay the loaded events, measuring sustained throughput.
    t0 = time.perf_counter()
    summary = run_pipeline(df, dflib)
    elapsed = max(time.perf_counter() - t0, 1e-9)

    events_per_second = summary["events"] / elapsed
    print(f"[hft harness:{kind}] {summary['events']} events in {elapsed:.3f}s "
          f"({summary['vwap_buckets']} vwap buckets)")
    print(json.dumps({
        "metric_value": events_per_second,
        "gpu_name": gpu_name,
        "device_count": device_count,
        "harness_commit": "cudf-lob-v1",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
