"""Predicted-vs-observed residuals for the HFT pipeline (Task 1 demo).

This is a stand-in for the full nsys-sqlite -> gitm.tracer.schema.Trace
converter (the larger W2 piece). It hardcodes the observed kernel times
from benchmarks/hft/results.md (seed 42, 25M-event run, new pod) and
computes the same residual gitm.optimizer.monitor.residuals() would:

    r_kt = (t_obs - t_pred) / t_pred

A large positive r_kt means the kernel ran far slower than the
memory-bandwidth-bound floor predicts -- i.e. the op is NOT
bandwidth-limited, and roofline's compute/memory model doesn't capture
what's actually limiting it (e.g. sequential entropy decoding for zstd,
branchy per-row page decode for Parquet).

Run:
    python3 -m gitm.planner.hft_residual_demo
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.planner.hft_graph import predict_hft_graph


@dataclass
class ObservedKernel:
    op: str
    t_obs_ms: float
    source: str


# From benchmarks/hft/results.md, "Top GPU Kernels (nsys, seed 42)" / Run 2
# (new pod, 200-shard format, 25M-event warm window).
OBSERVED = [
    ObservedKernel("zstd_decompress", 252.076091, "zstd::decompression_kernel"),
    ObservedKernel("parquet_decode", 139.100450, "decode_page_data_generic"),
    # DeviceMergeSortMergeKernel (26.555531ms) + DeviceMergeSortBlockSortKernel
    # (17.371249ms) -- both are part of the top-of-book sort.
    ObservedKernel("merge_sort", 26.555531 + 17.371249, "DeviceMergeSort*"),
]


def main() -> None:
    g = predict_hft_graph()
    pred_by_op = {n.op: n.prediction for n in g.nodes}

    print(f"{'op':<18} {'pred (ms)':>10} {'obs (ms)':>10} {'r_kt':>10}  bound")
    print("-" * 60)

    for ok in OBSERVED:
        pred = pred_by_op[ok.op]
        t_pred_ms = pred.t_pred_s * 1e3
        r_kt = (ok.t_obs_ms - t_pred_ms) / t_pred_ms
        print(
            f"{ok.op:<18} {t_pred_ms:10.4f} {ok.t_obs_ms:10.3f} "
            f"{r_kt:10.1f}  {pred.bound}"
        )

    vwap = pred_by_op["vwap_reduce"]
    print(f"{'vwap_reduce':<18} {vwap.t_pred_s * 1e3:10.4f} {'?':>10} {'?':>10}  {vwap.bound}")

    print()
    print("All three profiled ops are ~70-600x slower than the memory-bandwidth")
    print("floor predicts. roofline()'s compute-vs-memory model doesn't have a")
    print("category for these: zstd entropy decode and Parquet page decode are")
    print("bound by sequential/branchy per-element work, not by HBM bandwidth.")
    print()
    print("Granger causality (gitm.optimizer.attribution.attribute) needs >= 4")
    print("samples per op (max_lag=2). This demo has 1 sample per op (the")
    print("seed-42 aggregate kernel sum). Next step: profile each of the 5")
    print("shards in the 25M run separately -> 5 samples/op, enough to run")
    print("attribute() and rank which stage Granger-causes the others.")


if __name__ == "__main__":
    main()
