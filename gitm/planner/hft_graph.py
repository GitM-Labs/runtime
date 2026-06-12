"""Predicted execution graph for the HFT LOB-replay benchmark.

Mirrors gitm.planner.graph (which predicts LLM decode steps) but for the
HFT data-loading + LOB-replay pipeline: zstd decompression, Parquet page
decode, merge sort (top-of-book ordering by symbol_id/ts_ns), and the
VWAP/microprice reductions.

All four ops here have flops=0, so roofline() always returns
bound="memory" and t_pred_s = bytes_moved / peak_mem_bw_bytes_per_s.

Numbers are derived from benchmarks/hft/results.md (seed 42, 25M-event
warm window, 200-shard zstd-1 Parquet, schema: ts_ns i64, symbol_id i32,
side i8, price i64, size i32, type i8 = 26 bytes/row uncompressed).
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, roofline


@dataclass(frozen=True)
class HFTDatasetSpec:
    """Shape of one HFT harness run, used to size the predicted graph.

    compression_ratio is an ESTIMATE (zstd-1 on mixed int columns is
    typically 2.5-4x). TODO: replace with the real ratio once file sizes
    can be checked on the pod (compressed_size / uncompressed_size).
    """

    name: str = "hft_seed42_25M"
    n_events: int = 25_000_000
    bytes_per_event_uncompressed: int = 26  # i64+i32+i8+i64+i32+i8
    compression_ratio: float = 3.0  # ESTIMATE — verify against real files

    @property
    def uncompressed_bytes(self) -> int:
        return self.n_events * self.bytes_per_event_uncompressed

    @property
    def compressed_bytes(self) -> int:
        return int(self.uncompressed_bytes / self.compression_ratio)


def predict_hft_graph(
    dataset: HFTDatasetSpec | None = None,
    hw: HardwareSpec | None = None,
) -> Graph:
    """Emit a predicted execution graph for the HFT LOB-replay harness.

    Pipeline stages (ordered as they appear in the nsys kernel summary):
      1. zstd_decompress  - read compressed column chunks, write decompressed
      2. parquet_decode   - decode columnar pages into cuDF columns
      3. merge_sort       - sort by (symbol_id, ts_ns) for top-of-book replay
      4. vwap_reduce      - cummax/cumsum reductions for microprice + VWAP

    Each PredictedNode.prediction.t_pred_s is the *memory-bandwidth-bound*
    floor for that op — the fastest it could possibly run on this GPU if it
    were purely bandwidth limited. Comparing this floor to the observed
    kernel time (via gitm.optimizer.monitor.residuals) tells us whether the
    op is actually bandwidth-bound or bound by something roofline doesn't
    model (e.g. sequential entropy decoding, branchy page-decode logic).
    """
    dataset = dataset or HFTDatasetSpec()
    hw = hw or HardwareSpec()

    # model/batch are unused for HFT but required by the Graph dataclass —
    # pass defaults so this stays type-compatible with the LLM planner.
    g = Graph(model=ModelSpec(), hw=hw, batch=BatchConfig())

    comp = dataset.compressed_bytes
    uncomp = dataset.uncompressed_bytes

    # 1. zstd decompression: read compressed bytes in, write uncompressed out
    g.nodes.append(
        PredictedNode(
            "zstd_decompress",
            layer=None,
            prediction=roofline(
                "zstd_decompress", flops=0, bytes_moved=comp + uncomp, hw=hw
            ),
            expected_stream_id=0,
        )
    )

    # 2. Parquet page decode: ~1x uncompressed in, ~1x out -> 2x uncompressed
    g.nodes.append(
        PredictedNode(
            "parquet_decode",
            layer=None,
            prediction=roofline(
                "parquet_decode", flops=0, bytes_moved=2 * uncomp, hw=hw
            ),
            expected_stream_id=0,
        )
    )

    # 3. Merge sort by (symbol_id, ts_ns): full read + write of reordered rows
    g.nodes.append(
        PredictedNode(
            "merge_sort",
            layer=None,
            prediction=roofline(
                "merge_sort", flops=0, bytes_moved=2 * uncomp, hw=hw
            ),
            expected_stream_id=0,
        )
    )

    # 4. VWAP / microprice reductions: reads price+size, writes derived cols
    g.nodes.append(
        PredictedNode(
            "vwap_reduce",
            layer=None,
            prediction=roofline(
                "vwap_reduce", flops=0, bytes_moved=int(0.5 * uncomp), hw=hw
            ),
            expected_stream_id=0,
        )
    )

    return g


if __name__ == "__main__":
    g = predict_hft_graph()
    print(f"Dataset: {HFTDatasetSpec().name}")
    print(f"{'op':<18} {'t_pred (ms)':>12} {'bound':>8} {'bytes_moved':>14}")
    for n in g.nodes:
        p = n.prediction
        print(f"{n.op:<18} {p.t_pred_s * 1e3:12.4f} {p.bound:>8} {p.bytes:14,.0f}")
    print(f"\ntotal predicted: {g.total_pred_s * 1e3:.4f} ms")
