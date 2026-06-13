"""Measure the REAL compression ratio + per-shard byte counts.

Replaces the estimated ``compression_ratio = 3.0`` in hft_graph.py with a
number measured from the actual Parquet shards. Run this on the pod where
/workspace/hft_numba_seed42/ lives, then paste the printed compression_ratio
back into HFTDatasetSpec.

What it measures, per shard:
  - on-disk compressed bytes (os.path.getsize)
  - uncompressed bytes (num_rows * 26, the fixed-width row size)
  - parquet metadata's own uncompressed estimate (total_uncompressed_size)

The (compressed, uncompressed) pair gives the real ratio. We print both the
26-bytes/row figure and Parquet's own accounting because they can differ
(Parquet stores some per-page overhead + dictionary pages that the naive
26*rows ignores).

Run:
    python3 measure_compression.py /workspace/hft_numba_seed42
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

BYTES_PER_ROW = 26  # ts_ns i64 + symbol_id i32 + side i8 + price i64 + size i32 + type i8


def main(stage: str) -> None:
    paths = sorted(Path(stage).glob("part-*.parquet"))
    if not paths:
        print(f"no parquet shards in {stage}")
        return

    total_disk = 0
    total_rows = 0
    total_pq_uncompressed = 0

    # Sample first 5 shards (matches the 25M-event warm window) + report all-200 totals.
    print(f"{'shard':<22} {'rows':>12} {'disk_MB':>10} {'pq_uncomp_MB':>14} {'ratio':>7}")
    print("-" * 70)
    for i, p in enumerate(paths):
        md = pq.ParquetFile(p).metadata
        disk = p.stat().st_size
        rows = md.num_rows
        pq_uncomp = sum(
            md.row_group(rg).total_byte_size for rg in range(md.num_row_groups)
        )
        total_disk += disk
        total_rows += rows
        total_pq_uncompressed += pq_uncomp
        if i < 5:
            ratio = pq_uncomp / disk if disk else 0
            print(
                f"{p.name:<22} {rows:>12,} {disk / 1e6:>10.2f} "
                f"{pq_uncomp / 1e6:>14.2f} {ratio:>7.2f}"
            )

    naive_uncomp = total_rows * BYTES_PER_ROW
    ratio_naive = naive_uncomp / total_disk if total_disk else 0
    ratio_pq = total_pq_uncompressed / total_disk if total_disk else 0

    print("-" * 70)
    print(f"shards:                 {len(paths)}")
    print(f"total rows:             {total_rows:,}")
    print(f"total on-disk:          {total_disk / 1e9:.3f} GB")
    print(f"naive uncompressed:     {naive_uncomp / 1e9:.3f} GB (26 B/row)")
    print(f"parquet uncompressed:   {total_pq_uncompressed / 1e9:.3f} GB (metadata)")
    print()
    print(f"compression_ratio (naive 26B/row): {ratio_naive:.3f}")
    print(f"compression_ratio (parquet meta):  {ratio_pq:.3f}")
    print()
    print("ACTION: replace compression_ratio=3.0 in HFTDatasetSpec")
    print(f"        with the parquet-meta ratio above ({ratio_pq:.3f}).")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "/workspace/hft_numba_seed42"
    main(stage)
