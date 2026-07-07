"""Synthetic limit-order-book generator (the GITM stand-in for GoMask).

Produces the frozen HFT dataset at the scale target: a stream of order events
with the exact schema the benchmark pins — `ts_ns, symbol_id, side, price,
size, type` — as Parquet (row-group 128 MiB, zstd-1), sharded so memory stays
bounded regardless of total event count. Seed-deterministic: the same
`(events, seed, events_per_file)` always yields byte-identical shards, which is
what makes `manifest.yaml` reproducible.

Scales from a laptop smoke (10 M events) to the cluster target (1 B events per
seed) with the same command — only `--events` changes. The model is a
per-symbol random-walk mid price with Poisson-spaced timestamps and a realistic
add/cancel/trade mix; it is intentionally inexpensive (no matching engine) because the
benchmark stresses *replay + indicator* throughput, not market realism.

    python generate.py --events 10_000_000 --seed 42 --out $STAGE/hft_10m_seed42
    python generate.py --events 1_000_000_000 --seed 42 --out $STAGE/hft_1b_seed42

If a real GoMask binary is adopted, swap the call site in the Makefile; the
schema and sharding contract here are what the rest of the pipeline depends on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Schema — pinned. Order and dtypes match benchmarks/hft/datasets.md exactly.
SIDE_BID, SIDE_ASK = 0, 1
TYPE_ADD, TYPE_CANCEL, TYPE_TRADE = 0, 1, 2

DEFAULT_SYMBOLS = 512
DEFAULT_EVENTS_PER_FILE = 5_000_000  # ~128 MiB row-group at this schema width


@dataclass(frozen=True)
class GenConfig:
    events: int
    seed: int
    symbols: int = DEFAULT_SYMBOLS
    events_per_file: int = DEFAULT_EVENTS_PER_FILE
    start_ts_ns: int = 0
    mean_gap_ns: int = 1_000  # ~1 µs mean inter-event time


def _arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("ts_ns", pa.int64()),
            ("symbol_id", pa.int32()),
            ("side", pa.int8()),
            ("price", pa.int64()),  # integer ticks
            ("size", pa.int32()),
            ("type", pa.int8()),
        ]
    )


def _shard_arrays(cfg: GenConfig, shard_index: int, n: int, ts_start: int) -> dict:
    """Generate one shard's columns deterministically.

    Each shard derives its RNG from ``(seed, shard_index)`` so output is
    independent of how many shards a run is split into — a 10 M run's first
    shard is byte-identical to the first shard of a 1 B run with the same seed.
    """
    rng = np.random.default_rng([cfg.seed, shard_index])

    # Monotonic timestamps: cumulative Poisson-ish gaps from this shard's start.
    gaps = rng.integers(1, 2 * cfg.mean_gap_ns, size=n, dtype=np.int64)
    ts_ns = ts_start + np.cumsum(gaps)

    symbol_id = rng.integers(0, cfg.symbols, size=n, dtype=np.int32)
    side = rng.integers(0, 2, size=n, dtype=np.int8)

    # Price: per-symbol base + a bounded random-walk wiggle, in integer ticks.
    base = (10_000 + (symbol_id.astype(np.int64) * 7) % 5_000)
    wiggle = rng.integers(-50, 51, size=n, dtype=np.int64)
    price = base + wiggle

    size = rng.integers(1, 1_000, size=n, dtype=np.int32)
    # Event mix ~ 55% add / 35% cancel / 10% trade.
    roll = rng.random(n)
    etype = np.where(roll < 0.55, TYPE_ADD, np.where(roll < 0.90, TYPE_CANCEL, TYPE_TRADE))

    return {
        "ts_ns": ts_ns,
        "symbol_id": symbol_id,
        "side": side,
        "price": price,
        "size": size,
        "type": etype.astype(np.int8),
        "_ts_end": int(ts_ns[-1]) if n else ts_start,
    }


def generate(cfg: GenConfig, out_dir: str | Path) -> list[Path]:
    """Generate the full dataset into ``out_dir`` as sharded Parquet.

    Returns the list of written shard paths (sorted). Timestamps remain globally
    monotonic across shards by threading each shard's last ts into the next.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema()

    n_full, rem = divmod(cfg.events, cfg.events_per_file)
    shard_sizes = [cfg.events_per_file] * n_full + ([rem] if rem else [])

    written: list[Path] = []
    ts_cursor = cfg.start_ts_ns
    for i, n in enumerate(shard_sizes):
        cols = _shard_arrays(cfg, i, n, ts_cursor)
        ts_cursor = cols.pop("_ts_end") + cfg.mean_gap_ns
        table = pa.table({k: cols[k] for k in schema.names}, schema=schema)
        path = out_dir / f"part-{i:05d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            compression_level=1,
            row_group_size=cfg.events_per_file,
        )
        written.append(path)
    return sorted(written)


def first_row_sample(out_dir: str | Path) -> dict:
    """Return the first row of the first shard — for the datasets.md sample."""
    import pyarrow.parquet as pq

    shards = sorted(Path(out_dir).glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards in {out_dir}")
    tbl = pq.read_table(shards[0])
    return {k: tbl[k][0].as_py() for k in tbl.column_names}


def _human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n}{unit}"
        n //= 1000
    return f"{n}T"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Synthetic LOB generator (GoMask stand-in).")
    p.add_argument("--events", type=lambda s: int(s.replace("_", "")), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--symbols", type=int, default=DEFAULT_SYMBOLS)
    p.add_argument("--events-per-file", type=lambda s: int(s.replace("_", "")),
                   default=DEFAULT_EVENTS_PER_FILE)
    args = p.parse_args(argv)

    cfg = GenConfig(
        events=args.events,
        seed=args.seed,
        symbols=args.symbols,
        events_per_file=args.events_per_file,
    )
    shards = generate(cfg, args.out)
    total_bytes = sum(s.stat().st_size for s in shards)
    print(
        f"generated {_human(args.events)} events seed={args.seed} -> "
        f"{len(shards)} shard(s), {total_bytes / 1e9:.3f} GB in {args.out}"
    )
    print(f"first row: {first_row_sample(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
