# HFT benchmark — datasets

> Owner: Erin (Leo) — dataset + reproducibility. Fill the TODOs as data lands.

## Sources
- **Generator:** [`generate.py`](generate.py) — a seed-deterministic synthetic
  LOB generator (the GITM stand-in for GoMask; swap in the real GoMask binary at
  the `generate.py` call site if/when adopted). Same command at every scale —
  only `--events` changes.
- **Config:** 1×10⁹ order events per seed, seeds `{42, 43, 44}`.
- **Invocation:**
  ```bash
  # full scale (cluster, ~3.2 TB raw / ~600 GB zstd per seed):
  python generate.py --events 1_000_000_000 --seed 42 --out $GITM_SCRATCH/staging/hft/hft_1b_seed42
  # laptop smoke (drives the whole freeze/verify/baseline/reproduce loop):
  make smoke
  ```

## Fields & units
Per-event row (Parquet, row-group 128 MiB, zstd-1):

| Field | Type | Unit |
| --- | --- | --- |
| `ts_ns` | int64 | nanoseconds since session open (globally monotonic) |
| `symbol_id` | int32 | dense symbol index |
| `side` | int8 | 0 = bid, 1 = ask |
| `price` | int64 | integer ticks |
| `size` | int32 | lots |
| `type` | int8 | 0 add / 1 cancel / 2 trade |

First-row sample (seed 42): `{'ts_ns': 179, 'symbol_id': 376, 'side': 1,
'price': 12584, 'size': 285, 'type': 0}`

## Scale target
- ~3.2 TB raw, ~600 GB compressed **per seed**.
- 1×10⁹ events per seed.

## Seed protocol
`{42, 43, 44}` → `$GITM_S3_ROOT/datasets/hft/{hft_1b_seed42, hft_1b_seed43, hft_1b_seed44}/`.

## Freeze & verify
```bash
make manifest      # -> manifest.yaml (sha256 + byte count per Parquet file)
make verify        # re-hash $GITM_SCRATCH/staging/hft and confirm byte-identical
```
Manifest sha256: <!-- TODO: paste after first freeze -->
