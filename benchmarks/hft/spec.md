# HFT benchmark — spec

> Owner: Ash  baseline + profiling + spec doc.

## 1. Input definition
Synthetic limit-order-book stream at 1×10⁹ events per seed, Parquet
(row-group 128 MiB, zstd-1), schema in [datasets.md](datasets.md). Bytes live in
`$GITM_S3_ROOT/datasets/hft/hft_1b_seed{42,43,44}/`, frozen by
[manifest.yaml](manifest.yaml).

## 2. Work unit
One million events end-to-end through:
`ingest → order-book update → top-of-book snapshot → derived metric
(microprice + VWAP-1s window)`. Baseline harness: a CUDA kernel set plus a
host-side Arrow ingest pipeline. Phases above are the rows of the stall table.
<!-- TODO: pin harness commit + config hash. -->

## 3. Success metric
`events_per_second` over a 60 s warm window. **Baseline target: ≥ 25 M events/s
on a single A100.** Three seeds must agree within 2 % (the recorded baseline is
their mean). No auxiliary metrics.

## 4. Expected stall profile
Matches `[expected_stall]` in [bench.toml](bench.toml):

| | CPU | Data-stall | Sync | GPU active |
| --- | --- | --- | --- | --- |
| Expected | < 5 % | 10–25 % | 5–15 % | 60–80 % |

Data-stall is Parquet decode + host→device copy; sync is top-of-book
reductions; CPU is low because Arrow handles ingest off the hot path.

**Saturation rule:** if measured GPU active > 85 %, flag Adit same day — fall
back to a 500 M-event shard and document.

## 5. Profiling Methodology

Profiled using NVIDIA Nsight Systems (nsys) on RunPod A100 80GB PCIe.

Command:
nsys profile --trace cuda,nvtx,osrt --output /root/data/hft_baseline_1 \
  python3 benchmarks/hft/harness.py --seed 42 --stage /root/data --max-events 25000000

Results saved to:
- /root/data/hft_baseline_1.nsys-rep
- /root/data/hft_baseline_1.sqlite

View with:
nsys stats /root/data/hft_baseline_1.nsys-rep

## 6. Reproduction Steps

**1. Clone repo:**
```bash
git clone git@github.com:GitM-Labs/runtime.git
cd runtime && git checkout hft_datageneration
pip install -e ".[dev]"
```

**2. Build the generator:**
```bash
cd benchmarks/hft/generator && mkdir -p build && cd build
export ARROW_LIB=$(python3 -c "import pyarrow; print(pyarrow.get_library_dirs()[0])")
ln -sf $ARROW_LIB/libparquet.so.2400 $ARROW_LIB/libparquet.so
cmake .. && make -j$(nproc)
```

**3. Generate datasets:**
```bash
export LD_LIBRARY_PATH=$ARROW_LIB:$LD_LIBRARY_PATH
./hft_gen 1000000000 42 /root/data/hft_1b_seed42/part-00000.parquet
./hft_gen 1000000000 43 /root/data/hft_1b_seed43/part-00000.parquet
./hft_gen 1000000000 44 /root/data/hft_1b_seed44/part-00000.parquet
```

**4. Run baseline (3x per seed, take means):**
```bash
export GITM_BENCH_STAGE="/root/data"
python3 benchmarks/hft/harness.py --seed 42 --stage /root/data --max-events 25000000
```

**5. Run nsys profile:**
```bash
nsys profile --trace cuda,nvtx,osrt --output /root/data/hft_baseline_1 \
  python3 benchmarks/hft/harness.py --seed 42 --stage /root/data --max-events 25000000
```

**6. View results:**
```bash
nsys stats /root/data/hft_baseline_1.nsys-rep
```
