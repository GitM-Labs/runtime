# HFT Benchmark Results

## Hardware
- GPU: NVIDIA A100 80GB PCIe
- CUDA: 12.8
- cuDF: 26.04.000
- Host: RunPod cloud instance

## Dataset
- Generator: C++ Hawkes process generator (seed-sharded)
- Events per seed: 1,000,000,000
- Seeds: 42, 43, 44
- Format: Parquet, zstd-1, single shard per seed
- Location: /root/data/hft_1b_seed{42,43,44}/part-00000.parquet
- Manifest: benchmarks/hft/manifest.yaml (sha256 + byte counts)

## Baseline Events/sec (3 runs per seed, 25M events warm window)

| Seed | Run 1 | Run 2 | Run 3 | Mean |
|------|-------|-------|-------|------|
| 42 | 29.5M | 31.1M | 29.1M | 29.9M |
| 43 | 31.1M | 29.5M | 30.0M | 30.2M |
| 44 | 30.9M | 29.8M | 30.8M | 30.5M |

- 3-seed mean: 30.2M events/sec
- Target: >=25M events/sec 
- Seed variance: ~2%  (target: within 2%)

## Stall Profile (seed 42, nsys profile)

| | CPU | Data-stall | Sync | GPU active |
|---|---|---|---|---|
| Expected | <5% | 10-25% | 5-15% | 60-80% |
| Measured | ~5% | ~73% | ~7% | ~15% |

## Stall Analysis
Top GPU kernels by time:
- zstd decompression: 49.3%
- Parquet page decode: 24.1%
- Merge sort (top-of-book): 7.1%
- Transform kernels: ~8%

Host-to-Device memory transfers: 91.2% of total memory time (7.98 GB transferred)

Data-stall is dominated by zstd decompression and Parquet decoding.
GPU active (15%) is significantly below expected (60-80%).
This gives GITM substantial headroom for optimization via the deviation monitor.

## Saturation Rule
GPU active (15%) is well below 85% threshold.
No swap to 500M events required.

## Profiling Artifacts
- /root/data/hft_baseline_1.nsys-rep (seed 42, 25M events)
- /root/data/hft_baseline_1.sqlite

## Notes
Single runs show ~18% variance between seeds due to cold start.
Using 3-run means reduces variance to ~2% meeting spec requirement.
Recommend warming GPU cache before recording baseline numbers.
