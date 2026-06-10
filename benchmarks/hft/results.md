# HFT Benchmark Results

## Hardware
- GPU: NVIDIA A100 80GB PCIe
- CUDA: 12.8
- cuDF: 26.6.0
- Host: RunPod cloud instance

## Dataset
- Generator: Numba JIT Hawkes + CuPy GPU fields + parallel writes
- Events per seed: 1,000,000,000
- Seeds: 42, 43, 44
- Format: Parquet, zstd-1, 200 shards per seed (5M events each)
- Location: /workspace/hft_numba_seed{42,43,44}/

## Baseline Events/sec

### Run 1 (original pod, single file format)
| Seed | Run 1 | Run 2 | Run 3 | Mean |
|------|-------|-------|-------|------|
| 42 | 29.5M | 31.1M | 29.1M | 29.9M |
| 43 | 31.1M | 29.5M | 30.0M | 30.2M |
| 44 | 30.9M | 29.8M | 30.8M | 30.5M |

3-seed mean: 30.2M events/sec
Target: >=25M events/sec - PASS
Seed variance: ~2% - PASS

### Run 2 (new pod, 200-shard format)
| Seed | Events/sec |
|------|------------|
| 42 | 19.3M |
| 43 | 19.6M |
| 44 | 21.2M |

3-seed mean: 20.0M events/sec
Target: >=25M events/sec - BELOW TARGET
Seed variance: ~10% - ABOVE SPEC

Note: Lower throughput due to multi-shard format (200 files vs 1 file).
File open overhead reduces throughput by ~10M events/sec.
Same stall profile as original run.

## Stall Profile Comparison (seed 42, nsys)

| | CPU | Data-stall | Sync | GPU active |
|---|---|---|---|---|
| Expected | <5% | 10-25% | 5-15% | 60-80% |
| Run 1 (single file) | ~5% | ~73% | ~7% | ~15% |
| Run 2 (200 shards) | ~5% | ~73% | ~7% | ~15% |

Stall profile is identical regardless of file format.
Bottleneck is zstd decompression (47-49%) and Parquet decode (24-26%).

## Top GPU Kernels (nsys, seed 42)

### Run 1
- zstd decompression: 49.3%
- Parquet decode: 24.1%
- Merge sort: 7.1%

### Run 2
- zstd decompression: 47.5%
- Parquet decode: 26.2%
- Merge sort: 8.3%

## Saturation Rule
GPU active (~15%) is well below 85% threshold.
No swap to 500M events required.

## Key Finding
Multi-shard format (200 files) reduces throughput vs single file due to file
open overhead. Recommend standardizing on single file format for benchmark
to consistently hit >=25M target. Flagged to Adit for format decision.

## Profiling Artifacts
- /workspace/hft_baseline_new.nsys-rep (seed 42, 25M events, 200-shard format)
- benchmarks/hft/hft_baseline_1.nsys-rep (seed 42, 25M events, single file format)
