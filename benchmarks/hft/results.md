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

## Baseline Configuration Tests (25M events warm window)

### Sequential (best stability)
| Seed | Events/sec |
|------|------------|
| 42 | 21.1M |
| 43 | 20.8M |
| 44 | 20.4M |

Mean: 20.7M events/sec
Variance: 3.4%
Target (>=25M): Below target
Variance spec (<2%): Above spec

### Parallel with sleep=3s stagger
| Seed | Events/sec |
|------|------------|
| 42 | 12.5M |
| 43 | 12.6M |
| 44 | 12.7M |

Mean: 12.6M events/sec
Variance: 1.5%
Target (>=25M): Below target
Variance spec (<2%): PASS

### Parallel with sleep=2s stagger
| Seed | Events/sec |
|------|------------|
| 42 | 19.5M |
| 43 | 18.9M |
| 44 | 18.3M |

Mean: 18.9M events/sec
Variance: 6.3%
Target (>=25M): Below target
Variance spec (<2%): Above spec

### Event count comparison (sequential)
| Max Events | Mean Events/sec | Variance |
|------------|-----------------|----------|
| 10M | 8.9M | 10% |
| 15M | 11.6M | 22% |
| 25M | 20.7M | 3.4% |

25M is clearly the best configuration.

## Original Baseline (single file format, different pod)
| Seed | Run 1 | Run 2 | Run 3 | Mean |
|------|-------|-------|-------|------|
| 42 | 29.5M | 31.1M | 29.1M | 29.9M |
| 43 | 31.1M | 29.5M | 30.0M | 30.2M |
| 44 | 30.9M | 29.8M | 30.8M | 30.5M |

Mean: 30.2M events/sec
Variance: ~2%
Target (>=25M): PASS
Variance spec (<2%): PASS

## Key Finding - Multi-shard vs Single File
Original data was 1 file per seed (1B rows).
Current data is 200 files per seed (5M rows each).

The harness reads all files and concatenates them which adds overhead.
With 200 files: ~20M events/sec
With 1 file: ~30M events/sec

Hypothesis: harness reads all 200 files even when max_events=25M.
Fix: modify harness to read only files needed for max_events.
This should restore ~30M events/sec without regenerating data.

## Stall Profile (nsys, seed 42, 25M events)

| | CPU | Data-stall | Sync | GPU active |
|---|---|---|---|---|
| Expected | <5% | 10-25% | 5-15% | 60-80% |
| Measured | ~5% | ~73% | ~7% | ~15% |

Top GPU kernels:
- zstd decompression: 47.5%
- Parquet decode: 26.2%
- Merge sort: 8.3%

## Saturation Rule
GPU active (15%) is well below 85% threshold.
No swap to 500M events required.

## Next Steps
1. Fix harness to read only needed files (not all 200)
2. Test if this restores 30M events/sec
3. If yes, document as the standard benchmark configuration
4. If no, regenerate data as single file
