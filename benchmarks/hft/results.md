# HFT Benchmark Results

## Hardware
- GPU: NVIDIA A100 80GB PCIe
- CUDA: 12.8
- cuDF: 26.6.0
- Host: RunPod cloud instance

## Dataset
- Generator: Numba JIT Hawkes process + CuPy GPU fields + 3 parallel write threads
  (see benchmarks/hft/generator_comparison.md for the full generator comparison)
- Events per seed: 1,000,000,000
- Seeds: 42, 43, 44
- Format: Parquet, snappy/zstd-1, 200 shards per seed (5M events each)
- Location: /workspace/hft_numba_seed42, hft_numba_seed43, hft_numba_seed44
- Schema: ts_ns int64, symbol_id int32, side int8, price int64, size int32, type int8
- Event mix: 55% add / 35% cancel / 10% trade (realistic order-flow ratio)
- Generation throughput: ~12.4M events/sec for 1B events (~1m22s), identical
  across all 3 seeds (0% variance)

## Baseline Configuration Tests (200-shard format, new pod)

### Sequential, 25M events (recommended baseline)
| Seed | Events/sec |
|------|------------|
| 42 | 21.1M |
| 43 | 20.8M |
| 44 | 20.4M |

Mean: 20.7M events/sec | Variance: 3.4%
Target (>=25M): below target | Variance spec (<2%): slightly above spec

### Event count comparison (sequential)
| Max Events | Mean Events/sec | Variance |
|------------|-----------------|----------|
| 10M | 8.9M | 10% |
| 15M | 11.6M | 22% |
| 25M | 20.7M | 3.4% |

25M is clearly the best-balanced configuration of those tested.

### Parallel runs (3 seeds simultaneously, staggered start)
| Config | Mean Events/sec | Variance | Notes |
|---|---|---|---|
| sleep=3s stagger, 15M events | 12.6M | 1.5% | passes variance spec, below throughput target |
| sleep=2s stagger, 25M events | 18.9M | 6.3% | unstable |
| no stagger, 25M events | crashes (Exit 1) | - | cuDF memory-pool conflict per process |

Parallel harness execution on a single A100 is unreliable: each Python
process creates its own cuDF memory pool, so 2-3 simultaneous processes at
25M events frequently OOM. A short stagger (2-3s) avoids crashes but adds
variance. Sequential execution is the recommended approach for this harness.

## Original Baseline (single-file format, original pod, C++ generator)
| Seed | Run 1 | Run 2 | Run 3 | Mean |
|------|-------|-------|-------|------|
| 42 | 29.5M | 31.1M | 29.1M | 29.9M |
| 43 | 31.1M | 29.5M | 30.0M | 30.2M |
| 44 | 30.9M | 29.8M | 30.8M | 30.5M |

Mean: 30.2M events/sec | Variance: ~2%
Target (>=25M): PASS | Variance spec (<2%): PASS

## Key Finding - Multi-shard vs Single File Format

Original data: 1 file per seed (1B rows). Current data: 200 files per seed
(5M rows each, matching generate.py's shard size).

| Format | Events/sec |
|---|---|
| Single file (1 x 1B rows) | ~30M |
| 200 shards (200 x 5M rows) | ~20M |

Patched harness.load_events() to read only the files needed for max_events
(5 files for 25M events) instead of all 200 - throughput barely changed
(20.6M vs 20.7M), so file count alone isn't the main cause. The remaining
gap is likely pod/hardware variability (different RunPod instance, cuDF
26.6.0 vs 26.04) rather than the shard format itself.

## Stall Profile (nsys, seed 42, 25M events)

| | CPU | Data-stall | Sync | GPU active |
|---|---|---|---|---|
| Expected | <5% | 10-25% | 5-15% | 60-80% |
| Measured (original pod) | ~5% | ~73% | ~7% | ~15% |
| Measured (new pod) | ~5% | ~73% | ~8% | ~15% |

Stall profile is consistent across pods and formats.

### Top GPU Kernels
| Kernel | Original pod | New pod |
|---|---|---|
| zstd decompression | 49.3% | 47.5% |
| Parquet page decode | 24.1% | 26.2% |
| Merge sort (top-of-book) | 7.1% | 8.3% |

Host-to-Device memory transfers: 91.2% of total memory time (7.98 GB, original pod).

## Saturation Rule
GPU active (~15%) is well below the 85% threshold. No swap to 500M events required.

## Task 1 - Predicted Graph Integration (W2)

Built gitm/planner/hft_graph.py - a predicted execution graph for the HFT
pipeline (zstd_decompress, parquet_decode, merge_sort, vwap_reduce), using
the same roofline model as gitm/planner/graph.py (LLM decode predictions).

### Residuals (predicted vs observed, seed 42, 25M events)

| Op | Predicted (ms) | Observed (ms) | r_kt | Bound |
|---|---|---|---|---|
| zstd_decompress | 0.425 | 252.076 | 592x | memory |
| parquet_decode | 0.638 | 139.100 | 217x | memory |
| merge_sort | 0.638 | 43.927 | 68x | memory |
| vwap_reduce | 0.159 | not separately profiled | - | memory |

### Key Finding

All profiled ops run 68-592x slower than the memory-bandwidth-bound floor
predicts. roofline()'s compute-vs-memory model has no category for ops
bound by sequential/branchy algorithms (zstd entropy decoding, Parquet
page decode) rather than HBM bandwidth or FLOPs. This is a gap in the
planner's prediction model, not specific to HFT - it would affect any
data-loading stage prediction.

### Task 3 - Granger Causality Next Step

gitm.optimizer.attribution.attribute() requires >=4 samples per op
(max_lag=2 default). Current data has 1 sample per op (aggregate kernel
sum from one nsys run on seed 42). Next step: profile each of the 5
shards in the 25M-event run separately to get 5 samples/op, enabling
Granger analysis of which stage causally drives the others' slowdown.

Code: gitm/planner/hft_graph.py, gitm/planner/hft_residual_demo.py
Run: python3 -m gitm.planner.hft_residual_demo

## Profiling Artifacts
- nsys profiles generated during testing (not committed to git - see
  .gitignore for *.nsys-rep / *.sqlite)

## Next Steps
1. Investigate remaining ~10M events/sec gap between pods (cuDF version,
   pod hardware) since file-count fix alone didn't close it
2. Profile per-shard kernel times to enable Granger causality (Task 3)
3. Extend hft_graph.py with a non-bandwidth prediction category for
   decode/decompress-bound ops (Task 1 follow-up)
