# HFT Generator Comparison Report

## Overview
I tested and built four different approaches to generate 1B HFT events, progressively improving speed while maintaining data realism.

## Generators Tested

### 1. C++ Generator (benchmarks/hft/generator/main.cpp)
Sequential loop per event. Hawkes process for timestamps. Writes to Parquet in 5M chunks. Single threaded — generation and writing happen one after another.

- Time for 1B events: ~10 minutes
- Events/sec: ~1.67M/sec
- Hawkes: exact

### 2. GPU Single Thread (CuPy)
Generates all 6 fields for 5M events at once on GPU using CuPy. Transfers to CPU, writes to Parquet. Still sequential — waits for write before generating next chunk.

- Time for 1B events: ~4m 25s
- Events/sec: ~3.8M/sec
- Hawkes: simple uniform gaps

### 3. GPU Parallel simple
GPU generates chunk N while 3 CPU threads write simultaneously. ThreadPoolExecutor with max_workers=3. GPU never waits for disk.

- Time for 1B events: 1m 18s
- Events/sec: 12.9M/sec
- Hawkes: simple uniform gaps

### 4. Numba + GPU Parallel (final)
Best of all approaches:
- Numba JIT compiles Hawkes loop to native CPU code — 53x faster than plain Python
- CuPy generates all other fields on GPU
- 3 parallel write threads
- Exact Hawkes timestamps preserved

- Time for 1B events: ~1m 22s
- Events/sec: ~12.4M/sec
- Hawkes: exact

## Results

| Generator | Time for 1B | Events/sec | Speedup vs C++ | Exact Hawkes |
|---|---|---|---|---|
| Adit Python (numpy) | ~30 min | 0.56M/sec | 0.3x | No |
| C++ CPU | ~10 min | 1.67M/sec | 1x | Yes |
| GPU single thread | ~4m 25s | 3.8M/sec | 2.3x | No |
| GPU parallel simple | 1m 18s | 12.9M/sec | 7.7x | No |
| Numba + GPU parallel | 1m 22s | 12.4M/sec | 7.4x | Yes |

## Key Insights

**Why Numba is fast:**
Numba JIT compiles Python to native machine code using LLVM. The sequential Hawkes loop that takes 1.5s in plain Python takes 0.029s in Numba — 53x faster. Same math, compiled to CPU instructions.

**Why GPU parallel is fast:**
Three things happen simultaneously:
- GPU generates fields for chunk N (thousands of cores in parallel)
- Write thread 1 writes chunk N-2 to disk
- Write thread 2 writes chunk N-1 to disk
Nothing waits. Like a pit crew where refueling, tire change, and wing adjustment all happen at once.

**Why not fully vectorize Hawkes on GPU:**
Hawkes process has sequential dependency — lambda_i depends on lambda_{i-1}. Attempted parallel prefix scan but thinning approximation produced non-monotonic timestamps. Numba JIT is the right solution — exact and fast.

## RNG Speed Test
- numpy CPU: 24M/sec
- CuPy GPU: 10B+/sec (423x faster at 5M numbers)
- GPU wins at scale, CPU wins for tiny batches

## Three Seeds Validated
| Seed | Rows | Files | Events/sec |
|---|---|---|---|
| 42 | 1,000,000,000 | 200 | 12.4M/sec |
| 43 | 1,000,000,000 | 200 | 12.4M/sec |
| 44 | 1,000,000,000 | 200 | 12.4M/sec |

Seed variance: 0% — identical throughput across all seeds

## Conclusion
Final generator (Numba + GPU + parallel writes) is 7.4x faster than C++ with exact Hawkes timestamps. The bottleneck shifted from RNG generation to disk I/O bandwidth — adding more than 3 write threads gives no benefit.

Next step: explore Arrow IPC format instead of Parquet to reduce write overhead further.
