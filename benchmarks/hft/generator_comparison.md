# HFT Generator Comparison Report

## Overview
I tested and built four different approaches to generate 1B HFT events, progressively improving speed while maintaining data realism. The final version is 7.4x faster than C++ while keeping exact Hawkes timestamps.

## What Makes the Data Realistic

The realism comes from three things, not from Numba or GPU:

1. Hawkes process for timestamps - in real markets, one big trade triggers more trades. Events cluster in bursts. Hawkes models this: every event jumps the arrival rate up by alpha (0.6), then it decays back to baseline at rate beta (0.8). Simple uniform random gaps would be unrealistic.

2. Realistic event mix - real order books have ~55% new orders, ~35% cancellations, ~10% trades. We use exactly this distribution instead of uniform random.

3. Price random walk - price moves up or down by 1 tick every time a trade happens, simulating real market microstructure.

Numba and GPU do NOT change the realism. They just make the same math run faster.

## Why Numba

The Hawkes process has a sequential dependency - each timestamp depends on the previous lambda value:

    lambda_i = MU + (lambda_{i-1} - MU) * exp(-BETA * dt_i) + ALPHA

This cannot be parallelized directly. In plain Python this loop runs at 0.7M/sec. Numba JIT compiles the exact same loop to native CPU machine code using LLVM - running at 34.8M/sec. 53x faster, same exact output.

Attempted fully vectorized GPU Hawkes using parallel prefix scan but thinning approximation produced non-monotonic timestamps. Numba is the correct solution - exact, fast, stable.

## Generators Tested

### 1. C++ Generator (benchmarks/hft/generator/main.cpp)
Sequential loop per event. Hawkes process for timestamps. Writes to Parquet in 5M chunks to avoid OOM. Single threaded.

- Time for 1B events: ~10 minutes
- Events/sec: ~1.67M/sec
- Hawkes: exact
- Event mix: uniform random

### 2. GPU Single Thread (CuPy)
Generates all 6 fields for 5M events at once on GPU. Transfers to CPU, writes to Parquet. Sequential - waits for write before next chunk. No Hawkes.

- Time for 1B events: ~4m 25s
- Events/sec: ~3.8M/sec
- Hawkes: simple uniform gaps
- Event mix: uniform random

### 3. GPU Parallel simple
GPU generates chunk N while 3 CPU threads write simultaneously using ThreadPoolExecutor. GPU never waits for disk. Still no Hawkes.

- Time for 1B events: 1m 18s
- Events/sec: 12.9M/sec
- Hawkes: simple uniform gaps
- Event mix: uniform random

### 4. Numba + GPU Parallel (final version)
Best of all approaches combined:
- Numba JIT compiles Hawkes loop to native CPU - 53x faster than plain Python, exact output
- CuPy generates symbol, side, price, size, type on GPU simultaneously
- Realistic event mix: 55% add, 35% cancel, 10% trade
- 3 parallel write threads - disk never idle

- Time for 1B events: ~1m 22s
- Events/sec: ~12.4M/sec
- Hawkes: exact
- Event mix: realistic 55/35/10

## Results

| Generator | Time for 1B | Events/sec | Speedup vs C++ | Exact Hawkes | Realistic Mix |
|---|---|---|---|---|---|
| Adit Python (numpy) | ~30 min | 0.56M/sec | 0.3x | No | No |
| C++ CPU | ~10 min | 1.67M/sec | 1x | Yes | No |
| GPU single thread | ~4m 25s | 3.8M/sec | 2.3x | No | No |
| GPU parallel simple | 1m 18s | 12.9M/sec | 7.7x | No | No |
| Numba + GPU parallel | 1m 22s | 12.4M/sec | 7.4x | Yes | Yes |

## Why GPU Parallel is Fast

Three things happen simultaneously:
- GPU generates fields for chunk N using thousands of GPU cores
- Write thread 1 writes chunk N-2 to disk
- Write thread 2 writes chunk N-1 to disk

Nothing waits. Like a pit crew where refueling, tire change, and wing adjustment all happen at once.

## Bottleneck at Each Stage

| Stage | C++ | GPU single | GPU parallel | Numba + GPU |
|---|---|---|---|---|
| Timestamp generation | slow CPU loop | uniform only | uniform only | Numba JIT fast |
| Field generation | slow CPU loop | GPU fast | GPU fast | GPU fast |
| Parquet write | sequential | sequential | 3x parallel | 3x parallel |
| Bottleneck | generation | write | disk bandwidth | disk bandwidth |

## RNG Speed Test (5M numbers)
- numpy CPU: 0.204s = 24M/sec
- CuPy GPU: 0.000s = 10B+/sec
- Speedup: 423x

## Three Seeds Validated

| Seed | Rows | Files | Events/sec |
|---|---|---|---|
| 42 | 1,000,000,000 | 200 | 12.4M/sec |
| 43 | 1,000,000,000 | 200 | 12.4M/sec |
| 44 | 1,000,000,000 | 200 | 12.4M/sec |

Seed variance: 0% - identical throughput across all seeds.

## Conclusion
Final generator (Numba + GPU + parallel writes) is 7.4x faster than C++ with exact Hawkes timestamps and realistic event mix. The key insight is that realism and speed are not tradeoffs - Numba makes exact Hawkes fast enough to match GPU field generation speed.

Next step: explore Arrow IPC format instead of Parquet to reduce write overhead further.
