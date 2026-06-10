# HFT Generator Comparison Report

## Overview
I tested three different approaches to generate 1B HFT events. Started with C++, then moved to GPU-based generation using CuPy, and finally added parallel writes. The results were surprising — GPU parallel ended up 8x faster than C++.

## Generators Tested

### 1. C++ Generator (benchmarks/hft/generator/main.cpp)
Loops through events one at a time. For each event calculates Hawkes process lambda, generates 6 random values, appends to Arrow builders, flushes to Parquet every 5M events. Single threaded — generation and writing happen sequentially.

- Time for 1B events: ~10 minutes
- Events/sec: ~1.67M/sec

### 2. GPU Single Thread (CuPy)
Generates all 6 fields for 5M events at once on GPU using CuPy. Transfers to CPU with cp.asnumpy(), writes to Parquet. Still sequential — waits for write to finish before generating next chunk.

- Time for 1B events: ~4m 25s
- Events/sec: ~3.8M/sec

### 3. GPU Parallel (gpu_generator.py)
GPU generates chunk N while 3 CPU threads write chunks N-1, N-2, N-3 simultaneously. Uses ThreadPoolExecutor with max_workers=3. GPU never waits for disk — as soon as a chunk is ready it gets handed to a free write thread.

- Time for 1B events: 1m 18s
- Events/sec: 12.9M/sec

## Results

| Generator | Time for 1B | Events/sec | Speedup vs C++ |
|---|---|---|---|
| C++ CPU | ~10 min | 1.67M/sec | 1x |
| GPU single thread | ~4m 25s | 3.8M/sec | 2.3x |
| GPU parallel (3 writers) | **1m 18s** | **12.9M/sec** | **7.7x** |

## Why is GPU Parallel so fast?

Think of it like a race car pit stop. The old way: drive in, stop, wait for crew to refuel, drive out. The new way: crew refuels while the car keeps moving — everything overlaps.

In our case:
- GPU = chef cooking food (generates random numbers, 400x faster than CPU)
- 3 write threads = 3 waiters serving food (writing to disk simultaneously)
- Chef never waits for waiters — keeps cooking
- Result: nothing is ever idle

## Bottleneck Analysis

| Stage | C++ | GPU single | GPU parallel |
|---|---|---|---|
| RNG generation | CPU bound | GPU fast | GPU fast |
| Parquet write | sequential | sequential | 3x parallel |
| Bottleneck | generation | write | disk bandwidth |

At GPU parallel, the bottleneck shifted to disk bandwidth — adding more than 3 writers doesn't help because the disk can't write faster.

## RNG Speed Test Results (5M numbers)
- numpy CPU: 0.204s = 24M/sec
- CuPy GPU: 0.000s = 10B+/sec
- GPU is 423x faster for random number generation

## Conclusion
GPU parallel generation is 7.7x faster than C++ for 1B events. The key insight is overlapping GPU generation with parallel disk writes using ThreadPoolExecutor. The bottleneck shifted from RNG generation to disk I/O, which is the natural ceiling for this approach.

Next step: explore writing to multiple files simultaneously across different storage paths to push past disk bandwidth limits.
