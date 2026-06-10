import time
import numpy as np
import cupy as cp

# Warmup CuPy first
print("Warming up CuPy...")
_warm = cp.random.default_rng(0).integers(0, 100, size=100_000, dtype=cp.int32)
cp.cuda.Stream.null.synchronize()
print("Done\n")

for n in [5_000_000, 50_000_000, 100_000_000, 500_000_000]:
    # CPU numpy - 3 runs, take mean
    cpu_times = []
    for _ in range(3):
        rng_cpu = np.random.default_rng(42)
        t0 = time.perf_counter()
        _ = rng_cpu.integers(1, 1000, size=n, dtype=np.int32)
        cpu_times.append(time.perf_counter() - t0)
    cpu_time = sum(cpu_times) / 3

    # GPU cupy - 3 runs, take mean
    gpu_times = []
    for _ in range(3):
        rng_gpu = cp.random.default_rng(42)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = rng_gpu.integers(1, 1000, size=n, dtype=cp.int32)
        cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.perf_counter() - t0)
    gpu_time = sum(gpu_times) / 3

    print(f"n={n/1e6:.0f}M: numpy={cpu_time:.3f}s cupy={gpu_time:.3f}s speedup={cpu_time/gpu_time:.1f}x")
