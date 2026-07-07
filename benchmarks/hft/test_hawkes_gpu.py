import time

import numpy as np
from numba import njit

MU = 100.0
ALPHA = 0.6
BETA = 0.8

@njit(cache=True)
def hawkes_numba(u, n, t0, lam0):
    """JIT-compiled Hawkes — exact, fast, sequential."""
    ts = np.empty(n, dtype=np.int64)
    t = t0
    lam = lam0
    for i in range(n):
        dt = -np.log(u[i]) / lam
        t += dt
        lam = MU + (lam - MU) * np.exp(-BETA * dt) + ALPHA
        ts[i] = int(t * 1e9)
    return ts, t, lam

def hawkes_cpu_plain(n, t0, lam0, seed):
    rng = np.random.default_rng(seed)
    ts = np.empty(n, dtype=np.int64)
    t = t0
    lam = lam0
    for i in range(n):
        u = rng.random()
        dt = -np.log(u) / lam
        t += dt
        lam = MU + (lam - MU) * np.exp(-BETA * dt) + ALPHA
        ts[i] = int(t * 1e9)
    return ts

n = 1_000_000
rng_np = np.random.default_rng(42)
u = rng_np.random(n)

# Warmup numba JIT
print("Warming up Numba JIT...")
_ = hawkes_numba(u[:100], 100, 0.0, MU)

# Numba
t0 = time.perf_counter()
ts_numba, _, _ = hawkes_numba(u, n, 0.0, MU)
numba_time = time.perf_counter() - t0
print(f"Numba JIT: {numba_time:.3f}s = {n/numba_time/1e6:.1f}M/sec")

# Plain CPU
t0 = time.perf_counter()
ts_cpu = hawkes_cpu_plain(n, 0.0, MU, 42)
cpu_time = time.perf_counter() - t0
print(f"Plain CPU: {cpu_time:.3f}s = {n/cpu_time/1e6:.1f}M/sec")

print(f"Speedup: {cpu_time/numba_time:.1f}x")
print(f"Monotonic: {bool(np.all(np.diff(ts_numba) > 0))}")
