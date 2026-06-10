import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from numba import njit

MU = 100.0
ALPHA = 0.6
BETA = 0.8

@njit(cache=True)
def hawkes_numba(u, n, t0, lam0):
    ts = np.empty(n, dtype=np.int64)
    t = t0
    lam = lam0
    for i in range(n):
        dt = -np.log(u[i]) / lam
        t += dt
        lam = MU + (lam - MU) * np.exp(-BETA * dt) + ALPHA
        ts[i] = int(t * 1e9)
    return ts, t, lam

def warmup():
    u = np.random.default_rng(0).random(100)
    hawkes_numba(u, 100, 0.0, MU)
    cp.random.default_rng(0).integers(0, 100, size=10_000)
    cp.cuda.Stream.null.synchronize()

def generate_chunk(rng, rng_np, n, t_hawkes, lambda_hawkes):
    # Hawkes timestamps via Numba (fast CPU)
    u = rng_np.random(n)
    ts_ns, t_hawkes, lambda_hawkes = hawkes_numba(u, n, t_hawkes, lambda_hawkes)

    # All other fields on GPU
    symbol_id = cp.asnumpy(rng.integers(0, 512, size=n, dtype=cp.int32))
    side = cp.asnumpy(rng.integers(0, 2, size=n, dtype=cp.int8))
    price = cp.asnumpy(rng.integers(9000, 11000, size=n, dtype=cp.int64))
    size_arr = cp.asnumpy(rng.integers(1, 1000, size=n, dtype=cp.int32))
    roll = cp.asnumpy(rng.random(n, dtype=cp.float32))
    etype = np.where(roll < 0.55, 0, np.where(roll < 0.90, 1, 2)).astype(np.int8)

    return {
        'ts_ns': ts_ns, 'symbol_id': symbol_id, 'side': side,
        'price': price, 'size': size_arr, 'type': etype,
    }, t_hawkes, lambda_hawkes

def write_chunk(data, path):
    table = pa.table({
        'ts_ns': pa.array(data['ts_ns'], type=pa.int64()),
        'symbol_id': pa.array(data['symbol_id'], type=pa.int32()),
        'side': pa.array(data['side'], type=pa.int8()),
        'price': pa.array(data['price'], type=pa.int64()),
        'size': pa.array(data['size'], type=pa.int32()),
        'type': pa.array(data['type'], type=pa.int8()),
    })
    pq.write_table(table, path, compression='zstd', compression_level=1)

def generate(n_events, seed, out_dir, chunk_size=5_000_000, n_writers=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Warming up...")
    warmup()
    rng = cp.random.default_rng(seed)
    rng_np = np.random.default_rng(seed)
    n_chunks = n_events // chunk_size
    t_hawkes = 0.0
    lambda_hawkes = MU
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_writers) as executor:
        futures = []
        for i in range(n_chunks):
            data, t_hawkes, lambda_hawkes = generate_chunk(
                rng, rng_np, chunk_size, t_hawkes, lambda_hawkes)
            path = out_dir / f"part-{i:05d}.parquet"
            futures.append(executor.submit(write_chunk, data, path))
            if len(futures) > n_writers:
                futures.pop(0).result()
            if i % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"Progress: {(i+1)*chunk_size/1e6:.0f}M/{n_events/1e6:.0f}M, {elapsed:.1f}s", end='\r')
        for f in futures:
            f.result()

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_events/1e6:.0f}M events in {elapsed:.1f}s = {n_events/elapsed/1e6:.1f}M events/sec")
    print("FINISHED")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=100_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="/workspace/test_gpu_output")
    p.add_argument("--chunk-size", type=int, default=5_000_000)
    p.add_argument("--writers", type=int, default=3)
    args = p.parse_args()
    generate(args.events, args.seed, args.out, args.chunk_size, args.writers)
