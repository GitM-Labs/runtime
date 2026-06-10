import time
import threading
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def generate_chunk_gpu(rng, n, ts_start):
    gaps = rng.integers(1, 2000, size=n, dtype=cp.int64)
    ts_ns = ts_start + cp.cumsum(gaps)
    symbol_id = rng.integers(0, 512, size=n, dtype=cp.int32)
    side = rng.integers(0, 2, size=n, dtype=cp.int8)
    price = rng.integers(9000, 11000, size=n, dtype=cp.int64)
    size_arr = rng.integers(1, 1000, size=n, dtype=cp.int32)
    etype = rng.integers(0, 3, size=n, dtype=cp.int8)
    return {'ts_ns': cp.asnumpy(ts_ns), 'symbol_id': cp.asnumpy(symbol_id),
            'side': cp.asnumpy(side), 'price': cp.asnumpy(price),
            'size': cp.asnumpy(size_arr), 'type': cp.asnumpy(etype)}, int(ts_ns[-1])

def write_chunk(data, path):
    table = pa.table({'ts_ns': pa.array(data['ts_ns'], type=pa.int64()),
        'symbol_id': pa.array(data['symbol_id'], type=pa.int32()),
        'side': pa.array(data['side'], type=pa.int8()),
        'price': pa.array(data['price'], type=pa.int64()),
        'size': pa.array(data['size'], type=pa.int32()),
        'type': pa.array(data['type'], type=pa.int8())})
    pq.write_table(table, path, compression='zstd', compression_level=1)

def generate(n_events, seed, out_dir, chunk_size=5_000_000, n_writers=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp.random.default_rng(0).integers(0, 100, size=10_000)
    cp.cuda.Stream.null.synchronize()
    rng = cp.random.default_rng(seed)
    n_chunks = n_events // chunk_size
    ts_cursor = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_writers) as executor:
        futures = []
        for i in range(n_chunks):
            data, ts_cursor = generate_chunk_gpu(rng, chunk_size, ts_cursor)
            path = out_dir / f"part-{i:05d}.parquet"
            futures.append(executor.submit(write_chunk, data, path))

            # Keep only last n_writers futures in flight
            if len(futures) > n_writers:
                futures.pop(0).result()

            if i % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"Progress: {(i+1)*chunk_size/1e6:.0f}M/{n_events/1e6:.0f}M, {elapsed:.1f}s", end='\r')

        # Wait for remaining writes
        for f in futures:
            f.result()

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_events/1e6:.0f}M events in {elapsed:.1f}s = {n_events/elapsed/1e6:.1f}M events/sec")
    print("FINISHED")

if __name__ == "__main__":
    generate(1_000_000_000, 42, '/workspace/workspace/hft_gpu_seed42')
