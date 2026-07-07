import time

import cupy as cp
import pyarrow as pa
import pyarrow.parquet as pq

n = 5_000_000
seed = 42

rng = cp.random.default_rng(seed)
symbol_id = rng.integers(0, 512, size=n, dtype=cp.int32)
side = rng.integers(0, 2, size=n, dtype=cp.int8)
price = rng.integers(9000, 11000, size=n, dtype=cp.int64)
size = rng.integers(1, 1000, size=n, dtype=cp.int32)
etype = rng.integers(0, 3, size=n, dtype=cp.int8)
gaps = rng.integers(1, 2000, size=n, dtype=cp.int64)
ts_ns = cp.cumsum(gaps)

table = pa.table({
    'ts_ns': pa.array(cp.asnumpy(ts_ns)),
    'symbol_id': pa.array(cp.asnumpy(symbol_id)),
    'side': pa.array(cp.asnumpy(side)),
    'price': pa.array(cp.asnumpy(price)),
    'size': pa.array(cp.asnumpy(size)),
    'type': pa.array(cp.asnumpy(etype)),
})

# Test no compression
t0 = time.perf_counter()
pq.write_table(table, '/workspace/test_nocomp.parquet', compression='none')
print(f"No compression: {time.perf_counter()-t0:.3f}s")

# Test zstd
t0 = time.perf_counter()
pq.write_table(table, '/workspace/test_zstd.parquet', compression='zstd', compression_level=1)
print(f"zstd-1: {time.perf_counter()-t0:.3f}s")

# Test snappy
t0 = time.perf_counter()
pq.write_table(table, '/workspace/test_snappy.parquet', compression='snappy')
print(f"snappy: {time.perf_counter()-t0:.3f}s")
