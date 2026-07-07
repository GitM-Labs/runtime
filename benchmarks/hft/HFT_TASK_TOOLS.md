# HFT Task 1 + Task 3 — tools and run order

Three scripts that finish the predicted-graph integration (Task 1) and the
Granger causality analysis (Task 3). All three need the pod (the data and
nsys live there). Run them in this order.

## What's measured vs estimated (read this first)

The predicted-graph residuals depend on byte counts. Right now hft_graph.py
uses ESTIMATES:
  - compression_ratio = 3.0  (a guess; zstd-1 on int columns is ~2.5-4x)
  - per-stage bytes_moved (assumed multiples of the uncompressed size)

The observed kernel times ARE real (from nsys). So the residual MAGNITUDES
(the "592x" numbers) are only as good as the byte estimates. Step 1 below
replaces the compression-ratio guess with a measured value. The Granger
RANKING (step 3) is more robust to the byte estimates than the magnitudes,
because it depends on relative timing across shards, not absolute residual
size — but at 20 samples it's still suggestive, not conclusive.

## Step 1 — measure the real compression ratio (no nsys, just file reads)

    python3 measure_compression.py /workspace/hft_numba_seed42

Copy the printed "compression_ratio (parquet meta)" value and replace
compression_ratio=3.0 in HFTDatasetSpec (benchmarks or gitm/planner/hft_graph.py).
Re-run `python3 -m gitm.planner.hft_residual_demo` to get corrected magnitudes.

## Step 2 — profile 20 shards separately (needs nsys + GPU)

    bash profile_per_shard.sh /workspace/hft_numba_seed42 /workspace/shard_profiles 20

Profiles 20 single-shard harness runs (5M events each) -> 20 nsys profiles ->
20 samples per kernel op. This is the part that takes a while (20 nsys runs).

WHY 20 AND NOT 5: grangercausalitytests at maxlag=2 needs >= 8 observations
or statsmodels raises "Maximum allowable lag is 0". Even at 8-20 the p-values
are noisy. 20 is a practical floor; more shards = more reliable.

## Step 3 — run Granger causality on the residual series

    python3 parse_shard_profiles.py /workspace/shard_profiles

Parses the 20 CSV summaries, builds a per-op residual time series against the
predicted graph, and runs gitm.optimizer.attribution.attribute(). Output is a
ranked table of (cause -> effect, p_value). A low p for
zstd_decompress -> parquet_decode would confirm decompression is upstream of
the decode slowdown, not just correlated with it.

## Validated locally

The parse + Granger chain was tested with 20 synthetic shards carrying a known
zstd->parquet lagged dependence. attribute() correctly ranked
zstd_decompress -> parquet_decode at p=0.0000 and the unrelated merge_sort
pairs as non-significant (p>0.18). So the code path works; on real data the
result depends on what the actual per-shard timings show.

## Honest limitations to mention to Adit

1. compression_ratio is estimated until step 1 is run on real files.
2. Granger at n=20 is suggestive, not conclusive — more shards would help.
3. roofline still has no decompression-bound category; the residuals are vs
   the memory-bandwidth floor, which these ops were never going to hit. The
   "right" fix is a new prediction category calibrated against nvCOMP zstd
   throughput numbers — that's a proposal for Adit, not something to merge
   into the shared roofline.py unilaterally.
