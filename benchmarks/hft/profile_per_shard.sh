#!/usr/bin/env bash
# Per-shard profiling for Granger causality (Task 3).
#
# attribute() runs grangercausalitytests at maxlag=2, which needs at LEAST
# 8 observations (samples) per op -- not the naive max_lag+2=4. And even at
# 8-20 samples Granger p-values are noisy: treat the ranking as suggestive,
# not conclusive. One nsys run on the whole 25M window gives 1 sample/op,
# so we profile many shards separately, one profile per shard = 1 sample/op
# each. Default N=20 shards (=100M events profiled) for a usable series.
#
# After this completes, parse_shard_profiles.py turns the 5 .sqlite files
# into a residual time series and runs attribute().
#
# Run on the pod:
#   bash profile_per_shard.sh /workspace/hft_numba_seed42
#
# NOTE: this profiles 5 single-shard runs, NOT the 25M concatenated run.
# Each run loads exactly one part-0000N.parquet (5M events). That's the
# point -- we want per-shard kernel times as independent samples.

set -euo pipefail

STAGE="${1:-/workspace/hft_numba_seed42}"
OUT="${2:-/workspace/shard_profiles}"
HARNESS="benchmarks/hft/harness.py"

mkdir -p "$OUT"

# Make a temp staging dir holding one shard at a time, because the harness
# globs part-*.parquet from a directory. We symlink a single shard in, profile,
# then swap. seed is parsed from the stage dir name (…seed42 -> 42).
SEED="$(basename "$STAGE" | grep -oE '[0-9]+$')"

NSHARDS="${3:-20}"
for ((i=0; i<NSHARDS; i++)); do
  SHARD=$(printf "part-%05d.parquet" "$i")
  if [[ ! -f "$STAGE/$SHARD" ]]; then
    echo "missing $STAGE/$SHARD, stopping"
    break
  fi

  TMP="$OUT/stage_$i/hft_numba_seed${SEED}"
  mkdir -p "$TMP"
  ln -sf "$STAGE/$SHARD" "$TMP/part-00000.parquet"

  echo "=== profiling shard $i ($SHARD) ==="
  nsys profile \
    --trace cuda,nvtx,osrt \
    --output "$OUT/shard_$i" \
    --force-overwrite true \
    python3 "$HARNESS" --seed "$SEED" --stage "$OUT/stage_$i" --max-events 5000000

  # Export the kernel summary table to sqlite for parsing.
  nsys stats --report cuda_gpu_kern_sum --format csv \
    --output "$OUT/shard_$i" "$OUT/shard_$i.nsys-rep" || true
done

echo
echo "done. profiles in $OUT/shard_*.nsys-rep (+ .csv summaries)"
echo "next: python3 parse_shard_profiles.py $OUT"
