#!/usr/bin/env bash
# Run 6-seed KITTI PointPillars baselines (seeds 42–47) and check convergence.
#
# Usage:
#   bash harness/run_baselines.sh
#
# Required env vars:
#   GITM_DATA_ROOT      — root of kitti/training/velodyne/ + output dir
#   OPENPCDET_CFG       — path to pointpillar.yaml  (set by setup_openpcdet.sh)
#   OPENPCDET_CKPT      — path to pointpillar_7728.pth  (set by setup_openpcdet.sh)
#
# Each run writes to $GITM_DATA_ROOT/runs/kitti_baseline_{1..6}.json
# After all runs complete, the script prints a convergence summary.
#
# For a quick 3-seed check: N_SEEDS=3 bash harness/run_baselines.sh

set -euo pipefail

: "${GITM_DATA_ROOT:?ERROR: GITM_DATA_ROOT is not set}"
: "${OPENPCDET_CFG:?ERROR: OPENPCDET_CFG is not set (path to pointpillar.yaml)}"
: "${OPENPCDET_CKPT:?ERROR: OPENPCDET_CKPT is not set (path to pointpillar_7728.pth)}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="$GITM_DATA_ROOT/runs"
mkdir -p "$RUNS_DIR"

N_FRAMES="${N_FRAMES:-7481}"
N_SEEDS="${N_SEEDS:-6}"   # set to 3 for a quick 3-seed check

SEEDS=(42 43 44 45 46 47)

run_baseline() {
  local seed="$1"
  local run_num="$2"
  local out="$RUNS_DIR/kitti_baseline_${run_num}.json"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  Baseline $run_num  (seed=$seed, frames=$N_FRAMES)"
  echo "══════════════════════════════════════════"

  python -m gitm.benchmarks.kitti.baseline \
    --cfg        "$OPENPCDET_CFG" \
    --ckpt       "$OPENPCDET_CKPT" \
    --data-root  "$GITM_DATA_ROOT" \
    --seed       "$seed" \
    --frames     "$N_FRAMES" \
    --output     "$out"

  echo "  Wrote: $out"
}

for i in $(seq 1 "$N_SEEDS"); do
  run_baseline "${SEEDS[$((i-1))]}" "$i"
done

echo ""
echo "══════════════════════════════════════════"
echo "  Convergence check ($N_SEEDS seeds)"
echo "══════════════════════════════════════════"

# Build the list of result files written above
result_files=()
for i in $(seq 1 "$N_SEEDS"); do
  result_files+=("$RUNS_DIR/kitti_baseline_${i}.json")
done

python - "${result_files[@]}" << 'PYEOF'
import json, sys

paths = sys.argv[1:]
results = []
for p in paths:
    with open(p) as f:
        results.append(json.load(f))

fps_vals  = [r["frames_per_second"] for r in results]
gpu_vals  = [r["gpu_active_pct"] for r in results]
data_vals = [r["data_stall_pct"] for r in results]
seeds     = [r["seed"] for r in results]

spread = (max(fps_vals) - min(fps_vals)) / max(fps_vals) * 100
ok = spread <= 2.0

print(f"  seeds: {' | '.join(str(s) for s in seeds)}")
print(f"  fps:   {' | '.join(f'{v:.2f}' for v in fps_vals)}")
print(f"  gpu%:  {' | '.join(f'{v:.1f}' for v in gpu_vals)}")
print(f"  data%: {' | '.join(f'{v:.1f}' for v in data_vals)}")
print(f"  spread: {spread:.2f}%  ({'PASS' if ok else 'FAIL -- re-run and investigate'})")

# Print headroom if available
headroom_vals = [r.get("compute_headroom_pct") for r in results]
if any(v is not None for v in headroom_vals):
    print(f"  headroom%: {' | '.join(f'{v:.1f}' if v is not None else 'N/A' for v in headroom_vals)}")

max_gpu = max(gpu_vals)
if max_gpu > 85.0:
    print(f"\n  WARNING: GPU active {max_gpu:.1f}% > 85% -- flag Adit for 500-frame fallback.")
else:
    print(f"\n  GPU active < 85% OK")

if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "All $N_SEEDS baselines complete."
echo "Fill fps + stall numbers into benchmarks/kitti/results.md"
