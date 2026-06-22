#!/usr/bin/env bash
# Edge speed run — OpenPCDet PointPillars fps over the warm window, 3 seeds.
# Produces the frames_per_second numbers + 2%-agreement check from spec.md §3.
#
# Run ON THE GPU POD (needs torch + pcdet + CUDA + the staged edge dataset).
#   bash run_edge_speeds.sh            # full 5000-frame warm window
#   WARM=500 bash run_edge_speeds.sh   # quick sighter first
#
# Run it with `bash`, NOT `source`/`.` — see the guard below.

# Guard: if this file is sourced (`. script` / `source script`) instead of run
# (`bash script`), a failed preflight's `exit` runs in your LOGIN shell and
# drops your SSH session. Detect sourcing and re-exec in a child bash so `exit`
# only ends the script, never your pod session.
if (return 0 2>/dev/null); then
  _self="${BASH_SOURCE[0]:-}"
  if [ -n "$_self" ]; then
    bash "$_self" "$@"; return $?
  fi
  echo "Don't source this script — run it:  bash run_edge_speeds.sh" >&2
  return 1
fi

set -euo pipefail

# Run from this script's own directory so `harness.py` / `build_manifest.py`
# resolve no matter where the caller invokes it from (e.g. repo root).
cd "$(dirname "${BASH_SOURCE[0]}")"

# Data is staged under /workspace/edge/data on the pod (manifest + checkpoints
# both live there). Override any of these via the environment.
STAGE="${GITM_BENCH_STAGE:-/workspace/edge/data}"
WARM="${WARM:-5000}"
export OPENPCDET_CFG="${OPENPCDET_CFG:-/workspace/edge/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml}"
export OPENPCDET_CKPT="${OPENPCDET_CKPT:-/workspace/edge/data/checkpoints/kitti/pointpillar_7728.pth}"
export GITM_BENCH_STAGE="$STAGE"

echo "stage=$STAGE  warm=$WARM"
echo "cfg=$OPENPCDET_CFG"
echo "ckpt=$OPENPCDET_CKPT"

# Preflight: the things that most often blow up a pod run.
[ -f "$STAGE/manifest.jsonl" ] || { echo "MISSING $STAGE/manifest.jsonl — build it: GITM_DATA_ROOT=$STAGE python build_manifest.py (or make keyframes)"; exit 1; }
[ -f "$OPENPCDET_CKPT" ] || { echo "MISSING checkpoint $OPENPCDET_CKPT"; exit 1; }
[ -f "$OPENPCDET_CFG" ] || { echo "MISSING config $OPENPCDET_CFG"; exit 1; }
echo "manifest rows: $(wc -l < "$STAGE/manifest.jsonl")"

declare -a FPS
for seed in 42 43 44; do
  echo "=== seed $seed ==="
  out="$(python harness.py --seed "$seed" --warm-frames "$WARM")"
  echo "$out" | grep '^\[edge harness'
  fps="$(echo "$out" | tail -1 | python -c 'import sys,json; print(json.load(sys.stdin)["metric_value"])')"
  FPS+=("$fps")
done

python - "${FPS[@]}" <<'PY'
import sys
v=[float(x) for x in sys.argv[1:]]
mean=sum(v)/len(v)
spread=(max(v)-min(v))/mean if mean else 0
print("\n==== EDGE SPEED SUMMARY ====")
for s,f in zip((42,43,44), v): print(f"  seed {s}: {f:8.1f} fps")
print(f"  mean   : {mean:8.1f} fps")
print(f"  spread : {spread*100:6.2f}%  ({'PASS' if spread<=0.02 else 'FAIL'} vs 2% tolerance)")
PY
