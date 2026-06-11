#!/usr/bin/env bash
# One-shot KITTI benchmark setup + baseline run for a fresh RunPod pod.
#
# Run AFTER kitti data is downloaded (fetch.py --step kitti already ran).
# Checks the download is complete, sets up OpenPCDet, generates the manifest,
# runs 6-seed baselines, and prints a convergence summary.
#
# Usage (from /workspace/runtime on the RunPod pod):
#   bash harness/pod_setup_and_run.sh
#
# Override defaults:
#   GITM_DATA_ROOT=/workspace/edge bash harness/pod_setup_and_run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GITM_DATA_ROOT="${GITM_DATA_ROOT:-/workspace/edge}"
OPENPCDET_DIR="${OPENPCDET_DIR:-/workspace/edge/OpenPCDet}"
CKPT_DIR="${CKPT_DIR:-$GITM_DATA_ROOT/checkpoints/kitti}"
CKPT_PATH="$CKPT_DIR/pointpillar_7728.pth"
CKPT_URL="${CKPT_URL:-}"

CFG_PATH="$OPENPCDET_DIR/tools/cfgs/kitti_models/pointpillar.yaml"
KITTI_VELODYNE="$GITM_DATA_ROOT/kitti/training/velodyne"
EXPECTED_FRAMES=7481
EXPECTED_CKPT_SHA="4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2"

step() { echo ""; echo "==> $*"; }

# ── 1. Verify KITTI data ────────────────────────────────────────────────────

step "Checking KITTI data at $KITTI_VELODYNE …"
if [ ! -d "$KITTI_VELODYNE" ]; then
  echo "ERROR: KITTI velodyne dir not found: $KITTI_VELODYNE" >&2
  echo "  Run first: python $REPO_ROOT/benchmarks/edge/fetch.py --step kitti --out $GITM_DATA_ROOT" >&2
  exit 1
fi
n_bins=$(find "$KITTI_VELODYNE" -name "*.bin" | wc -l)
if [ "$n_bins" -lt "$EXPECTED_FRAMES" ]; then
  echo "ERROR: only $n_bins .bin files found (expected $EXPECTED_FRAMES). Download still running?" >&2
  exit 1
fi
echo "  KITTI: $n_bins frames found OK"

# ── 2. Clone + install OpenPCDet (pinned commit) ────────────────────────────

PINNED_COMMIT="233f849829b6ac19afb8af8837a0246890908755"

step "Setting up OpenPCDet …"
if [ ! -d "$OPENPCDET_DIR/.git" ]; then
  echo "  Cloning OpenPCDet …"
  git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_DIR"
fi

actual_commit=$(git -C "$OPENPCDET_DIR" rev-parse HEAD)
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "  Pinning to commit $PINNED_COMMIT …"
  git -C "$OPENPCDET_DIR" checkout "$PINNED_COMMIT"
fi
echo "  OpenPCDet at $PINNED_COMMIT"

step "Installing PyTorch + OpenPCDet deps …"
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 -q
pip install spconv-cu124 -q
pip install -r "$OPENPCDET_DIR/requirements.txt" -q
pip install -e "$OPENPCDET_DIR" -q

step "Installing gitm package …"
pip install -e "$REPO_ROOT[dev,bench,nvidia]" -q

# ── 3. Pull + verify checkpoint ──────────────────────────────────────────────

step "Checking checkpoint …"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_PATH" ]; then
  if [ -z "$CKPT_URL" ]; then
    echo "ERROR: checkpoint not found and CKPT_URL is not set." >&2
    echo "  Download manually or set CKPT_URL to the pre-signed URL from Adit." >&2
    echo "  Checkpoint: https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm" >&2
    exit 1
  fi
  echo "  Downloading checkpoint …"
  curl -L "$CKPT_URL" -o "$CKPT_PATH"
fi
actual_sha=$(sha256sum "$CKPT_PATH" | awk '{print $1}')
if [ "$actual_sha" != "$EXPECTED_CKPT_SHA" ]; then
  echo "ERROR: checkpoint sha256 mismatch" >&2
  echo "  expected: $EXPECTED_CKPT_SHA" >&2
  echo "  actual:   $actual_sha" >&2
  exit 1
fi
echo "  Checkpoint sha256 OK"

# ── 4. Generate sha256 manifest ──────────────────────────────────────────────

step "Generating sha256 manifest …"
python "$REPO_ROOT/harness/gen_kitti_manifest.py" \
  --root "$GITM_DATA_ROOT/kitti/training" \
  --out  "$REPO_ROOT/benchmarks/kitti/manifest.yaml"
echo "  Manifest written."

step "Verifying manifest (fast mode) …"
python "$REPO_ROOT/harness/verify_manifest.py" --fast
echo "  Manifest verified."

# ── 5. Smoke test ────────────────────────────────────────────────────────────

step "Smoke test (10 frames) …"
export GITM_DATA_ROOT OPENPCDET_CFG="$CFG_PATH" OPENPCDET_CKPT="$CKPT_PATH"
python "$REPO_ROOT/harness/smoke_kitti.py" \
  --cfg "$CFG_PATH" --ckpt "$CKPT_PATH" --n-frames 10
echo "  Smoke test passed."

# ── 6. Run 6-seed baselines ──────────────────────────────────────────────────

step "Running 6-seed baselines (this takes ~1 hour) …"
GITM_DATA_ROOT="$GITM_DATA_ROOT" \
OPENPCDET_CFG="$CFG_PATH" \
OPENPCDET_CKPT="$CKPT_PATH" \
  bash "$REPO_ROOT/harness/run_baselines.sh"

# ── 7. Remind: commit results ────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  All done. Fill measured numbers into:"
echo "    benchmarks/kitti/spec.md    (Section 3 + 5 tables)"
echo "    benchmarks/kitti/results.md (baseline table)"
echo ""
echo "  Then commit and push:"
echo "    git add benchmarks/kitti/manifest.yaml benchmarks/kitti/spec.md benchmarks/kitti/results.md"
echo "    git commit -m 'KITTI: fill measured baseline numbers'"
echo "    git push"
echo "================================================================"
