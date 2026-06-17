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
OPENPCDET_DIR="${OPENPCDET_DIR:-${GITM_DATA_ROOT}/OpenPCDet}"
CKPT_DIR="${CKPT_DIR:-$GITM_DATA_ROOT/checkpoints/kitti}"
CKPT_PATH="$CKPT_DIR/pointpillar_7728.pth"
CKPT_URL="${CKPT_URL:-}"

CFG_PATH="$OPENPCDET_DIR/tools/cfgs/kitti_models/pointpillar.yaml"
# Accept either $GITM_DATA_ROOT/kitti/training or $GITM_DATA_ROOT/data/kitti/training
if [ -d "$GITM_DATA_ROOT/kitti/training/velodyne" ]; then
  KITTI_TRAINING="$GITM_DATA_ROOT/kitti/training"
elif [ -d "$GITM_DATA_ROOT/data/kitti/training/velodyne" ]; then
  KITTI_TRAINING="$GITM_DATA_ROOT/data/kitti/training"
else
  KITTI_TRAINING="$GITM_DATA_ROOT/kitti/training"
fi
KITTI_VELODYNE="$KITTI_TRAINING/velodyne"
EXPECTED_FRAMES=7481
EXPECTED_CKPT_SHA="4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2"

step() { echo ""; echo "==> $*"; }

# ── 1. Verify KITTI data ─────────────────────────────────────────────────────

step "Checking KITTI data at $KITTI_VELODYNE ..."
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
echo "  KITTI: $n_bins frames found OK at $KITTI_VELODYNE"

# ── 2. Clone + install OpenPCDet (pinned commit) ─────────────────────────────

PINNED_COMMIT="233f849829b6ac19afb8af8837a0246890908755"

step "Setting up OpenPCDet ..."
if [ ! -d "$OPENPCDET_DIR/.git" ]; then
  echo "  Cloning OpenPCDet ..."
  git clone https://github.com/open-mmlab/OpenPCDet.git "$OPENPCDET_DIR"
fi

actual_commit=$(git -C "$OPENPCDET_DIR" rev-parse HEAD)
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "  Pinning to commit $PINNED_COMMIT ..."
  git -C "$OPENPCDET_DIR" fetch --quiet
  git -C "$OPENPCDET_DIR" checkout "$PINNED_COMMIT"
fi
echo "  OpenPCDet at $PINNED_COMMIT"

step "Installing OpenPCDet deps ..."
# Do NOT install a fresh torch -- use whatever CUDA-matched version is already
# on the pod. Installing a mismatched torch breaks torchaudio/torchvision.
python -c "import torch; print('  torch', torch.__version__, '| CUDA', torch.version.cuda)"

# spconv: check if already importable (pod may have it from prior work), else
# try versioned wheels newest-to-oldest, fall back to building from source.
# spconv does not publish cu128 wheels; cu122 is the closest and is ABI-compatible
# with CUDA 12.x when torch is pre-installed.
if python -c "import spconv; print('  spconv', spconv.__version__, 'already installed')" 2>/dev/null; then
  echo "  spconv already present -- skipping install"
else
  echo "  Installing spconv (trying pre-built wheels then source) ..."
  pip install spconv-cu122 -q 2>/dev/null || \
  pip install spconv-cu121 -q 2>/dev/null || \
  pip install spconv-cu118 -q 2>/dev/null || {
    echo "  No pre-built wheel found -- building spconv from source (~10 min) ..."
    pip install "spconv @ git+https://github.com/traveller59/spconv.git@v2.3.6" --no-build-isolation -q
  }
fi

pip install -r "$OPENPCDET_DIR/requirements.txt" -q

# Install pcdet editable from OPENPCDET_DIR, then explicitly compile CUDA
# extensions if the .so files are missing (pip may skip build_ext on a
# re-install when it sees pcdet is "already satisfied" at a different path).
echo "  pip install -e OpenPCDet (no-build-isolation) ..."
pip install -e "$OPENPCDET_DIR" --no-build-isolation -q

SO_COUNT=$(find "$OPENPCDET_DIR/pcdet/ops" -name "*.so" 2>/dev/null | wc -l)
if [ "$SO_COUNT" -eq 0 ]; then
  echo "  No .so files found — building CUDA extensions (~5-10 min) ..."
  # Log to a file and surface the full log on failure; piping to `tail` would
  # both hide the real compile error and mask the non-zero exit status.
  build_log=$(mktemp)
  if ! ( cd "$OPENPCDET_DIR" && python setup.py build_ext --inplace ) >"$build_log" 2>&1; then
    echo "  ERROR: CUDA extension build failed. Full log:" >&2
    cat "$build_log" >&2
    exit 1
  fi
  rm -f "$build_log"
else
  echo "  CUDA extensions OK ($SO_COUNT .so files found)"
fi
# Verify pcdet AND a CUDA op import — a count-based .so check can't catch
# extensions built against a different Python/CUDA ABI, which is exactly the
# stale-install failure mode this step guards against.
python -c "import pcdet; from pcdet.ops.iou3d_nms import iou3d_nms_utils; print('  pcdet', pcdet.__version__, 'ready (CUDA ops import OK)')"

# ── 3. Install gitm package ──────────────────────────────────────────────────

step "Installing gitm package ..."
pip install hatchling -q   # build backend required by pyproject.toml
pip install -e "$REPO_ROOT[dev,bench,nvidia]" -q
echo "  gitm package installed."

# ── 4. Pull + verify checkpoint ─────────────────────────────────────────────

step "Checking checkpoint ..."
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_PATH" ]; then
  if [ -z "$CKPT_URL" ]; then
    # Try gdown for the public OpenPCDet Google Drive release
    echo "  No CKPT_URL set -- trying gdown for public OpenPCDet checkpoint ..."
    pip install gdown -q
    gdown "1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm" -O "$CKPT_PATH" || {
      echo "ERROR: checkpoint not found and gdown failed." >&2
      echo "  Set CKPT_URL to a pre-signed download URL and re-run." >&2
      exit 1
    }
  else
    echo "  Downloading checkpoint from CKPT_URL ..."
    curl -L "$CKPT_URL" -o "$CKPT_PATH"
  fi
fi
actual_sha=$(sha256sum "$CKPT_PATH" | awk '{print $1}')
if [ "$actual_sha" != "$EXPECTED_CKPT_SHA" ]; then
  echo "ERROR: checkpoint sha256 mismatch." >&2
  echo "  expected: $EXPECTED_CKPT_SHA" >&2
  echo "  actual:   $actual_sha" >&2
  exit 1
fi
echo "  Checkpoint sha256 OK"

# ── 5. Generate + verify sha256 manifest ────────────────────────────────────

step "Generating sha256 manifest ..."
python "$REPO_ROOT/harness/gen_kitti_manifest.py" \
  --root "$KITTI_TRAINING" \
  --out  "$REPO_ROOT/benchmarks/kitti/manifest.yaml"
echo "  Manifest written."

step "Verifying manifest (fast mode) ..."
python "$REPO_ROOT/harness/verify_manifest.py" --fast
echo "  Manifest verified."

# ── 6. Smoke test ───────────────────────────────────────────────────────────

step "Smoke test (10 frames) ..."
export PYTHONPATH="$OPENPCDET_DIR:${PYTHONPATH:-}"
GITM_DATA_ROOT="$GITM_DATA_ROOT" \
OPENPCDET_CFG="$CFG_PATH" \
OPENPCDET_CKPT="$CKPT_PATH" \
  python "$REPO_ROOT/harness/smoke_kitti.py" \
    --cfg "$CFG_PATH" --ckpt "$CKPT_PATH" --n-frames 10
echo "  Smoke test passed."

# ── 7. Run 6-seed baselines ─────────────────────────────────────────────────

step "Running 6-seed baselines (approx 75-100 min) ..."
GITM_DATA_ROOT="$GITM_DATA_ROOT" \
OPENPCDET_CFG="$CFG_PATH" \
OPENPCDET_CKPT="$CKPT_PATH" \
  bash "$REPO_ROOT/harness/run_baselines.sh"

# ── 8. Auto-fill results docs ────────────────────────────────────────────────

step "Filling spec.md + results.md ..."
GITM_DATA_ROOT="$GITM_DATA_ROOT" python "$REPO_ROOT/harness/fill_results.py"

# ── 9. Remind: commit ───────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  All done. Commit and push:"
echo "    cd $REPO_ROOT"
echo "    git add benchmarks/kitti/manifest.yaml benchmarks/kitti/spec.md benchmarks/kitti/results.md"
echo "    git commit -m 'KITTI: fill measured baseline numbers'"
echo "    git push"
echo "================================================================"
