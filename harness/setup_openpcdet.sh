#!/usr/bin/env bash
# Set up PyTorch + OpenPCDet and verify the PointPillars checkpoint on RunPod.
#
# Usage:
#   bash harness/setup_openpcdet.sh
#
# Expects:
#   - CUDA-capable GPU (A100 on RunPod)
#   - OpenPCDet already cloned at /workspace/edge/OpenPCDet (done by Kevin)
#   - CKPT_URL set to the checkpoint download URL (ask Adit), OR checkpoint
#     already present at /workspace/edge/checkpoints/pointpillar_7728.pth
#
# Sets:
#   OPENPCDET_CFG  — pointpillar.yaml path (used by harness + baseline scripts)
#   OPENPCDET_CKPT — checkpoint path

set -euo pipefail

OPENPCDET_DIR="${OPENPCDET_DIR:-/workspace/edge/OpenPCDet}"
CKPT_DIR="${CKPT_DIR:-/workspace/edge/checkpoints}"
CKPT_PATH="$CKPT_DIR/pointpillar_7728.pth"
CKPT_EXPECTED_SHA="4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2"
# Set CKPT_URL to the download URL Adit provides (Google Drive / S3 pre-signed link).
CKPT_URL="${CKPT_URL:-}"

CFG_PATH="$OPENPCDET_DIR/tools/cfgs/kitti_models/pointpillar.yaml"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 1. Install PyTorch (must come before OpenPCDet) ─────────────────────────

echo "==> Installing PyTorch 2.4.1 (cu124) …"
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

# ── 2. Install OpenPCDet ────────────────────────────────────────────────────

if [ ! -d "$OPENPCDET_DIR/.git" ]; then
  echo "ERROR: OpenPCDet not found at $OPENPCDET_DIR" >&2
  echo "  Clone it first: git clone https://github.com/open-mmlab/OpenPCDet.git $OPENPCDET_DIR" >&2
  exit 1
fi

echo "==> Installing OpenPCDet from $OPENPCDET_DIR …"
pip install -r "$OPENPCDET_DIR/requirements.txt" --quiet
pip install -e "$OPENPCDET_DIR" --quiet

ACTUAL_COMMIT=$(git -C "$OPENPCDET_DIR" rev-parse HEAD)
CFG_HASH=$(sha256sum "$CFG_PATH" | awk '{print $1}')

echo ""
echo "OpenPCDet commit       : $ACTUAL_COMMIT"
echo "pointpillar.yaml sha256: $CFG_HASH"
echo ""
echo "Fill these into benchmarks/edge/spec.md Section 2."

# ── 3. Pull checkpoint ───────────────────────────────────────────────────────

mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_PATH" ]; then
  echo "==> Checkpoint already present — verifying sha256 …"
else
  if [ -z "$CKPT_URL" ]; then
    echo "ERROR: CKPT_URL is not set and checkpoint is missing." >&2
    echo "  Set CKPT_URL to the download URL from Adit, then re-run." >&2
    exit 1
  fi
  echo "==> Downloading checkpoint …"
  curl -L "$CKPT_URL" -o "$CKPT_PATH"
fi

ACTUAL_SHA=$(sha256sum "$CKPT_PATH" | awk '{print $1}')
if [ "$ACTUAL_SHA" != "$CKPT_EXPECTED_SHA" ]; then
  echo "ERROR: checkpoint sha256 mismatch." >&2
  echo "  expected: $CKPT_EXPECTED_SHA" >&2
  echo "  actual  : $ACTUAL_SHA" >&2
  exit 1
fi
echo "==> Checkpoint sha256 OK"

# ── 4. Install gitm package ─────────────────────────────────────────────────

echo "==> Installing gitm package …"
pip install -e "$REPO_ROOT[dev,bench,nvidia]" -c "$REPO_ROOT/constraints.txt" --quiet

# ── 5. Build keyframe manifest (manifest.jsonl) ─────────────────────────────

echo "==> Building keyframe manifest …"
python -m gitm.bench edge-manifest \
  --root /workspace/edge \
  --out /workspace/edge/manifest.jsonl
echo "Keyframe manifest written to /workspace/edge/manifest.jsonl"

# ── 6. Smoke test: 10 frames end-to-end ─────────────────────────────────────

echo "==> Running smoke test (10 frames) …"
OPENPCDET_CFG="$CFG_PATH" \
OPENPCDET_CKPT="$CKPT_PATH" \
GITM_BENCH_STAGE=/workspace/edge \
  python "$REPO_ROOT/benchmarks/edge/harness.py" \
    --seed 42 --warm-frames 10 \
    --stage /workspace/edge

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Generate SHA256 manifest:"
echo "     python -m gitm.bench manifest build \\"
echo "       --root /workspace/edge/kitti \\"
echo "       --benchmark edge \\"
echo "       --out /workspace/runtime/benchmarks/edge/manifest.yaml"
echo ""
echo "  2. Run 3-seed baselines:"
echo "     cd /workspace/runtime/benchmarks/edge"
echo "     STAGE=/workspace/edge \\"
echo "     GITM_BENCH_STAGE=/workspace/edge \\"
echo "     OPENPCDET_CFG=$CFG_PATH \\"
echo "     OPENPCDET_CKPT=$CKPT_PATH \\"
echo "     make baseline"
