#!/usr/bin/env bash
# Build the gitm-ready job image on this node, once: the vLLM image plus what
# gitm's CUPTI collector needs to compile and run inside a job.
#
#   sudo ./build-image.sh                       # from the pre-imported vLLM image
#   sudo ./build-image.sh --gitm-ref main       # which gitm to test-build against
#
# Adds to the image:
#   * a C compiler (build-essential)
#   * CUDA toolkit headers matching the image's own CUDA (cupti.h, cuda_runtime.h + crt/)
#   * libcupti matching the DRIVER's CUDA major (pip nvidia-cuda-cupti): CUPTI's
#     activity API refuses to start when the two majors differ
# then installs gitm and compiles the collector once as a check, so a broken
# toolchain fails here instead of inside someone's job.
# The result becomes the default job image (DEFAULT_IMAGE in /etc/gitm-gpu/config).
# Run on every GPU node. Idempotent; rebuilds when re-run.
set -euo pipefail

SRC=""
OUT="/var/lib/gitm/images/gitm-vllm.sqsh"
GITM_REF="main"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)     SRC="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    --gitm-ref) GITM_REF="$2"; shift 2 ;;
    -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
SRC=${SRC:-$(ls /var/lib/gitm/images/vllm_vllm-openai_latest.sqsh 2>/dev/null || true)}
[[ -f "$SRC" ]] || { echo "source image not found; pass --from /path/to/image.sqsh" >&2; exit 1; }
DRIVER_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)
[[ -n "$DRIVER_MAJOR" ]] || { echo "could not read the driver's CUDA version from nvidia-smi" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

export ENROOT_DATA_PATH=/var/lib/enroot/data/root ENROOT_CACHE_PATH=/var/lib/enroot/cache/root
export ENROOT_RUNTIME_PATH=/run/enroot/root
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH"
NAME="gitm-build-$$"
cleanup() { enroot remove --force "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

step "Unpacking $SRC"
enroot create --name "$NAME" "$SRC" >/dev/null

step "Installing build tools, CUDA headers and libcupti ${DRIVER_MAJOR} inside the image"
# --root: root inside the container only. NVIDIA_VISIBLE_DEVICES=void: no GPUs
# needed (or touched) to build.
enroot start --root --rw --env NVIDIA_VISIBLE_DEVICES=void --env DRIVER_MAJOR="$DRIVER_MAJOR" \
  --env GITM_REF="$GITM_REF" "$NAME" bash -s <<'IN'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore
CUDA_VER=$(python3 -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
if [[ -z "$CUDA_VER" && -f /usr/local/cuda/version.json ]]; then
  CUDA_VER=$(python3 -c 'import json; print(json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"])')
fi
[[ -n "$CUDA_VER" ]] || { echo "cannot tell which CUDA this image uses" >&2; exit 1; }
MAJ=${CUDA_VER%%.*}; MIN=${CUDA_VER#*.}; MIN=${MIN%%.*}
echo "image CUDA $MAJ.$MIN, driver CUDA major $DRIVER_MAJOR"

apt-get update -qq
if ! apt-cache show "cuda-cudart-dev-$MAJ-$MIN" >/dev/null 2>&1; then
  # NVIDIA's CUDA apt repo, for images that don't ship it configured.
  . /etc/os-release
  REPO="ubuntu${VERSION_ID//./}"
  apt-get install -y -qq curl ca-certificates >/dev/null
  curl -fsSLo /tmp/keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/keyring.deb >/dev/null && rm -f /tmp/keyring.deb
  apt-get update -qq
fi
apt-get install -y -qq --no-install-recommends build-essential \
  "cuda-cudart-dev-$MAJ-$MIN" "cuda-cupti-dev-$MAJ-$MIN" "cuda-nvtx-$MAJ-$MIN" >/dev/null
[[ -e /usr/local/cuda ]] || ln -s "/usr/local/cuda-$MAJ.$MIN" /usr/local/cuda
rm -rf /var/lib/apt/lists/*

# libcupti whose major matches the driver (e.g. libcupti.so.13 on a CUDA 13 driver).
if [[ "$DRIVER_MAJOR" -ge 13 ]]; then
  pip install -q "nvidia-cuda-cupti>=${DRIVER_MAJOR},<$((DRIVER_MAJOR + 1))"
else
  pip install -q "nvidia-cuda-cupti-cu${DRIVER_MAJOR}"
fi

# Install gitm and compile the collector once, as a check of the toolchain.
pip install -q "gitm-labs[nvidia] @ https://github.com/GitM-Labs/runtime/archive/${GITM_REF}.tar.gz"
export CUDA_HOME=/usr/local/cuda
python3 -m gitm.tracer._cupti.build
python3 - <<'PY'
from gitm.tracer import injection
p = injection.lib_path()
assert p.exists(), f"{p} was not built"
print("collector built:", p)
PY
# Let job users (non-root inside the container) rebuild it after reinstalling gitm.
chmod -R a+rwX "$(python3 -c 'import gitm, os; print(os.path.dirname(gitm.__file__))')"
echo 'export CUDA_HOME=/usr/local/cuda' > /etc/profile.d/gitm-cuda.sh
IN

step "Saving $OUT"
rm -f "$OUT.tmp"
enroot export --output "$OUT.tmp" "$NAME" >/dev/null
mv -f "$OUT.tmp" "$OUT"; chmod 644 "$OUT"
touch /etc/gitm-gpu/config
if grep -q '^DEFAULT_IMAGE=' /etc/gitm-gpu/config; then
  sed -i "s#^DEFAULT_IMAGE=.*#DEFAULT_IMAGE=$OUT#" /etc/gitm-gpu/config
else
  echo "DEFAULT_IMAGE=$OUT" >> /etc/gitm-gpu/config
fi
step "Done: $OUT ($(du -h "$OUT" | cut -f1)) is now the default job image on $(hostname -s)"
