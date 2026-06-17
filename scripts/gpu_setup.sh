#!/usr/bin/env bash
# Set up the GITM infra on a RunPod (or any) CUDA GPU box: install the package,
# the GPU compute libs, and build the CUPTI tracer shim. Idempotent-ish.
#
#   bash scripts/gpu_setup.sh
#
# Requires a CUDA *toolkit* image (nvcc + CUPTI headers under
# $CUDA_HOME/extras/CUPTI), e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04 or a
# RunPod PyTorch template. A runtime-only image lacks the headers and the shim
# build will fail with a clear message.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "==> GPU check"
command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi — this is not a GPU box"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "==> CUDA_HOME=$CUDA_HOME (toolkit optional — wheels work too)"

echo "==> system build deps (just a C compiler; nvcc not required)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq build-essential python3-dev git >/dev/null 2>&1 || \
    echo "WARN: apt-get deps step skipped/failed (may already be present)"
fi

echo "==> python package (+ dev + full GPU stack: cuDF/CuPy/CUPTI/NVML)"
pip install -q -U pip
# The [gpu] extra pulls cuDF/CuPy + the CUPTI/runtime wheels; cuDF/CuPy resolve
# from NVIDIA's index, hence --extra-index-url.
pip install -q --extra-index-url=https://pypi.nvidia.com -e ".[dev,bench,gpu]"

echo "==> CUPTI matching the driver's CUDA version"
# CUPTI's Activity API needs libcupti's major to match the *driver* (else
# CUPTI_ERROR_NOT_COMPATIBLE). RunPod A100s run a CUDA-13 driver while torch is
# 12.8, so install the latest 'nvidia-cuda-cupti' (libcupti.so.13); build.py
# then links the driver-matched one. The -cu12 wheel is the fallback header source.
pip install -q nvidia-cuda-cupti nvidia-cuda-cupti-cu12 nvidia-cuda-runtime-cu12 \
  || echo "WARN: could not install CUPTI wheels — shim build will look for a toolkit instead"

echo "==> GPU compute libs (RAPIDS cuDF + CuPy, CUDA 12)"
# cuDF/CuPy power the real HFT harness path. If this fails the harness still
# runs, falling back to pandas (gpu_name=cpu) — fine for capture tests, not for
# the real throughput numbers.
pip install -q --extra-index-url=https://pypi.nvidia.com cudf-cu12 cupy-cuda12x \
  || echo "WARN: cuDF/CuPy install failed — HFT harness will fall back to pandas"

echo "==> build CUPTI tracer shim (toolkit or wheels, whichever is present)"
# Explicit build so a missing toolchain fails here, not on first run. (gitm also
# auto-builds the shim on first capture, so this step is belt-and-suspenders.)
python -m gitm.tracer._cupti.build

echo "==> done."
echo "    Smoke check : ./scripts/verify_infra.sh"
echo "    Real run    : gitm run --workload hft-lob   # auto-stages a smoke dataset"
echo "    Full driver : gitm-run-workload --workload hft --seed 42 --stage \$GITM_BENCH_STAGE"
