# GITM reproducible runtime image — the shared environment for Adit + interns.
#
# Build ONCE, push, and share the resulting image *digest*. Everyone who runs
# that digest gets a byte-identical software stack, so results match (perf within
# the 2% spread gate on the same GPU SKU; everything else exactly). See
# docs/REPRODUCIBILITY.md.
#
#   docker build -t <registry>/gitm:<tag> .
#   docker push <registry>/gitm:<tag>
#   docker inspect --format='{{index .RepoDigests 0}}' <registry>/gitm:<tag>  # share this
#
# Needs the CUDA *devel* base for nvcc + the CUPTI headers the tracer shim links.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.12 (matches the tested stack) + build toolchain for the CUPTI shim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates git build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gitm

# Pinned GPU compute stack, tied to the CUDA 12.4 base. Confirm these resolve on
# first build; once the image is pushed the digest freezes them for everyone.
ARG TORCH_VERSION=2.4.1
ARG CUDF_VERSION=24.10.01
ARG CUPY_VERSION=13.3.0
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==${CUDF_VERSION}" "cupy-cuda12x==${CUPY_VERSION}"

# The package + its CPU deps, pinned via constraints.txt for reproducibility.
COPY pyproject.toml constraints.txt README.md ./
COPY gitm ./gitm
COPY benchmarks ./benchmarks
COPY tests ./tests
COPY docs ./docs
COPY scripts ./scripts
RUN python -m pip install -e ".[dev,bench,nvidia]" -c constraints.txt

# Build the CUPTI tracer shim against this image's CUDA toolkit.
RUN python -m gitm.tracer._cupti.build

# Freeze the fully-resolved stack into the image for auditing / exact re-pin.
RUN python -m pip freeze > /opt/gitm/requirements.lock

# Sealed / least-privilege runtime: drop to an unprivileged user. No root at
# run time, no driver replacement, no phone-home — the runtime only reads CUPTI
# and emits telemetry in-cluster. Build steps above need root (apt/pip/nvcc);
# everything from here runs as uid 10001.
RUN useradd --create-home --uid 10001 gitm \
    && chown -R gitm:gitm /opt/gitm
USER gitm

CMD ["./scripts/verify_infra.sh"]
