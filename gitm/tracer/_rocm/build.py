"""Compile the ROCm injection tool against the installed rocprofiler-sdk.

    python -m gitm.tracer._rocm.build

One target: ``libgitm_rocm_inject.so`` — a plain .so with no libpython, loaded
by rocprofiler-register into every process that initializes HIP (via
``$ROCP_TOOL_LIBRARIES``). This is the AMD counterpart of ``libgitm_inject.so``
and writes the identical per-pid JSONL shards.

Unlike the CUPTI build there is no driver/toolkit major-version matching dance:
rocprofiler-sdk ships with the ROCm the driver was installed from, and there is
exactly one on the box. Requirements are just the sdk headers + library
(package ``rocprofiler-sdk``, present in the ``rocm/pytorch`` and ``rocm/vllm``
images and in any full ROCm >= 6.2 install — MI355X hosts run ROCm 7.x) and a
host C compiler; no hipcc, the sdk is a plain C API.

On a host with no ROCm this exits non-zero and the tracer degrades to a no-op,
matching the CUPTI build's behavior on a CUDA-less host.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "rocm_inject.c"
LIB = HERE / "libgitm_rocm_inject.so"


def _rocm_home() -> Path | None:
    for env in ("ROCM_PATH", "ROCM_HOME"):
        if os.environ.get(env):
            p = Path(os.environ[env])
            if p.is_dir():
                return p
    p = Path("/opt/rocm")
    return p if p.is_dir() else None


def _toolchain() -> tuple[Path, Path]:
    """Locate rocprofiler-sdk headers + library, or exit with instructions."""
    rocm = _rocm_home()
    inc = lib = None
    if rocm:
        c = rocm / "include"
        if (c / "rocprofiler-sdk" / "rocprofiler.h").exists():
            inc = c
        for d in (rocm / "lib", rocm / "lib64"):
            if d.is_dir() and list(d.glob("librocprofiler-sdk.so*")):
                lib = d
                break
    if inc is None or lib is None:
        raise SystemExit(
            "Could not locate rocprofiler-sdk (headers under "
            "$ROCM_PATH/include/rocprofiler-sdk/ and librocprofiler-sdk.so). "
            "Needs ROCm >= 6.2; on a slim image install the rocprofiler-sdk "
            "package (apt: rocprofiler-sdk) and a C compiler. On non-ROCm "
            "hosts the tracer is a no-op and this build is skipped."
        )
    return inc, lib


def build() -> Path:
    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC}")
    inc, lib = _toolchain()
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc, "-shared", "-fPIC", "-O2", "-pthread",
        # The sdk's HIP api_args.h drags in hip_runtime.h, which refuses to
        # compile without a platform selection. hipcc would define this; a
        # plain host cc (all we need for a C tool) must say it explicitly.
        "-D__HIP_PLATFORM_AMD__",
        f"-I{inc}",
        str(SRC),
        f"-L{lib}",
        "-lrocprofiler-sdk",
        f"-Wl,-rpath,{lib}",
        "-o", str(LIB),
    ]
    print("compiling rocm injection tool:\n  " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print(f"built {LIB}")
    return LIB


if __name__ == "__main__":
    try:
        build()
    except subprocess.CalledProcessError as exc:
        print(f"compile failed (exit {exc.returncode})", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
