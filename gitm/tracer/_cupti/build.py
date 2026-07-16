"""Compile the CUPTI targets, linking the libcupti that matches the driver.

    python -m gitm.tracer._cupti.build

Two targets share cupti_core.c:

* ``_cupti_shim<EXT>.so`` — the CPython extension. Traces only the process that
  imports it.
* ``libgitm_inject.so`` — a plain .so with no libpython, loaded by the CUDA driver
  into every process that initializes CUDA (via ``$CUDA_INJECTION64_PATH``). This
  is the one that can see vLLM's kernels, which run in a child EngineCore process
  the parent interpreter cannot instrument.

The subtlety that bit us on real hardware: CUPTI's Activity API returns
``CUPTI_ERROR_NOT_COMPATIBLE`` when the loaded ``libcupti`` major version doesn't
match the NVIDIA driver's CUDA version. RunPod's A100s run a CUDA-13 driver, but
the toolkit + torch ship CUDA 12.8 — so linking the toolkit's ``libcupti.so.12``
fails to enable. The fix: link ``libcupti.so.<driver-cuda-major>`` (e.g. the
``nvidia-cuda-cupti`` wheel's ``libcupti.so.13``), which ``gpu_setup.sh`` installs.
So this script:
1. reads the driver's CUDA major from ``nvidia-smi`` (e.g. 13),
2. finds ``libcupti.so.<major>`` across the toolkit and the pip ``nvidia/cu*``
   wheels, preferring the driver-matched major (else the newest),
3. compiles against the **toolkit** headers (complete — they carry ``crt/`` and
   the ``CUpti_Activity*`` structs; the records are append-compatible), and
4. links that libcupti with an rpath so it loads at runtime.
Only a host C compiler is needed — not ``nvcc``. On a host with no CUDA this
exits non-zero and the tracer degrades to a no-op.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE / "cupti_core.c"
SRC = HERE / "cupti_shim.c"
INJECT_SRC = HERE / "cupti_inject.c"
INJECT_LIB = HERE / "libgitm_inject.so"


def _cuda_home() -> Path | None:
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            p = Path(os.environ[env])
            if p.is_dir():
                return p
    for cand in ("/usr/local/cuda", "/opt/cuda"):
        if Path(cand).is_dir():
            return Path(cand)
    return None
def _nvidia_wheel_bases() -> list[Path]:
    try:
        import nvidia  # provided by nvidia-*-cu1x wheels (PyTorch deps)
    except Exception:
        return []
    return [Path(p) for p in getattr(nvidia, "__path__", [])]
def _driver_cuda_major() -> int | None:
    """Driver's CUDA major from ``nvidia-smi`` (the version CUPTI must match)."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)", out)
    return int(m.group(1)) if m else None
def _all_libcupti(cuda: Path | None) -> list[tuple[Path, int]]:
    """Every ``libcupti.so.<major>`` on the box, as ``(dir, major)``."""
    dirs: list[Path] = []
    if cuda:
        dirs += [
            cuda / "extras/CUPTI/lib64", cuda / "lib64", cuda / "lib",
            cuda / "targets/x86_64-linux/lib",
        ]
    for base in _nvidia_wheel_bases():
        dirs += list(base.glob("cu*/lib")) + [base / "cuda_cupti/lib"]
    out: list[tuple[Path, int]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for so in d.glob("libcupti.so.*"):
            m = re.fullmatch(r"libcupti\.so\.(\d+)", so.name)
            if m:
                out.append((d, int(m.group(1))))
    return out
def _pick_libcupti(cuda: Path | None) -> tuple[Path | None, int | None]:
    """Pick the libcupti dir+major matching the driver (else the newest)."""
    cands = _all_libcupti(cuda)
    if not cands:
        return None, None
    drv = _driver_cuda_major()
    if drv is not None:
        matched = [c for c in cands if c[1] == drv]
        if matched:
            return matched[0]
    return max(cands, key=lambda c: c[1])  # newest major
def _cupti_include(cuda: Path | None) -> Path | None:
    if cuda:
        for c in (cuda / "include", cuda / "extras/CUPTI/include",
                  cuda / "targets/x86_64-linux/include"):
            if (c / "cupti.h").exists():
                return c
    for base in _nvidia_wheel_bases():
        c = base / "cuda_cupti/include"
        if (c / "cupti.h").exists():
            return c
    return None
def _cudart(cuda: Path | None) -> tuple[Path | None, Path | None]:
    """cuda_runtime.h include (must carry ``crt/``) + libcudart dir."""
    inc = lib = None
    if cuda:
        for c in (cuda / "include", cuda / "targets/x86_64-linux/include"):
            if (c / "cuda_runtime.h").exists() and (c / "crt/host_defines.h").exists():
                inc = c
                break
        for c in (cuda / "lib64", cuda / "lib", cuda / "targets/x86_64-linux/lib"):
            if c.is_dir() and list(c.glob("libcudart.so*")):
                lib = c
                break
    return inc, lib
def _output_path() -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return HERE / f"_cupti_shim{suffix}"


def _toolchain() -> tuple[Path, Path, int, Path, Path]:
    """Locate CUPTI/cudart headers + the driver-matched libcupti, or exit."""
    cuda = _cuda_home()
    cupti_inc = _cupti_include(cuda)
    cupti_lib, cupti_major = _pick_libcupti(cuda)
    cudart_inc, cudart_lib = _cudart(cuda)
    missing = [n for n, v in (
        ("cupti.h headers", cupti_inc),
        ("libcupti", cupti_lib),
        ("cuda_runtime.h headers (with crt/)", cudart_inc),
        ("libcudart", cudart_lib),
    ) if v is None]
    if missing:
        raise SystemExit(
            "Could not locate: " + ", ".join(missing) + ".\n"
            "Need the CUDA toolkit headers (set $CUDA_HOME) plus a libcupti "
            "matching the driver. On a stock PyTorch pod:\n"
            "  pip install nvidia-cuda-cupti        # libcupti.so.<driver major>\n"
            "  apt-get install -y build-essential   # a C compiler\n"
            "On CPU-only hosts the tracer is a no-op and this build is skipped."
        )
    assert cupti_inc and cupti_lib and cupti_major and cudart_inc and cudart_lib
    return cupti_inc, cupti_lib, cupti_major, cudart_inc, cudart_lib


def _compile(sources: list[Path], out: Path, extra_includes: list[str], label: str) -> Path:
    cupti_inc, cupti_lib, cupti_major, cudart_inc, cudart_lib = _toolchain()
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc, "-shared", "-fPIC", "-O2",
        *[f"-I{i}" for i in extra_includes],
        f"-I{HERE}",
        f"-I{cupti_inc}",
        f"-I{cudart_inc}",
        *[str(s) for s in sources],
        f"-L{cupti_lib}",
        f"-L{cudart_lib}",
        f"-l:libcupti.so.{cupti_major}",
        "-lcudart",
        f"-Wl,-rpath,{cupti_lib}",
        f"-Wl,-rpath,{cudart_lib}",
        "-o", str(out),
    ]
    print(f"compiling {label}:\n  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"  driver CUDA major: {_driver_cuda_major()}  ->  libcupti.so.{cupti_major}")
    print(f"  cupti lib:  {cupti_lib}")
    print(f"  headers:    {cupti_inc} | {cudart_inc}")
    subprocess.run(cmd, check=True)
    print(f"built {out}")
    return out


def build() -> Path:
    """Build the CPython extension (_cupti_shim). Traces this process only."""
    for src in (CORE, SRC):
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
    return _compile(
        [CORE, SRC], _output_path(), [sysconfig.get_path("include")], "cupti shim"
    )


def build_injection() -> Path:
    """Build libgitm_inject.so — the driver-injected, no-libpython collector.
    Deliberately does NOT link libpython: the CUDA driver dlopens this into
    arbitrary processes at CUDA init, and dragging an interpreter into that path
    would be both unnecessary and unsafe.
    """
    for src in (CORE, INJECT_SRC):
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
    return _compile([CORE, INJECT_SRC], INJECT_LIB, [], "cupti injection lib")


if __name__ == "__main__":
    try:
        build()
        build_injection()
    except subprocess.CalledProcessError as exc:
        print(f"compile failed (exit {exc.returncode})", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
