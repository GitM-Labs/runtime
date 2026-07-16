"""Driver/toolkit compatibility

A pod's NVIDIA driver is a property of the host, not the container.

Compatible with CUDA 13
and CUDA 12.4, 12.8

The rule this module enforces:
    a binary built against CUDA major M needs a driver whose CUDA major is >= M
CUDA guarantees minor version compatibility (a cu12.8 build runs on any 12.x
driver) and drivers stay compatible with older toolkits (a cu12 build runs on a
CUDA 13 driver). What never works is jumping a major forward: cu13 binaries on a
12.x driver, which is exactly the failure that keeps eating runs here.
``check()`` reports every mismatch with the command that fixes it. It's cheap and
importable, so the vLLM workload calls it before building an engine.
On torch vs vLLM: torch's CUDA variant is selectable, PyTorch publishes one wheel
index per CUDA (cu126/cu128/cu130), so ``torch_index_url`` picks the right one from
the driver. vLLM's is NOT: a given vLLM release ships wheels for exactly one CUDA,
and no pip flag overrides that. When vLLM is the thing that mismatches, the only
fixes are a different vLLM version or a different host, so we say so plainly
rather than pretending an install command exists.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# CUDA variants PyTorch publishes a wheel index for, oldest first.
_TORCH_CUDA_INDEXES: tuple[tuple[int, int], ...] = ((12, 6), (12, 8), (12, 9), (13, 0))


@dataclass(frozen=True)
class Stack:
    """An exact, known-good (torch, vLLM) pin for one driver CUDA major."""

    torch: str
    torch_index: str
    vllm: str
    note: str = ""

    def pip_commands(self) -> list[str]:
        """vLLM FIRST, then torch. The order is load-bearing.
        vLLM pins an exact torch version and pip resolves it from PyPI, whose default
        build is the cu130 one. Installing torch first would just get overwritten by
        that default when vLLM lands. So install vLLM, then force torch back to the
        right CUDA variant of the version vLLM asked for.
        """
        return [
            f"pip install vllm=={self.vllm}",
            f"pip install --force-reinstall --index-url {self.torch_index} torch=={self.torch}",
        ]

# Driver CUDA major -> the exact stack to install on it. Pinned, never resolved.
#
# The whole reason this table exists: a plain pip install vllm resolves torch from
# PyPI, whose default torch==2.11.0 build is cu130. That silently assumes a CUDA 13
# driver. Production fleets run CUDA 12, and RunPod reschedules a pod between hosts
# with different drivers, so "whatever pip picks" is never the right answer.
#
# The vLLM version differs per row, and that is NOT cosmetic — see the note on the
# CUDA 12 row. Every vLLM from 0.20.0 up ships wheels whose extensions link
# libcudart.so.13 (checked with readelf on the wheels themselves, not inferred from
# the torch pin). 0.19.1 is the newest release still linking libcudart.so.12.
SUPPORTED_STACKS: dict[int, Stack] = {
    # Verified end-to-end on an H100 (driver 580+): weight load, torch.compile,
    # CUDA-graph capture, 64 prompts decoded. The stack the existing report used.
    13: Stack(
        torch="2.11.0",
        torch_index="https://download.pytorch.org/whl/cu130",
        vllm="0.25.1",
    ),
    # H100 on a 570 driver (CUDA 12.8). torch 2.11.0+cu128 initialises fine here, but
    # vLLM 0.25.1 does not: its kernels are CUDA 13 binaries and it dies at import
    # with "libcudart.so.13: cannot open shared object file". No torch variant fixes
    # that. 0.19.1 is the newest vLLM with CUDA 12 wheels, and it pins torch 2.10.0.
    12: Stack(
        torch="2.10.0",
        torch_index="https://download.pytorch.org/whl/cu128",
        vllm="0.19.1",
        note=(
            "vLLM 0.19.1, not the 0.25.1 used on CUDA 13 hosts — every vLLM from "
            "0.20.0 up is a CUDA 13 build. This is a DIFFERENT ENGINE (scheduler, "
            "kernels, defaults), so decode numbers from a CUDA 12 host are not "
            "comparable with numbers from a CUDA 13 host. Record the vLLM version "
            "alongside any result taken here."
        ),
    ),
}

def stack_for(driver: tuple[int, int]) -> Stack | None:
    """The pinned stack for this driver, or None if the host can't run vLLM."""
    return SUPPORTED_STACKS.get(driver[0])


@dataclass(frozen=True)
class Problem:
    component: str          # "torch" | "vllm"
    detail: str             # what's wrong
    remediation: str        # how to fix it

    def __str__(self) -> str:
        return f"{self.component}: {self.detail}\n  fix: {self.remediation}"


def driver_cuda() -> tuple[int, int] | None:
    """Max CUDA version the host driver supports, from ``nvidia-smi``.
    This is the ceiling everything else must fit under, and the one thing that
    cannot be changed from inside the container.
    """
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def torch_cuda() -> tuple[int, int] | None:
    """CUDA version torch was built against (``torch.version.cuda``)."""
    try:
        import torch
    except Exception:
        return None
    raw = getattr(torch.version, "cuda", None)
    if not raw:
        return None
    parts = raw.split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except ValueError:
        return None


def vllm_cuda_major() -> int | None:
    """CUDA major vLLM's compiled kernels link against, read off the .so itself.
    vLLM's version string says nothing about its CUDA build, so we ask the linker:
    the extension links ``libcudart.so.<major>``. This is the check that told us a
    torch-only downgrade could never work — vLLM's own kernels were cu13 too.
    """
    try:
        import vllm
    except Exception:
        return None

    pkg = Path(vllm.__file__).resolve().parent
    majors: set[int] = set()
    # rglob, not glob: vLLM's extensions live in subpackages (vllm/_C*.so,
    # vllm/vllm_flash_attn/*.so), so a top-level-only scan misses them and the
    # whole vLLM check silently no-ops. readelf (-d) reads the dynamic section
    # without executing the .so, unlike ldd.
    for so in pkg.rglob("*.so"):
        try:
            out = subprocess.check_output(
                ["readelf", "-d", str(so)], text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            continue
        majors.update(int(m) for m in re.findall(r"libcudart\.so\.(\d+)", out))
    return max(majors) if majors else None


def torch_index_url(driver: tuple[int, int]) -> str | None:
    """PyTorch wheel index for the newest CUDA the driver can actually run.
    Compares the full (major, minor), not just the major: minor-version compatibility
    would technically let a cu12.9 build run on a 12.8 driver, but matching the driver
    exactly is free and removes a whole class of "should work" from the stack.
    """
    usable = [v for v in _TORCH_CUDA_INDEXES if v <= driver]
    if not usable:
        return None
    major, minor = max(usable)
    return f"https://download.pytorch.org/whl/cu{major}{minor}"


def _compatible(built: int, driver_major: int) -> bool:
    """A CUDA-built binary runs on a driver_major driver iff built <= driver."""
    return built <= driver_major


def check() -> list[Problem]:
    """Every CUDA major mismatch on this box, with its remediation. Empty = fine."""
    problems: list[Problem] = []
    driver = driver_cuda()
    if driver is None:
        return problems  # no GPU / no nvidia-smi — nothing to be incompatible with

    tc = torch_cuda()
    if tc is not None and not _compatible(tc[0], driver[0]):
        index = torch_index_url(driver)
        problems.append(Problem(
            component="torch",
            detail=(
                f"built for CUDA {tc[0]}.{tc[1]}, but the host driver only supports "
                f"CUDA {driver[0]}.{driver[1]}. torch.cuda will refuse to initialize."
            ),
            remediation=(
                f"pip install --force-reinstall --index-url {index} torch"
                if index else
                f"no PyTorch wheel index exists for a CUDA {driver[0]}.x driver"
            ),
        ))

    vc = vllm_cuda_major()
    if vc is not None and not _compatible(vc, driver[0]):
        problems.append(Problem(
            component="vllm",
            detail=(
                f"kernels link libcudart.so.{vc} (a CUDA {vc} build), but the host "
                f"driver only supports CUDA {driver[0]}.{driver[1]}."
            ),
            remediation=(
                "no pip flag selects vLLM's CUDA build — a release ships wheels for "
                f"exactly one CUDA. Either install a vLLM version whose wheels are "
                f"CUDA {driver[0]}, or move to a host whose driver supports CUDA {vc} "
                f"(driver >= 580 for CUDA 13). Redeploying with RunPod's CUDA version "
                f"filter set to {vc}.0+ keeps the engine you are measuring unchanged."
            ),
        ))

    return problems


def require_compatible() -> None:
    """Raise before an expensive build if the stack can't run on this driver.
    Called by the vLLM workload factory. Without it the failure surfaces ~90s in,
    inside ``torch._C._cuda_init()``, after the weights are already downloaded.
    """
    driver = driver_cuda()
    if driver is not None and stack_for(driver) is None:
        raise RuntimeError(
            f"unsupported host: this driver supports only CUDA {driver[0]}.{driver[1]}, "
            f"and vLLM's wheels are CUDA {max(SUPPORTED_STACKS)} builds. No torch "
            f"reinstall fixes this — the driver belongs to the host. Redeploy with the "
            f"CUDA version filter set to {max(SUPPORTED_STACKS)}.0+ (driver >= 580)."
        )

    problems = check()
    if not problems:
        return
    raise RuntimeError(
        "CUDA stack is incompatible with this host's driver:\n\n"
        + "\n\n".join(str(p) for p in problems)
        + "\n\nThe driver belongs to the host and cannot be changed from inside the "
        "container. Run python -m gitm.cuda_env to re-check."
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m gitm.cuda_env`` — diagnose. ``--plan`` — emit the pip commands.
    ``--plan`` prints nothing but shell commands (or nothing at all, on an
    unsupported host) so gpu_setup.sh can just run its output.
    """
    import sys

    argv = sys.argv[1:] if argv is None else argv
    plan_only = "--plan" in argv

    driver = driver_cuda()
    if driver is None:
        if not plan_only:
            print("no NVIDIA driver detected (CPU-only host) — nothing to check")
        return 0

    stack = stack_for(driver)

    if plan_only:
        if stack:
            print("\n".join(stack.pip_commands()))
        return 0 if stack else 1

    tc = torch_cuda()
    vc = vllm_cuda_major()
    print(f"host driver   : CUDA {driver[0]}.{driver[1]}  (host-owned; cannot change in-container)")
    print(f"torch         : CUDA {f'{tc[0]}.{tc[1]}' if tc else 'not installed'}")
    print(f"vllm kernels  : CUDA {vc if vc else 'not installed'}")

    if stack is None:
        print(
            f"\nUNSUPPORTED HOST: no pinned stack runs vLLM on a CUDA {driver[0]}.x driver.\n"
            f"  vLLM's wheels are CUDA {max(SUPPORTED_STACKS)} builds; a driver that old cannot "
            f"load them, and no torch reinstall changes that.\n"
            f"  fix: redeploy with RunPod's CUDA version filter set to "
            f"{max(SUPPORTED_STACKS)}.0+ (driver >= 580)."
        )
        return 1

    print(f"\npinned stack for CUDA {driver[0]}: torch=={stack.torch} vllm=={stack.vllm}")
    if stack.note:
        print(f"NOTE: {stack.note}")
    problems = check()
    if not problems:
        print("OK — every component runs on this driver.")
        return 0
    print()
    for p in problems:
        print(p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
