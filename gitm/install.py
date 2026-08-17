"""Host preparation for a GPU box: ``gitm install``.

Brings a CUDA host to the state the tracer and the vLLM workloads require. The
equivalent of ``scripts/gpu_setup.sh``, reimplemented inside the package so that
it is available from a wheel installation, where the repository's ``scripts/``
directory is not present.

The two are not identical. The shell script performs an editable install of the
repository, which has no meaning for a user who obtained the package from an
index; that step is omitted here. What remains is the work that is a property of
the *host* rather than of the checkout:

1. Verification that the driver is one for which a pinned stack exists.
2. Installation of a CUPTI whose major version matches the driver.
3. Installation of the pinned ``(vLLM, torch)`` pair for that driver.
4. Compilation of the CUPTI shim and injection library.
5. Verification that the result is coherent.

Step 3 mutates an existing environment, replacing torch with a specific CUDA
build. It is separable via ``--skip-stack`` for hosts whose framework versions
are managed externally, and the whole plan is printable without execution via
``--dry-run``.

Version selection is delegated to :mod:`gitm.cuda_env`, which holds the pinned
driver-to-stack table. This module contributes sequencing and execution only; it
does not decide versions.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from gitm import cuda_env


@dataclass
class Step:
    """One unit of host preparation.

    Attributes
    ----------
    name
        Short identifier reported in the plan and in progress output.
    argv
        Command to execute.
    rationale
        Why the step is required, shown in ``--dry-run`` output so the plan is
        reviewable without reading this source.
    optional
        Whether failure is tolerated. Optional steps report a warning and allow
        the sequence to continue; required steps abort it.
    """

    name: str
    argv: list[str]
    rationale: str
    optional: bool = False


@dataclass
class Plan:
    """An ordered sequence of steps, with the driver context that produced it."""

    steps: list[Step] = field(default_factory=list)
    driver: tuple[int, int] | None = None
    stack: cuda_env.Stack | None = None

    def render(self) -> str:
        drv = f"CUDA {self.driver[0]}.{self.driver[1]}" if self.driver else "unknown"
        lines = [f"host driver: {drv}"]
        if self.stack is not None:
            lines.append(f"pinned stack: torch=={self.stack.torch} vllm=={self.stack.vllm}")
        lines.append("")
        for i, s in enumerate(self.steps, 1):
            flag = "  (optional)" if s.optional else ""
            lines.append(f"{i}. {s.name}{flag}")
            lines.append(f"     {s.rationale}")
            lines.append(f"     $ {' '.join(s.argv)}")
        return "\n".join(lines)


def _pip(*args: str) -> list[str]:
    """Invoke pip through the running interpreter.

    ``sys.executable -m pip`` rather than a bare ``pip``, so that packages land
    in the environment this process is running from. A bare ``pip`` resolves
    through ``PATH`` and, in a container with several interpreters, installs into
    an environment the tracer will never import from.
    """
    return [sys.executable, "-m", "pip", *args]


def build_plan(
    *,
    skip_apt: bool = False,
    skip_stack: bool = False,
    with_gpu_extras: bool = False,
) -> Plan:
    """Construct the installation plan for this host.

    Raises
    ------
    RuntimeError
        If the driver cannot be read, or if no pinned stack exists for it. Both
        conditions are terminal: proceeding would install a combination that has
        not been verified against this driver, and the resulting failure would
        surface inside ``torch._C._cuda_init`` long after the fact.
    """
    driver = cuda_env.driver_cuda()
    if driver is None:
        raise RuntimeError(
            "no NVIDIA driver detected (nvidia-smi unavailable). "
            "gitm install prepares a CUDA host; there is nothing to prepare here."
        )
    stack = cuda_env.stack_for(driver)
    if stack is None and not skip_stack:
        raise RuntimeError(
            f"no pinned stack is known for a CUDA {driver[0]}.x driver. "
            f"Supported: {sorted(cuda_env.SUPPORTED_STACKS)}. "
            "Re-run with --skip-stack to prepare the tracer only."
        )

    plan = Plan(driver=driver, stack=stack)

    if not skip_apt and shutil.which("apt-get") and not shutil.which("cc"):
        plan.steps.append(
            Step(
                name="C compiler",
                argv=["apt-get", "install", "-y", "-qq", "build-essential", "python3-dev"],
                rationale=(
                    "The CUPTI shim is compiled on the host. Only a C compiler is "
                    "required; nvcc is not."
                ),
                optional=True,
            )
        )

    major = driver[0]
    plan.steps.append(
        Step(
            name="CUPTI",
            argv=_pip("install", "-q", "nvidia-cuda-cupti", f"nvidia-cuda-cupti-cu{major}"),
            rationale=(
                f"The CUPTI Activity API requires libcupti's major version to match "
                f"the driver ({major}); a mismatch yields CUPTI_ERROR_NOT_COMPATIBLE. "
                "The versioned wheel additionally supplies cupti.h for the build."
            ),
            optional=True,
        )
    )

    if not skip_stack and stack is not None:
        for cmd in stack.pip_commands():
            plan.steps.append(
                Step(
                    name=f"stack: {cmd.split()[2] if len(cmd.split()) > 2 else cmd}",
                    argv=_pip(*cmd.split()[1:]),
                    rationale=(
                        "vLLM is installed before torch deliberately: vLLM pins an "
                        "exact torch version which pip resolves to the default CUDA "
                        "build, so torch is forced back afterwards."
                    ),
                )
            )

    if with_gpu_extras:
        plan.steps.append(
            Step(
                name="RAPIDS (cuDF, CuPy)",
                argv=_pip(
                    "install", "-q", "--extra-index-url=https://pypi.nvidia.com",
                    "cudf-cu12", "cupy-cuda12x",
                ),
                rationale=(
                    "Required by the HFT harness only. Without it that harness falls "
                    "back to pandas, which is adequate for capture but not for "
                    "throughput measurement."
                ),
                optional=True,
            )
        )

    plan.steps.append(
        Step(
            name="CUPTI shim",
            argv=[sys.executable, "-m", "gitm.tracer._cupti.build"],
            rationale=(
                "Compiles the in-process shim and the injection library that the "
                "CUDA driver loads via CUDA_INJECTION64_PATH."
            ),
        )
    )
    return plan


def _run(step: Step) -> bool:
    print(f"==> {step.name}")
    try:
        rc = subprocess.call(step.argv)
    except OSError as exc:
        rc, _ = 1, print(f"    {exc}")
    if rc == 0:
        return True
    level = "WARN" if step.optional else "ERROR"
    print(f"    {level}: {step.name} failed (exit {rc})")
    return step.optional


def verify() -> int:
    """Report whether the prepared host is coherent. Returns a process exit code."""
    print("==> verification")
    problems = cuda_env.check()
    for p in problems:
        print(f"    {p}")

    try:
        from gitm.tracer import injection

        ts = injection.cupti_now()
    except Exception as exc:
        ts = None
        print(f"    CUPTI clock unreadable: {type(exc).__name__}: {exc}")

    if ts is None:
        print(
            "    ERROR: the CUPTI clock is unreadable, so a capture window cannot be "
            "bounded. `gitm capture attach` will fail preflight."
        )
        return 1

    print(f"    CUPTI clock readable ({ts})")
    if problems:
        print("    stack reports problems above; tracing is available regardless.")
        return 1
    print("OK — tracer built and the stack is consistent with this driver.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gitm install",
        description="Prepare a CUDA host: driver-matched CUPTI, pinned vLLM/torch, tracer shim.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without executing it.")
    ap.add_argument("--skip-stack", action="store_true",
                    help="Do not install or replace vLLM/torch.")
    ap.add_argument("--skip-apt", action="store_true",
                    help="Do not install system build dependencies.")
    ap.add_argument("--with-gpu-extras", action="store_true",
                    help="Additionally install RAPIDS cuDF and CuPy (HFT harness only).")
    args = ap.parse_args(argv)

    try:
        plan = build_plan(
            skip_apt=args.skip_apt,
            skip_stack=args.skip_stack,
            with_gpu_extras=args.with_gpu_extras,
        )
    except RuntimeError as exc:
        print(f"cannot prepare this host: {exc}")
        return 2

    print(plan.render())
    print()
    if args.dry_run:
        print("dry run — nothing was executed.")
        return 0

    for step in plan.steps:
        if not _run(step):
            print("\naborted; the host is partially prepared.")
            return 1

    print()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
