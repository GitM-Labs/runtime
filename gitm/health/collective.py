"""Minimal pre-loop NCCL/RCCL AllReduce readiness check."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HealthResult:
    vendor: str
    gpu_count: int
    status: str  # pass | fail | skipped
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _detect_in_process() -> tuple[str, int]:
    """(vendor, GPUs visible to PyTorch), asked of torch in the calling process.

    Initialises CUDA's runtime in whatever process calls it, so it runs in a
    child process (:func:`detect_gpus`), never in the one that will build the
    engine.
    """
    try:
        import torch
    except ImportError:
        return "none", 0
    if not torch.cuda.is_available():
        return "none", 0
    vendor = "amd" if torch.version.hip is not None else "nvidia"
    return vendor, int(torch.cuda.device_count())


def detect_gpus(timeout_s: float = 60.0) -> tuple[str, int]:
    """Return (vendor, number of GPUs visible to PyTorch), asked in a child process.

    ``torch.cuda.is_available()`` and ``device_count()`` initialise the CUDA
    runtime in the process that calls them, and a CUDA runtime initialised
    before a fork cannot be used in the child. vLLM starts its EngineCore by
    forking on NVIDIA, so detecting the GPUs in gitm's own process made every
    engine build fail with "Cannot re-initialize CUDA in forked subprocess".
    A short-lived child asks instead, and gitm's process never touches CUDA.
    Anything that goes wrong in the child reads as no GPUs, which skips the check.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import json; from gitm.health.collective import _detect_in_process; "
             "print(json.dumps(_detect_in_process()))"],
            capture_output=True, text=True, timeout=timeout_s)
        if out.returncode != 0:
            return "none", 0
        vendor, count = json.loads(out.stdout.strip().splitlines()[-1])
        return str(vendor), int(count)
    except Exception:
        return "none", 0


def run_collective_health(timeout_s: float = 60.0) -> HealthResult:
    """AllReduce once across every visible local GPU."""
    vendor, count = detect_gpus(timeout_s=timeout_s)
    if count < 2:
        return HealthResult(vendor, count, "skipped", "fewer than 2 GPUs visible")

    with tempfile.TemporaryDirectory(prefix="gitm-health-") as tmp:
        result_path = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={count}",
            "--module",
            "gitm.health.probe_worker",
            str(result_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            stderr = None
        finally:
            # start_new_session=True puts torchrun outside the terminal PG, so
            # Ctrl+C / timeout must kill the group explicitly or workers linger.
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

        if timed_out:
            return HealthResult(vendor, count, "fail", f"AllReduce timed out after {timeout_s:g}s")

        if proc.returncode:
            detail = (stderr or "").strip().splitlines()
            return HealthResult(
                vendor,
                count,
                "fail",
                detail[-1][:300] if detail else f"AllReduce exited {proc.returncode}",
            )
        if not result_path.is_file() or not json.loads(result_path.read_text()).get("ok"):
            return HealthResult(vendor, count, "fail", "AllReduce returned incorrect values")
        return HealthResult(vendor, count, "pass", "AllReduce reached every visible GPU")
