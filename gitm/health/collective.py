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


def detect_gpus() -> tuple[str, int]:
    """Return (vendor, number of GPUs visible to PyTorch)."""
    try:
        import torch
    except ImportError:
        return "none", 0
    if not torch.cuda.is_available():
        return "none", 0
    vendor = "amd" if torch.version.hip is not None else "nvidia"
    return vendor, int(torch.cuda.device_count())


def run_collective_health(timeout_s: float = 60.0) -> HealthResult:
    """AllReduce once across every visible local GPU."""
    vendor, count = detect_gpus()
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
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
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
