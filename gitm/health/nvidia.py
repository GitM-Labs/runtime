"""NVIDIA workstream: NCCL AllReduce readiness via torch.distributed."""

from __future__ import annotations

import os


def device_count() -> int:
    """Usable CUDA devices for the collective probe.

    Honours ``CUDA_VISIBLE_DEVICES`` only when torch can actually see CUDA —
    an env list alone must not trigger a multi-proc NCCL spawn on a CPU box.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def detect_sku() -> str | None:
    """Best-effort GPU name for soft-floor catalogue lookup."""
    env = os.environ.get("GITM_GPU_SKU")
    if env and env.strip():
        return env.strip()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None
