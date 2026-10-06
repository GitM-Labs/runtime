"""AMD workstream: RCCL AllReduce readiness via torch.distributed (nccl backend).

Under ROCm, PyTorch's ``backend="nccl"`` is backed by RCCL. Device visibility
uses ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` — never ``nvidia-smi``.
"""

from __future__ import annotations

import os


def _visible_from_env() -> int | None:
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if parts:
                return len(parts)
    return None


def device_count() -> int:
    """Visible HIP/ROCm devices."""
    env_n = _visible_from_env()
    if env_n is not None:
        return env_n
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def detect_sku() -> str | None:
    """Best-effort GPU name for soft-floor catalogue lookup (e.g. MI355X)."""
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
