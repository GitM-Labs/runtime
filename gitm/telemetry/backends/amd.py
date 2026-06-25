"""AMD backend — ROCm SMI.

Symmetric to ``NvidiaBackend``. Implementation pending — sets up the import
shape so ``discover_backends()`` can find it once rocm-smi is wired up.
"""

from __future__ import annotations

import socket
import time

from gitm.telemetry.schema import Sample, ThrottleReason, WorkloadLabels


class AmdBackend:
    """ROCm SMI sampler. Raises ImportError until rocm-smi is wired up."""

    vendor = "amd"

    def __init__(self) -> None:
        # When rocm-smi bindings are added, import here. Until then we raise
        # so discover_backends() skips this vendor on the host.
        raise ImportError("ROCm SMI backend not yet implemented")

    def device_count(self) -> int:  # pragma: no cover - not yet implemented
        return 0

    def sample(self, gpu_index: int, labels: WorkloadLabels | None = None) -> Sample:  # pragma: no cover
        return Sample(
            ts_ns=time.time_ns(),
            node=socket.gethostname(),
            gpu_uuid=f"amd-{gpu_index}",
            gpu_index=gpu_index,
            vendor=self.vendor,
            throttle_reasons=ThrottleReason.NONE,
            labels=labels,
        )

    def close(self) -> None:  # pragma: no cover - not yet implemented
        return None
