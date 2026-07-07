"""NVIDIA backend — NVML via pynvml.

Per-field error tolerance via ``_try``: one failing NVML call doesn't drop the
whole sample. ``_NVML_THROTTLE_MAP`` translates NVML's bitmask into the
canonical ``ThrottleReason`` flag set.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from typing import Any, TypeVar

from gitm.telemetry.schema import Sample, ThrottleReason, WorkloadLabels

T = TypeVar("T")


# NVML throttle reason bits → canonical ThrottleReason flag.
# Bit values come from nvml.h::nvmlClocksThrottleReasons*.
_NVML_THROTTLE_MAP: dict[int, ThrottleReason] = {
    0x0000000000000001: ThrottleReason.GPU_IDLE,
    0x0000000000000002: ThrottleReason.APPLICATIONS_CLOCKS_SETTING,
    0x0000000000000004: ThrottleReason.SW_POWER_CAP,
    0x0000000000000008: ThrottleReason.HW_SLOWDOWN,
    0x0000000000000010: ThrottleReason.SYNC_BOOST,
    0x0000000000000020: ThrottleReason.SW_THERMAL_SLOWDOWN,
    0x0000000000000040: ThrottleReason.HW_THERMAL_SLOWDOWN,
    0x0000000000000080: ThrottleReason.HW_POWER_BRAKE_SLOWDOWN,
    0x0000000000000100: ThrottleReason.DISPLAY_CLOCK_SETTING,
}


def _try(fn: Callable[[], T], default: T | None = None) -> T | None:
    """Call ``fn``; return ``default`` on any exception.

    Used per NVML call so a single failure doesn't drop the whole sample.
    """
    try:
        return fn()
    except Exception:
        return default


def _decode_throttle(bitmask: int) -> ThrottleReason:
    out = ThrottleReason.NONE
    for bit, flag in _NVML_THROTTLE_MAP.items():
        if bitmask & bit:
            out |= flag
    return out


class NvidiaBackend:
    """NVML-backed GPU sampler.

    Requires ``pynvml``. Construction fails (raises ImportError or NVMLError)
    if NVML is unavailable; ``discover_backends()`` catches and skips.
    """

    vendor = "nvidia"

    def __init__(self) -> None:
        import pynvml  # type: ignore[import-not-found]

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._driver = _try(lambda: pynvml.nvmlSystemGetDriverVersion())
        if isinstance(self._driver, bytes):
            self._driver = self._driver.decode()
        self._node = socket.gethostname()
        self._n = pynvml.nvmlDeviceGetCount()
        self._handles: dict[int, Any] = {}

    def _handle(self, gpu_index: int) -> Any:
        h = self._handles.get(gpu_index)
        if h is None:
            h = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._handles[gpu_index] = h
        return h

    def device_count(self) -> int:
        return self._n

    def sample(self, gpu_index: int, labels: WorkloadLabels | None = None) -> Sample:
        nv = self._pynvml
        h = self._handle(gpu_index)

        uuid = _try(lambda: nv.nvmlDeviceGetUUID(h))
        if isinstance(uuid, bytes):
            uuid = uuid.decode()

        util = _try(lambda: nv.nvmlDeviceGetUtilizationRates(h))
        mem = _try(lambda: nv.nvmlDeviceGetMemoryInfo(h))
        power_mw = _try(lambda: nv.nvmlDeviceGetPowerUsage(h))
        temp = _try(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        sm_clock = _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
        mem_clock = _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))
        throttle_bits = _try(lambda: nv.nvmlDeviceGetCurrentClocksThrottleReasons(h), 0) or 0

        per_proc: dict[int, float] = {}
        procs = _try(lambda: nv.nvmlDeviceGetComputeRunningProcesses(h), []) or []
        for p in procs:
            per_proc[int(p.pid)] = 0.0  # NVML doesn't expose per-process util directly

        ecc_sbe = _try(
            lambda: nv.nvmlDeviceGetTotalEccErrors(
                h, nv.NVML_MEMORY_ERROR_TYPE_CORRECTED, nv.NVML_VOLATILE_ECC
            )
        )
        ecc_dbe = _try(
            lambda: nv.nvmlDeviceGetTotalEccErrors(
                h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC
            )
        )

        return Sample(
            ts_ns=time.time_ns(),
            node=self._node,
            gpu_uuid=uuid or f"nvidia-unknown-{gpu_index}",
            gpu_index=gpu_index,
            vendor=self.vendor,
            driver_version=self._driver,
            util_pct=float(util.gpu) if util is not None else None,
            mem_used_bytes=int(mem.used) if mem is not None else None,
            mem_total_bytes=int(mem.total) if mem is not None else None,
            power_w=(float(power_mw) / 1000.0) if power_mw is not None else None,
            temp_c=float(temp) if temp is not None else None,
            sm_clock_mhz=int(sm_clock) if sm_clock is not None else None,
            mem_clock_mhz=int(mem_clock) if mem_clock is not None else None,
            throttle_reasons=_decode_throttle(int(throttle_bits)),
            per_process=per_proc,
            ecc_volatile_sbe=int(ecc_sbe) if ecc_sbe is not None else None,
            ecc_volatile_dbe=int(ecc_dbe) if ecc_dbe is not None else None,
            labels=labels,
        )

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
