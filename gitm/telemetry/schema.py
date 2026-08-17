"""Vendor-neutral state-telemetry schema.

Canonical fields at the top layer; vendor-specific extensions go in ``extra``.
Dashboards built on canonical fields keep working when new vendors land.
"""

from __future__ import annotations

from enum import Flag, auto

from pydantic import BaseModel, ConfigDict, Field


class ThrottleReason(Flag):
    """Canonical throttle reasons. Vendor backends map their bitmask onto this."""

    NONE = 0
    GPU_IDLE = auto()
    APPLICATIONS_CLOCKS_SETTING = auto()
    SW_POWER_CAP = auto()
    HW_SLOWDOWN = auto()
    SYNC_BOOST = auto()
    SW_THERMAL_SLOWDOWN = auto()
    HW_THERMAL_SLOWDOWN = auto()
    HW_POWER_BRAKE_SLOWDOWN = auto()
    DISPLAY_CLOCK_SETTING = auto()


class WorkloadLabels(BaseModel):
    """Labels attached to every sample so dashboards can slice by workload."""

    workload_id: str
    fingerprint: str | None = None
    run_id: str | None = None
    phase: str | None = None  # "capture" | "attribute" | "replay" | "apply" | "stabilize"


class Sample(BaseModel):
    """One GPU state sample at a single timestamp.

    Canonical fields are vendor-neutral. Vendor-specific fields go in ``extra``.
    """

    model_config = ConfigDict(extra="forbid")

    ts_ns: int = Field(..., description="Sample timestamp in nanoseconds since epoch.")
    node: str
    gpu_uuid: str
    gpu_index: int
    vendor: str  # "nvidia" | "amd"
    driver_version: str | None = None

    # Canonical numeric fields
    util_pct: float | None = None
    mem_used_bytes: int | None = None
    mem_total_bytes: int | None = None
    power_w: float | None = None
    temp_c: float | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None

    throttle_reasons: ThrottleReason = ThrottleReason.NONE

    # Per-process utilization (pid -> util_pct), optional
    per_process: dict[int, float] = Field(default_factory=dict)

    # NVLink / xGMI throughput in bytes/s, keyed by link index
    interconnect_bytes_per_s: dict[int, int] = Field(default_factory=dict)

    # ECC counters, vendor-specific encoding
    ecc_volatile_sbe: int | None = None
    ecc_volatile_dbe: int | None = None

    # Vendor-specific fields keep landing here, dashboards on canonical fields
    # keep working unchanged.
    extra: dict[str, float] = Field(default_factory=dict)

    # Per-field backend failures. A partial sample remains usable, but consumers
    # must not read a missing throttle/process field as an observed zero.
    diagnostics: list[str] = Field(default_factory=list)

    labels: WorkloadLabels | None = None
