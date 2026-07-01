"""Intervention spec — schema for every entry in the library."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SafetyTier = Literal["low_risk", "moderate", "high_risk"]


class Applicability(BaseModel):
    """When this lever applies. All conditions are AND-ed."""

    model_config = ConfigDict(extra="forbid")
    workloads: list[str] = Field(default_factory=lambda: ["vllm-decode"])
    requires_dtype: list[str] | None = None  # e.g. ["fp16", "bf16"]
    requires_hardware: list[str] | None = None  # e.g. ["A100", "H100"]
    min_kv_cache_len: int | None = None
    max_kv_cache_len: int | None = None
    min_gpus: int | None = None
    requires_collective: bool = False
    requires_interconnect: bool = False
    other: str | None = None  # free-form caveat


class SafetyGate(BaseModel):
    """Conditions that must hold before this lever is applied live."""

    model_config = ConfigDict(extra="forbid")
    tier: SafetyTier = "moderate"
    requires_rollback_window_s: int = 60
    forbid_if_oom_history: bool = True
    requires_qualification_commit: bool = False
    notes: str = ""


class InterventionSpec(BaseModel):
    """One curated lever."""

    model_config = ConfigDict(extra="forbid")

    name: str
    summary: str
    knob: str  # vLLM config key, e.g. "max_num_batched_tokens"
    value: int | float | str | bool | None = None  # value to set the knob to on apply
    applies_to_kernels: list[str] = Field(default_factory=list)  # substring match
    expected_delta_mean: float  # signed, e.g. +0.08 = 8% improvement
    expected_delta_lo: float
    expected_delta_hi: float
    source: str  # paper, blog, vLLM docs URL — required
    applicability: Applicability = Field(default_factory=Applicability)
    safety: SafetyGate = Field(default_factory=SafetyGate)
    review: str | None = None  # reviewer sign-off note (None until reviewed)
