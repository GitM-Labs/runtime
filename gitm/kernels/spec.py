"""Intervention spec — schema for every entry in the library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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
    knob: str  # vLLM config key, e.g. "max_num_batched_tokens" — or a display
    # label ("k1=v1,k2=v2") for a joint candidate; see ``knobs`` below.
    value: int | float | str | bool | None = None  # value to set on apply (single-knob)
    # Scale the engine's CURRENT value by this factor instead of hardcoding one
    # number (see gitm.optimizer.vllm_knobs.resolve_relative_value). None (the
    # default): value is a static literal. value is the offline/predict-only
    # fallback when there's no live engine to read a current setting from.
    value_multiplier: float | None = None
    # A sweep of multipliers (e.g. [0.5, 2.0, 4.0]) instead of one — expands
    # this entry into one candidate per point (expand_relative_candidates).
    # Empty (default): no sweep.
    value_multiplier_grid: list[float] = Field(default_factory=list)
    value_max: float | None = None  # clamp the scaled result (e.g. a fraction < 1.0)
    value_min: float | None = None
    # >1 knob=value pair applied/rolled back together as one atomic unit. Empty
    # (default) means single-knob — use knob/value instead. See knob_values.
    knobs: dict[str, Any] = Field(default_factory=dict)
    applies_to_kernels: list[str] = Field(default_factory=list)  # substring match
    #: The lever acts on the whole decode step rather than on named ops — batch
    #: shape, admission order, graph capture, sharding degree. Kept apart from
    #: ``applies_to_kernels`` because "every op" is a property of the lever, not a
    #: list that happens to name the ops one architecture has: enumerating a dense
    #: model's ops made these levers invisible on a sparse-MoE one, where the same
    #: knob applies just as much and none of those op names occur.
    whole_step: bool = False
    #: The lever's gain comes from making the ops it names *faster*, so a region
    #: already at its roofline floor has nothing for it to recover.
    #:
    #: Separate from ``applies_to_kernels``, which says which kernels the lever
    #: touches — what coverage needs — and *not* where its gain comes from. Most
    #: op-scoped levers in the catalogue fail this: five of the six scoped to
    #: ``attn_score_value`` work through cache capacity, host swap or avoided
    #: recomputation (``kv_cache_dtype_fp8`` doubles capacity,
    #: ``preemption_mode_swap`` swaps instead of recomputing), and none of those
    #: need the attention kernels to be above their floor to pay off. Reading the
    #: two fields as one would reject them on a sound measurement and wrong
    #: reasoning.
    #:
    #: "Makes the op faster" is also narrower than "shortens the step by way of
    #: that op". ``enable_eplb`` cuts expert stragglers, but a straggler rank
    #: runs *more* expert GEMMs rather than slower ones, so each kernel sits at
    #: its floor while the step waits — a distribution problem the per-op gap
    #: cannot see. Only a lever whose gain is the slack between one op and its
    #: own floor belongs here.
    #:
    #: Only consulted by the selection gate, and defaults false so a lever is
    #: never filtered on a mechanism nobody has stated.
    recovers_kernel_time: bool = False
    expected_delta_mean: float  # signed, e.g. +0.08 = 8% improvement
    expected_delta_lo: float
    expected_delta_hi: float
    source: str  # paper, blog, vLLM docs URL — required
    applicability: Applicability = Field(default_factory=Applicability)
    safety: SafetyGate = Field(default_factory=SafetyGate)
    review: str | None = None  # reviewer sign-off note (None until reviewed)
    # An accuracy sentinel the apply path runs after measuring and before
    # keeping: None to pass, else the reason. The live keep gate sees throughput
    # alone, so a lever that can change model output carries its own check here
    # and apply_intervention rolls back on failure whatever applicator is used.
    # Runtime-only: a callable, never read from or written to library YAML.
    correctness_gate: Callable[[Any], str | None] | None = Field(
        default=None, exclude=True, repr=False
    )

    @property
    def knob_values(self) -> dict[str, Any]:
        """The knob=value pairs this spec wants applied — the one shape every
        applicator should read, whether the spec is single-knob or joint."""
        return dict(self.knobs) if self.knobs else {self.knob: self.value}
