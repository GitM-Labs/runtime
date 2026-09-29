"""The experiment contract: what a hypothesis submits, and the verdict it gets back.

Two contracts, both pydantic with ``extra="forbid"`` so a misspelled field is an
error rather than a silently ignored default:

* :class:`ExperimentContract`: the submission. It holds the claim, the
  intervention, the workload, both configurations, the budget, the gates and the
  measurement protocol. Every analysis choice is fixed here, **before** any
  candidate outcome is read.
* :class:`Verdict`: the result. It is one of five terminal states, with the
  observed effect, its interval, gate results, cost, rollback outcome and the raw
  samples it was computed from.

Effects are **signed relative changes where positive means better**, whatever
the metric's direction: ``+0.08`` on a throughput metric is 8% more throughput,
and ``+0.08`` on a latency metric is 8% less latency. This keeps one sign
convention for every threshold in the contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Schema identities. Bump on any field change a consumer could misread.
SCHEMA: Final = "gitm.experiment.contract/v1"
VERDICT_SCHEMA: Final = "gitm.experiment.verdict/v1"

TerminalState = Literal["improved", "regressed", "inconclusive", "invalid", "failed_to_execute"]
Direction = Literal["higher_is_better", "lower_is_better"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EffectInterval(_Model):
    """A signed relative effect and its interval (positive = better)."""

    mean: float
    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> EffectInterval:
        if not self.lo <= self.mean <= self.hi:
            raise ValueError(f"need lo <= mean <= hi, got {self.lo}, {self.mean}, {self.hi}")
        return self


class Hypothesis(_Model):
    id: str
    submitter: str
    claim: str
    mechanism: str
    expected_effect: EffectInterval
    #: The claim is rejected when the effect's interval lies entirely below this.
    #: Reported alongside the terminal state: a candidate can be ``improved``
    #: yet still reject a claim that promised more than it delivered.
    reject_if_effect_below: float


class Metric(_Model):
    name: str
    unit: str
    direction: Direction


class Workload(_Model):
    #: Replay identity: the source trace's sha256, or a synthetic trace's
    #: parameter digest (see :mod:`gitm.traffic.schema`).
    trace_id: str
    source: str
    #: The traffic operating point, e.g. a :meth:`gitm.traffic.regime.Regime.label`
    #: plus the offered concurrency. Noise-floor entries are keyed on it.
    operating_point: str
    max_concurrency: int | None = Field(default=None, ge=1)
    duration_s: float = Field(gt=0)


class ArmConfig(_Model):
    label: str
    #: The engine's full launch configuration for this arm.
    engine_args: dict[str, Any]


class Budget(_Model):
    max_wall_clock_s: float = Field(gt=0)
    max_accelerator_s: float = Field(gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)


class CorrectnessGate(_Model):
    method: Literal["exact_match", "token_agreement", "eval_score"]
    #: Reference outputs or eval set the candidate is scored against.
    reference: str
    #: Every candidate repetition must score at least this.
    min_score: float = Field(ge=0.0, le=1.0)


class LatencyGate(_Model):
    metric: str
    direction: Direction = "lower_is_better"
    #: Largest tolerated worsening of this metric's mean, as a fraction of the
    #: baseline mean (0.05 = up to 5% worse is acceptable).
    max_regression: float = Field(ge=0.0)


class Protocol(_Model):
    #: Kept samples per arm, after warmup is discarded.
    repetitions: int = Field(ge=2)
    #: Samples discarded after **every** switch between arms. A restart-applied
    #: knob pays a cold engine on every switch, not only on the first.
    warmup_reps: int = Field(default=0, ge=0)
    comparison_order: Literal["ABAB", "ABBA", "randomized"]
    order_seed: int | None = None
    pairing: Literal["paired", "unpaired"]
    #: Only fixed-N is supported in v1. Sequential rules come later and need
    #: their own error-spending definition before they can be offered.
    stopping_rule: Literal["fixed_n"] = "fixed_n"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _seeded(self) -> Protocol:
        if self.comparison_order == "randomized" and self.order_seed is None:
            raise ValueError("comparison_order 'randomized' requires order_seed")
        return self


class ExperimentContract(_Model):
    schema_version: Literal["gitm.experiment.contract/v1"] = SCHEMA
    candidate_id: str
    hypothesis: Hypothesis
    #: The knob=value change under test. The candidate config must equal the
    #: baseline config with exactly these knobs changed.
    intervention: dict[str, Any] = Field(min_length=1)
    primary_metric: Metric
    workload: Workload
    baseline: ArmConfig
    candidate: ArmConfig
    budget: Budget
    correctness_gate: CorrectnessGate
    latency_gates: list[LatencyGate] = Field(default_factory=list)
    protocol: Protocol
    #: The noise-floor entry this experiment's detectability is read from. With
    #: none, detectability is ``not_established``; no entry from another
    #: operating point may stand in.
    noise_floor_ref: str | None = None

    @model_validator(mode="after")
    def _one_variable(self) -> ExperimentContract:
        base, cand = self.baseline.engine_args, self.candidate.engine_args
        if cand != {**base, **self.intervention}:
            raise ValueError(
                "candidate.engine_args must equal baseline.engine_args with exactly "
                "the intervention's knobs changed"
            )
        unchanged = [k for k, v in self.intervention.items() if base.get(k, object()) == v]
        if unchanged:
            raise ValueError(f"intervention does not change the baseline for: {unchanged}")
        return self

    def sha256(self) -> str:
        """Digest of the canonical JSON form, recorded on every verdict."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def load_contract(path: str | Path) -> ExperimentContract:
    """Read and validate a contract from YAML (JSON is valid YAML too)."""
    return ExperimentContract.model_validate(yaml.safe_load(Path(path).read_text()))


# --- verdict -----------------------------------------------------------------


class GateResult(_Model):
    name: str
    status: Literal["pass", "fail", "not_run"]
    detail: str = ""


class Cost(_Model):
    wall_clock_s: float
    accelerator_s: float
    estimated_usd: float | None = None


class RawSample(_Model):
    arm: Literal["baseline", "candidate"]
    #: Position in the run order, counting warmup samples.
    order_index: int
    warmup: bool
    metrics: dict[str, float]
    correctness: float | None = None
    wall_s: float
    accelerator_s: float


class Verdict(_Model):
    schema_version: Literal["gitm.experiment.verdict/v1"] = VERDICT_SCHEMA
    candidate_id: str
    hypothesis_id: str
    contract_sha256: str
    state: TerminalState
    reason: str
    #: Signed relative effect on the primary metric (positive = better).
    effect: float | None = None
    interval: tuple[float, float] | None = None
    #: The estimator and interval method, named exactly.
    method: str
    n_baseline: int
    n_candidate: int
    correctness: GateResult
    latency: list[GateResult]
    control_status: Literal["passed", "failed", "not_run"]
    #: ``None`` when there is no interval to judge the claim with.
    claim_rejected: bool | None = None
    detectability: str
    cost: Cost
    rollback: Literal["kept", "rolled_back", "not_applied", "restore_failed"]
    provenance: dict[str, Any]
    raw: list[RawSample]
