"""Customer-verification export — the A/B result as data, not prose.

The provenance report (:mod:`gitm.optimizer.report`) is written for a human: it
says a lever was kept and by how much. That is not enough for a customer who
wants to re-measure the claim on their own harness, because the report gives a
percentage and a sentence — not the configuration each number was measured
under, nor how much the measurement itself scattered.

This module writes the same A/B as structured data: both throughputs, the full
engine configuration on each side, the scatter and rep count behind them, and
the environment they were measured in. Nothing here measures anything new — the
numbers all come from :class:`~gitm.optimizer.apply.EngineABResult`, which the
loop already produces and currently reads two fields from.

Two things are deliberate:

* **``kept`` comes from the gate, not the measurement.** ``EngineABResult.kept``
  is a measure-time ``delta >= 0`` indicator; the authoritative keep/rollback
  decision is ``ApplyResult.rolled_back``, which applies the caller's
  ``min_keep_delta``. Exporting the indicator would occasionally claim a lever
  was kept that the gate actually rolled back.
* **An agreement band travels with every number.** At ``reps=1`` the measured
  scatter is exactly 0, so a customer re-measuring would "disagree" with us over
  ordinary run-to-run jitter. :data:`MIN_NOISE_BAND` floors the band so the
  export states the precision it actually has instead of implying certainty it
  doesn't.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gitm.kernels.spec import InterventionSpec
    from gitm.optimizer.apply import ApplyResult, EngineABResult
    from gitm.optimizer.report import Provenance

#: Export format version. Bump on any breaking field change so a consumer can
#: tell which shape it is reading.
SCHEMA = 1

#: Floor on the relative band within which a re-measurement counts as agreeing
#: with ours. A single-rep A/B reports zero scatter, which would imply a
#: precision no GPU benchmark has; 2% is a conservative stand-in for ordinary
#: run-to-run variation on decode throughput.
MIN_NOISE_BAND = 0.02


@dataclass
class VerificationRecord:
    """One baseline↔candidate comparison, in full.

    ``baseline_config`` / ``candidate_config`` are the engine's complete kwargs
    on each side, not just the knob that moved: a customer reproducing at a
    different ``gpu_memory_utilization`` measures a different system, and
    without both configs neither side can tell a real disagreement from a setup
    difference.
    """

    intervention_name: str
    summary: str
    knob: str
    value: Any
    source: str  # the citation the lever came from

    baseline_tps: float
    candidate_tps: float
    speedup: float
    delta: float  # speedup - 1, the signed change

    baseline_std: float
    candidate_std: float
    reps: int
    agreement_band: float  # relative; a re-measurement inside this agrees
    significant: bool  # the gain cleared the measured noise band

    kept: bool  # from the rollback gate, not the measure-time indicator
    via: str  # "hot-swap" | "restart"

    baseline_config: dict[str, Any] = field(default_factory=dict)
    candidate_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_record(
    spec: InterventionSpec,
    ab: EngineABResult,
    apply_result: ApplyResult,
    *,
    baseline_config: dict[str, Any] | None = None,
    candidate_config: dict[str, Any] | None = None,
) -> VerificationRecord:
    """Assemble one record from a live A/B and its gate decision.

    ``baseline_config`` must be captured *before* the apply — the engine's
    kwargs are mutated in place by a hot-swap and replaced entirely by a
    restart, so reading them afterwards yields the candidate on both sides.
    """
    return VerificationRecord(
        intervention_name=spec.name,
        summary=spec.summary,
        knob=spec.knob,
        value=spec.value,
        source=spec.source,
        baseline_tps=ab.baseline_tps,
        candidate_tps=ab.candidate_tps,
        speedup=ab.speedup,
        delta=ab.speedup - 1.0,
        baseline_std=ab.baseline_std,
        candidate_std=ab.candidate_std,
        reps=ab.reps,
        agreement_band=max(ab.rel_std, MIN_NOISE_BAND),
        significant=ab.significant,
        # The gate decides, not the measurement — see the module docstring.
        kept=not apply_result.rolled_back,
        via=ab.via,
        baseline_config=dict(baseline_config or {}),
        candidate_config=dict(candidate_config or {}),
    )


def _environment(gpu_sku: str | None) -> dict[str, Any]:
    """Best-effort description of the box the numbers were measured on.

    Every field degrades to ``None`` rather than a guess: an export that names
    the wrong GPU is worse than one that admits it doesn't know.
    """
    from gitm.cuda_env import driver_cuda, torch_cuda

    def _ver(v: tuple[int, int] | None) -> str | None:
        return f"{v[0]}.{v[1]}" if v else None

    return {
        "gpu_sku": gpu_sku,
        "driver_cuda": _ver(driver_cuda()),
        "torch_cuda": _ver(torch_cuda()),
    }


def build_export(
    records: list[VerificationRecord],
    provenance: Provenance,
    *,
    gpu_sku: str | None = None,
) -> dict[str, Any]:
    """The full export document: provenance + environment + every comparison."""
    return {
        "schema": SCHEMA,
        "provenance": {
            "workload_id": provenance.workload_id,
            "fingerprint": provenance.fingerprint,
            "run_id": provenance.run_id,
            "git_sha": provenance.git_sha,
            "gitm_version": provenance.gitm_version,
        },
        "environment": _environment(gpu_sku),
        "protocol": {
            "metric": "decode throughput (tokens/sec)",
            "reps": "each side benchmarked `reps` times; std is the sample stdev",
            "agreement_band": (
                "relative band around our numbers within which a re-measurement "
                f"agrees; floored at {MIN_NOISE_BAND:.0%} because a single-rep A/B "
                "reports zero scatter"
            ),
            "kept": "decided by the rollback gate (min_keep_delta), not by delta >= 0",
        },
        "results": [r.to_dict() for r in records],
    }


def write_verification(
    records: list[VerificationRecord],
    provenance: Provenance,
    out_path: str | Path,
    *,
    gpu_sku: str | None = None,
) -> str:
    """Write the export as JSON to ``out_path``. Returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_export(records, provenance, gpu_sku=gpu_sku)
    out_path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return str(out_path)
