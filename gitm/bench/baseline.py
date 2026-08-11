"""Baseline aggregation and the two sign-off gates.

A baseline is *locked* when three convergent runs (one per seed) agree on the
top-line metric within 2 % — the recorded baseline is their mean. Two gates
decide sign-off, both encoded here so every benchmark is judged identically:

* **spread gate** — ``max-min`` over ``mean`` of the three metric values must
  be under ``spread_tolerance`` (default 2 %). Convergence is what makes the
  number trustworthy as an optimization reference.
* **saturation gate** — wall-clock-weighted GPU active % must stay under
  ``gpu_active_ceiling`` (default 85 %). A saturated benchmark has no residual
  headroom for the runtime to find, so it trips the same-day swap rule.

A third, optional check compares the recorded mean against ``baseline_target``
(e.g. HFT ≥ 25 M events/s) in the configured direction.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from gitm.bench.schema import BaselineRun, BenchConfig


def load_runs(paths: list[str | Path]) -> list[BaselineRun]:
    runs = []
    for p in paths:
        runs.append(BaselineRun.model_validate_json(Path(p).read_text()))
    return runs


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class BaselineSummary:
    benchmark: str
    metric: str
    n: int
    mean: float
    stddev: float
    spread: float  # (max - min) / mean
    gpu_active_overall: float | None  # worst (max) across runs; None = unmeasured
    recorded: float  # the number we publish = mean
    gates: list[GateResult] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "metric": self.metric,
            "n": self.n,
            "seeds": self.seeds,
            "recorded": self.recorded,
            "mean": self.mean,
            "stddev": self.stddev,
            "spread": self.spread,
            "gpu_active_overall": self.gpu_active_overall,
            "passed": self.passed,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
        }


def aggregate(runs: list[BaselineRun], config: BenchConfig) -> BaselineSummary:
    """Aggregate baseline runs and evaluate the sign-off gates.

    Does not assume exactly three runs — fewer is a gate failure, more is fine —
    so a pair can iterate on two and still get a meaningful spread reading.
    """
    if not runs:
        raise ValueError("no baseline runs to aggregate")

    for r in runs:
        if r.benchmark != config.name:
            raise ValueError(
                f"run benchmark {r.benchmark!r} != config {config.name!r}"
            )
        if r.metric != config.metric:
            raise ValueError(f"run metric {r.metric!r} != config {config.metric!r}")

    values = [r.metric_value for r in runs]
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    spread = (max(values) - min(values)) / mean if mean else float("inf")
    gpu_values = [r.gpu_active_overall() for r in runs]
    gpu_coverage_ok = all(value is not None for value in gpu_values)
    gpu_overall = max(value for value in gpu_values if value is not None) if gpu_coverage_ok else None

    gates: list[GateResult] = []

    # Gate 1: three convergent seeds.
    n_ok = len(runs) >= 3
    spread_ok = spread <= config.spread_tolerance
    gates.append(
        GateResult(
            "count",
            n_ok,
            f"{len(runs)} run(s); need >= 3 convergent seeds",
        )
    )
    gates.append(
        GateResult(
            "spread",
            spread_ok,
            f"spread {spread:.2%} vs tolerance {config.spread_tolerance:.2%}",
        )
    )

    # Gate 2: saturation / swap rule.
    sat_ok = gpu_overall is not None and gpu_overall < config.gpu_active_ceiling
    gates.append(
        GateResult(
            "saturation",
            sat_ok,
            (
                f"GPU active {gpu_overall:.1%} vs ceiling {config.gpu_active_ceiling:.0%}"
                + ("" if sat_ok else " — trips swap rule, shard same day")
                if gpu_overall is not None
                else "GPU-active coverage unavailable: at least one run has no positive-wall-time "
                "stall breakdown; saturation sign-off refused"
            ),
        )
    )

    provenance_issues: list[str] = []
    for run in runs:
        if not run.git_sha or run.git_sha == "unknown":
            provenance_issues.append(f"seed {run.seed}: git SHA unavailable")
        if not run.manifest_sha256:
            provenance_issues.append(f"seed {run.seed}: dataset manifest digest unavailable")
        if not run.gpu_name or run.gpu_name.lower() in {"cpu", "gpu-unknown"}:
            provenance_issues.append(f"seed {run.seed}: GPU identity unavailable ({run.gpu_name!r})")
        if run.device_count < 1:
            provenance_issues.append(f"seed {run.seed}: no GPU devices reported")
        provenance_issues.extend(f"seed {run.seed}: {note}" for note in run.provenance_warnings)
    gates.append(
        GateResult(
            "provenance",
            not provenance_issues,
            "all runs pin code, dataset, and GPU identity"
            if not provenance_issues
            else "; ".join(provenance_issues),
        )
    )

    # Gate 3 (optional): metric vs target.
    if config.baseline_target is not None:
        if config.target_direction == "ge":
            tgt_ok = mean >= config.baseline_target
            cmp = ">="
        else:
            tgt_ok = mean <= config.baseline_target
            cmp = "<="
        gates.append(
            GateResult(
                "target",
                tgt_ok,
                f"mean {mean:.4g} {cmp} target {config.baseline_target:.4g}",
            )
        )

    return BaselineSummary(
        benchmark=config.name,
        metric=config.metric,
        n=len(runs),
        mean=mean,
        stddev=stddev,
        spread=spread,
        gpu_active_overall=gpu_overall,
        recorded=mean,
        gates=gates,
        seeds=sorted(r.seed for r in runs),
    )


def aggregate_files(paths: list[str | Path], config: BenchConfig) -> BaselineSummary:
    return aggregate(load_runs(paths), config)


def write_summary(summary: BaselineSummary, out: str | Path) -> Path:
    out = Path(out)
    out.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    return out
