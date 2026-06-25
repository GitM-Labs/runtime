"""The reproducibility test, shared across benchmarks.

"A non-author re-ran `make baseline` on a clean box and hit the recorded numbers
in under 60 minutes." This encodes exactly that, in two checks:

1. **byte-identical dataset** — re-hash the staged dataset against the committed
   ``manifest.yaml``. Anyone with the manifest must be able to re-materialize the
   same bytes; this is the freeze contract.
2. **reproduced metric** — re-run the baseline across all seeds and confirm the
   spread gate still holds, within a wall-clock budget.

Returns a structured report (pass/fail + time-to-reproduce) so it composes in CI
and CI, rather than relying on a human reading log output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from gitm.bench.baseline import BaselineSummary, aggregate
from gitm.bench.manifest import VerifyResult, verify_manifest
from gitm.bench.runner import run_seed
from gitm.bench.schema import BenchConfig


@dataclass
class ReproduceReport:
    benchmark: str
    dataset_ok: bool
    metric_ok: bool
    within_time: bool
    minutes: float
    limit_minutes: float
    verify: VerifyResult | None
    summary: BaselineSummary | None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.dataset_ok and self.metric_ok and self.within_time

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "passed": self.passed,
            "dataset_ok": self.dataset_ok,
            "metric_ok": self.metric_ok,
            "within_time": self.within_time,
            "minutes": round(self.minutes, 3),
            "limit_minutes": self.limit_minutes,
            "verify": self.verify.summary() if self.verify else None,
            "spread": self.summary.spread if self.summary else None,
            "notes": self.notes,
        }


def reproduce(
    config: BenchConfig,
    *,
    stage_dir: str | Path,
    manifest_path: str | Path,
    runs_dir: str | Path,
    limit_minutes: float = 60.0,
    run_metric: bool = True,
    now=time.monotonic,
) -> ReproduceReport:
    """Run the reproducibility test and return a structured report.

    ``run_metric=False`` checks only dataset reproducibility (useful when the
    baseline harness isn't present yet); the metric check is then reported as
    skipped rather than failed.
    """
    t0 = now()
    notes: list[str] = []

    # 1. dataset byte-identity
    verify = verify_manifest(manifest_path, stage_dir)
    dataset_ok = verify.ok
    if not dataset_ok:
        notes.append(f"dataset mismatch: {verify.summary()}")

    # 2. metric reproduction
    summary: BaselineSummary | None = None
    metric_ok = True
    if run_metric:
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for i, seed in enumerate(config.seeds, start=1):
            run = run_seed(config, seed, manifest_path=manifest_path,
                           config_dir=Path(manifest_path).parent)
            (runs_dir / f"{config.name}_baseline_{i}.json").write_text(
                run.model_dump_json(indent=2) + "\n"
            )
            runs.append(run)
        summary = aggregate(runs, config)
        metric_ok = summary.passed
        if not metric_ok:
            notes.append("baseline gates failed on re-run")
    else:
        notes.append("metric check skipped (run_metric=False)")

    minutes = (now() - t0) / 60.0
    within_time = minutes <= limit_minutes
    if not within_time:
        notes.append(f"exceeded {limit_minutes} min budget ({minutes:.1f} min)")

    return ReproduceReport(
        benchmark=config.name,
        dataset_ok=dataset_ok,
        metric_ok=metric_ok,
        within_time=within_time,
        minutes=minutes,
        limit_minutes=limit_minutes,
        verify=verify,
        summary=summary,
        notes=notes,
    )
