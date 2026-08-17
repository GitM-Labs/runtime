"""Canonical benchmark data shapes.

Three contracts, shared verbatim across all benchmarks:

* :class:`BenchConfig` — the parsed ``bench.toml``. The *only* place a benchmark
  declares anything domain-specific (metric name, seeds, expected stall bands,
  work-unit command). Keeps the benchmark layer dumb: the shared tooling reads
  this; interns never touch plumbing.
* :class:`StallPhase` — one row of the canonical stall-breakdown table.
* :class:`BaselineRun` — the ``$GITM_SCRATCH/runs/<name>_baseline_N.json``
  contract that every baseline run emits and that sign-off reads back.
"""

from __future__ import annotations

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vendor = Literal["nvidia", "amd"]
#: Whether the top-line metric should be ``>=`` target (throughput) or ``<=``
#: target (latency). Every current benchmark is throughput, but the gate is
#: written once so a latency benchmark drops in without code changes.
TargetDirection = Literal["ge", "le"]


class Band(BaseModel):
    """An inclusive ``[lo, hi]`` fraction band, e.g. expected data-stall 10–25 %.

    Stored as fractions in ``[0, 1]`` (0.10, 0.25), not percentages, so the
    monitor can compare against measured fractions directly.
    """

    model_config = ConfigDict(extra="forbid")
    lo: float = Field(ge=0.0, le=1.0)
    hi: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Band:
        if self.lo > self.hi:
            raise ValueError(f"band lo ({self.lo}) must be <= hi ({self.hi})")
        return self

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi


class ExpectedStall(BaseModel):
    """The spec's section-4 expected stall profile, as comparable bands."""

    model_config = ConfigDict(extra="forbid")
    cpu: Band
    data_stall: Band
    sync: Band
    gpu_active: Band


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: Path under ``$GITM_S3_ROOT/datasets/<root>/`` (canonical store).
    root: str
    #: Manifest filename, relative to ``benchmarks/<name>/``.
    manifest: str = "manifest.yaml"


class WorkUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    #: Baseline harness command. ``{seed}`` and ``{warm_window_s}`` are
    #: substituted before exec. The command must emit a JSON line with the
    #: top-line metric so :mod:`gitm.bench.baseline` can read it.
    command: str
    description: str = ""


class BenchConfig(BaseModel):
    """Parsed ``benchmarks/<name>/bench.toml`` — the whole domain surface.

    Everything the shared tooling needs to manifest, profile, run, and gate a
    benchmark without knowing what domain it is.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    vendor: Vendor  # selects nsys (nvidia) vs rocprof (amd)
    metric: str  # e.g. "events_per_second"
    warm_window_s: int = Field(gt=0)
    seeds: list[int] = Field(min_length=1)
    spread_tolerance: float = Field(default=0.02, gt=0.0, le=1.0)
    gpu_active_ceiling: float = Field(default=0.85, gt=0.0, le=1.0)
    baseline_target: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    target_direction: TargetDirection = "ge"
    dataset: DatasetRef
    work_unit: WorkUnit
    expected_stall: ExpectedStall

    @classmethod
    def from_toml(cls, path: str | Path) -> BenchConfig:
        data = tomllib.loads(Path(path).read_text())
        return cls.model_validate(data)


class StallPhase(BaseModel):
    """One row of the canonical stall-breakdown table.

    | Phase | CPU % | Data-stall % | Sync % | GPU active % | Throughput | Wall-clock |

    Percentages are stored as fractions in ``[0, 1]``. The four fraction fields
    are *time attribution* and should sum to roughly 1.0 within a phase
    (validated loosely — overlap on the host side makes exact sums impossible).
    """

    model_config = ConfigDict(extra="forbid")

    phase: str
    cpu: float = Field(ge=0.0, le=1.0)
    data_stall: float = Field(ge=0.0, le=1.0)
    sync: float = Field(ge=0.0, le=1.0)
    gpu_active: float = Field(ge=0.0, le=1.0)
    throughput: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    wall_clock_s: float = Field(ge=0.0)


class BaselineRun(BaseModel):
    """The ``<name>_baseline_N.json`` contract.

    One locked baseline run. Three of these (one per seed) constitute a signed
    baseline once :mod:`gitm.bench.baseline` confirms <2 % spread and the GPU
    active ceiling.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    seed: int
    vendor: Vendor
    metric: str
    metric_value: float = Field(gt=0.0, allow_inf_nan=False)
    warm_window_s: int = Field(gt=0)

    # Provenance: a baseline is only reproducible if these are pinned.
    git_sha: str
    gitm_version: str
    harness_commit: str | None = None
    manifest_sha256: str | None = None  # sha256 of the dataset manifest itself
    provenance_warnings: list[str] = Field(default_factory=list)

    gpu_name: str = ""
    device_count: int = 1
    started_at_ns: int = 0
    ended_at_ns: int = 0

    stall_breakdown: list[StallPhase] = Field(default_factory=list)

    def gpu_active_overall(self) -> float | None:
        """Wall-clock-weighted GPU active fraction across phases.

        This is the number checked against ``gpu_active_ceiling`` — a single
        hot phase can saturate even when the unweighted mean looks calm.
        """
        total = sum(p.wall_clock_s for p in self.stall_breakdown)
        if total <= 0.0:
            return None
        return sum(p.gpu_active * p.wall_clock_s for p in self.stall_breakdown) / total
