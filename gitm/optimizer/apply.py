"""Apply an intervention spec to a live workload, behind a rollback gate.

    apply_intervention(spec, applicator) -> ApplyResult

Every live apply is wrapped in a snapshot → apply → measure → (keep | rollback)
cycle so a bad lever can never leave the workload worse than it started:

1. **snapshot** the pre-intervention state,
2. **apply** the spec's ``knob = value`` change (may raise on a bad value),
3. **measure** the resulting delta (a callback supplied by the caller),
4. **keep** it only if the measured delta clears ``min_keep_delta``; otherwise
   **restore** the snapshot.

Any exception in apply or measure also triggers a restore. The GPU-specific part
is isolated behind the :class:`Applicator` seam — :class:`ConfigFileApplicator`
edits a config file, :class:`DictApplicator` an in-memory dict (used in tests).
The live vLLM/engine applicator implements the same three methods (roadmap).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from gitm.kernels.spec import InterventionSpec

#: A measurement callback: returns the signed fractional delta after an apply
#: (``+0.08`` = 8% faster), or ``None`` if no measurement was taken (apply-only).
MeasureFn = Callable[[InterventionSpec], "float | None"]


@dataclass
class ApplyResult:
    applied: bool
    rolled_back: bool
    measured_delta: float | None
    error: str | None = None


class Applicator(Protocol):
    """The live-mutation seam. Implementations must be snapshot/restore-safe."""

    def snapshot(self) -> Any: ...
    def apply(self, spec: InterventionSpec) -> None: ...
    def restore(self, snapshot: Any) -> None: ...
    def measure(self, spec: InterventionSpec) -> float | None: ...


def apply_intervention(
    spec: InterventionSpec,
    applicator: Applicator,
    *,
    min_keep_delta: float = 0.0,
) -> ApplyResult:
    """Apply ``spec`` through ``applicator`` behind a rollback gate.

    ``min_keep_delta`` is the regression threshold: a measured delta below it
    (e.g. a slowdown) is rolled back. With no measurement (``measure`` returns
    ``None``) the change is kept — apply-only mode.
    """
    snapshot = applicator.snapshot()

    # Step 2: apply. A bad value (validation error) rolls straight back.
    try:
        applicator.apply(spec)
    except Exception as exc:
        applicator.restore(snapshot)
        return ApplyResult(False, rolled_back=True, measured_delta=None,
                           error=f"apply failed, restored: {exc}")

    # Step 3: measure. A crash mid-measurement also rolls back.
    try:
        delta = applicator.measure(spec)
    except Exception as exc:
        applicator.restore(snapshot)
        return ApplyResult(False, rolled_back=True, measured_delta=None,
                           error=f"measure failed, restored: {exc}")

    # Step 4: keep-or-rollback on the regression threshold.
    if delta is not None and delta < min_keep_delta:
        applicator.restore(snapshot)
        return ApplyResult(True, rolled_back=True, measured_delta=delta,
                           error=f"regression {delta:+.3f} < keep threshold "
                                 f"{min_keep_delta:+.3f}, restored")

    return ApplyResult(True, rolled_back=False, measured_delta=delta)


# --- reference applicators ---------------------------------------------------


def _set_knob(config: dict, spec: InterventionSpec) -> None:
    if spec.value is None:
        raise ValueError(
            f"intervention {spec.name!r} has no value to set on knob {spec.knob!r}"
        )
    config[spec.knob] = spec.value


class DryRunApplicator:
    """No live target — predict-only. apply/restore are no-ops; measure is None.

    Used by the embedded loop when no engine is attached (the loop runs
    end-to-end without a GPU): candidates flow through the pipeline and land in
    the report as *unverified* (measured_delta is None), never claimed as won.
    """

    def snapshot(self) -> None:
        return None

    def apply(self, spec: InterventionSpec) -> None:
        return None

    def restore(self, snapshot: None) -> None:
        return None

    def measure(self, spec: InterventionSpec) -> float | None:
        return None


class DictApplicator:
    """In-memory config dict applicator — the testable reference."""

    def __init__(self, config: dict, *, measure_fn: MeasureFn | None = None):
        self.config = config
        self._measure_fn = measure_fn

    def snapshot(self) -> dict:
        return copy.deepcopy(self.config)

    def apply(self, spec: InterventionSpec) -> None:
        _set_knob(self.config, spec)

    def restore(self, snapshot: dict) -> None:
        self.config.clear()
        self.config.update(snapshot)

    def measure(self, spec: InterventionSpec) -> float | None:
        return self._measure_fn(spec) if self._measure_fn else None


class ConfigFileApplicator:
    """Applies the knob to a YAML config file; snapshots/restores its bytes."""

    def __init__(self, path: str | Path, *, measure_fn: MeasureFn | None = None):
        self.path = Path(path)
        self._measure_fn = measure_fn

    def snapshot(self) -> bytes:
        return self.path.read_bytes() if self.path.exists() else b""

    def apply(self, spec: InterventionSpec) -> None:
        data = yaml.safe_load(self.path.read_text()) if self.path.exists() else {}
        if not isinstance(data, dict):
            raise ValueError(f"{self.path}: expected a mapping at top level")
        _set_knob(data, spec)
        self.path.write_text(yaml.safe_dump(data, sort_keys=False))

    def restore(self, snapshot: bytes) -> None:
        if snapshot:
            self.path.write_bytes(snapshot)
        elif self.path.exists():
            self.path.unlink()

    def measure(self, spec: InterventionSpec) -> float | None:
        return self._measure_fn(spec) if self._measure_fn else None


def apply_intervention_from_file(
    path: str | Path,
    *,
    config: str | Path | None = None,
    min_keep_delta: float = 0.0,
) -> dict:
    """CLI helper: apply an intervention YAML to a target ``config`` file.

    Without a ``config`` target there is nothing safe to mutate, so this reports
    a no-op rather than pretending. A live engine applicator is the
    other implementation of the seam.
    """
    with open(path) as fh:
        spec = InterventionSpec.model_validate(yaml.safe_load(fh))

    if config is None:
        return {
            "intervention": spec.name,
            "applied": False,
            "rolled_back": False,
            "measured_delta": None,
            "error": "no target config given (--config); supply a config file or "
                     "a live engine applicator to apply.",
        }

    res = apply_intervention(spec, ConfigFileApplicator(config), min_keep_delta=min_keep_delta)
    return {
        "intervention": spec.name,
        "applied": res.applied,
        "rolled_back": res.rolled_back,
        "measured_delta": res.measured_delta,
        "error": res.error,
    }
