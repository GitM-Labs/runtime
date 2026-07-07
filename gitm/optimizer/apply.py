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
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.vllm_knobs import get_knob, knob_kind, set_knob

if TYPE_CHECKING:
    from gitm.safety.audit import AuditLog

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
    audit: AuditLog | None = None,
) -> ApplyResult:
    """Apply ``spec`` through ``applicator`` behind a rollback gate.

    ``min_keep_delta`` is the regression threshold: a measured delta below it
    (e.g. a slowdown) is rolled back. With no measurement (``measure`` returns
    ``None``) the change is kept — apply-only mode.

    When an ``audit`` log is supplied, every live mutation and every rollback is
    recorded to the durable safety trail (best-effort — a broken audit sink never
    blocks the apply). Pass one only where the applicator mutates a real target;
    a dry-run leaves it ``None`` so the trail stays free of no-op entries.
    """
    snapshot = applicator.snapshot()

    # Step 2: apply. A bad value (validation error) rolls straight back.
    try:
        applicator.apply(spec)
    except Exception as exc:
        applicator.restore(snapshot)
        _audit(audit, "revert", spec, cause=f"apply failed, restored: {exc}")
        return ApplyResult(False, rolled_back=True, measured_delta=None,
                           error=f"apply failed, restored: {exc}")
    _audit(audit, "apply", spec, cause="applied live", knob=spec.knob, value=spec.value)

    # Step 3: measure. A crash mid-measurement also rolls back.
    try:
        delta = applicator.measure(spec)
    except Exception as exc:
        applicator.restore(snapshot)
        _audit(audit, "revert", spec, cause=f"measure failed, restored: {exc}")
        return ApplyResult(False, rolled_back=True, measured_delta=None,
                           error=f"measure failed, restored: {exc}")

    # Step 4: keep-or-rollback on the regression threshold.
    if delta is not None and delta < min_keep_delta:
        applicator.restore(snapshot)
        _audit(audit, "revert", spec,
               cause=f"regression {delta:+.3f} < keep threshold {min_keep_delta:+.3f}")
        return ApplyResult(True, rolled_back=True, measured_delta=delta,
                           error=f"regression {delta:+.3f} < keep threshold "
                                 f"{min_keep_delta:+.3f}, restored")

    return ApplyResult(True, rolled_back=False, measured_delta=delta)


def _audit(
    audit: AuditLog | None, event: str, spec: InterventionSpec, *, cause: str, **detail: Any
) -> None:
    """Record one apply/revert to the safety trail — best-effort, never raises."""
    if audit is None:
        return
    try:
        audit.record(event, spec.name, cause, **detail)
    except Exception:
        pass


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


@dataclass
class EngineABResult:
    """Outcome of the live decode-throughput A/B for one knob change."""

    knob: str
    value: Any
    baseline_tps: float
    candidate_tps: float
    speedup: float  # candidate / baseline
    # measure-time indicator (delta >= 0); the *authoritative* keep/rollback
    # decision is ApplyResult.rolled_back from apply_intervention, which gates on
    # the caller's min_keep_delta. Report verdicts derive from ApplyResult, not this.
    kept: bool
    via: str = "hot-swap"  # "hot-swap" (scheduling knob) | "restart" (structural knob)

    @property
    def verdict(self) -> str:
        d = self.speedup - 1.0
        return f"{'kept' if self.kept else 'rolled back'} ({d:+.1%} decode throughput, via {self.via})"


class StructuralKnobRequiresRestart(RuntimeError):
    """A structural knob was applied with no restart hook to enact it.

    Raised by :meth:`LiveEngineApplicator.apply` so ``apply_intervention`` rolls
    the candidate back with a clear reason — never silently sets a structural
    field the running engine won't honor.
    """


class LiveEngineApplicator:
    """Apply a knob to a live (vLLM) engine, gated by a real decode-throughput A/B.

    Routes by knob taxonomy (:mod:`gitm.optimizer.vllm_knobs`):

    * **scheduling** knobs (``max_num_seqs``, ``max_num_batched_tokens``,
      ``scheduling_policy``) are *hot-swapped* in place — set on the live
      scheduler config, effective next step, restored by setting the old value.
    * **structural** knobs (parallelism, dtype, quantization, block size, …) can
      only take effect on a fresh engine, so they are routed through
      ``restart_fn(engine, knob, value) -> new_engine``: the candidate engine
      replaces the live one for the A/B, and restore swaps the original back
      (shutting the candidate down best-effort). With no ``restart_fn`` a
      structural knob raises :class:`StructuralKnobRequiresRestart`, which
      ``apply_intervention`` turns into a clean rollback — never a silent no-op.

    The Applicator protocol's three phases map to a measured A/B: ``snapshot``
    benchmarks baseline decode throughput; ``apply`` hot-swaps or restarts;
    ``measure`` benchmarks the candidate and returns the signed speedup
    (``candidate/baseline - 1``), so a slowdown trips ``min_keep_delta`` and is
    rolled back via ``restore``.

    ``throughput_fn(engine) -> tokens_per_second`` is injected (the caller owns
    what "a decode" means). ``getter``/``setter`` default to the knob-taxonomy
    resolver and are overridable for engines that gate config behind methods.
    """

    def __init__(
        self,
        engine: Any,
        *,
        throughput_fn: Callable[[Any], float],
        restart_fn: Callable[[Any, str, Any], Any] | None = None,
        getter: Callable[[Any, str], Any] | None = None,
        setter: Callable[[Any, str, Any], None] | None = None,
        reps: int = 1,
    ) -> None:
        self.engine = engine
        self._tps = throughput_fn
        self._restart_fn = restart_fn
        self._getter = getter or get_knob
        self._setter = setter or set_knob
        self._reps = max(1, reps)
        self._baseline_tps: float | None = None
        # Restore record: ("hotswap", knob, old_value) | ("restart", old_engine) | None.
        self._prev: tuple[Any, ...] | None = None
        self.last_result: EngineABResult | None = None

    def _bench(self) -> float:
        return sum(self._tps(self.engine) for _ in range(self._reps)) / self._reps

    def snapshot(self) -> dict[str, Any]:
        # Reset both the restore record AND last_result: snapshot() runs at the
        # start of every apply_intervention, so a candidate whose apply() fails
        # (e.g. a structural knob with no restart hook, where measure() never
        # runs) must not leave the *previous* candidate's A/B result visible.
        self._prev = None
        self.last_result = None
        self._baseline_tps = self._bench()
        return {"baseline_tps": self._baseline_tps}

    def apply(self, spec: InterventionSpec) -> None:
        if spec.value is None:
            raise ValueError(f"intervention {spec.name!r} has no value for knob {spec.knob!r}")

        if knob_kind(spec.knob) == "scheduling":
            # Hot-swap in place. Record the restore point only AFTER a successful
            # set: if the knob can't be located the setter raises, _prev stays
            # None, and restore() is a no-op (nothing changed → nothing to undo).
            try:
                prev = self._getter(self.engine, spec.knob)
            except AttributeError:
                prev = None
            self._setter(self.engine, spec.knob, spec.value)
            self._prev = ("hotswap", spec.knob, prev)
            return

        # Structural knob — needs a restart to take effect.
        if self._restart_fn is None:
            raise StructuralKnobRequiresRestart(
                f"knob {spec.knob!r} is structural (needs an engine restart); "
                "no restart_fn supplied, so it cannot be applied live"
            )
        old_engine = self.engine
        new_engine = self._restart_fn(old_engine, spec.knob, spec.value)
        if new_engine is None:
            raise StructuralKnobRequiresRestart(
                f"restart_fn produced no engine for structural knob {spec.knob!r}"
            )
        self.engine = new_engine
        self._prev = ("restart", old_engine)

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self._prev is None:
            return
        tag = self._prev[0]
        if tag == "hotswap":
            _, knob, old = self._prev
            self._setter(self.engine, knob, old)
        elif tag == "restart":
            _, old_engine = self._prev
            self._shutdown(self.engine)  # drop the candidate engine we built
            self.engine = old_engine
        # Consume the restore record so a second restore() can't re-undo (or
        # re-shutdown the already-discarded candidate engine) a second time.
        self._prev = None

    @staticmethod
    def _shutdown(engine: Any) -> None:
        """Best-effort release of a candidate engine built for a restart A/B."""
        for path in ("shutdown", "llm_engine.shutdown", "engine.shutdown"):
            obj: Any = engine
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if callable(obj):
                try:
                    obj()
                except Exception:
                    pass
                return

    def measure(self, spec: InterventionSpec) -> float | None:
        baseline = self._baseline_tps if self._baseline_tps is not None else self._bench()
        # A non-positive baseline means the A/B has no valid reference — an idle
        # engine, a probe that returned 0, no tokens produced. Raising (vs forcing
        # speedup=1.0) makes apply_intervention roll the candidate back instead of
        # silently *keeping* an unmeasurable change as a non-regression.
        if baseline <= 0:
            raise ValueError(
                f"baseline decode throughput is {baseline}; cannot run a valid A/B "
                f"for knob {spec.knob!r}"
            )
        candidate = self._bench()
        speedup = candidate / baseline
        delta = speedup - 1.0
        via = "restart" if (self._prev and self._prev[0] == "restart") else "hot-swap"
        self.last_result = EngineABResult(
            knob=spec.knob, value=spec.value, baseline_tps=baseline,
            candidate_tps=candidate, speedup=speedup, kept=delta >= 0.0, via=via,
        )
        return delta


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
