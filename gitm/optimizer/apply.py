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
import gc
import math
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
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
    #: The baseline could not be put back after this candidate, so whatever the
    #: applicator mutates is in a state nothing measured. A caller holding more
    #: candidates must stop: every A/B after this one would be taken against an
    #: unknown baseline, or against no engine at all. Also set when an earlier
    #: gate is still running against the target.
    restore_failed: bool = False

    @property
    def kept(self) -> bool:
        """Whether the gate kept this candidate. The one place that is decided.

        Not ``not rolled_back``. A candidate whose restore failed was not rolled
        back, but it was not kept either: every restore follows a rejection, so
        the gate had already said no. Reading ``not rolled_back`` as kept turned a
        candidate measured at -20% whose rollback then failed into a kept result,
        which history counts as a win.
        """
        return self.applied and not self.rolled_back and not self.restore_failed


class EngineTimeout(RuntimeError):
    """An engine call ran past its deadline and the engine's processes were killed.

    Raised by the workload's watchdog (K-2). The engine it was raised for is
    gone: a candidate that timed out is a failed candidate, and a baseline that
    timed out leaves nothing to measure against.
    """


class RestoreFailed(RuntimeError):
    """The baseline could not be restored, and the target is in an unknown state.

    Raised by an applicator rather than letting the rebuild's own exception
    escape, so :func:`apply_intervention` can tell "the candidate failed" (the
    normal case, which it rolls back) from "the rollback failed" (which it cannot
    do anything about and must not try twice).
    """


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
    ``None``) the change is kept — apply-only mode. A spec carrying a
    ``correctness_gate`` is additionally rolled back when that gate fails,
    regardless of the measured delta, or when it gives no verdict within
    ``spec.safety.correctness_gate_timeout_s``.

    When an ``audit`` log is supplied, every live mutation and every rollback is
    recorded to the durable safety trail (best-effort — a broken audit sink never
    blocks the apply). Pass one only where the applicator mutates a real target;
    a dry-run leaves it ``None`` so the trail stays free of no-op entries.
    """
    # The gate callable is dropped when a spec is saved; don't apply ungated.
    if spec.safety.requires_correctness_gate and spec.correctness_gate is None:
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error="spec requires a correctness gate and carries none; "
                                 "nothing applied")
    # An earlier gate is still loading the server, so no A/B can be trusted.
    # restore_failed=True makes the loops stop.
    if running := _gates_still_running():
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error=f"correctness gate still running ({', '.join(running)}); "
                                 "nothing applied",
                           restore_failed=True)

    # Step 1: snapshot. On a live engine this benchmarks the baseline, so it can
    # fail the same ways a measurement can — and when it does, nothing has been
    # applied yet. Returned as an error, not raised: one candidate whose baseline
    # could not be taken must not end a run that has other candidates to try.
    try:
        snapshot = applicator.snapshot()
    except EngineTimeout as exc:
        # The baseline itself hung and was killed, so there is no engine left to
        # measure the next candidate against: the run has lost its engine.
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error=f"baseline timed out, nothing applied: {exc}",
                           restore_failed=True)
    except Exception as exc:
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error=f"snapshot failed, nothing applied: {exc}")

    restore_started = False

    def _unrestored(applied: bool, delta: float | None, cause: str,
                    exc: BaseException) -> ApplyResult:
        """The result for a candidate whose rollback itself failed.

        Returned, never raised. A restore that fails is a baseline rebuild that
        could not get the memory back, and raising it here ended the whole run
        with no report — losing every A/B already measured at exactly the point
        the run most needed to write them down. ``rolled_back`` is False because
        nothing was rolled back; saying otherwise would tell the reader the
        baseline is in place.
        """
        nonlocal restore_started
        restore_started = True
        _audit(audit, "restore_failed", spec, knobs=_knob_values(spec),
               cause=f"{cause}; restore failed: {exc}")
        return ApplyResult(applied, rolled_back=False, measured_delta=delta,
                           error=f"{cause}; restore failed: {exc}",
                           restore_failed=True)

    def _rollback(applied: bool, delta: float | None, cause: str) -> ApplyResult | None:
        """Restore the snapshot. ``None`` on success, else the unrestored result."""
        nonlocal restore_started
        restore_started = True
        try:
            applicator.restore(snapshot)
        except Exception as exc:
            return _unrestored(applied, delta, cause, exc)
        return None

    def _decide() -> ApplyResult:
        """Steps 2-4: apply, measure, judge. The result says whether it was kept."""
        # Step 2: apply. A bad value (validation error) rolls straight back.
        try:
            applicator.apply(spec)
        except RestoreFailed as exc:
            # The applicator already tried to put the baseline back inside apply
            # and could not. Calling restore() again would retry the same
            # rebuild that just failed.
            return _unrestored(False, None, "apply failed", exc)
        except Exception as exc:
            if (bad := _rollback(False, None, f"apply failed: {exc}")) is not None:
                return bad
            _audit(audit, "revert", spec, cause=f"apply failed, restored: {exc}",
                   knobs=_knob_values(spec))
            return ApplyResult(False, rolled_back=True, measured_delta=None,
                               error=f"apply failed, restored: {exc}")
        _audit(audit, "apply", spec, cause="applied live", knobs=_knob_values(spec))

        # Step 3: measure. A crash mid-measurement also rolls back.
        # An applicator that leaves the engine it measured last (not
        # necessarily the candidate) has to know the bar the keep below will
        # use, so that putting the candidate back happens here, where a
        # failure rolls back, and not after the keep has been decided.
        if hasattr(applicator, "keep_at"):
            applicator.keep_at = min_keep_delta
        try:
            delta = applicator.measure(spec)
        except RestoreFailed as exc:
            # Validation rebuilds the baseline between rounds; when that rebuild
            # fails there is no baseline left, which is a lost engine, not a
            # measurement that failed and was rolled back.
            _audit(audit, "restore_failed", spec, knobs=_knob_values(spec),
                   cause=f"measure failed: {exc}")
            return ApplyResult(True, rolled_back=False, measured_delta=None,
                               error=f"measure failed: {exc}", restore_failed=True)
        except Exception as exc:
            if (bad := _rollback(False, None, f"measure failed: {exc}")) is not None:
                return bad
            _audit(audit, "revert", spec, cause=f"measure failed, restored: {exc}",
                   knobs=_knob_values(spec))
            return ApplyResult(False, rolled_back=True, measured_delta=None,
                               error=f"measure failed, restored: {exc}")

        # Step 3b: the spec's own correctness gate, before any keep decision. A
        # faster candidate that fails it is restored with the reason, so the
        # throughput number alone can never keep a change that alters output.
        if spec.correctness_gate is not None:
            # The gate is a benchmark against a live server: it can time out, lose
            # the connection, or crash. Any of those is "not judged", and an
            # unjudged change is restored exactly like a failed one — the same
            # shape as the measure step above, so a crash mid-gate can never leave
            # the candidate applied.
            try:
                why = _run_gate(spec.correctness_gate, spec,
                                spec.safety.correctness_gate_timeout_s)
            except Exception as exc:
                what = "timed out" if isinstance(exc, _GateDeadline) else "crashed"
                cause = f"correctness gate {what}: {exc}"
                if (bad := _rollback(True, delta, cause)) is not None:
                    return bad
                _audit(audit, "revert", spec, knobs=_knob_values(spec),
                       cause=f"correctness gate {what}, restored: {exc}")
                return ApplyResult(True, rolled_back=True, measured_delta=delta,
                                   error=f"correctness gate {what}, restored: {exc}")
            if why is not None:
                if (bad := _rollback(True, delta, f"correctness gate failed: {why}")) is not None:
                    return bad
                _audit(audit, "revert", spec, knobs=_knob_values(spec),
                       cause=f"correctness gate failed: {why}")
                return ApplyResult(True, rolled_back=True, measured_delta=delta,
                                   error=f"correctness gate failed: {why}, restored")

        # Step 4: keep-or-rollback on the regression threshold.
        if delta is not None and delta < min_keep_delta:
            cause = f"regression {delta:+.3f} < keep threshold {min_keep_delta:+.3f}"
            if (bad := _rollback(True, delta, cause)) is not None:
                return bad
            _audit(audit, "revert", spec, knobs=_knob_values(spec),
                   cause=f"regression {delta:+.3f} < keep threshold {min_keep_delta:+.3f}")
            return ApplyResult(True, rolled_back=True, measured_delta=delta,
                               error=f"regression {delta:+.3f} < keep threshold "
                                     f"{min_keep_delta:+.3f}, restored")
        return ApplyResult(True, rolled_back=False, measured_delta=delta)

    # Ctrl-C or SystemExit: restore once, then re-raise. A restore that already
    # started is not retried.
    try:
        result = _decide()
    except BaseException as exc:
        cause = f"aborted by {type(exc).__name__}"
        if restore_started:
            _audit(audit, "interrupted", spec, knobs=_knob_values(spec),
                   cause=f"{cause} after a restore began; whether it finished is unknown")
        elif _rollback(True, None, cause) is None:
            _audit(audit, "revert", spec, knobs=_knob_values(spec),
                   cause=f"{cause}; applicator restore() completed")
        raise
    if not result.kept:
        return result
    delta = result.measured_delta

    # Kept: whatever restore() would have needed is now garbage (for a restart,
    # the whole previous engine). Optional hook — only applicators holding
    # releasable state implement it.
    # A release that fails (or warns under -W error) must not undo a keep the
    # gate already made: the candidate is live and measured.
    commit = getattr(applicator, "commit", None)
    if callable(commit):
        try:
            commit()
        except Exception as exc:
            return ApplyResult(True, rolled_back=False, measured_delta=delta,
                               error=f"kept; releasing the replaced engine failed: {exc}")
    return ApplyResult(True, rolled_back=False, measured_delta=delta)


class _GateDeadline(Exception):
    """The correctness gate returned no verdict before its deadline."""


@dataclass(frozen=True)
class _GateRun:
    thread: threading.Thread
    finished: threading.Event


# Every gate thread that has not returned. A thread can't be killed, so a
# timed-out or interrupted gate stays here until it ends. Not Thread.is_alive():
# on CPython 3.10-3.12 an interrupted join() marks a running thread as stopped.
_unfinished_gates: list[_GateRun] = []


def _gates_still_running() -> list[str]:
    _unfinished_gates[:] = [g for g in _unfinished_gates
                           if g.thread.ident is not None and not g.finished.is_set()]
    return [g.thread.name for g in _unfinished_gates]


def _run_gate(
    gate: Callable[[Any], str | None], spec: InterventionSpec, timeout_s: float
) -> str | None:
    """Run ``gate(spec)`` on a thread and return its verdict, or raise after ``timeout_s``."""
    box: dict[str, Any] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            box["why"] = gate(spec)
        except BaseException as exc:
            box["exc"] = exc
        finally:
            finished.set()

    t = threading.Thread(target=run, name=f"correctness-gate:{spec.name}", daemon=True)
    _unfinished_gates.append(_GateRun(t, finished))  # before start(), so Ctrl-C can't skip it
    t.start()
    t.join(min(timeout_s, threading.TIMEOUT_MAX))
    if not finished.is_set():
        raise _GateDeadline(f"no verdict within {timeout_s:g} s, not judged")
    if "exc" in box:
        raise box["exc"]
    return box.get("why")


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


def _warn_release(step: str, exc: Exception) -> None:
    warnings.warn(f"engine release step {step!r} failed: {exc}", RuntimeWarning, stacklevel=3)


def _knob_values(spec: Any) -> dict[str, Any]:
    """The knob=value pairs ``spec`` wants applied — single or joint. Duck-type
    friendly: also works for a bare test double exposing just .knob/.value."""
    knobs = getattr(spec, "knobs", None)
    return dict(knobs) if knobs else {spec.knob: spec.value}


# --- reference applicators ---------------------------------------------------


def _set_knob(config: dict, spec: InterventionSpec) -> None:
    values = _knob_values(spec)
    if not values or any(v is None for v in values.values()):
        raise ValueError(f"intervention {spec.name!r} has no value(s) to set")
    config.update(values)


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
    # measure-time indicator (delta > noise_band); the *authoritative* keep/rollback
    # decision is ApplyResult.rolled_back from apply_intervention, which gates on
    # the caller's min_keep_delta. Report verdicts derive from ApplyResult, not this.
    kept: bool
    via: str = "hot-swap"  # "hot-swap" (scheduling knob) | "restart" (structural knob)
    # Confidence over ``reps`` A/B repetitions (GITM_AB_REPS). At reps=1 std is 0,
    # the noise band is 0, and ``significant`` reduces to delta>0 (pre-reps behaviour).
    baseline_std: float = 0.0
    candidate_std: float = 0.0
    reps: int = 1
    significant: bool = True  # the gain cleared the measurement noise band
    #: ``(baseline_tps, candidate_tps)`` per validation round, in the order run.
    #: ``speedup`` is the mean of the per-round ratios, so drift within a round
    #: cancels; ``baseline_tps`` / ``candidate_tps`` are the rounds' means.
    rounds: tuple[tuple[float, float], ...] = ()
    #: The gain this candidate had to beat, and the A/A noise floor it came from
    #: (``None`` when no A/A sample existed yet and the default was used).
    threshold: float | None = None
    noise_floor: float | None = None

    @property
    def rel_std(self) -> float:
        """Combined relative scatter of baseline+candidate — the noise band."""
        return (self.baseline_std + self.candidate_std) / self.baseline_tps if self.baseline_tps else 0.0

    @property
    def verdict(self) -> str:
        d = self.speedup - 1.0
        conf = "" if self.reps < 2 else (
            f", ±{self.rel_std:.1%} over {self.reps} reps"
            f" ({'significant' if self.significant else 'within noise'})"
        )
        return (
            f"{'kept' if self.kept else 'rolled back'} "
            f"({d:+.1%} decode throughput, via {self.via}{conf})"
        )


@dataclass
class NoiseFloor:
    """How far an identical engine moves when it is rebuilt and measured again.

    Every sample is an A/A delta: the same configuration measured on two
    separate builds, ``second / first - 1``. They come from the opening A/A
    pairs a run measures before its first candidate, and from every later
    re-measurement of the same baseline on a fresh build (after a rollback, or
    between validation rounds), so the estimate improves at no extra cost as a
    run goes on.

    The floor is the root mean square of the samples. An A/A delta has a true
    mean of zero, so this is the standard deviation without spending a degree of
    freedom on estimating a mean that is known, and it is defined from one
    sample. Repeated decodes on one engine (``GITM_AB_REPS``) never enter it:
    they miss the build-to-build part of the noise.
    """

    samples: list[float] = field(default_factory=list)
    #: Baseline measurements left out because they were not the first on their
    #: build (see ``LiveEngineApplicator._note_baseline``).
    skipped_not_fresh: int = 0
    #: The calibration's warm-up pair, left out of ``samples``: the first rebuilt
    #: baseline against the engine running when calibration began (see
    #: ``LiveEngineApplicator.calibrate``). Kept so a cold start stays visible.
    warmup: float | None = None

    @property
    def floor(self) -> float | None:
        if not self.samples:
            return None
        return math.sqrt(sum(d * d for d in self.samples) / len(self.samples))

    def threshold(self, factor: float, *, minimum: float, without_floor: float) -> float:
        """The mean gain a candidate must beat: twice the noise of its measured
        mean, ``factor * floor`` (see :func:`mean_noise_factor`), and never less
        than ``minimum``, the smallest gain worth a change. ``without_floor``
        stands in until the first A/A sample exists."""
        floor = self.floor
        if floor is None:
            return max(without_floor, minimum)
        return max(2.0 * floor * factor, minimum)


def mean_noise_factor(rounds: int, *, rebuilt: bool) -> float:
    """How much of the A/A floor survives averaging ``rounds`` paired rounds.

    The floor is the spread of one A/A pair: two builds. A round's ratio is one
    candidate build against one baseline build, so a single round's spread is
    the floor itself (factor 1).

    Averaging only helps where the rounds measured *different* builds. Serial
    validation alternates and reuses the running engine (A1 B1, B1 A2, A2 B2,
    ...), so neighbouring rounds share a build and are not independent: with
    two rounds the mean is ``B1 - (A1 + A2) / 2``, whose spread is 0.87 of the
    floor, not the 0.71 that ``1 / sqrt(2)`` would claim. This adds up each
    build's weight in the mean for that pattern and returns the exact factor,
    ``sqrt(sum(w^2) / 2)``. Rounds that rebuild nothing (a hot-swapped knob, or
    two engines resident in parallel) re-measure the same two builds, so their
    mean has the same build noise as one round: factor 1.
    """
    if not rebuilt or rounds <= 1:
        return 1.0
    weights: dict[tuple[str, int], float] = {}
    a, b = 0, 0
    for k in range(1, rounds + 1):
        if k > 1 and k % 2 == 0:
            a += 1          # B then A: a fresh baseline build
        elif k > 1:
            b += 1          # A then B: a fresh candidate build
        weights[("B", b)] = weights.get(("B", b), 0.0) + 1.0 / rounds
        weights[("A", a)] = weights.get(("A", a), 0.0) - 1.0 / rounds
    return math.sqrt(sum(w * w for w in weights.values()) / 2.0)


class StructuralKnobRequiresRestart(RuntimeError):
    """A structural knob was applied with no restart hook to enact it.

    Raised by :meth:`LiveEngineApplicator.apply` so ``apply_intervention`` rolls
    the candidate back with a clear reason — never silently sets a structural
    field the running engine won't honor.
    """


#: What vLLM gives an engine when nobody says otherwise. A restart candidate
#: inherits the baseline's kwargs, so both engines ask for the same fraction.
_DEFAULT_GPU_FRACTION = 0.9


def gpu_fraction(engine: Any) -> float | None:
    """The device fraction this engine was built to hold, or ``None`` if unknown.

    Read off the kwargs it was built with rather than measured off the device.
    Measuring would mean initialising CUDA in this process to ask, which on the
    offline engine is a context the parent does not otherwise carry — and the
    number we need is the one the *next* engine will ask for, which is this one
    by construction: a restart candidate inherits the baseline's kwargs.

    Two absences that are not the same. An engine with ``gitm_llm_kwargs`` and no
    ``gpu_memory_utilization`` in them is one gitm built without an explicit cap,
    so it holds vLLM's own default — which is the case the MI355X run was in.
    An engine with no ``gitm_llm_kwargs`` at all is somebody else's handle, and
    what it holds is genuinely unknown; guessing the default there would refuse
    a restart on a number nobody supplied.
    """
    kwargs = getattr(engine, "gitm_llm_kwargs", None)
    if kwargs is None:
        return None
    try:
        return float(kwargs.get("gpu_memory_utilization", _DEFAULT_GPU_FRACTION))
    except (TypeError, ValueError):
        return _DEFAULT_GPU_FRACTION


def parallel_restart_fits(
    engine: Any, values: dict[str, Any] | None = None
) -> tuple[bool, str]:
    """Whether a candidate engine can be built while the baseline is still up.

    Parallel mode holds both at once, so what matters is the *sum* of the two
    fractions, not twice the baseline's. Those are usually the same number,
    because a restart candidate inherits the baseline's kwargs — but not always:
    ``gpu_memory_utilization_dynamic`` is a catalogue lever whose whole purpose
    is to change this fraction, so a candidate carrying it asks for something
    else. Doubling the baseline gets that case wrong in both directions: it
    passes 0.45 beside a 0.9 candidate that needs 135%, and refuses 0.6 beside a
    0.4 candidate that fits exactly.

    The check stays arithmetic rather than a free-memory reading. Both numbers
    are knowable from kwargs, so no device query and no CUDA context in a
    process that does not otherwise carry one.

    This is the failure that cost the MI355X run 27 of its 29 candidates. The
    constraint was documented in ``workloads.py`` and enforced nowhere, so every
    structural candidate built into a device the baseline had 90% of, and died.
    """
    baseline = gpu_fraction(engine)
    if baseline is None:
        # Not a handle gitm built, so nothing says what it holds. Refusing here
        # would block a deployment that supplies its own restart_fn on a number
        # it never gave us.
        return True, "this engine does not say what fraction of the device it holds"

    candidate = baseline
    if values and "gpu_memory_utilization" in values:
        try:
            candidate = float(values["gpu_memory_utilization"])
        except (TypeError, ValueError):
            candidate = baseline

    total = baseline + candidate
    if total <= 1.0:
        return True, (
            f"baseline {baseline:.2f} + candidate {candidate:.2f} "
            f"= {total:.2f} of the device")
    same = "" if candidate == baseline else f" and the candidate asks for {candidate:.0%}"
    return False, (
        f"the baseline holds {baseline:.0%} of each device{same}, so building "
        f"them side by side needs {total:.0%}. Use restart_mode='serial' to "
        f"release the baseline first, or build with "
        f"GITM_VLLM_GPU_MEM<={0.5:.2f} to leave room for both"
    )


def _largest_fitting_candidate(engine: Any) -> float:
    """The biggest fraction a candidate can take and still fit beside this baseline.

    Asked of :func:`parallel_restart_fits` rather than computed alongside it,
    because the operator acts on this number and the check is what will judge
    them. Deriving it separately put the two at odds in both directions:
    rounding ``1 - 0.585`` to ``0.42`` offered a candidate the check then
    refused at 1.005, and flooring it understated the room at 13 of the 49
    two-decimal baselines — including 0.9, which is vLLM's own default, where
    0.10 fits and the warning said 0.09.

    Floor first, then ask whether one more hundredth is accepted. Two decimals
    because that is the precision the warning prints at, and a limit the
    operator cannot type is not a limit.
    """
    room = max(1.0 - (gpu_fraction(engine) or 0.0), 0.0)
    limit = math.floor(room * 100) / 100
    nxt = round(limit + 0.01, 2)
    if parallel_restart_fits(engine, {"gpu_memory_utilization": nxt})[0]:
        return nxt
    return limit


_RESTART_MODES = ("parallel", "serial", "sleep_swap")
#: Modes that release the baseline's device memory before building a candidate,
#: and so need ``baseline_restart_fn`` to get it back.
_SERIAL_MODES = ("serial", "sleep_swap")


def resolve_restart_mode(
    engine: Any, requested: str | None, baseline_restart_fn: Any = None
) -> tuple[str, str]:
    """``(mode, why)`` for this engine. ``requested`` wins when it is given.

    Serial is the better default wherever it is available: it releases the
    baseline before building the candidate, so the candidate gets the whole
    device instead of whatever the baseline left. Parallel's only advantage is
    not paying for a baseline rebuild, and it buys that by requiring both
    engines resident — which at any realistic ``gpu_memory_utilization`` is
    impossible. The default used to be parallel, and 93% of one run's candidates
    died of it.

    Parallel remains the fallback for a deployment that supplies no
    ``baseline_restart_fn``, because there serial has nothing to restore with.

    That callback is passed in rather than read off the engine, so the mode is
    decided by the same one the apply path will use. Reading the engine
    attribute here while the applicator held a constructor argument let the two
    disagree, and both disagreements were bad: a caller supplying only the
    argument got parallel despite having a rebuild available, and one supplying
    only the attribute got serial and then failed every structural apply for
    want of the argument.

    ``"sleep_swap"`` is serial that keeps the baseline asleep instead of
    destroying it, and is only ever chosen explicitly. It falls back to serial
    per candidate, so it needs the same baseline rebuild serial does.
    """
    if requested:
        if requested not in _RESTART_MODES:
            raise ValueError(
                "restart_mode must be 'parallel', 'serial' or 'sleep_swap', "
                f"got {requested!r}")
        return requested, "set explicitly"
    if (baseline_restart_fn or getattr(engine, "gitm_baseline_restart_fn", None)) is None:
        return "parallel", (
            "no baseline_restart_fn, so serial has nothing to rebuild the "
            "baseline with")
    return "serial", "the default: parallel needs both engines resident at once"


class LiveEngineApplicator:
    """Apply a knob (or a joint set — see ``InterventionSpec.knobs``) to a live
    (vLLM) engine, gated by a real decode-throughput A/B.

    Routes by knob taxonomy (:mod:`gitm.optimizer.vllm_knobs`):

    * **scheduling** knobs are rare deployment-specific live controls. A joint
      candidate whose knobs are ALL scheduling is hot-swapped as one set.
      The curated vLLM EngineArg map intentionally treats scheduler-looking
      EngineArgs as structural because vLLM reads them at construction time.
    * **structural** knobs (parallelism, dtype, quantization, block size, …) can
      only take effect on a fresh engine, so they (or a joint set containing
      any structural knob) are routed through
      ``restart_fn(engine, {knob: value, ...}) -> new_engine``: the candidate
      engine replaces the live one for the A/B, and restore swaps the original
      back (shutting the candidate down best-effort). With no ``restart_fn`` a
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
        restart_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
        baseline_restart_fn: Callable[[Any], Any] | None = None,
        restart_mode: str | None = None,
        getter: Callable[[Any, str], Any] | None = None,
        setter: Callable[[Any, str, Any], None] | None = None,
        reps: int = 1,
        force_restart: bool = False,
        noise_gate: bool = False,
        validation: int = 1,
        min_effect: float = 0.0,
        threshold_without_floor: float = 0.0,
        noise: NoiseFloor | None = None,
    ) -> None:
        self.engine = engine
        # True from the moment an engine is built until it is first measured.
        # The first measurement on a build is the only one that counts for an
        # A/A sample; see _note_baseline. The engine handed in is unmeasured.
        self._unmeasured_build = True
        self._last_measure_fresh = False
        self._tps = throughput_fn
        # The engine's own hook is the fallback, so a caller handing over an
        # engine gitm built need not re-pass what is already on it. One effective
        # callback, and the mode is decided from that same one rather than from a
        # second source that can disagree with it.
        self._baseline_restart_fn = baseline_restart_fn or getattr(
            engine, "gitm_baseline_restart_fn", None)
        # None means "work it out from this engine", so the choice is made by one
        # rule wherever an applicator is built. Defaulting the parameter to
        # "parallel" put the trap one level below the loop: a direct caller got
        # the mode that cannot build a candidate at any realistic memory cap.
        restart_mode, _ = resolve_restart_mode(
            engine, restart_mode, self._baseline_restart_fn)
        self.restart_mode_warning: str | None = None
        if restart_mode == "parallel" and restart_fn is not None:
            # Against a candidate that inherits the baseline's fraction, since
            # no real candidate is in hand yet. That covers most of the
            # catalogue but not all of it, so this says what it actually
            # checked: a lever that *lowers* gpu_memory_utilization can still
            # fit, and is re-checked with its own value at the rebuild. Claiming
            # structural candidates are impossible would have an operator
            # dismiss a measurement that would have worked.
            fits, _ = parallel_restart_fits(engine)
            if not fits:
                room = _largest_fitting_candidate(engine)
                # Recorded, not raised: a run whose candidates are all
                # hot-swappable never reaches a rebuild and should not be
                # stopped here. The caller surfaces this so the operator learns
                # it when the run starts rather than per dead candidate.
                self.restart_mode_warning = (
                    f"the baseline holds {gpu_fraction(engine):.0%} of each "
                    f"device, so in restart_mode='parallel' only a candidate "
                    f"that lowers gpu_memory_utilization to {room:.2f} or less "
                    f"can be built beside it. Every other structural candidate "
                    f"will be refused. Use restart_mode='serial' to release the "
                    f"baseline first, or build with GITM_VLLM_GPU_MEM<=0.50")
        self._restart_fn = restart_fn
        self._restart_mode = restart_mode
        self._getter = getter or get_knob
        self._setter = setter or set_knob
        self._reps = max(1, reps)
        # force_restart is kept for custom deployments that still classify a
        # knob as scheduling but want to measure it through the restart path.
        self._force_restart = force_restart
        self._baseline_tps: float | None = None
        self._baseline_std: float = 0.0
        # How hard a candidate is tested before it is kept. The defaults are the
        # old rule exactly (one A/B, keep at >= 0), so callers that do not ask
        # for validation see no change; the loop asks for it.
        #: Paired A/B rounds a candidate is measured over (serial restart only).
        self.validation = max(1, int(validation))
        #: The smallest mean gain worth keeping a change for.
        self.min_effect = float(min_effect)
        #: The threshold used until the first A/A sample exists.
        self.threshold_without_floor = float(threshold_without_floor)
        self.noise = noise if noise is not None else NoiseFloor()
        #: Judge candidates against the noise floor at all. Off by default: a
        #: caller that does not ask for it keeps the old rule (keep at >= 0),
        #: even though A/A samples are still collected.
        self.noise_gate = bool(noise_gate)
        # The last baseline measurement: (configuration generation, the engine it
        # was measured on, tok/s). A later measurement of the same generation on
        # a different engine object is a rebuild of the same configuration, so
        # their difference is an A/A sample. A kept candidate starts a new
        # generation: from then on the baseline is a different configuration.
        self._generation = 0
        self._last_baseline: tuple[int, int, float] | None = None
        # Restore record: ("hotswap", knob, old_value) | ("restart", old_engine) |
        # ("serial_restart", restore_baseline_fn) |
        # ("sleep_swap", baseline, candidate_or_None) | None.
        self._prev: tuple[Any, ...] | None = None
        self.last_result: EngineABResult | None = None
        #: The ``min_keep_delta`` apply_intervention will keep at. Set by it
        #: before each measure(), so the engine left live agrees with the keep.
        self.keep_at = 0.0
        # sleep_swap: engines put to sleep and not yet woken or shut down, by id.
        # Each holds weights in CPU RAM and some device memory, so close()
        # releases whatever is left here.
        self._sleeping: dict[int, Any] = {}
        self._swap_rec: dict[str, Any] = {"sleep_s": [], "wake_s": []}
        #: sleep_swap: one record per restart candidate — how it was built (or
        #: why it fell back to serial) and each sleep and wake-up it cost, in
        #: seconds. None of that time is inside a throughput measurement.
        self.swap_log: list[dict[str, Any]] = []

    def _bench(self) -> float:
        return sum(self._tps(self.engine) for _ in range(self._reps)) / self._reps

    def _bench_stats(self) -> tuple[float, float]:
        """(mean, sample stdev) of decode throughput over ``reps`` runs.

        stdev is 0.0 for a single rep → the noise band is 0 and keep falls back to
        ``delta > 0``, i.e. reps=1 behaves exactly as before reps were added.
        """
        self._last_measure_fresh = self._unmeasured_build
        self._unmeasured_build = False
        samples = [self._tps(self.engine) for _ in range(self._reps)]
        mean = sum(samples) / len(samples)
        if len(samples) < 2:
            return mean, 0.0
        var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
        return mean, var**0.5

    def snapshot(self) -> dict[str, Any]:
        # Reset both the restore record AND last_result: snapshot() runs at the
        # start of every apply_intervention, so a candidate whose apply() fails
        # (e.g. a structural knob with no restart hook, where measure() never
        # runs) must not leave the *previous* candidate's A/B result visible.
        self._prev = None
        self.last_result = None
        self._state_is_baseline = False
        self._baseline_tps, self._baseline_std = self._bench_stats()
        self._note_baseline(self._baseline_tps)
        return {"baseline_tps": self._baseline_tps}

    def _note_baseline(self, tps: float) -> None:
        """Record a baseline measurement, and the A/A sample it makes with the
        previous one: the same configuration, each measured first on its build.

        Only a build's first measurement counts. A second measurement of the
        same engine is not comparable with a fresh one: on a laptop on battery
        the first decode after a rebuild (after the GPU had cooled through it)
        ran 4-5% faster than the next, and pairing the two made that heat
        effect read as build-to-build noise, multiplying the floor by seven.
        Two first measurements on separate builds of the same configuration
        are exactly an A/A pair, which is what the opening calibration takes.
        """
        if not self._last_measure_fresh:
            self.noise.skipped_not_fresh += 1
            return
        last = self._last_baseline
        if last is not None and last[0] == self._generation and last[2] > 0 and tps > 0:
            self.noise.samples.append(tps / last[2] - 1.0)
        self._last_baseline = (self._generation, id(self.engine), tps)

    def calibrate(self, pairs: int, *, deadline_ns: int | None = None) -> list[float]:
        """Measure the baseline against itself, rebuilt each time, for ``pairs``
        A/A samples, so the first candidate is judged against a measured floor.

        The engine running when calibration begins is measured as a warm-up and
        left out of the floor, so ``pairs`` samples cost ``pairs + 1`` rebuilds.
        It is the first engine after the opening capture, and on Kimi K2.5
        (MI355X, TP=8) it decoded at 1110 tok/s where every later build of the
        same configuration ran at 1380-1470: one 24.8% sample, and an RMS floor
        of 8.9% that rejected a +5% CUDA-graph win against a 25% bar. Its pair
        is recorded as ``noise.warmup``.

        Serial restart only: it needs ``baseline_restart_fn``. Stops early at
        ``deadline_ns``. Returns the A/A samples it added. A rebuild or a
        baseline measurement that fails raises :class:`RestoreFailed`, the same
        as a failed rollback: the run has lost its engine.
        """
        # sleep_swap calibrates exactly as serial does: the floor is
        # build-to-build noise, so it needs fresh builds, not woken ones.
        if (pairs < 1 or self._baseline_restart_fn is None
                or self._restart_mode not in _SERIAL_MODES):
            return []
        before = len(self.noise.samples)
        warm = self._bench_baseline()
        self._last_baseline = None          # nothing pairs with the warm-up
        for i in range(pairs + 1):
            if deadline_ns is not None and time.time_ns() >= deadline_ns:
                break
            old = self.engine
            self._shutdown(old)
            self._rebuild_baseline(lambda old=old: self._baseline_restart_fn(old),
                                   "calibrating the noise floor")
            tps = self._bench_baseline()
            if i == 0 and warm > 0 and tps > 0:
                self.noise.warmup = tps / warm - 1.0
            self._note_baseline(tps)
        return self.noise.samples[before:]

    def _bench_baseline(self) -> float:
        """The running baseline's throughput, where losing it loses the run.

        Used where nothing else can stand in for the baseline (calibration, a
        validation round after the baseline was rebuilt). A measurement that
        fails there, a hang the watchdog killed included, leaves no baseline to
        compare against, so it is :class:`RestoreFailed` rather than a candidate
        failure that rolls back to an engine that is not there.
        """
        try:
            return self._bench_stats()[0]
        except RestoreFailed:
            raise
        except Exception as exc:
            # A probe that raised has not necessarily stopped the engine. Release
            # it before letting go of the handle, or its workers keep their GPU
            # memory after the run stops and the next run cannot build.
            # _shutdown never raises.
            self._shutdown(self.engine)
            self.engine = None
            self._prev = None
            raise RestoreFailed(f"could not measure the baseline: {exc}") from exc

    def apply(self, spec: InterventionSpec) -> None:
        values = _knob_values(spec)  # single knob, or a joint candidate's set
        if not values or any(v is None for v in values.values()):
            raise ValueError(f"intervention {spec.name!r} has no value(s) to set")

        if not self._force_restart and all(knob_kind(k) == "scheduling" for k in values):
            # Record each restore point only after its successful set, so a
            # later setter raising only leaves what was actually changed to undo.
            applied: dict[str, Any] = {}
            self._prev = ("hotswap", applied)
            for knob, value in values.items():
                try:
                    prev = self._getter(self.engine, knob)
                except AttributeError:
                    prev = None
                self._setter(self.engine, knob, value)
                applied[knob] = prev
            return

        # >=1 structural knob — the set can only take effect together via a
        # rebuild, so the whole dict goes through restart_fn in one call.
        if self._restart_fn is None:
            raise StructuralKnobRequiresRestart(
                f"knob(s) {', '.join(values)} are structural (need an engine "
                "restart); no restart_fn supplied, so they cannot be applied live"
            )
        old_engine = self.engine
        if self._restart_mode in _SERIAL_MODES:
            if self._baseline_restart_fn is None:
                raise StructuralKnobRequiresRestart(
                    f"{self._restart_mode} restart mode requires baseline_restart_fn "
                    "to rebuild the baseline"
                )
            if self._restart_mode == "sleep_swap":
                new_engine = self._build_beside_sleeping(spec, old_engine, values)
                if new_engine is not None:
                    self.engine = new_engine
                    self._unmeasured_build = True
                    self._activate(new_engine)
                    return
                # Fell back: the baseline has been released, and this candidate
                # is measured as a plain serial restart.

            def restore_baseline() -> Any:
                return self._baseline_restart_fn(old_engine)

            if self._restart_mode == "serial":
                self._shutdown(old_engine)
            self._prev = ("serial_restart", restore_baseline)
            try:
                new_engine = self._restart_fn(old_engine, values)
            except Exception as exc:
                self._rebuild_baseline(restore_baseline, f"candidate build failed: {exc}")
                raise
        else:
            # Refuse a build that cannot fit rather than let it OOM. The failure
            # is identical either way for this candidate, but an OOM traceback
            # from inside vLLM says nothing about which mode caused it or what
            # to do, and it repeats once per candidate for the whole run.
            fits, why = parallel_restart_fits(old_engine, values)
            if not fits:
                raise StructuralKnobRequiresRestart(
                    f"knob(s) {', '.join(values)} need an engine rebuild, and "
                    f"restart_mode='parallel' cannot provide one: {why}")
            new_engine = self._restart_fn(old_engine, values)
        if new_engine is None:
            if self._restart_mode in _SERIAL_MODES:
                _, restore_baseline = self._prev
                self._rebuild_baseline(restore_baseline, "restart_fn produced no engine")
            raise StructuralKnobRequiresRestart(
                f"restart_fn produced no engine for knob(s) {', '.join(values)}"
            )
        self.engine = new_engine
        self._unmeasured_build = True
        self._activate(new_engine)
        if self._restart_mode not in _SERIAL_MODES:
            self._prev = ("restart", old_engine)

    def _build_beside_sleeping(self, spec: InterventionSpec, base: Any,
                               values: dict[str, Any]) -> Any:
        """sleep_swap: put the baseline to sleep and build the candidate beside it.

        Returns the candidate, with ``_prev`` recording both engines, or ``None``
        after releasing the baseline when the candidate cannot be built that
        way; the caller then measures it as a serial restart, and the record
        says why. The candidate's ``gpu_memory_utilization`` is never lowered to
        make it fit: the A/B would then compare two KV budgets.
        """
        rec: dict[str, Any] = {"candidate": spec.name, "via": "sleep_swap",
                               "fallback": None, "sleep_s": [], "wake_s": []}
        self.swap_log.append(rec)
        self._swap_rec = rec
        kwargs = getattr(base, "gitm_llm_kwargs", None)
        why: str | None = None
        if not callable(getattr(base, "sleep", None)) or not callable(
                getattr(base, "wake_up", None)):
            why = "the baseline engine has no sleep()/wake_up()"
        elif kwargs is not None and not kwargs.get("enable_sleep_mode"):
            why = "the baseline was not built with enable_sleep_mode=True"
        if why is None:
            # Recorded before the build, so a Ctrl-C during it still leaves
            # restore() able to wake the baseline.
            self._prev = ("sleep_swap", base, None)
            try:
                self._sleep(base)
            except Exception as exc:
                why = f"the baseline could not be put to sleep: {exc}"
        if why is None:
            why = self._candidate_cannot_fit(base, values)
        if why is None:
            try:
                cand = self._restart_fn(base, values)
                if cand is None:
                    raise StructuralKnobRequiresRestart("restart_fn produced no engine")
            except Exception as exc:
                why = f"the candidate failed to build beside the sleeping baseline: {exc}"
            else:
                self._prev = ("sleep_swap", base, cand)
                return cand
        rec["via"] = "serial"
        rec["fallback"] = why
        self._prev = None
        self._release(base)
        return None

    def _candidate_cannot_fit(self, base: Any, values: dict[str, Any]) -> str | None:
        """Why vLLM would refuse the candidate with the baseline asleep, or None.

        vLLM starts an engine only if free device memory is at least
        ``gpu_memory_utilization`` of the total. Asked of the engine's own
        ``gitm_free_gpu_fraction_fn`` (it reads the device without a CUDA
        context in this process); an engine without one, or a reading that is
        unavailable, is left to the build to decide.
        """
        free_fn = getattr(base, "gitm_free_gpu_fraction_fn", None)
        need = gpu_fraction(base)
        if "gpu_memory_utilization" in values:
            try:
                need = float(values["gpu_memory_utilization"])
            except (TypeError, ValueError):
                pass
        if not callable(free_fn) or need is None:
            return None
        try:
            # The candidate's own settings: a lever that changes
            # tensor_parallel_size runs on a different set of GPUs.
            free = free_fn(base, values)
        except Exception:
            return None
        if free is None or free >= need:
            return None
        return (f"with the baseline asleep {free:.0%} of the device is free and "
                f"the candidate asks for gpu_memory_utilization={need:.2f}")

    def _sleep(self, engine: Any) -> None:
        self._sleeping[id(engine)] = engine     # before: a failed sleep is still released
        t0 = time.perf_counter()
        engine.sleep(level=1)
        self._swap_rec["sleep_s"].append(round(time.perf_counter() - t0, 3))

    def _wake(self, engine: Any) -> None:
        t0 = time.perf_counter()
        engine.wake_up()
        self._sleeping.pop(id(engine), None)
        self._swap_rec["wake_s"].append(round(time.perf_counter() - t0, 3))

    def _release(self, engine: Any) -> None:
        """Shut an engine down whether it is awake or asleep. Never raises."""
        self._sleeping.pop(id(engine), None)
        self._shutdown(engine)

    def _swap_live(self, target: Any) -> None:
        """sleep_swap: sleep the running engine and wake ``target`` in its place.

        Not a build, so ``target``'s next measurement is not a fresh one. A
        failure raises with ``_prev`` intact, and restore() sorts out which of
        the two is still usable.
        """
        self._sleep(self.engine)
        self._wake(target)
        self.engine = target
        self._unmeasured_build = False
        self._activate(target)

    def close(self) -> None:
        """Shut down every engine still asleep.

        Every path through a sleep_swap candidate already leaves one engine; this
        is the backstop for one that was interrupted part-way (a restore that
        began and did not finish, or a swap stopped between its sleep and its
        wake-up, which leaves the running engine itself asleep). An engine
        still asleep at this point is never going to be woken, so it is
        released even when it is ``self.engine``, and the handle is dropped
        rather than handed back as live. A no-op in every other mode.
        """
        for engine in list(self._sleeping.values()):
            if engine is self.engine:
                self.engine = None
            self._release(engine)
        self._sleeping.clear()

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self._prev is None:
            return
        tag = self._prev[0]
        if tag == "hotswap":
            _, applied = self._prev
            for knob, old in applied.items():
                self._setter(self.engine, knob, old)
        elif tag == "restart":
            _, old_engine = self._prev
            self._shutdown(self.engine)  # drop the candidate engine we built
            self.engine = old_engine
            # Not a build: the baseline that was already measured comes back.
            self._unmeasured_build = False
            self._activate(old_engine)
        elif tag == "serial_restart":
            _, restore_baseline = self._prev
            self._shutdown(self.engine)  # drop the candidate engine we built
            self._rebuild_baseline(restore_baseline, "rolling back the candidate")
            return
        elif tag == "sleep_swap":
            self._wake_baseline("rolling back the candidate")
            return
        elif tag == "baseline_running":
            # Validation ended on a baseline round: the baseline is already the
            # running engine, so there is nothing to put back.
            pass
        # Consume the restore record so a second restore() can't re-undo (or
        # re-shutdown the already-discarded candidate engine) a second time.
        self._prev = None

    def _wake_baseline(self, cause: str) -> None:
        """sleep_swap rollback: drop the candidate, asleep or not, and make the
        baseline live again, waking it if it sleeps. No rebuild.

        The record is consumed first, as for a serial rebuild. A baseline that
        will not wake is shut down rather than handed back: it is a lost
        engine, :class:`RestoreFailed`, exactly as a rebuild that failed.
        """
        _, base, cand = self._prev
        self._prev = None
        if cand is not None:
            self._release(cand)
        if id(base) in self._sleeping:
            try:
                self._wake(base)
            except Exception as exc:
                self._release(base)
                self.engine = None
                raise RestoreFailed(
                    f"could not wake the baseline engine ({cause}): {exc}") from exc
        self.engine = base
        # Not a build: the baseline that was already measured comes back.
        self._unmeasured_build = False
        self._activate(base)

    def _rebuild_baseline(self, restore_baseline: Callable[[], Any], cause: str) -> None:
        """Build the baseline engine again after a serial restart, and make it live.

        The record is consumed whether or not the rebuild works. A rebuild that
        fails is almost always one that could not get the device memory back, and
        trying it a second time from ``restore()`` fails the same way — that
        second attempt, with nothing around it to catch it, is what used to end
        the run without a report.

        Activated as well as assigned. The workload runner drives whichever
        engine was last activated, so a rebuilt baseline that is only assigned
        here leaves the runner on the engine that was just shut down.
        """
        self._prev = None
        try:
            self.engine = restore_baseline()
            self._unmeasured_build = True
        except Exception as exc:
            # The baseline was already shut down: hold nothing rather than a
            # dead handle the loop would pass back to the caller as live.
            self.engine = None
            raise RestoreFailed(
                f"could not rebuild the baseline engine ({cause}): {exc}") from exc
        self._activate(self.engine)

    def install(self, engine: Any) -> None:
        """Make ``engine``, built outside the applicator, the running one.

        The same configuration (a re-trace swaps in a traced build and then an
        untraced one), so its measurements still pair with the baseline's; and
        a new build, so its next measurement is a fresh one. Assigning
        ``engine`` directly kept the previous engine's freshness, and the first
        measurement after every re-trace was discarded as a warm re-measurement.
        """
        self.engine = engine
        self._unmeasured_build = True

    def commit(self) -> None:
        """The candidate was kept: release what restore() would have swapped back.

        In parallel mode that is the previous engine, still holding its GPUs.
        Without this, the next snapshot() drops the only handle to it and it stays
        resident for the rest of the process — the next candidate is then built
        beside two engines, not one.
        """
        if self._prev is not None and self._prev[0] == "restart":
            self._shutdown(self._prev[1])
        elif self._prev is not None and self._prev[0] == "sleep_swap":
            # The old baseline sleeps with its weights in CPU RAM; the kept
            # candidate is the baseline now, so nothing will wake it again.
            # measure() left the candidate live for this keep; release
            # whichever engine is not running regardless, so a keep that
            # disagreed with it can never shut down the one engine left.
            _, base, cand = self._prev
            self._release(cand if self.engine is base else base)
        self._prev = None
        # The candidate is the baseline now: a different configuration, so its
        # measurements are not A/A samples against the old one's.
        self._generation += 1
        self._last_baseline = None

    @staticmethod
    def _activate(engine: Any) -> None:
        fn = getattr(engine, "gitm_activate_fn", None)
        if callable(fn):
            try:
                fn(engine)
            except Exception:
                pass

    @staticmethod
    def _shutdown(engine: Any) -> None:
        """Best-effort release of an engine before/after a restart A/B.

        Never raises (a failed release must not abort the A/B), but every failed
        step is warned: a swallowed failure looks exactly like a clean release,
        and the next engine build is what pays for it.
        """
        custom = getattr(engine, "gitm_shutdown_fn", None)
        released = False
        if callable(custom):
            try:
                custom(engine)
                released = True
            except Exception as exc:
                _warn_release("gitm_shutdown_fn", exc)

        # Generic shutdown only as a fallback: after a working gitm_shutdown_fn
        # it would tear down a released engine twice, and warn when that fails.
        for path in () if released else ("shutdown", "llm_engine.shutdown", "engine.shutdown"):
            obj: Any = engine
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if callable(obj):
                try:
                    obj()
                except Exception as exc:
                    _warn_release(path, exc)
                break
        gc.collect()
        try:
            import torch
        except ImportError:
            return  # CPU-only box: nothing to empty
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            _warn_release("cuda cache release", exc)

    def measure(self, spec: InterventionSpec) -> float | None:
        baseline = self._baseline_tps if self._baseline_tps is not None else self._bench_stats()[0]
        # A non-positive baseline means the A/B has no valid reference — an idle
        # engine, a probe that returned 0, no tokens produced. Raising (vs forcing
        # speedup=1.0) makes apply_intervention roll the candidate back instead of
        # silently *keeping* an unmeasurable change as a non-regression.
        if baseline <= 0:
            raise ValueError(
                f"baseline decode throughput is {baseline}; cannot run a valid A/B "
                f"for knob {spec.knob!r}"
            )
        candidate, cand_std = self._bench_stats()
        tag = self._prev[0] if self._prev else None
        via = "restart" if tag in {"restart", "serial_restart", "sleep_swap"} else "hot-swap"
        rounds = [(baseline, candidate)]
        # sleep_swap rounds wake the same two builds rather than rebuilding
        # them, so like a hot-swap they average out no build noise.
        rebuilt = tag == "serial_restart"
        if self.validation > 1 and not self._clear_loss(rounds):
            if rebuilt:
                rounds = self._validation_rounds(spec, rounds)
            elif tag == "sleep_swap":
                rounds = self._rounds_sleep_swap(rounds)
            elif tag in {"hotswap", "restart"}:
                rounds = self._rounds_in_place(spec, rounds)
        ratios = [b / a for a, b in rounds]
        speedup = sum(ratios) / len(ratios)
        delta = speedup - 1.0
        # Noise band = combined relative scatter of baseline+candidate over reps on
        # one engine (GITM_AB_REPS). reps=1 → std=0 → band=0.
        noise_band = (self._baseline_std + cand_std) / baseline
        # The keep bar: twice the A/A noise of this mean (allowing for rounds
        # that share a build), the smallest gain worth keeping, and the reps
        # band, whichever is largest. Without the noise gate it is the reps band
        # alone: the old rule.
        threshold = (self.noise.threshold(
            mean_noise_factor(len(rounds), rebuilt=rebuilt),
            minimum=self.min_effect, without_floor=self.threshold_without_floor)
            if self.noise_gate else 0.0)
        bar = max(threshold, noise_band)
        # One comparison decides everything: apply_intervention keeps at
        # ``delta - bar >= keep_at``, and the candidate rebuild or wake-up below
        # must agree with it. Significance stays measured against zero.
        keep = delta - bar >= self.keep_at
        significant = delta - bar >= 0 if self.noise_gate else delta > noise_band
        if tag == "hotswap" and keep and self._state_is_baseline:
            self._set_knobs(_knob_values(spec))
        if keep and self._prev and self._prev[0] == "baseline_running":
            # Kept, but the last round left the baseline running: build the
            # candidate again so the keep leaves it live, and so a later
            # rollback (a failed correctness gate) rebuilds the baseline.
            self._swap_to_candidate(spec)
            self._prev = ("serial_restart", self._round_restore)
        if keep and tag == "sleep_swap" and self.engine is not self._prev[2]:
            # Kept with the baseline awake from the last round: wake the
            # candidate again so the keep leaves it live.
            self._swap_live(self._prev[2])
        self.last_result = EngineABResult(
            knob=spec.knob, value=(getattr(spec, "knobs", None) or spec.value),
            baseline_tps=sum(a for a, _ in rounds) / len(rounds),
            candidate_tps=sum(b for _, b in rounds) / len(rounds),
            speedup=speedup, kept=significant, via=via,
            baseline_std=self._baseline_std, candidate_std=cand_std,
            reps=self._reps, significant=significant, rounds=tuple(rounds),
            threshold=bar, noise_floor=self.noise.floor,
        )
        return delta - bar

    def _validation_rounds(self, spec: InterventionSpec,
                           rounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Measure up to ``validation`` paired rounds, alternating the order.

        Round 1 (A then B) has been measured. Round 2 measures the running
        candidate again, then rebuilds the baseline and measures it (B then A);
        round 3 measures that baseline again, then rebuilds the candidate (A then
        B); and so on. Alternating cancels drift within a round, and each round
        after the first costs one build rather than two.

        A rebuild that fails rolls the candidate back (the baseline is put back
        first); a baseline that cannot be put back, or measured, is
        :class:`RestoreFailed`.
        """
        _, restore_baseline = self._prev
        self._round_restore = restore_baseline
        for k in range(2, self.validation + 1):
            if k % 2 == 0:
                # B then A: the candidate is running.
                b = self._bench_stats()[0]
                self._shutdown(self.engine)
                self._rebuild_baseline(restore_baseline, f"validation round {k}")
                self._prev = ("baseline_running",)
                a = self._bench_baseline()
                self._note_baseline(a)
            else:
                # A then B: the baseline is running.
                a = self._bench_baseline()
                self._note_baseline(a)
                self._swap_to_candidate(spec)
                self._prev = ("serial_restart", restore_baseline)
                b = self._bench_stats()[0]
            rounds.append((a, b))
        return rounds

    def _rounds_sleep_swap(self, rounds: list[tuple[float, float]]
                           ) -> list[tuple[float, float]]:
        """Validation rounds for sleep_swap, alternating order as serial does,
        but moving between the two engines by sleep and wake-up instead of a
        rebuild. Each round re-measures the same two builds.

        A baseline that cannot be measured once woken is a lost engine, as in
        serial; the candidate asleep beside it is released too.
        """
        _, base, cand = self._prev
        for k in range(2, self.validation + 1):
            if k % 2 == 0:
                # B then A: the candidate is running.
                b = self._bench_stats()[0]
                self._swap_live(base)
                a = self._bench_woken_baseline(cand)
            else:
                # A then B: the baseline is running.
                a = self._bench_woken_baseline(cand)
                self._swap_live(cand)
                b = self._bench_stats()[0]
            rounds.append((a, b))
        return rounds

    def _bench_woken_baseline(self, cand: Any) -> float:
        try:
            a = self._bench_baseline()
        except RestoreFailed:
            self._release(cand)
            raise
        self._note_baseline(a)      # not a fresh build, so never an A/A sample
        return a

    def _clear_loss(self, rounds: list[tuple[float, float]]) -> bool:
        """Round 1 already lost by more than one round's noise: not worth
        confirming. Without the noise gate, any loss."""
        a1, b1 = rounds[0]
        if not self.noise_gate:
            return b1 < a1
        bar = self.noise.threshold(1.0, minimum=self.min_effect,
                                   without_floor=self.threshold_without_floor)
        return b1 / a1 - 1.0 <= -bar

    _state_is_baseline = False

    def _set_knobs(self, values: dict[str, Any]) -> None:
        for knob, value in values.items():
            self._setter(self.engine, knob, value)
        self._state_is_baseline = False

    def _rounds_in_place(self, spec: InterventionSpec,
                         rounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Validation rounds where nothing is rebuilt, alternating order.

        A hot-swapped candidate toggles its knobs back and forth on the one
        engine; a parallel-restart candidate has both engines resident and
        measures each in turn. Neither rebuilds, so the rounds re-measure the
        same two builds: they average out decode noise and drift, not build
        noise, and :func:`mean_noise_factor` does not shrink the threshold for
        them.
        """
        tag = self._prev[0]
        values = _knob_values(spec)
        for k in range(2, self.validation + 1):
            if tag == "restart":
                old = self._prev[1]
                if k % 2 == 0:
                    b = self._bench_stats()[0]
                    a = self._bench_resident_baseline(old)
                else:
                    a = self._bench_resident_baseline(old)
                    b = self._bench_stats()[0]
            elif k % 2 == 0:
                # B then A: put the old values back for the baseline half.
                b = self._bench_stats()[0]
                for knob, old_value in self._prev[1].items():
                    self._setter(self.engine, knob, old_value)
                self._state_is_baseline = True
                a = self._bench_stats()[0]
            else:
                a = self._bench_stats()[0]
                self._set_knobs(values)
                b = self._bench_stats()[0]
            rounds.append((a, b))
        return rounds

    def _bench_resident_baseline(self, old: Any) -> float:
        """The parallel-mode baseline, still resident beside the candidate.

        A measurement that fails there (a hang the watchdog killed included)
        may have taken the baseline with it, and a rollback would then make a
        dead engine current and report success. It is a lost engine instead: the
        baseline is released, not restored, and the candidate stays where it is.
        """
        try:
            return sum(self._tps(old) for _ in range(self._reps)) / self._reps
        except Exception as exc:
            self._shutdown(old)
            self._prev = None
            raise RestoreFailed(f"could not measure the resident baseline: {exc}") from exc

    def _swap_to_candidate(self, spec: InterventionSpec) -> None:
        """Replace the running baseline with a fresh build of the candidate. If
        the build fails, the baseline is rebuilt and the error re-raised, so the
        candidate is rolled back rather than left half-applied."""
        base = self.engine
        self._shutdown(base)
        self.engine = None
        # A Ctrl-C during the build must still leave restore() able to rebuild the baseline.
        self._prev = ("serial_restart", lambda: self._baseline_restart_fn(base))
        try:
            new_engine = self._restart_fn(base, _knob_values(spec))
            if new_engine is None:
                raise StructuralKnobRequiresRestart("restart_fn produced no engine")
        except Exception:
            self._rebuild_baseline(lambda: self._baseline_restart_fn(base),
                                   "candidate rebuild failed during validation")
            self._prev = ("baseline_running",)
            raise
        self.engine = new_engine
        self._unmeasured_build = True
        self._activate(new_engine)


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
