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
    ``None``) the change is kept — apply-only mode. A spec carrying a
    ``correctness_gate`` is additionally rolled back when that gate fails,
    regardless of the measured delta.

    When an ``audit`` log is supplied, every live mutation and every rollback is
    recorded to the durable safety trail (best-effort — a broken audit sink never
    blocks the apply). Pass one only where the applicator mutates a real target;
    a dry-run leaves it ``None`` so the trail stays free of no-op entries.
    """
    # Step 1: snapshot. On a live engine this benchmarks the baseline, so it can
    # fail the same ways a measurement can — and when it does, nothing has been
    # applied yet. Returned as an error, not raised: one candidate whose baseline
    # could not be taken must not end a run that has other candidates to try.
    try:
        snapshot = applicator.snapshot()
    except Exception as exc:
        return ApplyResult(False, rolled_back=False, measured_delta=None,
                           error=f"snapshot failed, nothing applied: {exc}")

    # Step 2: apply. A bad value (validation error) rolls straight back.
    try:
        applicator.apply(spec)
    except Exception as exc:
        applicator.restore(snapshot)
        _audit(audit, "revert", spec, cause=f"apply failed, restored: {exc}",
               knobs=_knob_values(spec))
        return ApplyResult(False, rolled_back=True, measured_delta=None,
                           error=f"apply failed, restored: {exc}")
    _audit(audit, "apply", spec, cause="applied live", knobs=_knob_values(spec))

    # Step 3: measure. A crash mid-measurement also rolls back.
    try:
        delta = applicator.measure(spec)
    except Exception as exc:
        applicator.restore(snapshot)
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
            why = spec.correctness_gate(spec)
        except Exception as exc:
            applicator.restore(snapshot)
            _audit(audit, "revert", spec, knobs=_knob_values(spec),
                   cause=f"correctness gate crashed, restored: {exc}")
            return ApplyResult(True, rolled_back=True, measured_delta=delta,
                               error=f"correctness gate crashed, restored: {exc}")
        if why is not None:
            applicator.restore(snapshot)
            _audit(audit, "revert", spec, knobs=_knob_values(spec),
                   cause=f"correctness gate failed: {why}")
            return ApplyResult(True, rolled_back=True, measured_delta=delta,
                               error=f"correctness gate failed: {why}, restored")

    # Step 4: keep-or-rollback on the regression threshold.
    if delta is not None and delta < min_keep_delta:
        applicator.restore(snapshot)
        _audit(audit, "revert", spec, knobs=_knob_values(spec),
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
    """
    if requested:
        if requested not in {"parallel", "serial"}:
            raise ValueError(
                f"restart_mode must be 'parallel' or 'serial', got {requested!r}")
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
    ) -> None:
        self.engine = engine
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
        # Restore record: ("hotswap", knob, old_value) | ("restart", old_engine) |
        # ("serial_restart", restore_baseline_fn) | None.
        self._prev: tuple[Any, ...] | None = None
        self.last_result: EngineABResult | None = None

    def _bench(self) -> float:
        return sum(self._tps(self.engine) for _ in range(self._reps)) / self._reps

    def _bench_stats(self) -> tuple[float, float]:
        """(mean, sample stdev) of decode throughput over ``reps`` runs.

        stdev is 0.0 for a single rep → the noise band is 0 and keep falls back to
        ``delta > 0``, i.e. reps=1 behaves exactly as before reps were added.
        """
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
        self._baseline_tps, self._baseline_std = self._bench_stats()
        return {"baseline_tps": self._baseline_tps}

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
        if self._restart_mode == "serial":
            if self._baseline_restart_fn is None:
                raise StructuralKnobRequiresRestart(
                    "serial restart mode requires baseline_restart_fn to rebuild the baseline"
                )
            def restore_baseline() -> Any:
                return self._baseline_restart_fn(old_engine)

            self._shutdown(old_engine)
            self._prev = ("serial_restart", restore_baseline)
            try:
                new_engine = self._restart_fn(old_engine, values)
            except Exception:
                self.engine = restore_baseline()
                self._prev = None
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
            if self._restart_mode == "serial":
                _, restore_baseline = self._prev
                self.engine = restore_baseline()
                self._prev = None
            raise StructuralKnobRequiresRestart(
                f"restart_fn produced no engine for knob(s) {', '.join(values)}"
            )
        self.engine = new_engine
        self._activate(new_engine)
        if self._restart_mode != "serial":
            self._prev = ("restart", old_engine)

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
            self._activate(old_engine)
        elif tag == "serial_restart":
            _, restore_baseline = self._prev
            self._shutdown(self.engine)  # drop the candidate engine we built
            self.engine = restore_baseline()
            self._activate(self.engine)
        # Consume the restore record so a second restore() can't re-undo (or
        # re-shutdown the already-discarded candidate engine) a second time.
        self._prev = None

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
        """Best-effort release of an engine before/after a restart A/B."""
        custom = getattr(engine, "gitm_shutdown_fn", None)
        if callable(custom):
            try:
                custom(engine)
            except Exception:
                pass

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
                break
        try:
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

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
        speedup = candidate / baseline
        delta = speedup - 1.0
        # Noise band = combined relative scatter of baseline+candidate. A gain is
        # kept only if it clears the band: returning (delta - band) makes the
        # existing min_keep_delta=0 gate keep ONLY significant gains and roll back
        # anything within noise. reps=1 → std=0 → band=0 → keep iff delta>0.
        noise_band = (self._baseline_std + cand_std) / baseline
        significant = delta > noise_band
        via = "restart" if (self._prev and self._prev[0] in {"restart", "serial_restart"}) else "hot-swap"
        self.last_result = EngineABResult(
            knob=spec.knob, value=(getattr(spec, "knobs", None) or spec.value),
            baseline_tps=baseline,
            candidate_tps=candidate, speedup=speedup, kept=significant, via=via,
            baseline_std=self._baseline_std, candidate_std=cand_std,
            reps=self._reps, significant=significant,
        )
        return delta - noise_band


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
