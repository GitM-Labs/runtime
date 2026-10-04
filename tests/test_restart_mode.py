"""Choosing the restart mode, and refusing one that cannot work.

A structural knob can only be measured by rebuilding the engine. Parallel mode
builds the candidate while the baseline is still up; serial releases the
baseline first and rebuilds it afterwards. The default was parallel, and on the
MI355X run it cost **27 of 29 candidates** — each one building into a device the
baseline already held 90% of, and dying of OOM.

The constraint was written down in `workloads.py` ("in parallel mode
baseline+candidate must both fit") and enforced nowhere.
"""

from __future__ import annotations

import pytest

from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.apply import (
    LiveEngineApplicator,
    StructuralKnobRequiresRestart,
    gpu_fraction,
    parallel_restart_fits,
    resolve_restart_mode,
)


def _structural_spec():
    """A lever on a knob that can only take effect through a rebuild."""
    return InterventionSpec(
        name="tp4", summary="s", knob="tensor_parallel_size", value=4,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )


class _Engine:
    """An engine gitm built: it always carries the kwargs it was built with."""

    def __init__(self, fraction=None, baseline_restart=True):
        self.gitm_llm_kwargs = {} if fraction is None else {"gpu_memory_utilization": fraction}
        if baseline_restart:
            self.gitm_baseline_restart_fn = lambda old: _Engine(fraction)


# --- what fraction the next engine will ask for -----------------------------


def test_the_fraction_comes_from_the_kwargs_the_engine_was_built_with():
    """Not measured off the device: a restart candidate inherits the baseline's
    kwargs, so the number we need is this one by construction — and measuring
    would mean initialising CUDA in a process that does not otherwise carry a
    context."""
    assert gpu_fraction(_Engine(0.45)) == 0.45
    assert gpu_fraction(_Engine(None)) == 0.9       # built with no cap: vLLM's default


def test_an_engine_gitm_did_not_build_is_unknown_rather_than_assumed():
    """Two absences that are not the same. No cap in the kwargs means vLLM's
    default; no kwargs at all means somebody else's handle, and guessing there
    would refuse a restart on a number nobody supplied."""
    assert gpu_fraction(object()) is None
    fits, why = parallel_restart_fits(object())
    assert fits and "does not say" in why


def test_an_unreadable_fraction_falls_back_rather_than_raising():
    e = _Engine()
    e.gitm_llm_kwargs = {"gpu_memory_utilization": "not a number"}
    assert gpu_fraction(e) == 0.9


# --- can two engines fit at once --------------------------------------------


@pytest.mark.parametrize("fraction,fits", [
    (0.45, True), (0.5, True), (0.51, False), (0.9, False),
])
def test_parallel_needs_each_engine_to_fit_in_half_the_device(fraction, fits):
    got, why = parallel_restart_fits(_Engine(fraction))
    assert got is fits, why


def test_the_default_build_cannot_run_a_parallel_restart():
    """The case that actually happened: nobody set GITM_VLLM_GPU_MEM, so both
    engines asked for 90% of the device."""
    fits, why = parallel_restart_fits(_Engine(None))
    assert not fits
    assert "90%" in why and "180%" in why
    assert "serial" in why and "GITM_VLLM_GPU_MEM" in why   # both ways out


# --- which mode a run gets --------------------------------------------------


def test_serial_is_the_default_where_a_baseline_can_be_rebuilt():
    mode, why = resolve_restart_mode(_Engine(0.9), None)
    assert mode == "serial" and "default" in why


def test_parallel_remains_the_fallback_with_no_baseline_restart():
    """Serial has nothing to restore with there, so parallel is the only mode —
    and it is chosen for that reason rather than inherited as a default."""
    mode, why = resolve_restart_mode(_Engine(0.9, baseline_restart=False), None)
    assert mode == "parallel" and "baseline_restart_fn" in why


def test_an_explicit_choice_wins_either_way():
    for want in ("parallel", "serial"):
        mode, why = resolve_restart_mode(_Engine(0.9), want)
        assert mode == want and why == "set explicitly"


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="parallel.*serial"):
        resolve_restart_mode(_Engine(0.9), "sequential")


# --- the refusal, where it matters ------------------------------------------


def _applicator(fraction, **kw):
    return LiveEngineApplicator(
        _Engine(fraction), throughput_fn=lambda e: 1.0,
        restart_fn=lambda e, v: _Engine(fraction), **kw)


def test_a_parallel_rebuild_that_cannot_fit_is_refused_before_it_is_attempted():
    """The failure is the same for this candidate either way, but an OOM from
    inside vLLM names neither the mode nor the way out, and it repeats once per
    candidate for the whole run."""
    app = _applicator(0.9, restart_mode="parallel")
    with pytest.raises(StructuralKnobRequiresRestart, match="cannot provide one"):
        app.apply(_structural_spec())


def test_a_parallel_rebuild_that_fits_is_allowed():
    app = _applicator(0.45, restart_mode="parallel")
    app.apply(_structural_spec())
    assert app.engine is not None


def test_the_warning_is_raised_at_construction_not_per_candidate():
    """Recorded rather than raised: a run whose candidates are all hot-swappable
    never reaches a rebuild and must not be stopped here."""
    assert _applicator(0.9, restart_mode="parallel").restart_mode_warning
    assert _applicator(0.45, restart_mode="parallel").restart_mode_warning is None
    assert _applicator(0.9, restart_mode="serial").restart_mode_warning is None


# --- the candidate's own fraction, not twice the baseline's ----------------


def test_a_lever_that_changes_the_fraction_is_measured_on_its_own_value():
    """`gpu_memory_utilization_dynamic` exists to change this number, so the
    candidate does not always ask for what the baseline holds. Doubling the
    baseline gets that wrong in both directions."""
    # Would pass a doubling check (0.45 x 2 = 0.9) and needs 135%.
    fits, why = parallel_restart_fits(_Engine(0.45), {"gpu_memory_utilization": 0.9})
    assert not fits and "135%" in why and "candidate asks for 90%" in why

    # Would fail a doubling check (0.6 x 2 = 1.2) and fits exactly.
    fits, _ = parallel_restart_fits(_Engine(0.6), {"gpu_memory_utilization": 0.4})
    assert fits


def test_a_lever_on_another_knob_leaves_the_candidate_inheriting():
    fits, why = parallel_restart_fits(_Engine(0.45), {"tensor_parallel_size": 4})
    assert fits and "0.45 + candidate 0.45" in why


def test_an_unreadable_candidate_value_falls_back_to_the_baseline():
    fits, _ = parallel_restart_fits(_Engine(0.45), {"gpu_memory_utilization": "lots"})
    assert fits


# --- one source for "can the baseline be rebuilt" --------------------------


def test_a_baseline_rebuild_passed_as_an_argument_is_enough_for_serial():
    """It used to read the engine attribute while the apply path used the
    constructor argument, so a caller supplying only the argument got parallel
    despite having a rebuild available."""
    engine = _Engine(0.9, baseline_restart=False)
    app = LiveEngineApplicator(
        engine, throughput_fn=lambda e: 1.0,
        restart_fn=lambda e, v: engine,
        baseline_restart_fn=lambda old: engine)
    assert app._restart_mode == "serial"


def test_a_baseline_rebuild_on_the_engine_is_enough_for_serial():
    """The other direction, which was worse: it resolved to serial and then
    failed every structural apply because the argument was missing."""
    app = LiveEngineApplicator(
        _Engine(0.9), throughput_fn=lambda e: 1.0,
        restart_fn=lambda e, v: _Engine(0.9))
    assert app._restart_mode == "serial"
    assert app._baseline_restart_fn is not None
    app.apply(_structural_spec())      # reaches the rebuild rather than raising


def test_no_rebuild_anywhere_still_falls_back_to_parallel():
    app = LiveEngineApplicator(
        _Engine(0.45, baseline_restart=False), throughput_fn=lambda e: 1.0,
        restart_fn=lambda e, v: _Engine(0.45))
    assert app._restart_mode == "parallel"


def test_the_startup_warning_says_what_it_actually_checked():
    """It compares the baseline against a candidate that inherits its fraction,
    which is most of the catalogue but not all. Saying structural candidates are
    impossible would have an operator dismiss a measurement that would have
    worked — a 0.6 baseline still admits a candidate at 0.4."""
    warning = _applicator(0.6, restart_mode="parallel").restart_mode_warning
    assert warning
    assert "0.40 or less" in warning          # the room that is actually left
    assert "cannot be measured" not in warning
    assert "Every other structural candidate will be refused" in warning

    # And that candidate is indeed allowed when it turns up.
    fits, _ = parallel_restart_fits(_Engine(0.6), {"gpu_memory_utilization": 0.4})
    assert fits


@pytest.mark.parametrize("baseline", [0.51, 0.585, 0.6, 0.666, 0.75, 0.9, 0.99])
def test_the_limit_the_warning_prints_is_one_the_check_accepts(baseline):
    """The invariant, not an example. A rounded limit is worse than none: a
    baseline of 0.585 leaves 0.415, and printing "0.42" hands the operator a
    number the fit check then refuses at 1.005."""
    import re

    warning = _applicator(baseline, restart_mode="parallel").restart_mode_warning
    assert warning, f"no warning at baseline {baseline}"

    stated = float(re.search(r"to ([0-9.]+) or less", warning).group(1))
    fits, why = parallel_restart_fits(
        _Engine(baseline), {"gpu_memory_utilization": stated})
    assert fits, f"warning offered {stated} at baseline {baseline}, but: {why}"
