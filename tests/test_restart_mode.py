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
