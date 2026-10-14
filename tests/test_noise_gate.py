"""A candidate is kept only on a gain above the run's own noise floor.

The old rule kept anything that measured >= 0 in one A/B. An A/A through gitm's
own path on an RTX 4050 (the baseline rebuilt and measured against itself, 15
times) kept 8 of 15: noise alone is positive half the time. The new rule:

* the run measures its baseline against itself, rebuilt each time, for a noise
  floor, and keeps adding A/A samples whenever the same baseline is re-measured
  on a fresh build;
* a candidate is measured over ``validation`` paired rounds, alternating order;
* it is kept only if the mean of the rounds' ratios beats
  ``max(2 * floor / sqrt(rounds), min_effect)``.
"""

from __future__ import annotations

import math

import pytest

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import (
    LiveEngineApplicator,
    NoiseFloor,
    RestoreFailed,
    apply_intervention,
    mean_noise_factor,
)


def _spec(knob: str = "block_size", value=16) -> InterventionSpec:
    return InterventionSpec.model_validate(
        dict(name=knob, summary=f"set {knob}", knob=knob, value=value,
             expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
             source="test"))


class _Box:
    """Fake builds: every engine built gets the next throughput from a script,
    per configuration, so a test says exactly what each build measures."""

    def __init__(self, baseline: list[float], candidate: list[float] | None = None):
        self.script = {"A": list(baseline), "B": list(candidate or [])}
        self.built: list[str] = []
        self.measured: list[tuple[str, float]] = []

    def build(self, kind: str):
        self.built.append(kind)
        e = type("E", (), {})()
        e.kind = kind
        e.tps = self.script[kind].pop(0)
        return e

    def tps(self, e) -> float:
        self.measured.append((e.kind, e.tps))
        return e.tps

    def applicator(self, **kw) -> LiveEngineApplicator:
        first = self.build("A")
        return LiveEngineApplicator(
            first, throughput_fn=self.tps,
            restart_fn=lambda _old, _v: self.build("B"),
            baseline_restart_fn=lambda _old: self.build("A"),
            restart_mode="serial", **kw)


# --------------------------------------------------------------------------- #
# the floor and the threshold                                                 #
# --------------------------------------------------------------------------- #
def test_the_floor_is_the_rms_of_aa_samples():
    nf = NoiseFloor([0.004, -0.004, 0.004, -0.004])
    assert nf.floor == pytest.approx(0.004)
    assert NoiseFloor().floor is None


def test_the_threshold_is_twice_the_noise_of_the_mean_never_below_the_minimum():
    nf = NoiseFloor([0.01, -0.01])
    assert nf.threshold(1.0, minimum=0.0, without_floor=0.02) == pytest.approx(0.02)
    assert nf.threshold(0.5, minimum=0.0, without_floor=0.02) == pytest.approx(0.01)
    assert nf.threshold(0.5, minimum=0.015, without_floor=0.02) == pytest.approx(0.015)
    assert NoiseFloor().threshold(1.0, minimum=0.01, without_floor=0.02) == 0.02


def test_rounds_that_share_a_build_shrink_the_noise_less_than_root_n():
    """Alternating rounds reuse the running engine: two rounds are
    B1 - (A1 + A2) / 2, whose spread is 0.87 of the floor, not 1 / sqrt(2)."""
    assert mean_noise_factor(1, rebuilt=True) == 1.0
    assert mean_noise_factor(2, rebuilt=True) == pytest.approx(math.sqrt(0.75))
    assert mean_noise_factor(3, rebuilt=True) == pytest.approx(math.sqrt(5 / 9))
    assert mean_noise_factor(3, rebuilt=True) > 1 / math.sqrt(3)
    # Rounds that rebuild nothing re-measure the same two builds.
    assert mean_noise_factor(4, rebuilt=False) == 1.0


# --------------------------------------------------------------------------- #
# calibration and free A/A samples                                            #
# --------------------------------------------------------------------------- #
def test_calibration_measures_the_baseline_against_itself_rebuilt():
    box = _Box([990.0, 1000.0, 1004.0, 998.0])
    app = box.applicator()
    samples = app.calibrate(2)
    assert box.built == ["A", "A", "A", "A"]            # the warm-up, then N + 1 rebuilds
    assert samples == pytest.approx([0.004, 998 / 1004 - 1])
    assert app.noise.warmup == pytest.approx(1000 / 990 - 1)


def test_a_cold_first_engine_does_not_set_the_floor():
    """Kimi K2.5 on MI355X: the first engine after the opening capture decoded
    at 1110 tok/s, every later build at about 1400. Paired, that one 24.8%
    sample made the RMS floor 8.9% and rejected a +5% win against a 25% bar."""
    box = _Box([1109.69, 1385.49, 1424.68, 1387.71])
    app = box.applicator()
    app.calibrate(2)
    assert app.noise.warmup == pytest.approx(0.2485, abs=1e-3)
    assert app.noise.floor < 0.03
    bar = app.noise.threshold(mean_noise_factor(2, rebuilt=True),
                              minimum=0.01, without_floor=0.02)
    assert bar < 0.05                                   # the +5.0% graph-replay win clears it


def test_a_rollback_rebuild_adds_an_aa_sample_for_free():
    box = _Box([1000.0, 1010.0], candidate=[990.0])
    app = box.applicator(noise_gate=True, min_effect=0.01, threshold_without_floor=0.02)
    apply_intervention(_spec(), app)                    # loses, rolled back
    app.snapshot()                                      # the rebuilt baseline, measured again
    assert app.noise.samples == pytest.approx([0.01])


def test_a_keep_starts_a_new_configuration():
    box = _Box([1000.0], candidate=[1100.0, 1100.0])
    app = box.applicator(noise_gate=True, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.kept
    app.snapshot()          # the kept candidate, measured as the new baseline
    assert app.noise.samples == []                     # not an A/A sample against the old one


# --------------------------------------------------------------------------- #
# the keep rule                                                               #
# --------------------------------------------------------------------------- #
def test_a_gain_inside_the_noise_is_rolled_back():
    """The old rule kept +0.5% because it is >= 0."""
    box = _Box([1000.0, 1000.0], candidate=[1005.0])
    app = box.applicator(noise_gate=True, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.rolled_back and app.last_result.threshold == 0.02


def test_the_defaults_keep_the_old_rule():
    """Callers that do not ask for validation see no change."""
    box = _Box([1000.0], candidate=[1005.0])
    res = apply_intervention(_spec(), box.applicator())
    assert res.kept


def test_rounds_alternate_and_cost_one_build_each():
    # Round 1: A then B (build B). Round 2: B again, then rebuild A. Round 3:
    # A again, then rebuild B.
    box = _Box([1000.0, 1000.0], candidate=[1050.0, 1050.0])
    app = box.applicator(noise_gate=True, validation=3, min_effect=0.01, threshold_without_floor=0.02)
    app.snapshot()
    app.apply(_spec())
    app.measure(_spec())
    assert box.built == ["A", "B", "A", "B"]
    assert [k for k, _ in box.measured] == ["A", "B", "B", "A", "A", "B"]
    ab = app.last_result
    assert len(ab.rounds) == 3 and ab.speedup == pytest.approx(1.05)


def test_a_keep_that_ends_on_a_baseline_round_rebuilds_the_candidate():
    box = _Box([1000.0, 1000.0], candidate=[1050.0, 1050.0])
    app = box.applicator(noise_gate=True, validation=2, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.kept and app.engine.kind == "B"
    assert box.built == ["A", "B", "A", "B"]


def test_a_rollback_that_ends_on_a_baseline_round_builds_nothing_more():
    box = _Box([1000.0, 1000.0], candidate=[1003.0])
    app = box.applicator(noise_gate=True, validation=2, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.rolled_back and app.engine.kind == "A"
    assert box.built == ["A", "B", "A"]


def test_a_clear_loss_is_not_confirmed():
    box = _Box([1000.0, 1000.0], candidate=[900.0])
    app = box.applicator(noise_gate=True, validation=3, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.rolled_back and len(app.last_result.rounds) == 1
    assert box.built == ["A", "B", "A"]                 # the rollback only


def test_the_decision_is_the_mean_of_the_paired_ratios():
    box = _Box([1000.0, 1010.0], candidate=[1030.0, 1020.0])
    app = box.applicator(noise_gate=True, validation=2, min_effect=0.0, noise=NoiseFloor([0.005]))
    apply_intervention(_spec(), app)
    ab = app.last_result
    # round 1: A=1000 B=1030; round 2: B=1030 (same build) then A=1010
    assert ab.rounds == ((1000.0, 1030.0), (1010.0, 1030.0))
    assert ab.speedup == pytest.approx((1.03 + 1030 / 1010) / 2)
    # The baseline rebuilt in round 2 (1010 against 1000) is itself an A/A
    # sample, so the floor already includes it when the decision is made.
    assert app.noise.samples == pytest.approx([0.005, 0.01])
    floor = math.sqrt((0.005 ** 2 + 0.01 ** 2) / 2)
    assert ab.threshold == pytest.approx(2 * floor * math.sqrt(0.75))


def test_a_candidate_that_fails_to_rebuild_mid_validation_is_rolled_back():
    box = _Box([1000.0, 1000.0, 1000.0], candidate=[1050.0])
    app = box.applicator(noise_gate=True, validation=3, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)       # round 3 needs a second B: none scripted
    assert res.rolled_back and not res.restore_failed
    assert app.engine.kind == "A"


def test_a_baseline_lost_mid_validation_is_a_lost_engine():
    box = _Box([1000.0], candidate=[1050.0])     # no second A to rebuild
    app = box.applicator(noise_gate=True, validation=2, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.restore_failed and not res.rolled_back


def test_calibration_that_cannot_rebuild_is_a_lost_engine():
    box = _Box([1000.0])
    with pytest.raises(RestoreFailed):
        box.applicator().calibrate(1)


# --------------------------------------------------------------------------- #
# the loop                                                                    #
# --------------------------------------------------------------------------- #
def test_the_loop_measures_its_noise_and_says_what_it_used(tmp_path, monkeypatch):
    import json
    import sys
    from contextlib import contextmanager
    from pathlib import Path

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace
    from .test_vllm_knobs_and_restart import _FullEngine

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100,
                               end_ns=i * 100 + 80) for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    # Fake engines only: without this, autoresearch introspects a real vLLM.
    monkeypatch.setitem(sys.modules, "vllm", None)
    engine = _FullEngine()
    engine.gitm_baseline_restart_fn = lambda old: type(old)(
        max_num_seqs=old.scheduler_config.max_num_seqs)
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=3, validation=2))
    doc = json.loads((Path(out["run_dir"]) / "noise_floor.json").read_text())
    assert doc["validation"] == 2 and len(doc["calibration_samples"]) == 2
    assert out["summary"]["validation"] == 2
    for rec in json.loads((Path(out["run_dir"]) / "verification.json").read_text())["results"]:
        assert rec["threshold"] is not None and rec["rounds"]


# --------------------------------------------------------------------------- #
# review findings                                                             #
# --------------------------------------------------------------------------- #
def test_samples_collected_without_the_gate_change_nothing():
    """A caller that does not turn the gate on keeps the old rule, even after a
    rollback has collected an A/A sample."""
    box = _Box([1000.0, 1010.0, 1010.0], candidate=[990.0, 1015.0])
    app = box.applicator()
    apply_intervention(_spec(), app)                     # rolled back
    res = apply_intervention(_spec(), app)               # +0.5%: the old rule keeps it
    assert app.noise.samples == pytest.approx([0.01])    # the rebuilt baseline, measured
    assert res.kept


def test_a_gain_exactly_at_the_bar_is_kept_and_the_candidate_is_running():
    box = _Box([1000.0, 1000.0], candidate=[1000.0, 1000.0])
    app = box.applicator(noise_gate=True, validation=2, min_effect=0.0,
                         threshold_without_floor=0.0)
    res = apply_intervention(_spec(), app)
    assert res.kept and app.engine.kind == "B"


def test_a_baseline_that_cannot_be_measured_mid_validation_is_a_lost_engine():
    box = _Box([1000.0, 1000.0], candidate=[1050.0])

    def tps(e):
        if e.kind == "A" and box.built.count("A") == 2:
            raise TimeoutError("A/B decode ran past 1800s; its engine was killed")
        return box.tps(e)

    app = LiveEngineApplicator(
        box.build("A"), throughput_fn=tps,
        restart_fn=lambda _o, _v: box.build("B"),
        baseline_restart_fn=lambda _o: box.build("A"),
        restart_mode="serial", noise_gate=True, validation=2, min_effect=0.01,
        threshold_without_floor=0.02)
    released = []
    app._shutdown = lambda e: released.append(e.kind)
    res = apply_intervention(_spec(), app)
    assert res.restore_failed and not res.rolled_back and app.engine is None
    assert released[-1] == "A"           # the unmeasurable baseline was released, not dropped


def test_calibration_stops_at_the_deadline():
    box = _Box([1000.0, 1001.0, 1002.0])
    app = box.applicator()
    assert app.calibrate(2, deadline_ns=0) == []
    assert box.built == ["A"]


def test_the_recorded_threshold_is_the_bar_that_decided():
    box = _Box([1000.0], candidate=[1015.0])
    app = box.applicator(noise_gate=True, min_effect=0.01, threshold_without_floor=0.01,
                         reps=2)
    app._bench_stats = lambda: (1000.0, 20.0) if app.engine.kind == "A" else (1015.0, 20.0)
    apply_intervention(_spec(), app)
    assert app.last_result.threshold == pytest.approx(0.04)      # the reps band, not 1%


def test_a_hot_swapped_candidate_gets_its_rounds_too():
    """A knob set on the running engine is toggled back and forth, with no
    rebuild, and the threshold is not shrunk for those rounds."""
    state = {"v": 0}
    engine = type("E", (), {})()
    seen: list[int] = []

    def tps(_e):
        seen.append(state["v"])
        return 1000.0 + 30.0 * state["v"]

    app = LiveEngineApplicator(
        engine, throughput_fn=tps, getter=lambda _e, _k: state["v"],
        setter=lambda _e, _k, v: state.update(v=v), noise_gate=True, validation=3,
        min_effect=0.01, threshold_without_floor=0.02)
    spec = _spec("max_num_seqs", 1)
    from gitm.optimizer import apply as apply_mod

    orig = apply_mod.knob_kind
    apply_mod.knob_kind = lambda _k: "scheduling"
    try:
        res = apply_intervention(spec, app)
    finally:
        apply_mod.knob_kind = orig
    assert len(app.last_result.rounds) == 3
    assert seen == [0, 1, 1, 0, 0, 1]                    # A B | B A | A B
    assert res.kept and state["v"] == 1
    assert app.last_result.threshold == pytest.approx(0.02)


def test_the_loop_skips_calibration_without_an_engine_owned_probe(tmp_path, monkeypatch):
    """The fallback probe re-runs the opening runner and can measure only the
    original engine; calibrating with it would shut that engine down."""
    import sys
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace
    from .test_vllm_knobs_and_restart import _FullEngine

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100,
                               end_ns=i * 100 + 80) for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    called = []
    monkeypatch.setattr(LiveEngineApplicator, "calibrate",
                        lambda self, *a, **k: called.append(1) or [])
    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", None)
    engine = _FullEngine()
    del engine.gitm_throughput_fn
    engine.gitm_baseline_restart_fn = lambda old: old
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=1))
    assert out["summary"]["status"] != "no_data"           # it reached Phase 4
    assert called == []


def test_a_parallel_baseline_that_dies_mid_validation_is_a_lost_engine():
    """Measuring the resident baseline can take it down (a killed hang). A
    rollback would make that dead engine current and report success."""
    base = type("E", (), {"kind": "A", "dead": False})()
    cand = type("E", (), {"kind": "B", "dead": False})()
    calls = {"A": 0}
    released = []

    def tps(e):
        if e.kind == "A":
            calls["A"] += 1
            if calls["A"] == 2:
                e.dead = True
                raise TimeoutError("A/B decode ran past 1800s; its engine was killed")
            return 1000.0
        return 1050.0

    app = LiveEngineApplicator(
        base, throughput_fn=tps, restart_fn=lambda _o, _v: cand,
        restart_mode="parallel", noise_gate=True, validation=2, min_effect=0.01,
        threshold_without_floor=0.02)
    app._shutdown = lambda e: released.append(e.kind)
    from gitm.optimizer import apply as apply_mod

    orig = apply_mod.parallel_restart_fits
    apply_mod.parallel_restart_fits = lambda *_a, **_k: (True, "")
    try:
        res = apply_intervention(_spec(), app)
    finally:
        apply_mod.parallel_restart_fits = orig
    assert res.restore_failed and not res.rolled_back
    assert app.engine is not base and "A" in released


def test_only_first_measurements_on_a_build_make_aa_samples():
    """On a laptop on battery the first decode after a rebuild ran 4-5% faster
    than the next one on the same engine. Pairing a second measurement with a
    fresh one read that as noise and multiplied the floor by seven."""
    box = _Box([1000.0] * 8,
               candidate=[1003.0, 1003.0, 1003.0, 1003.0])
    seen: dict[int, int] = {}

    def tps(e):
        seen[id(e)] = seen.get(id(e), 0) + 1
        return e.tps * (1.0 if seen[id(e)] == 1 else 0.95)   # warm engines run slower

    app = LiveEngineApplicator(
        box.build("A"), throughput_fn=tps,
        restart_fn=lambda _o, _v: box.build("B"),
        baseline_restart_fn=lambda _o: box.build("A"),
        restart_mode="serial", noise_gate=True, validation=2, min_effect=0.01,
        threshold_without_floor=0.02)
    app.calibrate(1)
    apply_intervention(_spec(), app)        # round 2 rebuilds A: fresh
    apply_intervention(_spec(), app)        # its snapshot re-measures that A: not fresh
    # Every sample compares two fresh 1000s, so the floor is 0, not ~5%.
    assert app.noise.samples and all(abs(d) < 1e-9 for d in app.noise.samples)
    assert app.noise.skipped_not_fresh >= 1


def test_a_baseline_brought_back_in_parallel_mode_is_not_fresh():
    """A parallel rollback makes the already-measured baseline current again:
    no build happened, so its next measurement is not an A/A sample."""
    base = type("E", (), {"tps": 1000.0})()
    cand = type("E", (), {"tps": 900.0})()
    app = LiveEngineApplicator(
        base, throughput_fn=lambda e: e.tps, restart_fn=lambda _o, _v: cand,
        restart_mode="parallel", noise_gate=True, min_effect=0.01,
        threshold_without_floor=0.02)
    from gitm.optimizer import apply as apply_mod

    orig = apply_mod.parallel_restart_fits
    apply_mod.parallel_restart_fits = lambda *_a, **_k: (True, "")
    try:
        res = apply_intervention(_spec(), app)           # -10%: rolled back
    finally:
        apply_mod.parallel_restart_fits = orig
    assert res.rolled_back and app.engine is base
    app.snapshot()
    assert app._last_measure_fresh is False
    assert app.noise.samples == [] and app.noise.skipped_not_fresh == 1


# --------------------------------------------------------------------------- #
# the budget and the engine at exit                                           #
# --------------------------------------------------------------------------- #
def test_a_candidate_starts_only_if_the_budget_left_covers_one():
    """The first Kimi run started its last candidate 35 s before a 5 h budget
    ended and overran it by 19 minutes."""
    from gitm.scheduler.loop import _next_cycle_overruns

    assert _next_cycle_overruns(100.0, 3600.0, []) is None       # nothing to go by yet
    assert _next_cycle_overruns(3600.0, 3600.0, []) == "spent"
    assert _next_cycle_overruns(3000.0, 3600.0, [300.0, 500.0]) is None
    why = _next_cycle_overruns(17965.0, 18000.0, [2700.0, 3300.0])
    assert why is not None and "50 min" in why


def test_the_run_releases_an_engine_it_built_and_only_that(monkeypatch):
    """Left running, vLLM's spawned TP workers keep ``gitm run`` from exiting
    after it has written its report."""
    import gitm.api as api

    released: list[str] = []
    monkeypatch.setattr(LiveEngineApplicator, "_shutdown",
                        staticmethod(lambda e: released.append(e)))

    def fake_loop(cfg):
        cfg.engine = cfg.engine or "built-by-the-workload"
        return {}

    monkeypatch.setattr(api, "run_loop", fake_loop)
    api.optimize(workload="vllm-decode")
    assert released == ["built-by-the-workload"]

    released.clear()
    api.optimize("the-callers-engine")
    api.optimize(workload_runner=lambda: {})            # the runner's engine: not ours
    assert released == []

    def failing_loop(cfg):
        cfg.engine = "built-by-the-workload"
        raise RuntimeError("the loop failed")

    released.clear()
    monkeypatch.setattr(api, "run_loop", failing_loop)
    with pytest.raises(RuntimeError):
        api.optimize(workload="vllm-decode")
    assert released == ["built-by-the-workload"]


def test_an_engine_a_retrace_swaps_in_is_a_fresh_build():
    """A re-trace replaces the running baseline with a traced build and then an
    untraced one. That engine is new, so its first measurement pairs with the
    last fresh one instead of being discarded as a warm re-measurement."""
    box = _Box([1000.0, 1000.0, 1000.0, 1010.0])
    app = box.applicator()
    app.calibrate(1)                        # the warm-up, then one A/A pair
    assert app.noise.samples == pytest.approx([0.0])
    app.snapshot()                          # the same engine again: not fresh
    assert app.noise.skipped_not_fresh == 1
    app.install(box.build("A"))             # what the re-trace leaves running
    app.snapshot()
    assert app.noise.samples == pytest.approx([0.0, 0.01])
