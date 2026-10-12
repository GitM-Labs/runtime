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


def test_the_threshold_shrinks_with_rounds_but_never_below_the_minimum():
    nf = NoiseFloor([0.01, -0.01])
    assert nf.threshold(1, minimum=0.0, without_floor=0.02) == pytest.approx(0.02)
    assert nf.threshold(4, minimum=0.0, without_floor=0.02) == pytest.approx(0.01)
    assert nf.threshold(4, minimum=0.015, without_floor=0.02) == pytest.approx(0.015)
    assert NoiseFloor().threshold(1, minimum=0.01, without_floor=0.02) == 0.02


# --------------------------------------------------------------------------- #
# calibration and free A/A samples                                            #
# --------------------------------------------------------------------------- #
def test_calibration_measures_the_baseline_against_itself_rebuilt():
    box = _Box([1000.0, 1004.0, 998.0])
    app = box.applicator()
    samples = app.calibrate(2)
    assert box.built == ["A", "A", "A"]                 # N + 1 builds for N pairs
    assert samples == pytest.approx([0.004, 998 / 1004 - 1])


def test_a_rollback_rebuild_adds_an_aa_sample_for_free():
    box = _Box([1000.0, 1010.0], candidate=[990.0])
    app = box.applicator(min_effect=0.01, threshold_without_floor=0.02)
    apply_intervention(_spec(), app)                    # loses, rolled back
    app.snapshot()                                      # the rebuilt baseline, measured again
    assert app.noise.samples == pytest.approx([0.01])


def test_a_keep_starts_a_new_configuration():
    box = _Box([1000.0], candidate=[1100.0, 1100.0])
    app = box.applicator(min_effect=0.01, threshold_without_floor=0.02)
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
    app = box.applicator(min_effect=0.01, threshold_without_floor=0.02)
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
    app = box.applicator(validation=3, min_effect=0.01, threshold_without_floor=0.02)
    app.snapshot()
    app.apply(_spec())
    app.measure(_spec())
    assert box.built == ["A", "B", "A", "B"]
    assert [k for k, _ in box.measured] == ["A", "B", "B", "A", "A", "B"]
    ab = app.last_result
    assert len(ab.rounds) == 3 and ab.speedup == pytest.approx(1.05)


def test_a_keep_that_ends_on_a_baseline_round_rebuilds_the_candidate():
    box = _Box([1000.0, 1000.0], candidate=[1050.0, 1050.0])
    app = box.applicator(validation=2, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.kept and app.engine.kind == "B"
    assert box.built == ["A", "B", "A", "B"]


def test_a_rollback_that_ends_on_a_baseline_round_builds_nothing_more():
    box = _Box([1000.0, 1000.0], candidate=[1003.0])
    app = box.applicator(validation=2, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.rolled_back and app.engine.kind == "A"
    assert box.built == ["A", "B", "A"]


def test_a_clear_loss_is_not_confirmed():
    box = _Box([1000.0, 1000.0], candidate=[900.0])
    app = box.applicator(validation=3, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)
    assert res.rolled_back and len(app.last_result.rounds) == 1
    assert box.built == ["A", "B", "A"]                 # the rollback only


def test_the_decision_is_the_mean_of_the_paired_ratios():
    box = _Box([1000.0, 1010.0], candidate=[1030.0, 1020.0])
    app = box.applicator(validation=2, min_effect=0.0, noise=NoiseFloor([0.005]))
    apply_intervention(_spec(), app)
    ab = app.last_result
    # round 1: A=1000 B=1030; round 2: B=1030 (same build) then A=1010
    assert ab.rounds == ((1000.0, 1030.0), (1010.0, 1030.0))
    assert ab.speedup == pytest.approx((1.03 + 1030 / 1010) / 2)
    # The baseline rebuilt in round 2 (1010 against 1000) is itself an A/A
    # sample, so the floor already includes it when the decision is made.
    assert app.noise.samples == pytest.approx([0.005, 0.01])
    floor = math.sqrt((0.005 ** 2 + 0.01 ** 2) / 2)
    assert ab.threshold == pytest.approx(2 * floor / math.sqrt(2))


def test_a_candidate_that_fails_to_rebuild_mid_validation_is_rolled_back():
    box = _Box([1000.0, 1000.0, 1000.0], candidate=[1050.0])
    app = box.applicator(validation=3, min_effect=0.01, threshold_without_floor=0.02)
    res = apply_intervention(_spec(), app)       # round 3 needs a second B: none scripted
    assert res.rolled_back and not res.restore_failed
    assert app.engine.kind == "A"


def test_a_baseline_lost_mid_validation_is_a_lost_engine():
    box = _Box([1000.0], candidate=[1050.0])     # no second A to rebuild
    app = box.applicator(validation=2, min_effect=0.01, threshold_without_floor=0.02)
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
