"""The experiment contract and the local evaluation path.

These tests check the wiring and the result semantics against a deterministic
fake engine. They are not a hardware performance result.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from gitm.experiment import Sample, Verdict, evaluate, load_contract
from gitm.experiment.__main__ import main as cli_main
from gitm.experiment.evaluate import run_order
from gitm.optimizer.apply import DictApplicator

FIXTURE = Path(__file__).parent / "fixtures" / "experiment" / "kv_fp8_local.yaml"


class FakeServer:
    """Reads the applied config and returns metrics with seeded noise.

    ``effect`` is the true relative throughput change fp8 produces here.
    """

    def __init__(self, config: dict, *, effect: float = 0.10, noise: float = 0.01,
                 correctness: float = 1.0, ttft_change: float = 0.0, seed: int = 0,
                 fail_at: int | None = None):
        self.config = config
        self.effect, self.noise = effect, noise
        self.correctness, self.ttft_change = correctness, ttft_change
        self.rng = random.Random(seed)
        self.fail_at = fail_at

    def sample(self, arm: str, index: int) -> Sample:
        if index == self.fail_at:
            raise RuntimeError("server went away")
        fp8 = self.config.get("kv_cache_dtype") == "fp8"
        assert fp8 == (arm == "candidate"), "sampled arm does not match the applied config"
        jitter = 1.0 + self.rng.gauss(0.0, self.noise)
        tps = 1000.0 * (1.0 + self.effect if fp8 else 1.0) * jitter
        ttft = 800.0 * (1.0 + self.ttft_change if fp8 else 1.0)
        return Sample(
            metrics={"output_tok_s": tps, "ttft_ms_p95": ttft, "itl_ms_p95": 30.0},
            correctness=self.correctness if fp8 else 1.0,
            wall_s=180.0, accelerator_s=8 * 180.0,
        )


def _run(contract=None, **server_kw):
    contract = contract or load_contract(FIXTURE)
    config = dict(contract.baseline.engine_args)
    app = DictApplicator(config)
    server = FakeServer(config, **server_kw)
    return evaluate(contract, app, server, provenance={"hardware": "fake"}), config


def _edit(**changes):
    data = load_contract(FIXTURE).model_dump(mode="json")
    for dotted, value in changes.items():
        *path, last = dotted.split("__")
        node = data
        for key in path:
            node = node[key]
        node[last] = value
    return data


# --- contract validation ----------------------------------------------------


def test_fixture_validates():
    c = load_contract(FIXTURE)
    assert c.candidate_id == "fixture-kv-cache-fp8"
    assert len(c.sha256()) == 64


def test_candidate_must_be_baseline_plus_intervention():
    data = _edit(candidate__engine_args={"tensor_parallel_size": 4, "max_num_seqs": 256,
                                         "kv_cache_dtype": "fp8"})
    from gitm.experiment.contract import ExperimentContract
    with pytest.raises(ValidationError, match="exactly"):
        ExperimentContract.model_validate(data)


def test_intervention_must_change_something():
    from gitm.experiment.contract import ExperimentContract
    data = _edit(intervention={"kv_cache_dtype": "auto"},
                 candidate__engine_args=_edit()["baseline"]["engine_args"])
    with pytest.raises(ValidationError, match="does not change"):
        ExperimentContract.model_validate(data)


def test_unknown_field_rejected():
    from gitm.experiment.contract import ExperimentContract
    data = _edit()
    data["protocol"]["repititions"] = 5
    with pytest.raises(ValidationError):
        ExperimentContract.model_validate(data)


def test_randomized_order_needs_seed():
    from gitm.experiment.contract import ExperimentContract
    with pytest.raises(ValidationError, match="order_seed"):
        ExperimentContract.model_validate(_edit(protocol__comparison_order="randomized"))


@pytest.mark.parametrize("order", ["ABAB", "ABBA"])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_run_order_is_balanced(order, n):
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(protocol__comparison_order=order,
                                                protocol__repetitions=n))
    arms = run_order(c)
    assert arms.count("baseline") == arms.count("candidate") == n


def test_cli_validate(capsys, tmp_path):
    assert cli_main(["validate", str(FIXTURE)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("candidate_id: x\n")
    assert cli_main(["validate", str(bad)]) == 1


# --- fixture end to end ------------------------------------------------------


def test_fixture_end_to_end_improved_and_kept(tmp_path):
    v, config = _run(effect=0.10)
    assert v.state == "improved"
    assert v.rollback == "kept" and config["kv_cache_dtype"] == "fp8"
    assert v.interval[0] > 0 and v.effect == pytest.approx(0.10, abs=0.02)
    assert v.correctness.status == "pass"
    assert [g.status for g in v.latency] == ["pass", "pass"]
    assert v.n_baseline == v.n_candidate == 5
    assert v.detectability == "not_established"
    assert v.claim_rejected is False
    # Warmup is recorded but kept out of the estimate.
    assert sum(s.warmup for s in v.raw) > 0
    # Machine-readable and round-trips.
    out = tmp_path / "verdict.json"
    out.write_text(v.model_dump_json(indent=2))
    assert Verdict.model_validate(json.loads(out.read_text())) == v


def test_no_effect_is_inconclusive_and_rolled_back():
    v, config = _run(effect=0.0, noise=0.02)
    assert v.state == "inconclusive"
    assert v.rollback == "rolled_back" and config["kv_cache_dtype"] == "auto"


def test_slowdown_is_regressed():
    v, config = _run(effect=-0.10)
    assert v.state == "regressed" and v.interval[1] < 0
    assert config["kv_cache_dtype"] == "auto"


def test_small_real_gain_rejects_the_claim():
    v, _ = _run(effect=0.01, noise=0.001)
    assert v.state == "improved"
    assert v.claim_rejected is True  # promised at least 3%, delivered ~1%


def test_correctness_failure_is_invalid_even_if_faster():
    v, config = _run(effect=0.10, correctness=0.90)
    assert v.state == "invalid" and v.correctness.status == "fail"
    assert "correctness gate failed" in v.reason
    # The speed result is still reported, just not used as a verdict.
    assert v.effect == pytest.approx(0.10, abs=0.02) and v.interval[0] > 0
    assert config["kv_cache_dtype"] == "auto"


def test_latency_gate_failure_is_regressed():
    v, _ = _run(effect=0.10, ttft_change=0.20)
    assert v.state == "regressed"
    assert v.latency[0].name == "ttft_ms_p95" and v.latency[0].status == "fail"


def test_sampler_crash_is_failed_to_execute_and_restores():
    v, config = _run(fail_at=3)
    assert v.state == "failed_to_execute" and "server went away" in v.reason
    assert config["kv_cache_dtype"] == "auto"


def test_budget_exhausted_is_inconclusive():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(budget__max_wall_clock_s=600))
    v, config = _run(c)
    assert v.state == "inconclusive" and "budget" in v.reason
    assert v.cost.wall_clock_s >= 600
    assert config["kv_cache_dtype"] == "auto"


def test_failed_control_blocks_the_verdict():
    contract = load_contract(FIXTURE)
    config = dict(contract.baseline.engine_args)
    v = evaluate(contract, DictApplicator(config), FakeServer(config), control_status="failed")
    assert v.state == "invalid" and v.rollback == "not_applied" and v.raw == []


def test_unpaired_estimator_agrees_on_a_clear_effect():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(protocol__pairing="unpaired",
                                                protocol__comparison_order="randomized",
                                                protocol__order_seed=7))
    v, _ = _run(c, effect=0.10)
    assert v.state == "improved" and "Welch" in v.method


def test_cost_is_accounted():
    contract = load_contract(FIXTURE)
    config = dict(contract.baseline.engine_args)
    v = evaluate(contract, DictApplicator(config), FakeServer(config),
                 usd_per_accelerator_s=0.001)
    n = len(v.raw)
    assert v.cost.wall_clock_s == pytest.approx(180.0 * n)
    assert v.cost.estimated_usd == pytest.approx(0.001 * 8 * 180.0 * n)


# --- review fixes (Greptile on #136) -----------------------------------------


def test_improved_run_ending_on_baseline_reapplies_candidate():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(protocol__repetitions=4))
    assert run_order(c)[-1] == "baseline"  # ABBA with even reps ends on baseline
    v, config = _run(c, effect=0.10)
    assert v.state == "improved"
    assert v.rollback == "kept" and config["kv_cache_dtype"] == "fp8"


class _CountingApplicator:
    """DictApplicator that fails its Nth apply call."""

    def __init__(self, config: dict, fail_apply_call: int):
        self.inner, self.calls, self.fail_at = DictApplicator(config), 0, fail_apply_call

    def snapshot(self):
        return self.inner.snapshot()

    def apply(self, spec):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("engine would not come back on the candidate")
        self.inner.apply(spec)

    def restore(self, snap):
        self.inner.restore(snap)


def test_failed_reapply_is_failed_to_execute_and_restores():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(protocol__repetitions=4))
    config = dict(c.baseline.engine_args)
    # ABBA x4 applies the candidate twice during the run; the 3rd apply is the re-apply.
    v = evaluate(c, _CountingApplicator(config, 3), FakeServer(config, effect=0.10))
    assert v.state == "failed_to_execute" and "re-applying" in v.reason
    assert v.rollback == "rolled_back" and config["kv_cache_dtype"] == "auto"


def test_zero_metric_is_invalid_and_restores():
    contract = load_contract(FIXTURE)
    config = dict(contract.baseline.engine_args)

    class ZeroServer(FakeServer):
        def sample(self, arm, index):
            s = super().sample(arm, index)
            s.metrics["output_tok_s"] = 0.0
            return s

    v = evaluate(contract, DictApplicator(config), ZeroServer(config))
    assert v.state == "invalid" and "non-positive" in v.reason
    assert config["kv_cache_dtype"] == "auto"


def test_decision_error_is_failed_to_execute_and_restores(monkeypatch):
    import sys
    ev = sys.modules["gitm.experiment.evaluate"]  # the package re-exports evaluate() over the module name

    def boom(*a, **k):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(ev, "_decide", boom)
    v, config = _run(effect=0.10)  # fixture ends on the candidate arm
    assert v.state == "failed_to_execute" and "decision failed" in v.reason
    assert v.rollback == "rolled_back" and config["kv_cache_dtype"] == "auto"


def test_dollar_budget_stops_the_run():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(budget__max_cost_usd=5.0))
    config = dict(c.baseline.engine_args)
    # 8 x 180 accelerator-s per sample at $0.001 = $1.44 per sample.
    v = evaluate(c, DictApplicator(config), FakeServer(config), usd_per_accelerator_s=0.001)
    assert v.state == "inconclusive" and "budget" in v.reason
    assert 5.0 <= v.cost.estimated_usd < 5.0 + 1.44 + 1e-9
    assert config["kv_cache_dtype"] == "auto"


def test_dollar_budget_without_a_rate_is_refused():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(budget__max_cost_usd=5.0))
    config = dict(c.baseline.engine_args)
    v = evaluate(c, DictApplicator(config), FakeServer(config))
    assert v.state == "invalid" and "max_cost_usd" in v.reason
    assert v.rollback == "not_applied" and v.raw == []


class _ReloadingApplicator(DictApplicator):
    """Each apply/restore costs `reload_s` on a fake clock, like a model reload."""

    def __init__(self, config, clock, reload_s):
        super().__init__(config)
        self.clock, self.reload_s = clock, reload_s

    def apply(self, spec):
        self.clock[0] += self.reload_s
        super().apply(spec)

    def restore(self, snap):
        self.clock[0] += self.reload_s
        super().restore(snap)


def test_setup_time_counts_toward_wall_clock_and_cost():
    contract = load_contract(FIXTURE)
    config = dict(contract.baseline.engine_args)
    now = [0.0]
    app = _ReloadingApplicator(config, now, reload_s=600.0)
    v = evaluate(contract, app, FakeServer(config), clock=lambda: now[0],
                 accelerators_held=8, usd_per_accelerator_s=0.001)
    n = len(v.raw)
    assert v.cost.setup_s > 0 and v.cost.setup_s % 600.0 == 0
    assert v.cost.wall_clock_s == pytest.approx(180.0 * n + v.cost.setup_s)
    assert v.cost.accelerator_s == pytest.approx(8 * 180.0 * n + 8 * v.cost.setup_s)
    assert v.cost.estimated_usd == pytest.approx(0.001 * v.cost.accelerator_s)


def test_setup_time_can_exhaust_the_wall_clock_budget():
    from gitm.experiment.contract import ExperimentContract
    c = ExperimentContract.model_validate(_edit(budget__max_wall_clock_s=3000))
    config = dict(c.baseline.engine_args)
    now = [0.0]
    v = evaluate(c, _ReloadingApplicator(config, now, reload_s=1500.0), FakeServer(config),
                 clock=lambda: now[0])
    # Sampler time alone (~2 x 180 s before the first switch) would not hit 3000 s.
    assert v.state == "inconclusive" and "budget" in v.reason
    assert config["kv_cache_dtype"] == "auto"


def test_unpaired_interval_ignores_arm_scale():
    from gitm.experiment.evaluate import effect_interval
    rng = random.Random(3)
    b = [1000.0 * (1 + rng.gauss(0, 0.01)) for _ in range(6)]
    c = [1000.0 * (1 + rng.gauss(0, 0.01)) for _ in range(6)]
    kw = dict(higher_is_better=True, paired=False, alpha=0.05)
    _, lo1, hi1, m1 = effect_interval(b, c, **kw)
    # Same relative noise, candidate 10x larger: the relative interval and df
    # must not change.
    _, lo2, hi2, m2 = effect_interval(b, [10 * x for x in c], **kw)
    df = lambda m: m.split("Welch df=")[1].split(",")[0]  # noqa: E731
    assert df(m1) == df(m2)
    assert (hi2 - lo2) / 10 == pytest.approx(hi1 - lo1)


def test_cli_malformed_yaml_does_not_stop_the_batch(capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("candidate_id: [unclosed\n")
    assert cli_main(["validate", str(bad), str(FIXTURE)]) == 1
    out = capsys.readouterr().out
    assert f"FAIL {bad}" in out and f"ok   {FIXTURE}" in out


# --- real contracts ------------------------------------------------------------


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[1] / "experiments" / "contracts").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_registered_contracts_validate(path):
    c = load_contract(path)
    assert c.sha256()
