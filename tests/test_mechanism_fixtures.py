"""The derivations in docs/mechanism_model.md, checked through the real monitor.

Every test builds two executions with ``gitm.optimizer.mechanism_fixtures`` and
compares what ``monitor.residuals`` / ``check_invariants`` (and, for the raw-trace
layer, ``node_rollup.device_comm_stats``) make of them. A confound passes when the
observations are equal; a separating condition passes when they differ in the way
the derivation says. Test names match the ones the document cites.
"""

from __future__ import annotations

import copy
import json

import pytest

from gitm.optimizer.deviation import classify_op
from gitm.optimizer.mechanism_fixtures import (
    SIDE_STREAM,
    AdditiveCost,
    Misroute,
    RegimeGate,
    RegionSlowdown,
    Scenario,
    Serialization,
    TracerOverhead,
    TruthModel,
    fit_affine,
    generate,
    observe,
    same_residuals,
    saturating,
    step,
)
from gitm.optimizer.monitor import _serialized_fraction, check_invariants
from gitm.optimizer.replay import _load_trace_jsonl
from gitm.planner.roofline import BatchConfig, HardwareSpec

ATTN = "attn_score_value"
ETA = 0.8
BASE_R = 1 / ETA - 1  # 0.25: a healthy kernel under matched truth, inside the ±0.4 band
KV_SWEEP = (1024, 2048, 4096, 16384)


def _one(**kw):
    fx = generate(Scenario(**kw))[0]
    return fx, observe(fx)


def _events(fx):
    return [(e.name, e.start_ns, e.end_ns, e.stream_id) for e in fx.trace.events]


def _attn_ns(kv=2048, truth=None):
    fx = generate(Scenario(points=(BatchConfig(batch=8, kv_cache_len=kv),),
                           truth=truth or TruthModel.matched(ETA)))[0]
    node = next(n for n in fx.graph.nodes if n.op == ATTN)
    return (truth or TruthModel.matched(ETA)).duration_ns(node, fx.scenario.hw, fx.point)


def _sweep(kvs=KV_SWEEP, **kw):
    """``(t_pred(x), observed attention duration)`` per operating point, both in seconds."""
    points = tuple(BatchConfig(batch=8, kv_cache_len=kv) for kv in kvs)
    t, d = [], []
    for fx in generate(Scenario(points=points, **kw)):
        t.append(next(n.prediction.t_pred_s for n in fx.graph.nodes if n.op == ATTN))
        d.append(observe(fx).op_duration_s(ATTN))
    return t, d


def _steps_changed(obs, n_layers, threshold=BASE_R + 1e-3):
    """Steps whose attention rows moved off the healthy baseline."""
    return {i // n_layers for i, r in enumerate(obs.op_series(ATTN)) if r > threshold}


# ── sanity ──────────────────────────────────────────────────────────────────


def test_baseline_pairs_every_modeled_kernel():
    fx, obs = _one(side_stream=True)
    compute = [e for e in fx.trace.events if e.stream_id != SIDE_STREAM]
    side = [e for e in fx.trace.events if e.stream_id == SIDE_STREAM]
    # Every compute kernel is paired; the NCCL kernels have no node in the dense
    # graph and are dropped, exactly as a real all-reduce would be.
    assert len(obs.residuals.per_kernel) == len(compute)
    assert len(side) == fx.graph.model.n_layers * fx.scenario.n_steps
    assert all(classify_op(e.name) == e.range_op for e in compute)
    assert all(classify_op(e.name) == "tp_all_reduce" for e in side)


def test_baseline_r_kt_equals_inverse_eta_minus_one():
    _, obs = _one()
    assert all(abs(k.r_kt - BASE_R) < 1e-4 for k in obs.residuals.per_kernel)
    assert not [v for v in obs.violations if v.invariant == "kernel_time"]
    # Lemma 1's consequence on a single stream: r_sc = 1, so stream_concurrency
    # fires on a healthy run.
    assert obs.residuals.serialized_concurrency_fraction == 1.0
    assert [v.invariant for v in obs.violations] == ["stream_concurrency"]


def test_g05_healthy_kernel_violates_at_eta_07():
    _, obs = _one(truth=TruthModel.matched(0.7))
    kt = [v for v in obs.violations if v.invariant == "kernel_time"]
    # Nothing injected: 1/0.7 - 1 = 0.43 is outside a ±0.4 band around the peak point.
    assert len(kt) == len(obs.residuals.per_kernel)
    assert {v.severity for v in kt} == {1.0}


def test_deterministic_under_seed():
    a, _ = _one(noise_cv=0.02, seed=1)
    b, _ = _one(noise_cv=0.02, seed=1)
    c, _ = _one(noise_cv=0.02, seed=2)
    assert _events(a) == _events(b)
    assert _events(a) != _events(c)


def test_jsonl_round_trip(tmp_path):
    fx, obs = _one(side_stream=True)
    out = fx.write(tmp_path / "run")
    loaded = _load_trace_jsonl(out / "trace.jsonl")
    fx2 = copy.copy(fx)
    fx2.trace = loaded
    assert same_residuals(obs, observe(fx2), atol=0.0)
    off = json.loads((out / "off" / "serving_summary.json").read_text())
    assert off["tracing"] == "off"
    assert off["server"]["tpot_mean_s"] == pytest.approx(fx.true_tpot_s)


def _check_m1(base, obs, fx):
    assert all(abs(r - (1.5 * (1 + BASE_R) - 1)) < 1e-4 for r in obs.op_series(ATTN))
    assert obs.op_series("mlp_down") == base.op_series("mlp_down")


def _check_m2(base, obs, fx):
    assert same_residuals(base, obs)
    assert base.exposed_comm_ns == 0
    assert obs.exposed_comm_ns > 0
    assert obs.step_ns[0] > base.step_ns[0]


def _check_m3k(base, obs, fx):
    t = next(n.prediction.t_pred_s for n in fx.graph.nodes if n.op == ATTN)
    assert all(abs(r - (BASE_R + 20e-6 / t)) < 1e-4 for r in obs.op_series(ATTN))


def _check_m3g(base, obs, fx):
    assert same_residuals(base, obs)
    assert obs.step_ns[0] > base.step_ns[0]


def _check_m4(base, obs, fx):
    assert _steps_changed(obs, fx.graph.model.n_layers) == {1, 3}


def _check_m5e(base, obs, fx):
    assert obs.step_ns[0] > base.step_ns[0]
    assert obs.true_tpot_s == base.true_tpot_s


def _check_m5m(base, obs, fx):
    assert obs.op_series("attn_out_proj") == []
    assert len(obs.op_series(ATTN)) == 2 * len(base.op_series(ATTN))
    assert obs.true_tpot_s == base.true_tpot_s


@pytest.mark.parametrize(
    ("kw", "check"),
    [
        (dict(mechanisms=(RegionSlowdown(0.5, ops={ATTN}),)), _check_m1),
        (dict(side_stream=True, mechanisms=(Serialization(),)), _check_m2),
        (dict(mechanisms=(AdditiveCost(20_000, ops={ATTN}),)), _check_m3k),
        (dict(mechanisms=(AdditiveCost(20_000, "gap", ops={ATTN}),)), _check_m3g),
        (dict(step_z=(0, 1, 0, 1),
              mechanisms=(RegimeGate(RegionSlowdown(0.5, ops={ATTN}), 0.5),)), _check_m4),
        (dict(observation=(TracerOverhead(2_000),)), _check_m5e),
        (dict(observation=(Misroute("attn_out_proj", ATTN),)), _check_m5m),
    ],
    ids=["M1-slowdown", "M2-serialization", "M3k-additive", "M3g-gap", "M4-gate",
         "M5-overhead", "M5-misroute"],
)
def test_each_mechanism_moves_its_observable(kw, check):
    base_kw = {k: v for k, v in kw.items() if k in ("side_stream", "step_z")}
    _, base = _one(**base_kw)
    fx, obs = _one(**kw)
    check(base, obs, fx)


# ── lemmas ──────────────────────────────────────────────────────────────────


def test_lemma1_r_sc_is_stream_sequence_only():
    fx, obs = _one(side_stream=True)
    # Halve every duration, keeping every start: the start-order stream sequence is
    # unchanged, and so is r_sc, whatever the durations did.
    shrunk = [e.model_copy(update={"end_ns": e.start_ns + (e.end_ns - e.start_ns) // 2})
              for e in fx.trace.events]
    assert _serialized_fraction(shrunk) == obs.residuals.serialized_concurrency_fraction
    single = [e for e in fx.trace.events if e.stream_id != SIDE_STREAM]
    assert _serialized_fraction(single) == 1.0


def test_lemma2_equal_residuals_equal_violations():
    delta = 0.5 * _attn_ns()
    _, a = _one(mechanisms=(RegionSlowdown(0.5, ops={ATTN}),))
    _, b = _one(mechanisms=(AdditiveCost(delta, ops={ATTN}),))
    assert same_residuals(a, b)
    assert a.violation_signature() == b.violation_signature()
    assert check_invariants(copy.deepcopy(a.residuals)) == a.violations


def test_lemma3_saturating_efficiency_is_additive():
    eta_max, b_half = 0.9, 2e6
    bw = HardwareSpec().peak_mem_bw_bytes_per_s
    a, _ = _one(truth=TruthModel(eta_m=saturating(eta_max, b_half)))
    b, _ = _one(truth=TruthModel(eta_m=eta_max, c0_ns=b_half / (bw * eta_max) * 1e9))
    for ea, eb in zip(a.trace.events, b.trace.events, strict=True):
        assert abs((ea.end_ns - ea.start_ns) - (eb.end_ns - eb.start_ns)) <= 1


# ── P1: region slowdown vs in-kernel additive cost ──────────────────────────


def test_p1_confound_single_operating_point():
    """P1's confounded pair: indistinguishable at one operating point."""
    delta = 0.5 * _attn_ns()
    fa, a = _one(mechanisms=(RegionSlowdown(0.5, ops={ATTN}),))
    fb, b = _one(mechanisms=(AdditiveCost(delta, ops={ATTN}),))
    assert _events(fa) == _events(fb)  # confounded at the raw trace, so everywhere above it
    assert same_residuals(a, b, atol=0.0)
    assert a.violation_signature() == b.violation_signature()
    assert a.true_tpot_s == b.true_tpot_s  # and at the untraced arm
    # The monitor does see *a* deviation, on attention only; it cannot say which.
    assert {v.node_op for v in a.violations if v.invariant == "kernel_time"} == {ATTN}


def test_p1_sweep_separates():
    """P1's separating condition: the same mechanisms across operating points."""
    delta = 0.5 * _attn_ns(2048)
    slow = RegionSlowdown(0.5, ops={ATTN})
    add = AdditiveCost(delta, ops={ATTN})

    ratio_slow, ratio_add = [], []
    for kv in KV_SWEEP:
        pt = (BatchConfig(batch=8, kv_cache_len=kv),)
        ratio_slow.append((1 + _one(points=pt, mechanisms=(slow,))[1].op_series(ATTN)[0])
                          / (1 + BASE_R))
        ratio_add.append((1 + _one(points=pt, mechanisms=(add,))[1].op_series(ATTN)[0])
                         / (1 + BASE_R))
    assert max(ratio_slow) - min(ratio_slow) < 1e-4  # 1.5 at every kv
    assert ratio_slow[0] == pytest.approx(1.5, abs=1e-4)
    assert max(ratio_add) - min(ratio_add) > 0.5  # 1 + δ/d(kv): large at small kv

    fit_slow = fit_affine(*_sweep(mechanisms=(slow,)))
    fit_add = fit_affine(*_sweep(mechanisms=(add,)))
    assert abs(fit_slow.b) < 10e-9
    assert fit_slow.a == pytest.approx(1.5 / ETA, rel=1e-4)
    assert fit_add.b == pytest.approx(delta / 1e9, rel=1e-3)
    assert fit_add.a == pytest.approx(1 / ETA, rel=1e-4)


def test_p1_reference_ratio_attributes_change():
    truth = TruthModel(eta_m=ETA, c0_ns={ATTN: 5_000})
    ref = fit_affine(*_sweep(truth=truth))
    slow = fit_affine(*_sweep(truth=truth, mechanisms=(RegionSlowdown(0.5, ops={ATTN}),)))
    add = fit_affine(*_sweep(truth=truth, mechanisms=(AdditiveCost(20_000, ops={ATTN}),)))
    # M1 scales both components; M3k moves only the intercept.
    assert slow.a / ref.a == pytest.approx(1.5, rel=1e-4)
    assert slow.b / ref.b == pytest.approx(1.5, rel=1e-3)
    assert add.a / ref.a == pytest.approx(1.0, rel=1e-4)
    assert add.b - ref.b == pytest.approx(20e-6, rel=1e-3)


def test_p1_mismatched_c0_enters_intercept():
    truth = TruthModel(eta_m=ETA, c0_ns={ATTN: 5_000})
    add = fit_affine(*_sweep(truth=truth, mechanisms=(AdditiveCost(20_000, ops={ATTN}),)))
    assert add.b == pytest.approx(25e-6, rel=1e-3)
    # A pure slowdown on top of a baseline fixed cost also has b > 0: "b > 0" says
    # the deviation has an additive part, not that the mechanism was additive.
    slow = fit_affine(*_sweep(truth=truth, mechanisms=(RegionSlowdown(0.5, ops={ATTN}),)))
    assert slow.b == pytest.approx(7.5e-6, rel=1e-3)


def test_p1_efficiency_step_flagged_by_lack_of_fit():
    variant_switch = TruthModel(eta_m=step(0.8, 0.6, kv_threshold=4096))
    bent = fit_affine(*_sweep(truth=variant_switch))
    straight = fit_affine(*_sweep())
    # A straight line keeps only the 1 ns timestamp quantization (~1e-5 relative at
    # the 82 µs point); a variant switch bends it by ~0.9.
    assert straight.curvature < 1e-4
    assert bent.curvature > 0.1
    assert abs(bent.b) > 1e-6  # the straight line through it has a spurious intercept
    two_points = fit_affine(*_sweep(kvs=(1024, 16384), truth=variant_switch))
    assert two_points.curvature is None  # any two points fit a line


def test_p1_traffic_excess_moves_slope_only():
    fit = fit_affine(*_sweep(truth=TruthModel(eta_m=ETA, tau={ATTN: 0.1})))
    assert fit.a == pytest.approx(1.1 / ETA, rel=1e-4)
    assert abs(fit.b) < 10e-9


# ── P2: serialization vs additive gap ───────────────────────────────────────


def _p2_pair():
    base, _ = _one(side_stream=True)
    side = next(e for e in base.trace.events if e.stream_id == SIDE_STREAM)
    # Serializing costs the side kernel's duration plus its own launch gap.
    delta = (side.end_ns - side.start_ns) + base.scenario.launch_gap_ns
    n_layers = base.graph.model.n_layers
    ser = _one(side_stream=True, mechanisms=(Serialization(),))
    gap = _one(side_stream=True, mechanisms=(
        AdditiveCost(delta, "gap", ops={"qkv_proj"}, layers=set(range(1, n_layers))),
        AdditiveCost(delta, "gap", ops={"lm_head"}),
    ))
    return ser, gap, side.end_ns - side.start_ns


def test_p2_confound_serialization_vs_gap():
    (fs, s), (fg, g), _ = _p2_pair()
    assert same_residuals(s, g, atol=0.0)  # r_kt and r_sc
    assert s.violation_signature() == g.violation_signature()
    assert s.step_ns == g.step_ns
    assert s.true_tpot_s == g.true_tpot_s


def test_p2_exposed_comm_separates():
    (fs, s), (fg, g), side_ns = _p2_pair()
    n_side = fs.graph.model.n_layers * fs.scenario.n_steps
    assert s.exposed_comm_ns == n_side * side_ns
    assert g.exposed_comm_ns == 0
    assert _events(fs) != _events(fg)


# ── P3: regime gate vs region slowdown ──────────────────────────────────────


def test_p3a_gate_always_on_equals_slowdown():
    slow = RegionSlowdown(0.5, ops={ATTN})
    fa, _ = _one(step_z=(5, 5, 5, 5), mechanisms=(RegimeGate(slow, 1.0),))
    fb, _ = _one(mechanisms=(slow,))
    assert _events(fa) == _events(fb)


def test_p3b_monotone_gate_equals_onset():
    slow = RegionSlowdown(0.5, ops={ATTN})
    fa, _ = _one(step_z=(1, 2, 3, 4), mechanisms=(RegimeGate(slow, 2.5),))
    fb, _ = _one(mechanisms=(RegionSlowdown(0.5, ops={ATTN}, steps=range(2, 4)),))
    assert _events(fa) == _events(fb)


def test_p3_shuffled_z_separates():
    z = (3, 1, 4, 2)
    fx, obs = _one(step_z=z, mechanisms=(RegimeGate(RegionSlowdown(0.5, ops={ATTN}), 2.5),))
    changed = _steps_changed(obs, fx.graph.model.n_layers)
    assert changed == {s for s, v in enumerate(z) if v > 2.5} == {0, 2}
    # No time onset reproduces it.
    for onset in range(len(z)):
        _, o = _one(mechanisms=(RegionSlowdown(0.5, ops={ATTN}, steps=range(onset, len(z))),))
        assert not same_residuals(obs, o)


# ── P4: additive cost vs tracer overhead ────────────────────────────────────


def test_p4_confound_overhead_vs_additive():
    fa, a = _one(mechanisms=(AdditiveCost(2_000),))
    fb, b = _one(observation=(TracerOverhead(2_000),))
    assert _events(fa) == _events(fb)
    assert same_residuals(a, b, atol=0.0)


def test_p4_off_arm_separates():
    _, base = _one()
    _, a = _one(mechanisms=(AdditiveCost(2_000),))
    _, b = _one(observation=(TracerOverhead(2_000),))
    assert a.true_tpot_s > base.true_tpot_s
    assert b.true_tpot_s == base.true_tpot_s
