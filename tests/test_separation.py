"""The P1 separating experiment's decision rule, run on synthetic traces.

Each test injects a known mechanism with ``gitm.optimizer.mechanism_fixtures``,
reduces every window with the same ``window_stat`` the hardware analysis uses, and
checks that ``decide`` names the mechanism or abstains. The baseline carries a
5 µs fixed cost, so a test on the roofline intercept (``b > 0``) would be fooled
here; the rule, which regresses candidate on baseline, is not.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from gitm.optimizer import separation as sep
from gitm.optimizer.mechanism_fixtures import (
    AdditiveCost,
    RegionSlowdown,
    Scenario,
    TracerOverhead,
    TruthModel,
    generate,
    step,
)
from gitm.planner.roofline import BatchConfig

ATTN = "attn_score_value"
KVS = (2048, 4096, 8192, 16384, 32768)
BASE_TRUTH = TruthModel(eta_m=0.8, c0_ns={ATTN: 5_000})
ALL_CONTROL_OK = {f"L{kv}": 1.0 for kv in KVS}


def _sweep(*, mechanisms=(), observation=(), truth=BASE_TRUTH, reps=3, noise=0.01,
           seed0=0, kvs=KVS):
    out: dict[str, list[float]] = {f"L{kv}": [] for kv in kvs}
    points = tuple(BatchConfig(batch=8, kv_cache_len=kv) for kv in kvs)
    for r in range(reps):
        for fx in generate(Scenario(points=points, truth=truth, mechanisms=mechanisms,
                                    observation=observation, noise_cv=noise,
                                    seed=seed0 + r)):
            w = sep.window_stat(sep.kernels_from_trace(fx.trace))
            out[f"L{fx.point.kv_cache_len}"].append(w.launch_s)
    return out


@pytest.fixture(scope="module")
def base():
    return _sweep()


@pytest.fixture(scope="module")
def base_repeat():
    return _sweep(seed0=100)


def _decide(base, cand, base_repeat, **kw):
    return sep.decide(base, cand, base_repeat=base_repeat, control_ratio=ALL_CONTROL_OK,
                      **kw)


# ── the per-window statistic ────────────────────────────────────────────────


def test_window_groups_anchor_with_its_reduce_and_keeps_modal_grid():
    def k(name, start, dur, grid=(64, 1, 1)):
        return sep.Kernel(name, start, start + dur, 7, 0, grid)

    ks = [
        # Three full-batch launches: stage-1 split-KV plus its reduce.
        k("mla_decode_stage1", 0, 100), k("mla_decode_reduce", 110, 20), k("gemm", 140, 50),
        k("mla_decode_stage1", 200, 100), k("mla_decode_reduce", 310, 20), k("gemm", 340, 50),
        k("mla_decode_stage1", 400, 100), k("mla_decode_reduce", 510, 20),
        # One launch at a smaller batch (another grid): dropped.
        k("mla_decode_stage1", 600, 40, grid=(16, 1, 1)), k("mla_decode_reduce", 650, 20),
        # Cache insert and a projection are not attention core.
        k("reshape_and_cache_flash_kernel", 700, 5), k("attn_out_proj_gemm", 710, 30),
    ]
    w = sep.window_stat(ks)
    assert w.anchor == "mla_decode_stage1"
    assert w.targets == ["mla_decode_reduce", "mla_decode_stage1"]
    assert w.launches == 3
    assert w.launch_s == pytest.approx(120e-9)


def test_fixture_window_resolves_attention_only():
    fx = generate(Scenario(points=(BatchConfig(batch=8, kv_cache_len=4096),)))[0]
    w = sep.window_stat(sep.kernels_from_trace(fx.trace))
    assert w.targets == [f"fixture_{ATTN}_kernel"]
    node = next(n for n in fx.graph.nodes if n.op == ATTN)
    assert w.launch_s == pytest.approx(node.prediction.t_pred_s / 0.8, rel=1e-4)


# ── one outcome per mechanism ───────────────────────────────────────────────


def test_region_speedup_is_multiplicative(base, base_repeat):
    cand = _sweep(mechanisms=(RegionSlowdown(-0.4, ops={ATTN}),))
    d = _decide(base, cand, base_repeat)
    assert d.outcome == "multiplicative", d.reasons
    assert d.k == pytest.approx(0.6, abs=0.01)
    assert abs(d.m_s) < d.mu_s


def test_fixed_cost_is_additive(base, base_repeat):
    cand = _sweep(mechanisms=(AdditiveCost(10_000, ops={ATTN}),))
    d = _decide(base, cand, base_repeat)
    assert d.outcome == "additive", d.reasons
    assert d.k == pytest.approx(1.0, abs=0.01)
    assert d.m_s == pytest.approx(10e-6, rel=0.1)


def test_both_is_mixed(base, base_repeat):
    cand = _sweep(mechanisms=(RegionSlowdown(-0.4, ops={ATTN}), AdditiveCost(10_000, ops={ATTN})))
    d = _decide(base, cand, base_repeat)
    assert d.outcome == "mixed", d.reasons
    assert d.k == pytest.approx(0.6, abs=0.01)
    assert d.m_s == pytest.approx(10e-6, rel=0.15)


def test_nothing_is_no_effect(base, base_repeat):
    d = _decide(base, _sweep(seed0=200), base_repeat)
    assert d.outcome == "no_effect", d.reasons


# ── abstentions ─────────────────────────────────────────────────────────────


def test_effect_at_the_margin_abstains(base, base_repeat):
    # k sits on 1 − κ with 3% noise: the interval straddles the margin.
    cand = _sweep(mechanisms=(RegionSlowdown(-sep.KAPPA, ops={ATTN}),), noise=0.03, seed0=300)
    d = _decide(base, cand, base_repeat)
    assert d.outcome == "inconclusive"
    assert any("straddles" in r for r in d.reasons)


def test_variant_switch_fails_lack_of_fit(base, base_repeat):
    # The candidate's efficiency drops only above kv 8192: no single (k, m).
    cand = _sweep(truth=TruthModel(eta_m=step(0.8, 0.5, kv_threshold=8192),
                                   c0_ns={ATTN: 5_000}))
    d = _decide(base, cand, base_repeat)
    assert d.outcome == "inconclusive"
    assert d.curvature > sep.CURVATURE_MAX
    assert any("lack of fit" in r for r in d.reasons)


def test_one_operating_point_cannot_separate(base, base_repeat):
    one = {"L8192": base["L8192"]}
    cand = _sweep(mechanisms=(RegionSlowdown(-0.4, ops={ATTN}),), kvs=(8192,))
    d = _decide(one, cand, base_repeat)
    assert d.outcome == "inconclusive"
    assert d.k is None  # P1: nothing to fit


def test_drift_abstains(base):
    drifted = {p: [v * 1.05 for v in vals] for p, vals in base.items()}
    d = _decide(base, _sweep(seed0=200), drifted)
    assert d.outcome == "inconclusive"
    assert not d.gates["drift"]


def test_control_op_moving_abstains(base, base_repeat):
    cand = _sweep(mechanisms=(RegionSlowdown(-0.4, ops={ATTN}),))
    d = sep.decide(base, cand, base_repeat=base_repeat,
                   control_ratio={**ALL_CONTROL_OK, "L32768": 1.04})
    assert d.outcome == "inconclusive"
    assert not d.gates["control"]


def test_noisy_reps_abstain(base, base_repeat):
    d = _decide(base, _sweep(noise=0.3, seed0=400, reps=2), base_repeat)
    assert d.outcome == "inconclusive"


# ── the invalidating effect the margin is sized for ─────────────────────────


def test_tracer_inflation_leaks_into_intercept_within_margin():
    # Both arms traced with ε per kernel: under a pure slowdown the intercept
    # becomes (1 − k)·ε, which the additive margin must absorb.
    eps = TracerOverhead(int(sep.EPS_MAX_S * 1e9))
    b = _sweep(observation=(eps,))
    r = _sweep(observation=(eps,), seed0=100)
    c = _sweep(observation=(eps,), mechanisms=(RegionSlowdown(-0.5, ops={ATTN}),))
    d = _decide(b, c, r)
    assert d.m_s == pytest.approx(0.5 * sep.EPS_MAX_S, rel=0.3)
    assert d.outcome == "multiplicative", d.reasons


# ── the run-directory contract with scripts/kimi_loop/p1_separation.sh ──────


def _write_run(root, monkeypatch):
    monkeypatch.setattr(sep, "MIN_LAUNCHES", 1)
    monkeypatch.setattr(sep, "CONTROL_OP", "qkv_proj")  # fixtures have no norm kernel
    points = tuple(BatchConfig(batch=8, kv_cache_len=kv) for kv in KVS)
    arms = {"base1": ((), 0), "cand": ((RegionSlowdown(-0.4, ops={ATTN}),), 0),
            "base2": ((), 100)}
    for phase, (mech, seed0) in arms.items():
        (root / "p1" / "traced" / phase).mkdir(parents=True)
        (root / "p1" / "traced" / phase / "sanity.json").write_text(
            json.dumps({"failures": [], "outputs": []}))
        for r in range(3):
            for fx in generate(Scenario(points=points, truth=BASE_TRUTH, mechanisms=mech,
                                        noise_cv=0.01, seed=seed0 + r)):
                d = root / "p1" / "traced" / phase / f"L{fx.point.kv_cache_len}_r{r + 1}"
                fx.write(d / "cap")
                (d / "guidellm.json").write_text(_itl(30.0, 40.0))
                (d / "metrics.prom").write_text(
                    "### ts_ns=0\nvllm:num_requests_running 16\nvllm:num_preemptions_total 0\n"
                    "### ts_ns=1\nvllm:num_requests_running 16\nvllm:num_preemptions_total 0\n")


def _itl(p50, p95):
    return json.dumps({"benchmarks": [{"metrics": {"inter_token_latency_ms": {
        "successful": {"median": p50, "percentiles": {"p95": p95}}}}}]})


def test_analyze_reads_the_runner_layout(tmp_path, monkeypatch):
    _write_run(tmp_path, monkeypatch)
    report = sep.analyze(tmp_path, n_layers=61, concurrency=16)
    assert report["decision"]["outcome"] == "multiplicative", report["decision"]["reasons"]
    assert report["targets"]["cand"] == [f"fixture_{ATTN}_kernel"]
    json.dumps(report, default=list)


def test_analyze_drops_bad_windows_and_gates_the_run(tmp_path, monkeypatch):
    _write_run(tmp_path, monkeypatch)
    cand = tmp_path / "p1" / "traced" / "cand"
    (cand / "L8192_r2" / "guidellm.json").write_text(_itl(30.0, 120.0))  # latency gate
    (cand / "L2048_r1" / "metrics.prom").write_text(
        "vllm:num_requests_running 9\nvllm:num_preemptions_total 0\n"
        "vllm:num_requests_running 9\nvllm:num_preemptions_total 3\n")  # load gates
    (cand / "L4096_r3" / "guidellm.json").unlink()  # a failed load generator
    report = sep.analyze(tmp_path, n_layers=61, concurrency=16)
    dropped = " | ".join(report["dropped_windows"])
    for needle in ("ITL p95/p50", "median running", "preemptions", "no guidellm.json"):
        assert needle in dropped, dropped
    # Three windows dropped, every point still has ≥ 2 reps: the verdict stands.
    assert report["decision"]["outcome"] == "multiplicative", report["decision"]["reasons"]

    # A failed correctness check is run-level: no verdict, and a remedy.
    (cand / "sanity.json").write_text(json.dumps({"failures": ["prompt 3"], "outputs": []}))
    dec = sep.analyze(tmp_path, n_layers=61, concurrency=16)["decision"]
    assert dec["outcome"] == "inconclusive"
    assert any("correctness" in r for r in dec["reasons"])
    assert dec["remedies"]


def test_anchor_is_the_longest_kernel_even_when_counts_tie():
    # The window opens between a stage-1 and its reduce, so the reduce is seen
    # once more than stage-1. The anchor must still be stage-1.
    def k(name, start, dur):
        return sep.Kernel(name, start, start + dur, 7, 0, (64, 1, 1))

    ks = [k("mla_decode_reduce", 0, 3)]
    for i in range(1, 50):
        ks += [k("mla_decode_stage1", 100 * i, 40), k("mla_decode_reduce", 100 * i + 45, 3)]
    w = sep.window_stat(ks)
    assert w.anchor == "mla_decode_stage1"
    assert w.launch_s == pytest.approx(43e-9)
    assert w.kernels_per_launch == 2


def test_candidate_only_kernel_is_reported(tmp_path, monkeypatch):
    # An extra kernel only in the candidate (a dequantize step, say) is outside
    # the op by construction; it must still surface in the report.
    _write_run(tmp_path, monkeypatch)
    for d in (tmp_path / "p1" / "traced" / "cand").glob("L*_r*"):
        trace = d / "cap" / "trace.jsonl"
        lines = trace.read_text().splitlines()
        extra = json.loads(lines[1])
        extra.update(name="kv_dequant_fp8_kernel", start_ns=10**12, end_ns=10**12 + 5000,
                     range_op=None, range_layer=None)
        trace.write_text("\n".join([*lines, json.dumps(extra)]) + "\n")
    diff = sep.analyze(tmp_path, n_layers=61, concurrency=16)["kernel_diff"]
    top = {row["kernel"]: row for row in diff["L2048"]}
    assert top["kv_dequant_fp8_kernel"]["only_in"] == "cand"


# ── calibration at the hardware's scale (docs §6: 5–76 µs roofline) ─────────

REAL = np.array([5.01, 9.73, 19.17, 38.04, 75.79]) * 1e-6 / 0.7 + 1e-6
LABELS = [f"L{i}" for i in range(5)]


def _real(truth, cv, rng, n=3):
    return {p: list(truth[i] * (1 + cv * rng.standard_normal(n))) for i, p in enumerate(LABELS)}


def _real_decide(k, m, cv, rng):
    return sep.decide(_real(REAL, cv, rng), _real(k * REAL + m, cv, rng),
                      base_repeat=_real(REAL, cv, rng),
                      control_ratio=dict.fromkeys(LABELS, 1.0), kernels_per_launch=(2, 2),
                      draws=1000)


def test_small_fixed_cost_on_top_of_a_speedup_is_mixed_at_real_scale():
    # The case the fp8 prediction names: k ≈ 0.55 plus a 1.5 µs dequantize cost,
    # 30% of the shortest launch. It must not be read as purely multiplicative.
    rng = np.random.default_rng(7)
    outcomes = Counter(_real_decide(0.55, 1.5e-6, 0.01, rng).outcome for _ in range(40))
    assert outcomes["mixed"] >= 36, outcomes
    assert outcomes["multiplicative"] == 0, outcomes


def test_linear_truth_rarely_fails_the_fit_or_drift_gates():
    rng = np.random.default_rng(8)
    fails = Counter()
    for _ in range(100):
        d = _real_decide(0.55, 0.0, 0.02, rng)
        fails.update(g for g in ("fit", "drift") if not d.gates.get(g, True))
    assert fails["fit"] <= 3 and fails["drift"] <= 12, fails


def test_intervals_cover_the_truth():
    rng = np.random.default_rng(9)
    hits_k = hits_m = 0
    runs = 200
    for _ in range(runs):
        d = _real_decide(0.55, 1.0e-6, 0.01, rng)
        hits_k += d.k_ci[0] <= 0.55 <= d.k_ci[1]
        hits_m += d.m_ci_s[0] <= 1.0e-6 <= d.m_ci_s[1]
    assert hits_k / runs >= 0.9 and hits_m / runs >= 0.9, (hits_k, hits_m)


def test_noisy_control_op_does_not_void_the_run():
    # rms_norm itself has 2% rep-to-rep noise and is truly unchanged; the control
    # gate must allow for that noise rather than read it as drift.
    rng = np.random.default_rng(10)
    fails = 0
    for _ in range(60):
        cb, cc = (_real(np.full(5, 3e-6), 0.02, rng) for _ in range(2))
        ratio = {p: np.mean(cc[p]) / np.mean(cb[p]) for p in LABELS}
        se = dict.fromkeys(LABELS, (0.02**2 / 3 * 2) ** 0.5)
        d = sep.decide(_real(REAL, 0.01, rng), _real(0.55 * REAL, 0.01, rng),
                       base_repeat=_real(REAL, 0.01, rng), control_ratio=ratio,
                       control_se=se, kernels_per_launch=(2, 2), draws=1000)
        fails += not d.gates["control"]
    assert fails <= 6, fails


def test_tracer_leak_uses_each_arms_kernel_count():
    # Baseline launches 1 kernel, candidate 2; both traced with ε per kernel.
    # A pure k = 0.9 leaks (2 − 0.9·1)·ε into m, which the margin must absorb.
    eps = sep.EPS_MAX_S
    rng = np.random.default_rng(11)
    base = _real(REAL + eps, 0.002, rng)
    d = sep.decide(base, _real(0.9 * REAL + 2 * eps, 0.002, rng),
                   base_repeat=_real(REAL + eps, 0.002, rng),
                   control_ratio=dict.fromkeys(LABELS, 1.0), kernels_per_launch=(1, 2))
    assert d.tracer_leak_s == pytest.approx(1.1 * eps, rel=0.01)
    assert d.outcome == "multiplicative", d.reasons
