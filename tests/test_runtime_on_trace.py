"""Tests for the runtime monitor + attribution upgrades.

Covers the four genuine gaps: real stream-concurrency, the multi-basis filter,
the doubly-robust estimator, and the replay validation harness.
"""

from __future__ import annotations

import numpy as np
import pytest


def _kernel(name, start, end, stream=7):
    from gitm.tracer.schema import KernelEvent

    return KernelEvent(kind="kernel", start_ns=start, end_ns=end, stream_id=stream,
                       device_id=0, name=name)


def _trace(events):
    from gitm.tracer.schema import Trace

    dur = max((e.end_ns for e in events), default=0)
    return Trace(workload_id="t", fingerprint="f", run_id="r", device_count=1,
                 vendor="nvidia", captured_at_ns=0, duration_ns=dur, events=events)


# --- stream concurrency (was hardcoded 0.0) ---------------------------------


def test_serialized_fraction_sequential_vs_overlapped():
    from gitm.optimizer.monitor import _serialized_fraction

    # back-to-back on one stream -> fully serialized
    seq = [_kernel("k", i * 100, i * 100 + 100, stream=7) for i in range(6)]
    assert _serialized_fraction(seq) == pytest.approx(1.0)

    # heavily overlapping on different streams -> not serialized
    over = [_kernel("k", 0, 1000, stream=s) for s in range(6)]
    assert _serialized_fraction(over) == pytest.approx(0.0)


def test_residuals_compute_real_concurrency():
    from gitm.optimizer.monitor import residuals
    from gitm.planner.graph import predict_graph

    trace = _trace([_kernel("k", i * 100, i * 100 + 100) for i in range(8)])
    res = residuals(trace, predict_graph())
    assert res.serialized_concurrency_fraction == pytest.approx(1.0)  # all sequential, one stream


# --- multi-basis filter ------------------------------------------------------


def test_multibasis_confirms_spike_filters_noise():
    from gitm.optimizer.multibasis import multibasis_anomalies

    rng = np.random.default_rng(0)
    x = list(rng.normal(0, 0.05, 40))
    x[20] = 3.0  # a clear transient spike
    mask = multibasis_anomalies(x)
    assert mask[20]
    assert mask.sum() <= 2  # no flood of false positives


def test_multibasis_short_series_uses_position_basis():
    from gitm.optimizer.multibasis import multibasis_anomalies

    x = [0.0, 0.0, 5.0, 0.0]  # too short for the frequency basis
    mask = multibasis_anomalies(x)
    assert mask[2] and mask.sum() == 1


def test_check_invariants_multibasis_suppresses_single_basis_blip():
    from gitm.optimizer.monitor import KernelResidual, Residuals, check_invariants

    rng = np.random.default_rng(1)
    res = Residuals()
    # one op, mostly-zero residuals (within band 0.4) + a couple isolated spikes
    vals = list(rng.normal(0, 0.02, 30))
    vals[15] = 1.5  # transient, multi-basis-confirmable
    for v in vals:
        res.per_kernel.append(KernelResidual(op="attn", layer=0, r_kt=v, r_mt=None))

    kept = check_invariants(res, multi_basis=True)
    raw = check_invariants(res, multi_basis=False)
    kt_kept = [v for v in kept if v.invariant == "kernel_time"]
    kt_raw = [v for v in raw if v.invariant == "kernel_time"]
    assert len(kt_kept) <= len(kt_raw)  # filter never adds
    assert any(abs(v.residual - 1.5) < 1e-6 for v in kt_kept)  # the real spike survives


def test_check_invariants_keeps_systematic_shift():
    from gitm.optimizer.monitor import KernelResidual, Residuals, check_invariants

    res = Residuals()
    for _ in range(20):  # whole op systematically 60% slow (> 0.4 band), no outlier
        res.per_kernel.append(KernelResidual(op="mlp", layer=1, r_kt=0.6, r_mt=None))
    kt = [v for v in check_invariants(res, multi_basis=True) if v.invariant == "kernel_time"]
    assert kt, "systematic shift must still be flagged under multi-basis"


# --- doubly-robust estimator -------------------------------------------------


def test_doubly_robust_recovers_ate_under_confounding():
    from gitm.optimizer.dr import doubly_robust_ate

    rng = np.random.default_rng(2)
    n = 500
    X = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-X))).astype(float)  # confounded
    y = 0.5 * t + 0.3 * X + rng.normal(0, 0.1, n)  # true ATE = 0.5
    ate, se = doubly_robust_ate(y, t, X)
    assert abs(ate - 0.5) < 0.1
    assert se < 0.1


def test_doubly_robust_degenerate_inputs():
    from gitm.optimizer.dr import doubly_robust_ate

    y = np.array([1.0, 2.0, 3.0])
    t = np.zeros(3)  # no treated units
    ate, se = doubly_robust_ate(y, t, np.arange(3))
    assert ate == 0.0 and se == float("inf")


def test_attribute_dr_ranks_pairs():
    from gitm.optimizer.dr import attribute_dr
    from gitm.optimizer.monitor import KernelResidual, Residuals
    from gitm.planner.graph import predict_graph

    rng = np.random.default_rng(3)
    res = Residuals()
    # cause "A" anomalous drives effect "B" up
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0  # A spikes periodically (treatment)
    b = 0.8 * a + rng.normal(0, 0.05, 40)
    for i in range(40):
        res.per_kernel.append(KernelResidual(op="A", layer=None, r_kt=float(a[i]), r_mt=None))
        res.per_kernel.append(KernelResidual(op="B", layer=None, r_kt=float(b[i]), r_mt=None))
    ranked = attribute_dr(res, predict_graph())
    assert ranked.hypotheses
    top = ranked.top(1)[0]
    assert "doubly-robust ATE" in top.notes


# --- replay validation harness ----------------------------------------------


def test_replay_validation_within_tolerance():
    from gitm.optimizer.replay_validation import validate

    result = validate(n=200, seed=7)
    assert result.passed
    assert result.mean_abs_rel_err <= 0.20
    assert result.frac_within_tol > 0.7
