"""The doubly-robust estimator reports what its fits did.

``dr.py`` used to silence every warning and replace failed nuisance fits with
means, then return a normal-looking ATE and p-value. With step position as the
only covariate, anomalies that cluster in time separate the propensity model: it
fails to converge, propensities hit 0 and 1 and are clipped. These tests pin
that such pairs are counted as degraded, and that "nothing to estimate" has a
status of its own rather than an empty list.
"""

from __future__ import annotations

import numpy as np
import pytest

from gitm.optimizer.attribution import AnalysisStatus
from gitm.optimizer.monitor import KernelResidual, Residuals

pytest.importorskip("statsmodels.api")

from gitm.optimizer.dr import attribute_dr, doubly_robust_ate  # noqa: E402


def _residuals(series: dict[str, np.ndarray]) -> Residuals:
    res = Residuals()
    n = len(next(iter(series.values())))
    for i in range(n):
        for op, vals in series.items():
            res.per_kernel.append(KernelResidual(op=op, layer=None, r_kt=float(vals[i]), r_mt=None))
    return res


def test_separated_propensity_is_recorded_not_silenced():
    n = 60
    pos = np.arange(n, dtype=float)
    t = (pos >= 30).astype(float)  # anomalies only in the second half
    diag: dict = {}
    doubly_robust_ate(np.random.default_rng(0).normal(size=n), t, pos, diag=diag)
    assert diag["warnings"] or diag["fallbacks"]


def test_propensity_fit_error_is_a_recorded_fallback(monkeypatch):
    import statsmodels.api as sm

    class Failing:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(sm, "Logit", Failing)
    n = 40
    t = np.zeros(n)
    t[::5] = 1.0
    diag: dict = {}
    doubly_robust_ate(np.random.default_rng(0).normal(size=n), t, np.arange(n, dtype=float), diag=diag)
    assert [kind for kind, _ in diag["fallbacks"]] == ["propensity:LinAlgError"]


def test_too_few_treated_rows_for_the_outcome_model_is_a_recorded_fallback():
    # 3 treated rows pass the degenerate-input guard but cannot fit an outcome
    # model with a constant and 2 covariates (3 parameters).
    n = 20
    t = np.zeros(n)
    t[[3, 9, 15]] = 1.0
    X = np.column_stack([np.arange(n, dtype=float), np.random.default_rng(1).normal(size=n)])
    diag: dict = {}
    doubly_robust_ate(np.random.default_rng(0).normal(size=n), t, X, diag=diag)
    assert "outcome_treated:too_few_rows" in [kind for kind, _ in diag["fallbacks"]]


def test_degenerate_input_still_fills_diag():
    diag: dict = {}
    doubly_robust_ate(np.ones(3), np.zeros(3), np.arange(3, dtype=float), diag=diag)
    assert diag == {"fallbacks": [], "warnings": []}


def test_degraded_fits_are_estimates_on_record_not_a_failure():
    rng = np.random.default_rng(1)
    n = 60
    cause = rng.normal(0, 0.05, n)
    cause[30:] = 1.0  # out of band only late in the run: position separates it
    effect = 0.8 * cause + rng.normal(0, 0.05, n)
    ranked = attribute_dr(_residuals({"attn": cause, "mlp": effect}), None)

    # Both ops are out of band late in the run, so both directions are tried
    # and both propensity fits separate.
    # Both produced an estimate, so the run did not fail; both are degraded.
    assert ranked.pairs_attempted == 2
    assert (ranked.pairs_completed, ranked.pairs_degraded) == (2, 2)
    assert ranked.status is AnalysisStatus.DEGRADED
    # Which warning or error statsmodels raises for separation varies by
    # release (CI installs the newest); the pair must be degraded either way.
    assert ranked.warnings or ranked.failures
    assert all("degraded" in h.notes and h.degraded for h in ranked.hypotheses)


def test_clean_fits_complete():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0  # periodic anomalies: not separable by position
    b = 0.8 * a + rng.normal(0, 0.05, 40)
    ranked = attribute_dr(_residuals({"A": a, "B": b}), None)
    assert ranked.pairs_attempted >= 1
    assert ranked.status in (AnalysisStatus.OK, AnalysisStatus.PARTIAL)
    assert ranked.pairs_completed >= 1


def test_no_anomalies_is_insufficient_data_with_a_reason():
    quiet = np.full(40, 0.01)
    ranked = attribute_dr(_residuals({"a": quiet, "b": quiet}), None)
    assert ranked.status is AnalysisStatus.INSUFFICIENT_DATA
    assert ranked.hypotheses == []
    assert "out-of-band" in ranked.reason
    assert ranked.summary()["reason"] == ranked.reason


def test_too_few_ops_is_insufficient_data():
    ranked = attribute_dr(_residuals({"a": np.ones(10)}), None)
    assert ranked.status is AnalysisStatus.INSUFFICIENT_DATA
    assert "fewer than 2 ops" in ranked.reason


def test_measurement_families_below_the_threshold_are_on_record():
    from gitm.optimizer.measure import measure_trace
    from gitm.tracer.schema import KernelEvent, Trace

    events, t = [], 0
    for i in range(40):
        for name, base in (("gemm_kernel_a", 1000 + 37 * (i % 7)), ("softmax_kernel", 400 + 11 * (i % 5))):
            events.append(KernelEvent(name=name, start_ns=t, end_ns=t + base, stream_id=0, device_id=0))
            t += base + 10
    for i in range(10):
        events.append(KernelEvent(name="layernorm_kernel", start_ns=t, end_ns=t + 300 + i,
                                  stream_id=0, device_id=0))
        t += 310
    trace = Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                  captured_at_ns=0, duration_ns=t, events=events)

    granger = measure_trace(trace).granger
    small = {fam: n for fam, n in granger.excluded_ops.items() if "layernorm" in fam}
    assert list(small.values()) == [10]
    assert granger.min_obs == 16
    assert granger.series_lengths[next(iter(small))] == 10


def test_measurement_with_only_small_families_says_why():
    from gitm.optimizer.attribution import granger_evidence
    from gitm.optimizer.measure import measure_trace
    from gitm.tracer.schema import KernelEvent, Trace

    events = [KernelEvent(name="softmax_kernel", start_ns=i * 500, end_ns=i * 500 + 400 + i,
                          stream_id=0, device_id=0) for i in range(10)]
    trace = Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                  captured_at_ns=0, duration_ns=10_000, events=events)
    granger = measure_trace(trace).granger
    assert granger.status is AnalysisStatus.INSUFFICIENT_DATA
    assert granger.reason == "no kernel family had 16 samples"
    assert granger_evidence(granger) == "Granger not run: 0 ops had ≥ 16 samples, need 2"


def test_an_op_with_nan_residuals_is_left_out_on_record():
    # As a cause, abs(nan) > band is False: the NaN would silently read as an
    # in-band sample. As an effect, it poisons the estimate.
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0
    b = 0.8 * a + rng.normal(0, 0.05, 40)
    b[::5] = 1.0
    c = 0.5 * a + rng.normal(0, 0.05, 40)
    b[7] = np.nan
    ranked = attribute_dr(_residuals({"A": a, "B": b, "C": c}), None)
    assert ranked.failures["nonfinite_series"] == 1
    assert ranked.first_errors["nonfinite_series"] == "B: 1 non-finite residual(s)"
    assert ranked.excluded_ops["B"] == 40
    assert ranked.hypotheses
    assert all("B" not in (h.cause_op, h.effect_op) for h in ranked.hypotheses)
    assert all(np.isfinite(h.p_value) for h in ranked.hypotheses)


def test_a_rejected_estimate_keeps_its_fit_diagnostics(monkeypatch):
    import gitm.optimizer.dr as dr

    def warns_then_nan(y, t, X, *, diag=None):
        diag["fallbacks"] = [("propensity:LinAlgError", "Singular matrix")]
        diag["warnings"] = [(RuntimeWarning, "overflow encountered in exp")]
        return float("nan"), 1.0

    monkeypatch.setattr(dr, "doubly_robust_ate", warns_then_nan)
    a = np.zeros(40)
    a[::5] = 1.0
    ranked = attribute_dr(_residuals({"A": a, "B": a.copy()}), None)
    assert ranked.failures["nonfinite_estimate"] == 2
    assert ranked.failures["propensity:LinAlgError"] == 2
    assert ranked.warnings == {"RuntimeWarning": 2}


def test_one_pair_raising_keeps_the_other_pairs(monkeypatch):
    import gitm.optimizer.dr as dr

    real = dr.doubly_robust_ate
    calls = []

    def second_call_raises(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(dr, "doubly_robust_ate", second_call_raises)
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0
    b = 0.8 * a + rng.normal(0, 0.05, 40)
    ranked = attribute_dr(_residuals({"A": a, "B": b}), None)
    assert ranked.pairs_attempted == 2
    assert ranked.pairs_completed == 1
    assert ranked.failures == {"MemoryError": 1}
    assert ranked.status is AnalysisStatus.PARTIAL


def test_clean_estimates_rank_above_degraded_ones():
    from gitm.optimizer.attribution import Hypothesis, RankedHypotheses

    ranked = RankedHypotheses(hypotheses=[
        Hypothesis("a", "b", 0.0, "+ slower", degraded=True),
        Hypothesis("c", "d", 0.04, "+ slower"),
    ])
    ranked.rank(lambda h: h.p_value)
    assert [(h.cause_op, h.degraded) for h in ranked.hypotheses] == [("c", False), ("a", True)]


def test_statsmodels_that_will_not_import_is_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "statsmodels.api", None)  # import raises ImportError
    a = np.zeros(40)
    a[::5] = 1.0
    ranked = attribute_dr(_residuals({"A": a, "B": a.copy()}), None)
    assert ranked.status is AnalysisStatus.UNAVAILABLE
    assert list(ranked.failures) in (["ImportError"], ["ModuleNotFoundError"])


def test_measurement_reason_states_the_threshold_that_applied():
    from gitm.optimizer.measure import measure_trace
    from gitm.tracer.schema import KernelEvent, Trace

    # One family with 20 samples, one with 10: attribute() sees one op and
    # would say "at least 8"; the measurement threshold is 16.
    events, t = [], 0
    for i in range(20):
        events.append(KernelEvent(name="gemm_kernel_a", start_ns=t, end_ns=t + 1000 + 37 * (i % 7),
                                  stream_id=0, device_id=0))
        t += 1100
    for i in range(10):
        events.append(KernelEvent(name="softmax_kernel", start_ns=t, end_ns=t + 400 + i,
                                  stream_id=0, device_id=0))
        t += 500
    granger = measure_trace(Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1,
                                  vendor="nvidia", captured_at_ns=0, duration_ns=t,
                                  events=events)).granger
    assert granger.status is AnalysisStatus.INSUFFICIENT_DATA
    assert granger.reason == "fewer than 2 kernel families with at least 16 samples"


def test_degraded_pairs_are_flagged_in_the_published_record():
    rng = np.random.default_rng(1)
    cause = rng.normal(0, 0.05, 60)
    cause[30:] = 1.0
    effect = 0.8 * cause + rng.normal(0, 0.05, 60)
    top = attribute_dr(_residuals({"attn": cause, "mlp": effect}), None).summary()["top_pairs"]
    assert top and all(p["degraded"] for p in top)


def test_ops_without_enough_anomalies_are_recorded_as_skipped_causes():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0  # 8 anomalies: tried as a cause
    b = rng.normal(0, 0.05, 40)  # none: never a cause
    ranked = attribute_dr(_residuals({"A": a, "B": b}), None)
    assert ranked.pairs_attempted == 1
    assert ranked.skipped_causes == {"B": 0}
    assert ranked.summary()["skipped_causes"] == {"B": 0}
