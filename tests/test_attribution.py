"""Granger attribution reports what it ran.

An empty hypothesis list used to be the only output, whether statsmodels was
missing, rejected its arguments, or the data were too short, and claims printed
it as "no strong causal signal". These tests pin one status per outcome and the
claim text for each.
"""

from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pytest

from gitm.optimizer.attribution import (
    AnalysisStatus,
    RankedHypotheses,
    attribute,
    granger_evidence,
)
from gitm.optimizer.monitor import KernelResidual, Residuals

stattools = pytest.importorskip("statsmodels.tsa.stattools")

# An op-pair claim such as "mlp→attn (p=0.02)"; live A/B text ("baseline 1 → candidate 2")
# also contains an arrow, so the pattern requires the p-value.
_PAIR_TEXT = re.compile(r"\S+\s*(→|->)\s*\S+\s*\(p=")


def _residuals(series: dict[str, np.ndarray]) -> Residuals:
    res = Residuals()
    n = max(len(v) for v in series.values())
    for i in range(n):
        for op, vals in series.items():
            if i < len(vals):
                res.per_kernel.append(KernelResidual(op=op, layer=None, r_kt=float(vals[i]), r_mt=None))
    return res


def _planted(n: int = 200) -> Residuals:
    rng = np.random.default_rng(0)
    cause = rng.normal(0, 0.1, n)
    effect = np.r_[0.0, 0.9 * cause[:-1]] + rng.normal(0, 0.02, n)
    return _residuals({"attn": cause, "mlp": effect, "lm_head": rng.normal(0, 0.1, n)})


def test_real_statsmodels_completes_every_pair_and_finds_the_planted_link():
    # Fails before the fix on statsmodels >= 0.15: every pair raised TypeError on
    # verbose=, was swallowed, and the result was an empty list.
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.OK
    assert hyps.pairs_attempted == 6
    assert hyps.pairs_completed == hyps.pairs_attempted
    assert hyps.failures == {}
    top = hyps.top(1)[0]
    assert (top.cause_op, top.effect_op) == ("attn", "mlp")
    assert top.p_value < 1e-6


def test_missing_statsmodels_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", None)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.UNAVAILABLE
    assert "ModuleNotFoundError" in hyps.first_errors
    assert granger_evidence(hyps).startswith("Granger unavailable: ")
    assert "pip install" in granger_evidence(hyps)


def test_one_op_is_insufficient_data():
    hyps = attribute(_residuals({"attn": np.zeros(50)}), None)
    assert hyps.status is AnalysisStatus.INSUFFICIENT_DATA
    assert hyps.pairs_attempted == 0


def test_eligibility_boundary_at_3_lags_plus_2():
    # statsmodels needs more than 3*max_lag + 1 observations; 7 always raises at lag 2.
    rng = np.random.default_rng(1)
    short = attribute(_residuals({"a": rng.normal(size=7), "b": rng.normal(size=7)}), None, max_lag=2)
    assert short.status is AnalysisStatus.INSUFFICIENT_DATA
    assert short.excluded_ops == {"a": 7, "b": 7}
    assert short.min_obs == 8

    enough = attribute(_residuals({"a": rng.normal(size=8), "b": rng.normal(size=8)}), None, max_lag=2)
    assert enough.status is AnalysisStatus.OK
    assert enough.n_obs_used == 8


def test_series_lengths_and_truncation_are_recorded():
    rng = np.random.default_rng(2)
    hyps = attribute(_residuals({"a": rng.normal(size=8), "b": rng.normal(size=100)}), None)
    assert hyps.series_lengths == {"a": 8, "b": 100}
    assert hyps.n_obs_used == 8


def test_all_pairs_raising_typeerror_is_failed_with_the_message(monkeypatch):
    # The statsmodels 0.15 regression itself is the real-statsmodels test above; this one
    # checks that whatever the library raises reaches the claim text.
    def broken(*args, **kwargs):
        raise TypeError("sentinel message 7731")

    monkeypatch.setattr(stattools, "grangercausalitytests", broken)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.FAILED
    assert hyps.failures == {"TypeError": 6}
    assert hyps.first_errors == {"TypeError": "sentinel message 7731"}
    text = granger_evidence(hyps)
    assert text.startswith("Granger failed: 0/6 op pairs completed (TypeError: 6).")
    assert "Most common error: TypeError: sentinel message 7731." in text
    assert "statsmodels API change" in text


def test_degenerate_failures_get_a_data_hint(monkeypatch):
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(stattools, "grangercausalitytests", singular)
    text = granger_evidence(attribute(_planted(), None))
    assert "degenerate, constant or too-short series" in text
    assert "statsmodels API change" not in text


def test_some_pairs_failing_is_partial(monkeypatch):
    real = stattools.grangercausalitytests
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(stattools, "grangercausalitytests", flaky)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.PARTIAL
    assert (hyps.pairs_completed, hyps.pairs_attempted) == (4, 6)
    assert granger_evidence(hyps).endswith("; 2 of 6 op pairs failed (ValueError: 2)")


def test_nan_at_a_later_lag_is_a_failure_not_a_result(monkeypatch):
    # min([0.01, nan]) is 0.01, so checking only the minimum would keep this pair.
    def nan_at_lag_2(*args, **kwargs):
        return {1: ({"ssr_ftest": (1.0, 0.01)},), 2: ({"ssr_ftest": (1.0, float("nan"))},)}

    monkeypatch.setattr(stattools, "grangercausalitytests", nan_at_lag_2)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.FAILED
    assert hyps.failures == {"nan_pvalue": 6}
    assert hyps.hypotheses == []


def test_warnings_are_counted_with_their_first_message(monkeypatch):
    real = stattools.grangercausalitytests

    def noisy(*args, **kwargs):
        warnings.warn("divide by zero encountered", RuntimeWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(stattools, "grangercausalitytests", noisy)
    hyps = attribute(_planted(), None)
    assert hyps.warnings.get("RuntimeWarning") == 6
    assert hyps.first_warnings["RuntimeWarning"] == "divide by zero encountered"


def test_verbose_false_is_passed_only_where_accepted(monkeypatch):
    # statsmodels < 0.15: verbose exists, prints unless False, and warns that it
    # is deprecated. The warning is ours, so it is not counted against the data.
    real = stattools.grangercausalitytests
    seen = []

    def old_api(x, maxlag, verbose=True):
        seen.append(verbose)
        warnings.warn("verbose is deprecated since functions should not print results",
                      FutureWarning, stacklevel=2)
        return real(x, maxlag=maxlag)

    monkeypatch.setattr(stattools, "grangercausalitytests", old_api)
    hyps = attribute(_planted(), None)
    assert set(seen) == {False}
    assert "FutureWarning" not in hyps.warnings
    assert hyps.status is AnalysisStatus.OK


def test_first_error_drops_absolute_paths(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("cannot open /Users/someone/secret/dir/file.npz for reading")

    monkeypatch.setattr(stattools, "grangercausalitytests", broken)
    message = attribute(_planted(), None).first_errors["OSError"]
    assert "/Users" not in message
    assert "file.npz" in message


def test_summary_keeps_pairs_marked_exploratory():
    summary = attribute(_planted(), None).summary()
    assert summary["status"] == "ok"
    assert summary["top_pairs"][0]["cause"] == "attn"
    assert all(pair["exploratory"] is True for pair in summary["top_pairs"])


@pytest.mark.parametrize("status", list(AnalysisStatus))
def test_every_status_has_claim_text_without_an_op_pair(status):
    hyps = RankedHypotheses(hypotheses=[], status=status, pairs_attempted=6, pairs_completed=6)
    if status is AnalysisStatus.PARTIAL:
        hyps.pairs_completed, hyps.failures = 4, {"ValueError": 2}
    text = granger_evidence(hyps)
    assert text
    assert "no strong causal signal" not in text
    assert not _PAIR_TEXT.search(text)


def test_ok_claim_text_names_the_artifact_and_no_pair():
    hyps = attribute(_planted(), None)
    text = granger_evidence(hyps, pairs_in="measurement.json")
    assert text == (
        "Granger ran (6/6 op pairs) but is not used as evidence: series are ordered "
        "by launch, not step; pairs in measurement.json"
    )


def test_unknown_status_raises():
    hyps = RankedHypotheses(hypotheses=[])
    hyps.status = "mystery"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no evidence text"):
        granger_evidence(hyps)


def test_measurement_claims_carry_the_failure(monkeypatch):
    from gitm.optimizer.measure import measure_trace, measurement_claims
    from gitm.tracer.schema import KernelEvent, Trace

    def broken(*args, **kwargs):
        raise TypeError("unexpected keyword argument 'verbose'")

    monkeypatch.setattr(stattools, "grangercausalitytests", broken)
    rng = np.random.default_rng(3)
    events, t = [], 0
    for _ in range(64):
        for name, base in (("gemm_kernel_a", 1000), ("softmax_kernel", 400)):
            dur = int(base * (1 + rng.normal(0, 0.3)))
            events.append(KernelEvent(start_ns=t, end_ns=t + dur, stream_id=7, device_id=0, name=name))
            t += dur + 10
    trace = Trace(workload_id="t", fingerprint="f", run_id="r", device_count=1, vendor="nvidia",
                  captured_at_ns=0, duration_ns=t, events=events)
    result = measure_trace(trace)
    claims = measurement_claims(result)
    assert result.granger.status is AnalysisStatus.FAILED
    assert claims
    for claim in claims:
        assert claim.causal_evidence.startswith("Granger failed: 0/")
        assert "no ranked hypothesis" not in claim.causal_evidence


def test_a_broken_statsmodels_install_is_unavailable_not_a_crash(monkeypatch):
    # A binary mismatch raises ValueError at import, not ImportError; the loop
    # calls attribute() unguarded, so it must not propagate.
    import importlib.abc

    for name in [m for m in sys.modules if m.startswith("statsmodels.tsa")]:
        monkeypatch.delitem(sys.modules, name)

    class Broken(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.startswith("statsmodels.tsa"):
                raise ValueError("numpy.dtype size changed, may indicate binary incompatibility")

    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.UNAVAILABLE
    assert hyps.first_errors == {"ValueError": "numpy.dtype size changed, may indicate binary incompatibility"}


def test_granger_prints_nothing(capsys):
    # statsmodels < 0.15 prints every test table unless told not to, and 0.15
    # removed the argument that told it.
    attribute(_planted(), None)
    assert capsys.readouterr().out == ""


def test_first_error_drops_home_and_install_paths_but_not_relative_text(monkeypatch):
    from pathlib import Path

    from gitm.optimizer.attribution import _clean_message

    home = str(Path.home())
    assert _clean_message(f"cannot open {home}/My Projects/run/t.npz") == "cannot open ~/My Projects/run/t.npz"
    if sys.prefix not in ("/usr", "/usr/local"):  # a system root is not named <python>
        assert _clean_message(f"in {sys.prefix}/lib/statsmodels/api.py") == "in <python>/lib/statsmodels/api.py"
    assert _clean_message("failed at /opt/data/x/trace.npz") == "failed at trace.npz"
    assert _clean_message("ratio a/b/c too small") == "ratio a/b/c too small"


def test_message_cleaner_on_a_system_python_without_a_home(monkeypatch):
    # The production image runs a distro python3 (sys.prefix "/usr"), and
    # arbitrary-uid containers have no home directory.
    import pathlib

    from gitm.optimizer import attribution

    def no_home(cls):
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(no_home))
    attribution._private_prefixes.cache_clear()
    try:
        clean = attribution._clean_message
        assert clean("OSError: /usr/local/cuda/lib64/libcudart.so.12 missing") == (
            "OSError: libcudart.so.12 missing"  # not "<python>/local/cuda/…"
        )
        assert clean("see https://github.com/x/y/issues/1") == "see https://github.com/x/y/issues/1"
    finally:
        attribution._private_prefixes.cache_clear()


def test_home_is_replaced_only_as_a_whole_path():
    from pathlib import Path

    from gitm.optimizer.attribution import _clean_message

    home = str(Path.home())
    assert _clean_message(f"at {home}2/other/f.txt") == "at f.txt"  # not "~2/…"
    assert _clean_message(f"at {home}") == "at ~"


def test_a_pair_that_warned_is_degraded_but_an_api_warning_is_not(monkeypatch):
    real = stattools.grangercausalitytests

    def numerically_noisy(*args, **kwargs):
        warnings.warn("divide by zero encountered", RuntimeWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(stattools, "grangercausalitytests", numerically_noisy)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.DEGRADED
    assert hyps.pairs_degraded == hyps.pairs_completed == 6
    assert all(p["degraded"] for p in hyps.summary()["top_pairs"])
    assert "6 raised fit warnings (RuntimeWarning: 6)" in granger_evidence(hyps)

    def deprecation_only(*args, **kwargs):
        warnings.warn("some_arg is deprecated", DeprecationWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr(stattools, "grangercausalitytests", deprecation_only)
    hyps = attribute(_planted(), None)
    assert hyps.status is AnalysisStatus.OK
    assert hyps.warnings == {"DeprecationWarning": 6}
    assert hyps.pairs_degraded == 0


def test_failed_text_shows_the_error_the_hint_is_about(monkeypatch):
    hyps = RankedHypotheses(hypotheses=[], status=AnalysisStatus.FAILED, pairs_attempted=12)
    hyps.record_failure("LinAlgError", "Singular matrix")
    for _ in range(11):
        hyps.record_failure("TypeError", "unexpected keyword argument")
    text = granger_evidence(hyps)
    assert "Most common error: TypeError: unexpected keyword argument." in text
    assert "statsmodels API change" in text


def test_unavailable_text_points_at_the_pinned_statsmodels():
    hyps = RankedHypotheses(hypotheses=[], status=AnalysisStatus.UNAVAILABLE)
    hyps.record_failure("ImportError", "No module named 'statsmodels'")
    assert granger_evidence(hyps).endswith("pip install -c constraints.txt statsmodels)")


def test_message_cleaner_handles_brackets_and_file_uris():
    from pathlib import Path

    from gitm.optimizer.attribution import _clean_message

    home = str(Path.home())
    assert _clean_message(f"paths [{home}] and {home};") == "paths [~] and ~;"
    assert _clean_message(f"at file://{home}/a/b.py") == "at ~/a/b.py"
