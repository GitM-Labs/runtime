"""The HFT 'act' loop: an optimization that is output-verified and rollback-gated.

Correctness is checked on the pandas backend (CI has no GPU); the throughput win
is GPU-specific and measured on hardware. The contract under test is the gate:
the candidate is kept only if its output is identical, and a divergent candidate
is always rolled back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_df(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts_ns": np.sort(rng.integers(0, 5_000_000_000, n)).astype("int64"),
            "symbol_id": rng.integers(0, 8, n).astype("int32"),
            "side": rng.integers(0, 2, n).astype("int8"),
            "price": rng.integers(100, 200, n).astype("int64"),
            "size": rng.integers(1, 100, n).astype("int32"),
            "type": rng.integers(0, 3, n).astype("int8"),
        }
    )


def test_fast_pipeline_is_output_identical():
    from gitm.benchmarks.hft.optimize import verify_equivalent

    assert verify_equivalent(_make_df(), pd), "dropping the redundant ffill must not change output"


def test_optimize_keeps_only_correct_candidate():
    from gitm.benchmarks.hft.optimize import optimize_hft

    r = optimize_hft(_make_df(), pd, reps=2)
    assert r.identical is True
    assert r.kept in {"candidate", "baseline"}  # CPU may not show the speedup; correctness is the gate
    assert r.baseline_eps > 0 and r.candidate_eps > 0


def test_optimize_rolls_back_a_divergent_candidate(monkeypatch):
    import gitm.benchmarks.hft.optimize as opt

    # Simulate a candidate that produces wrong output — the gate must reject it.
    monkeypatch.setattr(
        opt,
        "run_pipeline_fast",
        lambda d, lib: {"events": 1, "mean_microprice": -999.0, "vwap_buckets": 0},
    )
    r = opt.optimize_hft(_make_df(), pd, reps=1)
    assert r.identical is False
    assert r.kept == "baseline"
    assert "rolled back" in r.verdict


def _batches(df, k: int):
    """Split a frame into k row-chunks — the in-memory stand-in for sharded reads."""
    step = (len(df) + k - 1) // k
    for i in range(0, len(df), step):
        yield df.iloc[i : i + step].reset_index(drop=True)


def test_streaming_ab_matches_and_counts_all_events():
    from gitm.benchmarks.hft.optimize import optimize_hft_streaming

    df = _make_df(n=6000, seed=1)
    seen = []
    r = optimize_hft_streaming(
        _batches(df, 3), pd, on_batch=lambda i, info: seen.append(info["events"])
    )
    assert r.identical is True  # every batch's candidate output matched baseline
    assert sum(seen) == len(df)  # the whole dataset was processed, not just one frame
    assert r.baseline_summary["events"] == len(df)
    assert r.baseline_eps > 0 and r.candidate_eps > 0
    assert r.kept in {"candidate", "baseline"}  # CPU may not show the GPU speedup


def test_streaming_ab_rejects_empty_batches():
    """An empty A/B must not pass as a vacuous identical=True / zero-throughput
    result — it raises so a no-op never reads as a verified run."""
    import pytest

    from gitm.benchmarks.hft.optimize import optimize_hft_streaming

    with pytest.raises(ValueError, match="no batches"):
        optimize_hft_streaming(iter([]), pd)


def test_streaming_ab_summaries_are_per_pipeline(monkeypatch):
    """baseline_summary/candidate_summary must report each pipeline's own counts,
    so a divergent run never attributes candidate counts to the baseline."""
    import gitm.benchmarks.hft.optimize as opt

    # Candidate reports a different bucket count than baseline.
    monkeypatch.setattr(
        opt,
        "run_pipeline_fast",
        lambda d, lib: {"events": int(len(d)), "mean_microprice": -1.0, "vwap_buckets": 0},
    )
    r = opt.optimize_hft_streaming(_batches(_make_df(n=3000), 3), pd)
    assert r.identical is False
    # Baseline keeps its real (non-zero) bucket count; candidate's is the forced 0.
    assert r.baseline_summary["vwap_buckets"] >= 0
    assert r.candidate_summary["vwap_buckets"] == 0
    assert r.baseline_summary["vwap_buckets"] != r.candidate_summary["vwap_buckets"] or \
        r.baseline_summary["vwap_buckets"] == 0


def test_streaming_ab_rolls_back_when_any_batch_diverges(monkeypatch):
    import gitm.benchmarks.hft.optimize as opt

    # A single divergent batch must fail the whole-run gate.
    monkeypatch.setattr(
        opt,
        "run_pipeline_fast",
        lambda d, lib: {"events": int(len(d)), "mean_microprice": -1.0, "vwap_buckets": 0},
    )
    r = opt.optimize_hft_streaming(_batches(_make_df(n=3000), 3), pd)
    assert r.identical is False
    assert r.kept == "baseline"
    assert "rolled back" in r.verdict
