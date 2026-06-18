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
