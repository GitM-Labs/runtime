"""HFT wired into the autonomous loop as a real, rollback-gated intervention.

The loop's HFT path must observe → attribute → select → apply → prove with a
*measured* delta (not a measurement-only report), and never keep a candidate
whose output diverges. Correctness is exercised on the pandas backend (CI has no
GPU); the throughput win itself is GPU-specific.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .conftest import make_kernel, make_trace


def _make_df(n: int = 5000, seed: int = 0):
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


# --- the applicator (apply gate) ---------------------------------------------


def test_applicator_keeps_identical_and_reports_a_real_delta():
    from gitm.benchmarks.hft.optimize import HftFewerScansApplicator, hft_intervention_spec
    from gitm.optimizer.apply import apply_intervention

    app = HftFewerScansApplicator(_make_df(), pd, reps=1)
    res = apply_intervention(hft_intervention_spec(), app, min_keep_delta=0.0)

    # The A/B actually ran and verified the candidate is byte-identical.
    assert app.last_result is not None
    assert app.last_result.identical is True
    # A measured delta is always produced (kept if faster, rolled back if slower —
    # on tiny CPU frames either is fine; correctness is the hard gate).
    assert res.measured_delta is not None
    assert res.applied is True


def test_applicator_rolls_back_a_divergent_candidate(monkeypatch):
    import gitm.benchmarks.hft.optimize as opt
    from gitm.optimizer.apply import apply_intervention

    # A candidate that produces wrong output must never be kept.
    monkeypatch.setattr(
        opt,
        "run_pipeline_fast",
        lambda d, lib: {"events": 1, "mean_microprice": -999.0, "vwap_buckets": 0},
    )
    app = opt.HftFewerScansApplicator(_make_df(n=2000), pd, reps=1)
    res = apply_intervention(opt.hft_intervention_spec(), app, min_keep_delta=0.0)

    assert app.last_result.identical is False
    assert res.rolled_back is True
    assert res.measured_delta is None  # measure raised → no false speedup
    assert app.active == "baseline"  # state restored


# --- the full loop -----------------------------------------------------------


def _fake_capture_cudf(out_path, *, workload_id="w", fingerprint="f", run_id=None):
    @contextmanager
    def _cap():
        kernels = [
            make_kernel(f"cudf_groupby_scan_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 90 + (i % 7))
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return _cap()


def test_loop_runs_hft_intervention_with_measured_delta(tmp_path: Path, monkeypatch):
    """A runner carrying an applicator drives the apply+prove path end-to-end."""
    import gitm.scheduler.loop as loop
    from gitm.benchmarks.hft.optimize import HftFewerScansApplicator

    monkeypatch.setattr(loop, "capture", _fake_capture_cudf)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    df = _make_df(6000)

    def runner():
        return {"events": len(df)}

    runner.applicator = HftFewerScansApplicator(df, pd, reps=1)

    from gitm import optimize

    result = optimize(
        workload="hft", budget="1s", scratch=str(tmp_path), workload_runner=runner
    )
    summary, md = result["summary"], result["report_md"]

    assert summary["status"] == "ok"
    assert summary["mode"] == "intervention"
    assert summary["n_claims"] == 1
    assert summary["speedup"] is not None  # the A/B produced a real ratio
    # Provenance artifacts written.
    run_dir = Path(result["run_dir"])
    assert (run_dir / "apply_result.json").exists()
    assert (run_dir / "ranked_candidates.json").exists()
    # It's an HFT claim, not a vLLM knob.
    assert "scan" in md.lower()
    for knob in ("max_num_batched_tokens", "gpu_memory_utilization", "max_num_seqs"):
        assert knob not in md


def test_loop_hft_without_applicator_stays_measurement(tmp_path: Path, monkeypatch):
    """A bare runner (no applicator, e.g. fakes/older callers) is unchanged:
    measurement-only, never fabricated intervention claims."""
    import gitm.scheduler.loop as loop

    monkeypatch.setattr(loop, "capture", _fake_capture_cudf)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    result = optimize(
        workload="hft", budget="1s", scratch=str(tmp_path), workload_runner=lambda: {"events": 1}
    )
    assert result["summary"]["mode"] == "measurement"
    assert result["summary"]["n_claims"] == 0
