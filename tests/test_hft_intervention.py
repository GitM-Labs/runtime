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
import pytest

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
    # This encodes apply_intervention's contract: a CorrectnessError raised in
    # measure() is caught and converted to a rollback (not propagated). If that
    # contract changes, this call would raise instead of returning a result.
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


# --- the --seed/--stage/--max-events/--stream run flags ----------------------


def test_cli_run_rejects_hft_flags_on_non_hft_workload():
    """The hft data-selection flags are meaningless on other workloads → error,
    don't silently ignore."""
    from gitm.cli import main

    with pytest.raises(SystemExit, match="workload hft only"):
        main(["run", "--workload", "vllm-decode", "--stream"])


def test_cli_run_hft_stream_end_to_end(tmp_path: Path, monkeypatch):
    """`gitm run --workload hft --stream …` drives the streaming apply+prove path
    end-to-end: CLI flags → GITM_BENCH_* env → factory → HftStreamingApplicator."""
    monkeypatch.setenv("GITM_BENCH_EVENTS", "3000")  # tiny autogen smoke dataset

    from gitm.cli import main

    report = tmp_path / "report.md"
    rc = main([
        "run", "--workload", "hft", "--stream", "--shards-per-batch", "1",
        "--stage", str(tmp_path / "stage"), "--scratch", str(tmp_path / "scratch"),
        "--report", str(report),
    ])
    assert rc == 0
    md = report.read_text()
    assert "hft_top_of_book_fewer_scans" in md  # the streaming A/B ran and was claimed


def test_streaming_factory_builds_streaming_applicator(tmp_path: Path, monkeypatch):
    """The factory wires a streaming applicator that verifies over batches."""
    from gitm.benchmarks.hft.optimize import HftStreamingApplicator, hft_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    monkeypatch.setenv("GITM_BENCH_STAGE", str(tmp_path / "stage"))
    monkeypatch.setenv("GITM_BENCH_SEED", "42")
    monkeypatch.setenv("GITM_BENCH_EVENTS", "3000")
    monkeypatch.setenv("GITM_BENCH_STREAM", "1")
    monkeypatch.setenv("GITM_BENCH_SHARDS_PER_BATCH", "1")

    runner = get_factory("hft")(LoopConfig(workload="hft"))
    assert isinstance(runner.applicator, HftStreamingApplicator)
    assert runner()["events"] > 0  # observe pass streams the data

    res = apply_intervention(hft_intervention_spec(), runner.applicator, min_keep_delta=0.0)
    assert runner.applicator.last_result.identical is True  # verified over all batches
    assert res.measured_delta is not None
