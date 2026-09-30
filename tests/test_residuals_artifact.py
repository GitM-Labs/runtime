"""residuals.json as a reader sees it, from a CPU loop run with a recorded trace.

Needs no GPU: the capture is replaced by a fixed trace and the loop runs
predict-only (no engine), which is also the path where catalog claims carry the
residual and the Granger status rather than a live A/B.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from gitm.optimizer.monitor import LIMITATIONS, UNCLASSIFIED

from .conftest import make_kernel, make_trace

FLASH = "_ZN5flash24flash_fwd_splitkv_kernel"  # classifies to attn_score_value
SILU = "silu_and_mul_kernel"  # classifies to mlp_gate_up
BARE_GEMM = "ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_stages_32x5_tn"  # no op


def _capture(names_and_counts: list[tuple[str, int, int]]):
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        rng = np.random.default_rng(0)
        kernels, t = [], 0
        for name, base_ns, count in names_and_counts:
            for _ in range(count):
                dur = int(base_ns * (1 + rng.normal(0, 0.2)))
                kernels.append(make_kernel(name, start_ns=t, end_ns=t + dur))
                t += dur + 100
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return fake_capture


def _run(tmp_path, monkeypatch, capture):
    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    monkeypatch.setattr(loop, "capture", capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    out = run_loop(LoopConfig(engine=None, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=5))
    return out, json.loads((Path(out["run_dir"]) / "residuals.json").read_text())


def test_residuals_json_records_status_coverage_and_limitations(tmp_path, monkeypatch):
    pytest.importorskip("statsmodels.tsa.stattools")
    out, residuals = _run(tmp_path, monkeypatch, _capture([
        (FLASH, 30_000, 40), (SILU, 8_000, 40), (BARE_GEMM, 20_000, 10),
    ]))

    assert residuals["schema_version"] == 1
    assert residuals["granger"]["status"] == "ok"
    assert residuals["limitations"] == list(LIMITATIONS)

    cov = residuals["coverage"]
    assert cov["status"] == "ok"
    assert (cov["n_kernels_seen"], cov["n_kernels_matched"], cov["n_kernels_dropped"]) == (90, 80, 10)
    assert cov["dropped_ops"] == {UNCLASSIFIED: 10}
    assert 0.0 < cov["matched_time_fraction"] < 1.0
    assert "lm_head" in cov["unobserved_predicted_ops"]
    assert cov["residual_method"] == "duration_weighted"
    assert isinstance(cov["kt_residual"], float)
    # Legacy keys unchanged for existing readers.
    assert residuals["n_kernel_residuals"] == 80
    assert "top_hypotheses_granger" in residuals


def test_run_with_no_matched_kernel_says_so_instead_of_zero(tmp_path, monkeypatch):
    out, residuals = _run(tmp_path, monkeypatch, _capture([(BARE_GEMM, 20_000, 30)]))

    cov = residuals["coverage"]
    assert cov["status"] == "no_matches"
    assert cov["kt_residual"] is None
    assert cov["residual_method"] == "none"
    assert cov["dropped_ops"] == {UNCLASSIFIED: 30}
    assert residuals["granger"]["status"] == "insufficient_data"

    rows = [line for line in out["report_md"].splitlines() if line.startswith("| 1 |")]
    assert rows, "expected catalog claims in a predict-only run"
    assert "`kernel_time`: no kernels matched the graph |" in rows[0]
    assert "+0.0%" not in rows[0]
