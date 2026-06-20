"""AlphaFold2 (OpenFold) wired into the autonomous loop as an observed workload.

AF2 has no intervention library yet, so the loop must report an honest
*measurement* over the real kernels — never vLLM serving knobs and never a
fabricated intervention/speedup. These tests run without OpenFold/torch/GPU
(CI has none): they cover registration, clean no-data degradation when the
framework/data are absent, and the measurement-routing contract via an injected
runner + a faked trace.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .conftest import make_kernel, make_trace


def test_openfold_is_registered():
    from gitm.workloads import get_factory, registered

    names = set(registered())
    assert {"openfold", "alphafold", "af2"} <= names
    assert get_factory("openfold") is not None
    assert get_factory("alphafold") is not None


def test_openfold_no_deps_or_data_degrades_to_no_data(tmp_path: Path, monkeypatch):
    """No staged data (and/or no OpenFold) → honest no-data, not a crash or fake."""
    monkeypatch.setenv("GITM_BENCH_STAGE", str(tmp_path / "missing"))

    from gitm import optimize

    result = optimize(workload="openfold", budget="1s", scratch=str(tmp_path))
    summary = result["summary"]
    assert summary["status"] == "no_data"
    assert summary["n_claims"] == 0
    assert Path(summary["report_path"]).exists()


def _fake_capture_evoformer(out_path, *, workload_id="w", fingerprint="f", run_id=None):
    @contextmanager
    def _cap():
        # torch/cutlass-style kernel names — what a real AF2 trace looks like.
        kernels = [
            make_kernel(f"cutlass_sm90_evoformer_gemm_{i % 4}",
                        start_ns=i * 100, end_ns=i * 100 + 90 + (i % 9))
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return _cap()


def test_openfold_routes_to_measurement_not_intervention(tmp_path: Path, monkeypatch):
    """With real kernels in the trace, openfold reports a measurement — never a
    vLLM knob, and never the hft intervention path (no fabricated speedup)."""
    import gitm.scheduler.loop as loop

    monkeypatch.setattr(loop, "capture", _fake_capture_evoformer)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    result = optimize(
        workload="openfold", budget="1s", scratch=str(tmp_path),
        workload_runner=lambda: {"structures": 3, "events": 3, "median_plddt": 90.0},
    )
    summary, md = result["summary"], result["report_md"]

    assert summary["status"] == "ok"
    assert summary["mode"] == "measurement"  # not "intervention"
    assert summary["n_claims"] == 0
    for knob in ("max_num_batched_tokens", "gpu_memory_utilization", "max_num_seqs"):
        assert knob not in md, f"measurement report must not contain vLLM knob {knob!r}"
