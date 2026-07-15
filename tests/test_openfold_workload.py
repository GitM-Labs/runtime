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


def test_openfold_routes_to_measurement_not_intervention(tmp_path: Path, monkeypatch):
    """With real kernels in the trace, openfold reports a measurement — never a
    vLLM knob, and never the hft intervention path (no fabricated speedup)."""
    import gitm.scheduler.loop as loop

    entered = {"capture": False, "runner": False}

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        entered["capture"] = True
        kernels = [
            make_kernel(f"cutlass_sm90_evoformer_gemm_{i % 4}",
                        start_ns=i * 100, end_ns=i * 100 + 90 + (i % 9))
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    def runner():
        entered["runner"] = True
        return {"structures": 3, "median_plddt": 90.0}

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    from gitm import optimize

    result = optimize(
        workload="openfold", budget="1s", scratch=str(tmp_path), workload_runner=runner
    )
    summary, md = result["summary"], result["report_md"]

    # Guard against a vacuous pass: the workload actually ran under the trace.
    assert entered["capture"] and entered["runner"]
    assert summary["status"] == "ok"
    assert summary["mode"] == "measurement"  # not "intervention"
    assert summary["n_claims"] == 0
    for knob in ("max_num_batched_tokens", "gpu_memory_utilization", "max_num_seqs"):
        assert knob not in md, f"measurement report must not contain vLLM knob {knob!r}"


def _fake_capture_with(entered, prefix="cutlass_sm90_gemm"):
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        entered["capture"] = True
        kernels = [
            make_kernel(f"{prefix}_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 90 + (i % 9))
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return fake_capture


def test_openfold_loop_runs_intervention_with_applicator(tmp_path: Path, monkeypatch):
    """A runner carrying the bf16 applicator drives the FULL apply+prove path:
    `gitm run --workload openfold` does observe→attribute→select→apply→prove and
    emits a verified provenance claim (the no-corners loop, like HFT)."""
    import gitm.scheduler.loop as loop
    from benchmarks.biotech.optimize import AF2ABResult

    entered: dict = {}
    monkeypatch.setattr(loop, "capture", _fake_capture_with(entered))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    class FakeApplicator:
        def __init__(self):
            self.active = "baseline"
            self.last_result = None

        def snapshot(self):
            return self.active

        def apply(self, spec):
            self.active = "candidate"

        def restore(self, s):
            self.active = s

        def measure(self, spec):  # bf16 faster, plDDT within tolerance
            self.last_result = AF2ABResult(
                400.0, 600.0, 1.5, 89.0, 90.0, 1.0, 1.5, equivalent=True, kept="candidate"
            )
            return 0.5

    def runner():
        return {"structures": 3}

    runner.applicator = FakeApplicator()

    from gitm import optimize

    result = optimize(
        workload="openfold", budget="1s", scratch=str(tmp_path), workload_runner=runner
    )
    s, md = result["summary"], result["report_md"]

    assert entered.get("capture")
    assert s["mode"] == "intervention"
    assert s["n_claims"] == 1
    assert s["speedup"] == 1.5
    assert s["kept"] == "candidate"
    assert "af2_bf16_inference" in md
    for knob in ("max_num_batched_tokens", "gpu_memory_utilization", "max_num_seqs"):
        assert knob not in md
    # The claim's residual is a serialized-concurrency fraction, not a kernel-time
    # ratio — it must be labeled as such, not mislabeled "kernel_time".
    assert "`stream_concurrency`" in md
    assert "`kernel_time`" not in md
    assert (Path(result["run_dir"]) / "apply_result.json").exists()
    # The live bf16 apply is recorded on the durable safety trail.
    from gitm.safety import AuditLog

    audit_path = Path(result["run_dir"]) / "audit.jsonl"
    assert audit_path.exists()
    assert "apply" in [e.event for e in AuditLog(audit_path).entries()]


def test_openfold_loop_rolls_back_on_plddt_regression(tmp_path: Path, monkeypatch):
    """If bf16 moves plDDT past tolerance the applicator raises; the loop rolls
    back to fp32 and lists it rolled-back — no speedup reported on degraded
    structures."""
    import gitm.scheduler.loop as loop
    from benchmarks.biotech.optimize import AF2ABResult, CorrectnessError

    entered: dict = {}
    monkeypatch.setattr(loop, "capture", _fake_capture_with(entered))
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    class FakeApplicator:
        def __init__(self):
            self.active = "baseline"
            self.last_result = None

        def snapshot(self):
            return self.active

        def apply(self, spec):
            self.active = "candidate"

        def restore(self, s):
            self.active = s

        def measure(self, spec):  # plDDT regressed → gate must roll back
            self.last_result = AF2ABResult(
                400.0, 600.0, 1.5, 89.0, 80.0, -9.0, 1.5, equivalent=False, kept="baseline"
            )
            raise CorrectnessError("bf16 degraded plDDT")

    def runner():
        return {"structures": 3}

    runner.applicator = FakeApplicator()

    from gitm import optimize

    result = optimize(
        workload="openfold", budget="1s", scratch=str(tmp_path), workload_runner=runner
    )
    s = result["summary"]
    assert s["mode"] == "intervention"
    assert s["n_rolled_back"] == 1
    # The trail shows the bf16 apply and its rollback, in order.
    from gitm.safety import AuditLog

    events = [e.event for e in AuditLog(Path(result["run_dir"]) / "audit.jsonl").entries()]
    assert events == ["apply", "revert"]


def test_openfold_factory_filters_by_len_and_msa(tmp_path: Path, monkeypatch):
    """Protein selection (len + MSA filter, the 'no proteins' guard) is exercised
    without OpenFold/torch — it runs before the model is built. Here max_len=0
    excludes every protein, so the factory raises the clear no-proteins error."""
    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    stage = tmp_path / "stage"
    (stage / "msas" / "P1").mkdir(parents=True)
    (stage / "msas" / "P1" / "bfd_uniref_hits.a3m").write_text(">x\nMKT\n")
    (stage / "proteins_50k.fasta").write_text(">P1\nMKT\n")

    monkeypatch.setenv("GITM_BENCH_STAGE", str(stage))
    monkeypatch.setenv("GITM_BENCH_MAX_LEN", "0")  # excludes everything

    import pytest

    with pytest.raises(FileNotFoundError, match="no proteins"):
        get_factory("openfold")(LoopConfig(workload="openfold"))
