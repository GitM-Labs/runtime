"""Unit tests for pre-loop collective health (no live multi-GPU required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gitm.health.collective import (
    Check,
    HealthReport,
    algbw_busbw_gbs,
    expected_allreduce_sum,
    numerical_ok,
    run_collective_health,
    soft_bw_check,
    write_collective_health,
)


def test_expected_allreduce_sum():
    assert expected_allreduce_sum(8) == 8.0
    assert expected_allreduce_sum(4, fill=2.0) == 8.0


def test_numerical_ok_tight():
    assert numerical_ok(8.0, 8.0)
    assert numerical_ok(8.0000001, 8.0)
    assert not numerical_ok(7.0, 8.0)
    assert not numerical_ok(0.0, 8.0)


def test_algbw_busbw_allreduce_formula():
    # 4 GiB in 1s on 8 ranks → algbw = 4.0? nbytes=4e9 → 4 GB/s algbw
    nbytes = 4_000_000_000
    algbw, busbw = algbw_busbw_gbs(nbytes, 1.0, 8)
    assert algbw == pytest.approx(4.0)
    # busbw = algbw * 2 * 7/8 = 4 * 1.75 = 7.0
    assert busbw == pytest.approx(7.0)


def test_soft_bw_warn_below_floor():
    # catalogue 900 GB/s → 10% floor = 90 GB/s
    check = soft_bw_check(10.0, 900e9)
    assert check.status == "warn"
    assert "below soft floor" in check.detail


def test_soft_bw_pass_above_floor():
    check = soft_bw_check(100.0, 900e9)
    assert check.status == "pass"


def test_soft_bw_pass_when_no_catalogue():
    check = soft_bw_check(1.0, 0.0)
    assert check.status == "pass"
    assert "no catalogue" in check.detail


def test_write_collective_health_json(tmp_path: Path):
    report = HealthReport(
        vendor="nvidia",
        world_size=1,
        skipped=True,
        checks=[Check("collective_allreduce", "pass", "skipped")],
    )
    path = write_collective_health(tmp_path, report)
    assert path.name == "collective_health.json"
    text = path.read_text()
    assert '"ok": true' in text
    assert '"skipped": true' in text


def test_run_collective_health_skips_env(monkeypatch):
    monkeypatch.setenv("GITM_SKIP_COLLECTIVE_HEALTH", "1")
    report = run_collective_health()
    assert report.ok
    assert report.skipped
    assert report.checks[0].status == "pass"
    assert "skipped" in report.checks[0].detail


def test_run_collective_health_skips_flag():
    report = run_collective_health(skip=True)
    assert report.ok and report.skipped


def test_run_collective_health_skips_single_gpu(monkeypatch):
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 1)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")
    report = run_collective_health()
    assert report.ok and report.skipped
    assert report.world_size == 1


def test_run_collective_health_fail_on_timeout(monkeypatch):
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 2)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")

    def _boom(*_a, **_k):
        raise TimeoutError("collective AllReduce timed out after 60s (world_size=2)")

    monkeypatch.setattr("gitm.health.collective.run_torch_nccl_allreduce", _boom)
    report = run_collective_health()
    assert not report.ok
    assert any(c.status == "fail" for c in report.checks)
    assert "timed out" in report.diagnostic()


def test_run_collective_health_numerical_fail(monkeypatch):
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 2)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")
    monkeypatch.setattr(
        "gitm.health.collective.run_torch_nccl_allreduce",
        lambda *a, **k: {
            "elapsed_s": 0.01,
            "observed_sum": 1.0,  # wrong — expect 2.0
            "expected_sum": 2.0,
            "nbytes": 4 * 1024 * 1024,
        },
    )
    report = run_collective_health()
    assert not report.ok
    assert "numerical mismatch" in report.diagnostic()
    # busbw check still present as warn or pass — never fail
    bw = [c for c in report.checks if c.name == "collective_busbw"]
    assert bw and bw[0].status in ("pass", "warn")


def test_run_loop_aborts_on_collective_fail(tmp_path: Path, monkeypatch):
    """Failed health check must return no_data before factory / capture."""
    from gitm.health.collective import Check, HealthReport
    from gitm.scheduler.loop import LoopConfig, run_loop

    fail = HealthReport(
        vendor="nvidia",
        world_size=2,
        checks=[Check("collective_allreduce", "fail", "numerical mismatch")],
    )
    monkeypatch.setattr(
        "gitm.health.run_collective_health", lambda **_k: fail
    )
    # If factory is somehow reached, blow up — proves we aborted early.
    monkeypatch.setattr(
        "gitm.scheduler.loop.get_factory",
        lambda _w: (_ for _ in ()).throw(AssertionError("factory must not run")),
    )
    out = run_loop(
        LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path))
    )
    summary = out["summary"]
    assert summary["status"] == "no_data"
    assert "collective health" in summary["diagnostic"].lower()
    health_path = Path(summary["report_path"]).parent / "collective_health.json"
    assert health_path.exists()


def test_run_loop_proceeds_when_health_skipped(tmp_path: Path, monkeypatch):
    from gitm.scheduler.loop import LoopConfig, run_loop

    monkeypatch.setenv("GITM_SKIP_COLLECTIVE_HEALTH", "1")
    # Default path with no GPU → no_data from empty trace, not health fail
    out = run_loop(
        LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path))
    )
    assert out["summary"]["status"] == "no_data"
    # Health artifact still written
    run_dir = Path(out["summary"]["report_path"]).parent
    assert (run_dir / "collective_health.json").exists()
    assert "collective health check failed" not in out["summary"]["diagnostic"].lower()


def test_rccl_is_comm_kernel():
    from gitm.importers.node_rollup import is_comm_kernel

    assert is_comm_kernel("rcclDevKernel_AllReduce_Sum_f32")
    assert is_comm_kernel("void rcclKernel_AllGather(...)")


def test_rccl_classify_kernel_collective():
    from gitm.tracer.kernel_taxonomy import classify_kernel

    assert classify_kernel("rcclDevKernel_AllReduce_Sum_f16_RING_LL") == "collective"


def test_rccl_classify_op_tp_all_reduce():
    from gitm.optimizer.deviation import classify_op

    assert classify_op("rcclDevKernel_AllReduce_Sum_f32") == "tp_all_reduce"
