"""Unit tests for pre-loop collective health (no live multi-GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gitm.health.collective import (
    Check,
    HealthReport,
    _stop_procs,
    algbw_busbw_gbs,
    expected_allreduce_sum,
    numerical_ok,
    resolve_local_probe_world_size,
    resolve_probe_world_size,
    run_collective_health,
    run_torch_nccl_allreduce,
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
    nbytes = 4_000_000_000
    algbw, busbw = algbw_busbw_gbs(nbytes, 1.0, 8)
    assert algbw == pytest.approx(4.0)
    assert busbw == pytest.approx(7.0)


def test_soft_bw_warn_below_floor():
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


def test_resolve_probe_world_size_skips_default_vllm(monkeypatch):
    monkeypatch.delenv("GITM_VLLM_TP", raising=False)
    monkeypatch.delenv("GITM_VLLM_EXTRA_JSON", raising=False)
    assert resolve_probe_world_size(workload="vllm-decode") == 1


def test_resolve_probe_world_size_honours_gitm_vllm_tp(monkeypatch):
    monkeypatch.delenv("GITM_VLLM_EXTRA_JSON", raising=False)
    monkeypatch.setenv("GITM_VLLM_TP", "4")
    assert resolve_probe_world_size(workload="vllm-decode") == 4


def test_resolve_probe_world_size_extra_json_tp(monkeypatch):
    """Factory merges EXTRA_JSON after GITM_VLLM_TP — probe must see TP=2 there."""
    monkeypatch.delenv("GITM_VLLM_TP", raising=False)
    monkeypatch.setenv("GITM_VLLM_EXTRA_JSON", '{"tensor_parallel_size": 2}')
    assert resolve_probe_world_size(workload="vllm-decode") == 2


def test_resolve_probe_world_size_extra_json_overrides_tp_env(monkeypatch):
    monkeypatch.setenv("GITM_VLLM_TP", "1")
    monkeypatch.setenv("GITM_VLLM_EXTRA_JSON", '{"tensor_parallel_size": 2}')
    assert resolve_probe_world_size(workload="vllm-decode") == 2


def test_resolve_probe_world_size_engine_parallel_config():
    """Embedded engines keep TP under parallel_config, not top-level attrs."""
    from types import SimpleNamespace as NS

    engine = NS(llm_engine=NS(vllm_config=NS(parallel_config=NS(world_size=2))))
    assert resolve_probe_world_size(workload="vllm-decode", engine=engine) == 2


def test_resolve_probe_world_size_engine_parallel_config_tp_field():
    from types import SimpleNamespace as NS

    engine = NS(engine=NS(parallel_config=NS(tensor_parallel_size=2)))
    assert resolve_probe_world_size(workload="vllm-decode", engine=engine) == 2


def test_resolve_probe_world_size_engine_top_level_tp():
    class _Eng:
        tensor_parallel_size = 2

    assert resolve_probe_world_size(workload="vllm-decode", engine=_Eng()) == 2


def test_resolve_probe_world_size_engine_beats_env(monkeypatch):
    """A live TP=2 engine must not be skipped because GITM_VLLM_TP is unset."""
    from types import SimpleNamespace as NS

    monkeypatch.delenv("GITM_VLLM_TP", raising=False)
    monkeypatch.delenv("GITM_VLLM_EXTRA_JSON", raising=False)
    engine = NS(vllm_config=NS(parallel_config=NS(world_size=2)))
    assert resolve_probe_world_size(workload="vllm-decode", engine=engine) == 2


def test_resolve_probe_world_size_non_collective_workload():
    assert resolve_probe_world_size(workload="hft") == 1
    assert resolve_probe_world_size(workload="openfold") == 1


def test_run_collective_health_skips_env(monkeypatch):
    monkeypatch.setenv("GITM_SKIP_COLLECTIVE_HEALTH", "1")
    report = run_collective_health(world_size=4)
    assert report.ok
    assert report.skipped
    assert report.checks[0].status == "pass"
    assert "skipped" in report.checks[0].detail


def test_run_collective_health_skips_flag():
    report = run_collective_health(skip=True, world_size=4)
    assert report.ok and report.skipped


def test_run_collective_health_skips_when_workload_tp1(monkeypatch):
    """Multi-GPU host + TP=1 must skip — unused GPUs must not block the run."""
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 8)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")
    report = run_collective_health(world_size=1)
    assert report.ok and report.skipped
    assert "no multi-GPU collectives" in report.checks[0].detail


def test_resolve_local_probe_world_size_caps_to_visible():
    assert resolve_local_probe_world_size(8, 4) == 4
    assert resolve_local_probe_world_size(2, 8) == 2
    assert resolve_local_probe_world_size(8, 1) == 1


def test_resolve_local_probe_world_size_ignores_process_local_world_size(monkeypatch):
    """One process may own multiple TP GPUs; torchrun's process count is not GPU count."""
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.delenv("GITM_LOCAL_WORLD_SIZE", raising=False)
    assert resolve_local_probe_world_size(2, 2) == 2


def test_resolve_local_probe_world_size_honours_explicit_gitm_override(monkeypatch):
    monkeypatch.setenv("GITM_LOCAL_WORLD_SIZE", "2")
    assert resolve_local_probe_world_size(8, 4) == 2


def test_multi_node_global_ws_probes_local_gpus_only(monkeypatch):
    """world_size=8 on a 4-GPU node must probe 4 locally — not fail as missing remotes."""
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("GITM_LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 4)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")

    seen: dict = {}

    def _fake_allreduce(ws, **_k):
        seen["world_size"] = ws
        return {
            "elapsed_s": 0.01,
            "observed_sum": float(ws),
            "observed_min": float(ws),
            "observed_max": float(ws),
            "expected_sum": float(ws),
            "all_ranks_ok": True,
            "nbytes": 4 * 1024 * 1024,
        }

    monkeypatch.setattr(
        "gitm.health.collective.run_torch_nccl_allreduce", _fake_allreduce
    )
    report = run_collective_health(world_size=8)
    assert report.ok
    assert not report.skipped
    assert seen["world_size"] == 4
    assert report.world_size == 4
    assert report.global_world_size == 8
    assert "local_probe=4" in report.checks[0].detail
    assert "global_world_size=8" in report.checks[0].detail


def test_multi_node_single_local_gpu_skips_not_fails(monkeypatch):
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 1)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")
    report = run_collective_health(world_size=8)
    assert report.ok and report.skipped
    assert report.global_world_size == 8
    assert "local GPU" in report.checks[0].detail


def test_run_collective_health_fail_on_timeout(monkeypatch):
    monkeypatch.delenv("GITM_SKIP_COLLECTIVE_HEALTH", raising=False)
    monkeypatch.setattr("gitm.tracer.injection.detect_vendor", lambda: "nvidia")
    import gitm.health.nvidia as nvidia_mod

    monkeypatch.setattr(nvidia_mod, "device_count", lambda: 2)
    monkeypatch.setattr(nvidia_mod, "detect_sku", lambda: "H100")

    def _boom(*_a, **_k):
        raise TimeoutError("collective AllReduce timed out after 60s (world_size=2)")

    monkeypatch.setattr("gitm.health.collective.run_torch_nccl_allreduce", _boom)
    report = run_collective_health(world_size=2)
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
            "observed_sum": 2.0,
            "observed_min": 1.0,  # wrong elsewhere in the buffer
            "observed_max": 2.0,
            "expected_sum": 2.0,
            "all_ranks_ok": False,
            "nbytes": 4 * 1024 * 1024,
        },
    )
    report = run_collective_health(world_size=2)
    assert not report.ok
    assert "numerical mismatch" in report.diagnostic()
    bw = [c for c in report.checks if c.name == "collective_busbw"]
    assert bw and bw[0].status in ("pass", "warn")


def test_partial_start_failure_stops_started_workers(monkeypatch):
    """If a later rank fails to start, earlier ranks must be reaped."""
    started: list[MagicMock] = []

    def _popen(cmd, **_kwargs):
        rank = int(cmd[cmd.index("--rank") + 1])
        if rank >= 1:
            raise OSError("simulated launch failure")
        proc = MagicMock()
        proc.poll.return_value = None  # still running until stopped
        proc.pid = 1000 + rank
        proc.returncode = None
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        started.append(proc)
        return proc

    monkeypatch.setattr("gitm.health.collective.subprocess.Popen", _popen)

    stopped: list[MagicMock] = []

    def _stop(procs):
        stopped.extend(procs)
        for p in procs:
            p.poll.return_value = 0
            p.returncode = -15

    monkeypatch.setattr("gitm.health.collective._stop_procs", _stop)

    with pytest.raises(OSError, match="simulated launch failure"):
        run_torch_nccl_allreduce(2, timeout_s=5.0)

    assert len(started) == 1
    assert stopped == started


def test_stop_procs_terminates_then_kills():
    proc = MagicMock()
    # First poll (build alive list) → running; second (still list) → still running.
    polls = [None, None]
    proc.poll.side_effect = lambda: polls.pop(0) if polls else -9
    proc.wait.return_value = None

    _stop_procs([proc])
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_run_loop_aborts_on_collective_fail(tmp_path: Path, monkeypatch):
    """Failed health check must return no_data before factory / capture."""
    from gitm.health.collective import Check, HealthReport
    from gitm.scheduler.loop import LoopConfig, run_loop

    fail = HealthReport(
        vendor="nvidia",
        world_size=2,
        checks=[Check("collective_allreduce", "fail", "numerical mismatch")],
    )
    monkeypatch.setattr("gitm.health.run_collective_health", lambda **_k: fail)
    monkeypatch.setattr(
        "gitm.scheduler.loop.get_factory",
        lambda _w: (_ for _ in ()).throw(AssertionError("factory must not run")),
    )
    out = run_loop(LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path)))
    summary = out["summary"]
    assert summary["status"] == "no_data"
    assert "collective health" in summary["diagnostic"].lower()
    health_path = Path(summary["report_path"]).parent / "collective_health.json"
    assert health_path.exists()


def test_run_loop_passes_resolved_world_size(tmp_path: Path, monkeypatch):
    """Loop must scope the probe to resolve_probe_world_size, not all GPUs."""
    from gitm.scheduler.loop import LoopConfig, run_loop

    seen: dict = {}

    def _fake_health(**kwargs):
        seen.update(kwargs)
        return HealthReport(
            vendor="nvidia",
            world_size=kwargs.get("world_size") or 0,
            skipped=True,
            checks=[Check("collective_allreduce", "pass", "skipped")],
        )

    monkeypatch.setenv("GITM_VLLM_TP", "2")
    monkeypatch.setattr("gitm.health.run_collective_health", _fake_health)
    # Empty capture path → no_data after health
    out = run_loop(LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path)))
    assert seen.get("world_size") == 2
    assert out["summary"]["status"] == "no_data"


def test_run_loop_proceeds_when_health_skipped(tmp_path: Path, monkeypatch):
    from gitm.scheduler.loop import LoopConfig, run_loop

    monkeypatch.setenv("GITM_SKIP_COLLECTIVE_HEALTH", "1")
    out = run_loop(LoopConfig(workload="vllm-decode", budget="1s", scratch=str(tmp_path)))
    assert out["summary"]["status"] == "no_data"
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
