"""The loop must run a real workload and refuse to fake a result from nothing.

Covers the wiring added so ``gitm run`` actually drives a workload under the
tracer (instead of capturing an empty ``pass`` block) and the guard that reports
*no-data* rather than fabricating claims when the trace is empty.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .conftest import make_kernel, make_trace

# Digest of an empty kernel list — sha256(repr([]))[:16]. A real trace must not
# produce this; if it does, the workload didn't actually run.
EMPTY_DIGEST = "4f53cda18c2baa0c"


def test_no_data_guard_does_not_fabricate_claims(tmp_path: Path):
    """No GPU/shim and no registered runner → honest no-data, zero claims."""
    from gitm import optimize

    result = optimize(workload="vllm-decode", budget="1s", target=0.15, scratch=str(tmp_path))
    summary = result["summary"]

    assert summary["status"] == "no_data"
    assert summary["n_claims"] == 0
    assert summary["commit"] is False
    assert summary["diagnostic"]  # explains why nothing was measured
    assert Path(summary["report_path"]).exists()
    assert "NO DATA" in result["report_md"]


def test_runner_runs_inside_capture_and_produces_real_trace(tmp_path: Path, monkeypatch):
    """An injected runner is invoked inside the capture window; with kernels in
    the trace the loop proceeds to real claims with a non-empty fingerprint."""
    import gitm.scheduler.loop as loop

    called = {"runner": False, "sync": False}

    # Fake capture: yields a populated nvidia trace so the guard passes and the
    # fingerprint reflects real kernels.
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [
            make_kernel(f"cudf_groupby_{i % 5}", start_ns=i * 100, end_ns=i * 100 + 90)
            for i in range(60)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: called.__setitem__("sync", True))

    def runner():
        called["runner"] = True
        return {"events": 1_000}

    from gitm import optimize

    result = optimize(
        workload="hft", budget="1s", target=0.15, scratch=str(tmp_path), workload_runner=runner
    )
    summary = result["summary"]

    assert called["runner"], "runner must be invoked inside the capture window"
    assert called["sync"], "device must be synced so kernels land in the trace"
    assert summary["status"] == "ok"
    assert summary["fingerprint"].startswith("nvidia:")
    assert summary["fingerprint"] != f"nvidia:{EMPTY_DIGEST}"
    assert Path(summary["report_path"]).exists()


def test_cli_run_returns_nonzero_on_no_data(tmp_path: Path, capsys):
    """Automation must see a failure exit when a run measures nothing."""
    from gitm.cli import main

    rc = main(["run", "--workload", "vllm-decode", "--budget", "1s", "--scratch", str(tmp_path)])
    assert rc == 3


def test_hft_harness_importable_from_package():
    """The harness must ship in the wheel, i.e. be importable from the package."""
    from gitm.benchmarks.hft import harness

    assert harness.run_pipeline and harness.load_events and harness.select_backend


def test_hft_is_registered():
    from gitm.workloads import get_factory, registered

    assert "hft" in registered() and "hft-lob" in registered()
    assert get_factory("hft") is not None
    assert get_factory("vllm-decode") is None
