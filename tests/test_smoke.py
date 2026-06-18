"""End-to-end smoke tests for the W1 skeleton.

These run without a GPU. They exercise every public interface on the critical
path so any regression in the scaffold is caught on commit.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_version_importable():
    import gitm

    assert isinstance(gitm.__version__, str)
    assert gitm.__version__


def test_optimize_importable():
    from gitm import optimize  # noqa: F401


def test_planner_v0_graph_has_nodes():
    from gitm.planner import predict_graph

    g = predict_graph()
    assert len(g.nodes) > 0
    assert g.total_pred_s > 0


def test_trace_capture_writes_jsonl(tmp_path: Path):
    from gitm.tracer import capture

    out = tmp_path / "smoke.jsonl"
    with capture(out, workload_id="smoke"):
        pass
    lines = out.read_text().strip().splitlines()
    assert lines, "capture wrote no header"
    header = json.loads(lines[0])
    assert "_header" in header
    assert header["_header"]["workload_id"] == "smoke"


def test_intervention_library_loads_and_validates():
    from gitm.kernels import load_library

    specs = load_library()
    assert len(specs) >= 2
    for spec in specs:
        assert spec.expected_delta_lo <= spec.expected_delta_mean <= spec.expected_delta_hi
        assert spec.source.startswith(("http://", "https://"))


def test_qualification_returns_diagnostic_on_empty_trace():
    from gitm.optimizer.qualification import qualify
    from gitm.tracer.schema import Trace

    trace = Trace(
        workload_id="w",
        fingerprint="f",
        run_id="r",
        device_count=0,
        vendor="none",
        captured_at_ns=0,
        duration_ns=0,
    )
    result = qualify(trace)
    assert result.commit is False
    assert "No kernels" in result.diagnostic


def test_report_renders_with_zero_claims():
    from gitm.optimizer.report import build_provenance, write_report

    prov = build_provenance(
        workload_id="vllm-decode",
        fingerprint="nvidia:abcd1234",
        run_id="run_test",
        started_at_ns=0,
    )
    md = write_report(claims=[], provenance=prov, qualification_diagnostic="example")
    assert "GITM provenance report" in md
    assert "example" in md
    assert "vllm-decode" in md


def test_run_loop_end_to_end(tmp_path: Path, monkeypatch):
    """Embedded ``optimize()`` runs end-to-end without a GPU."""
    monkeypatch.setenv("GITM_SCRATCH", str(tmp_path))

    from gitm import optimize

    result = optimize(workload="vllm-decode", budget="1s", target=0.15)
    assert "summary" in result
    assert "report_md" in result
    assert Path(result["summary"]["report_path"]).exists()


def test_cli_help_does_not_crash():
    """``gitm --help`` and ``gitm run --help`` parse cleanly."""
    from gitm.cli import _parser

    p = _parser()
    # Argparse's default behavior is to print + raise SystemExit; here we just
    # verify the parser was constructed without error and has the run subcommand.
    args = p.parse_args(["run", "--workload", "vllm-decode", "--target", "15%"])
    assert args.workload == "vllm-decode"


def test_doctor_returns_jsonable():
    from gitm.doctor import doctor

    info = doctor()
    # Must be JSON-serializable for piped use.
    json.dumps(info)
    assert info["gitm_version"]
    assert "telemetry_backends" in info
