"""How much of a trace the residuals describe.

``residuals()`` used to skip kernels it could not pair with a graph node without
counting them, and the run-level residual read as if it covered the whole step.
These tests pin the coverage record and the run-level residual built from it.
"""

from __future__ import annotations

import pytest

from gitm.optimizer.monitor import UNCLASSIFIED, KernelResidual, Residuals, residuals
from gitm.optimizer.report import Claim, Provenance, write_report
from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, RooflinePrediction
from gitm.scheduler.loop import _agg_kt_residual, _agg_kt_residual_info
from gitm.tracer.schema import KernelEvent, Trace


def _node(op: str, layer: int | None, t_pred: float) -> PredictedNode:
    return PredictedNode(
        op, layer,
        RooflinePrediction(op=op, flops=0.0, bytes=0.0, t_compute_s=0.0, t_memory_s=t_pred,
                           t_pred_s=t_pred, bound="memory"),
    )


def _graph(nodes: list[PredictedNode]) -> Graph:
    return Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(), nodes=nodes)


def _k(name: str, dur_s: float) -> KernelEvent:
    return KernelEvent(name=name, start_ns=0, end_ns=round(dur_s * 1e9), stream_id=0, device_id=0)


def _trace(events: list[KernelEvent]) -> Trace:
    return Trace(workload_id="w", fingerprint="f", run_id="r", device_count=1,
                 vendor="nvidia", captured_at_ns=0, duration_ns=10**9, events=events)


DENSE = _graph([_node("attn_score_value", 0, 1e-3), _node("mlp_gate_up", 0, 2e-3)])


def test_every_kernel_is_matched_or_counted_as_dropped():
    trace = _trace([
        _k("flash_attn_fwd", 1e-3),              # attn_score_value: matched
        _k("ampere_sgemm_128x64_nn", 4e-3),      # bare GEMM: no op
        _k("rms_norm_kernel", 5e-4),             # rms_norm: classified, not in this graph
    ])
    res = residuals(trace, DENSE)

    assert res.n_kernels_seen == 3
    assert len(res.per_kernel) == 1
    assert res.n_kernels_dropped == 2
    assert res.dropped_ops == {UNCLASSIFIED: 1, "rms_norm": 1}
    assert res.dropped_time_s == pytest.approx(4.5e-3)
    assert res.unobserved_predicted_ops == ["mlp_gate_up"]

    cov = res.coverage()
    assert cov["status"] == "ok"
    assert cov["n_kernels_matched"] == 1
    assert cov["matched_time_fraction"] == pytest.approx(1e-3 / 5.5e-3)


def test_no_matched_kernel_has_no_fraction_and_no_residual():
    res = residuals(_trace([_k("ampere_sgemm_128x64_nn", 1e-3)]), DENSE)
    assert res.coverage()["status"] == "no_matches"
    assert res.coverage()["matched_time_fraction"] == pytest.approx(0.0)
    assert _agg_kt_residual(res) is None

    empty = residuals(_trace([]), DENSE)
    assert empty.coverage()["matched_time_fraction"] is None
    assert empty.unobserved_predicted_ops == ["attn_score_value", "mlp_gate_up"]
    info = _agg_kt_residual_info(empty)
    assert (info.value, info.method) == (None, "none")


def test_interval_row_inside_the_band_aggregates_to_zero():
    """Regression: an in-band kernel on a heterogeneous op has r_kt = 0 but keeps
    the nearest class's t_pred. Rebuilding from t_pred reported +100%."""
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 3e-3)]), g)
    kr = res.per_kernel[0]
    assert kr.interval_based and kr.r_kt == 0.0

    info = _agg_kt_residual_info(res)
    assert info.value == pytest.approx(0.0)
    assert info.method == "duration_weighted"
    assert not info.clamped


def test_interval_row_outside_the_band_keeps_its_deviation():
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 6e-3)]), g)
    # 6 ms against a 5 ms upper edge: +20%, matching the row's own residual.
    assert res.per_kernel[0].r_kt == pytest.approx(0.2)
    assert _agg_kt_residual(res) == pytest.approx(0.2)


def test_clamp_and_method_are_reported():
    big = Residuals(per_kernel=[
        KernelResidual(op="a", layer=None, r_kt=2.0, r_mt=None, t_obs_s=3e-3, t_pred_s=1e-3),
    ])
    info = _agg_kt_residual_info(big)
    assert (info.value, info.clamped) == (1.0, True)

    untimed = Residuals(per_kernel=[
        KernelResidual(op="a", layer=None, r_kt=0.1, r_mt=None),
        KernelResidual(op="b", layer=None, r_kt=0.3, r_mt=None),
    ])
    info = _agg_kt_residual_info(untimed)
    assert info.method == "median_ratio"
    assert info.value == pytest.approx(0.2)


def test_report_renders_a_missing_residual_with_its_reason():
    claim = Claim(summary="s", residual_invariant="kernel_time", residual_value=None,
                  residual_note="no kernels matched the graph", causal_evidence="e",
                  intervention_name="i", predicted_delta=0.0, measured_delta=None)
    prov = Provenance(workload_id="w", fingerprint="f", run_id="r", git_sha="x",
                      gitm_version="0", started_at_ns=0, ended_at_ns=0)
    md = write_report(claims=[claim], provenance=prov)
    assert "`kernel_time`: no kernels matched the graph |" in md


# ── stream concurrency with nothing to measure ──────────────────────────────


def test_serialized_fraction_is_not_measured_below_two_kernels():
    from gitm.optimizer.monitor import measured_serialized_fraction, serialized_text

    # _serialized_fraction returns 0.0 ("fully overlapped") for 0 or 1 kernels;
    # artifacts must not repeat that as a measurement.
    assert measured_serialized_fraction(0.0, 0) is None
    assert measured_serialized_fraction(0.0, 1) is None
    assert measured_serialized_fraction(0.25, 2) == 0.25
    assert serialized_text(0.0, 1) == "n/a (fewer than 2 kernels)"
    assert serialized_text(0.25, 5) == "0.250"


def test_measurement_with_one_kernel_has_no_concurrency_value():
    from gitm.optimizer.measure import measure_trace, measurement_summary

    result = measure_trace(_trace([_k("gemm_kernel_a", 1e-3)]))
    assert result.serialized_fraction == 0.0  # internal value, unchanged
    assert result.serialized_measured is None
    assert "serialized-concurrency=n/a (fewer than 2 kernels)" in measurement_summary("w", result)


# ── gitm deviate: records it could not read ─────────────────────────────────


def _jsonl(path, rows: list[str]):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _kernel_row(start: int, end: int, name: str = "flash_fwd_x") -> str:
    import json

    return json.dumps({"kind": "kernel", "name": name, "start_ns": start, "end_ns": end,
                       "stream_id": 0, "device_id": 0})


def test_stream_observed_counts_unreadable_records(tmp_path):
    import json

    from gitm.optimizer.deviation import stream_observed

    rows = [json.dumps({"_header": {"run_id": "t"}})]  # not a kernel: filtered, not counted
    rows += [_kernel_row(i * 100, i * 100 + 50) for i in range(8)]
    rows += ['{"kind": "kernel", "name": "flash_fwd_x", "start_ns": 900, "end_',  # cut off
             _kernel_row(1000, 1000),  # zero length
             '"just a string"']
    skipped: dict[str, int] = {}
    _, n_kernels, _, _ = stream_observed(_jsonl(tmp_path / "t.jsonl", rows), skipped=skipped)
    assert n_kernels == 8
    assert skipped == {"malformed_line": 2, "bad_timestamps": 1}


def test_deviate_reports_skipped_records(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main

    p = _jsonl(tmp_path / "t.jsonl",
               [_kernel_row(i * 100, i * 100 + 50) for i in range(4)] + ['{"kind": "kern'])
    assert main([str(p), "--no-graph"]) == 0
    assert "skipped 1 unreadable record(s): malformed_line 1" in capsys.readouterr().out

    assert main([str(p), "--no-graph", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_records"] == {"malformed_line": 1}


def test_deviate_clean_trace_reports_nothing_skipped(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main

    p = _jsonl(tmp_path / "t.jsonl", [_kernel_row(i * 100, i * 100 + 50) for i in range(4)])
    assert main([str(p), "--no-graph", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["skipped_records"] == {}


def test_measurement_json_records_granger_limitations_and_unmeasured_concurrency(tmp_path):
    import json

    from gitm.optimizer.measure import MEASURE_LIMITATIONS
    from gitm.optimizer.qualification import QualificationResult
    from gitm.scheduler.loop import _measurement_result

    _measurement_result(
        run_dir=tmp_path, run_id="r", workload="hft", trace=_trace([_k("gemm_kernel_a", 1e-3)]),
        qual=QualificationResult(commit=False, floor=0.0, fingerprint="f"),
        started_ns=0, trace_path=tmp_path / "t.jsonl",
    )
    record = json.loads((tmp_path / "measurement.json").read_text())
    assert record["serialized_concurrency_fraction"] is None
    assert record["granger"]["status"] == "insufficient_data"
    assert record["limitations"] == list(MEASURE_LIMITATIONS)


def test_deviate_trace_with_nothing_readable_reports_the_skips(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main

    p = _jsonl(tmp_path / "t.jsonl", ['{"kind": "kern', '{"kind": "kernel", "start'])
    assert main([str(p), "--no-graph"]) == 1
    out = capsys.readouterr().out
    assert "nothing to subtract" in out and "skipped 2 unreadable record(s): malformed_line 2" in out

    assert main([str(p), "--no-graph", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "trace": str(p), "n_kernels": 0, "skipped_records": {"malformed_line": 2},
        "error": "no kernel records",
    }


def test_deviate_by_phase_counts_skips_and_fails_on_an_unreadable_trace(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main

    p = _jsonl(tmp_path / "t.jsonl",
               [_kernel_row(i * 100, i * 100 + 50) for i in range(4)] + ['{"kind": "kern'])
    assert main([str(p), "--by-phase"]) == 0
    assert "skipped 1 unreadable record(s): malformed_line 1" in capsys.readouterr().out
    assert main([str(p), "--by-phase", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["phase_stats"]["skipped_records"] == {"malformed_line": 1}

    bad = _jsonl(tmp_path / "bad.jsonl", ['{"kind": "kern', '{"kind": "kernel", "start'])
    assert main([str(bad), "--by-phase", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["phase_stats"]["skipped_records"] == {"malformed_line": 2}


def test_reader_counts_bad_bytes_types_and_boolean_timestamps(tmp_path):
    import json

    from gitm.optimizer.deviation import stream_observed

    p = tmp_path / "t.jsonl"
    good = _kernel_row(0, 50) + "\n"
    rows = [
        good.encode(),
        b"\xff\xfe\n",  # not UTF-8
        (json.dumps({"kind": "kernel", "name": 123, "start_ns": 0, "end_ns": 50}) + "\n").encode(),
        (json.dumps({"kind": "kernel", "name": "k", "range_op": {"op": "x"},
                     "start_ns": 0, "end_ns": 50}) + "\n").encode(),
        (json.dumps({"kind": "kernel", "name": "k", "start_ns": False, "end_ns": True}) + "\n").encode(),
        _kernel_row(100, 150).encode() + b"\n",
    ]
    p.write_bytes(b"".join(rows))
    skipped: dict[str, int] = {}
    _, n_kernels, _, _ = stream_observed(p, skipped=skipped)
    assert n_kernels == 2
    assert skipped == {"bad_encoding": 1, "bad_fields": 2, "bad_timestamps": 1}


def test_a_rejected_record_is_not_a_phase_anchor(tmp_path):
    from gitm.optimizer.deviation import phase_anchors

    # A decode-named kernel whose end precedes its start is counted as
    # bad_timestamps by the phase readers; it must not label its neighbours.
    p = _jsonl(tmp_path / "t.jsonl", [_kernel_row(100, 90, name="fused_recurrent_fwd"),
                                      _kernel_row(200, 300, name="ampere_sgemm_128x64_nn")])
    assert phase_anchors(p) == []


def test_deviate_with_a_model_survives_a_bad_byte(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main, observed_scopes

    p = tmp_path / "t.jsonl"
    p.write_bytes((_kernel_row(0, 50) + "\n").encode() + b"\xff\n"
                  + (json.dumps({"kind": "kernel", "name": "k", "start_ns": 0, "end_ns": 9,
                                 "pid": [1], "device_id": 0}) + "\n").encode())
    # Neither the bad byte nor the list-valued pid adds a worker or crashes.
    assert observed_scopes(p) == {(None, 0)}
    assert main([str(p), "--model", "kimi-k2.5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["skipped_records"] == {"bad_encoding": 1, "bad_fields": 1}
