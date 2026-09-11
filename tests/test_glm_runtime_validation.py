"""Production seams which graph arithmetic tests cannot exercise."""

import json

import pytest

from gitm.optimizer.deviation import main as deviate
from gitm.optimizer.deviation import observed_op
from gitm.optimizer.validate_trace import audit_trace
from gitm.planner.model_catalogue import predict
from gitm.serve import discover, model_config
from gitm.tracer import injection
from gitm.tracer._cupti_decode import normalize_range_name
from tests.test_glm_graph import GLM_CONFIG


@pytest.mark.parametrize(("path", "op"), [
    ("self_attn.q_a_proj", "attn_q_a"),
    ("self_attn.kv_a_proj_with_mqa", "attn_kv_a"),
    ("self_attn.q_b_proj", "attn_q_b"),
    ("self_attn.kv_b_proj", "attn_kv_b"),
    ("self_attn.indexer.wq_b", "attn_index_proj"),
    ("self_attn.indexer.wk", "attn_index_proj"),
    ("self_attn.indexer.weights_proj", "attn_index_proj"),
    ("self_attn.indexer", "attn_index_score"),
    ("mlp.shared_experts.down_proj", "moe_shared"),
    ("mlp.experts.0.gate_up_proj", "moe_routed"),
    ("input_layernorm", "rms_norm"),
])
def test_glm_module_ranges_match_graph_vocabulary(path, op):
    module = f"model.layers.4.{path}"
    for label in (str({"Module": module}), json.dumps({"Module": module})):
        assert normalize_range_name(label) == f"L4/{op}"


def test_nested_helpers_do_not_inflate_projection():
    assert observed_op("scaled_fp8_quant", "attn_q_a") == "act_quant"
    assert observed_op("rms_norm", "layer") == "rms_norm"
    assert observed_op("opaque_gemm", "attn_q_a") == "attn_q_a"
    assert observed_op("moe_sum", "moe_routed") == "moe_combine"


def test_live_glm_reader_preserves_glm_spec(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(GLM_CONFIG))
    target = discover.Target(pid=1, cmdline=["vllm", "serve", str(tmp_path),
                             "-tp", "8", "--enable-expert-parallel"])
    live = model_config.live_moe_spec(target, environ={})
    assert isinstance(live, model_config.LiveSpec)
    assert live.family == "glm_moe_dsa"
    assert live.spec.n_full_indexer_layers == 21
    assert live.sharding.tp == live.sharding.ep == 8


def test_production_shards_do_not_cross_correlate(tmp_path, monkeypatch):
    base = tmp_path / "trace.jsonl"
    monkeypatch.setenv(injection.ENV_OUT, str(base))
    for pid, op in ((100, "attn_q_a"), (200, "attn_kv_a")):
        rows = [
            {"kind": "marker", "marker_id": 1, "marker_flags": 0,
             "timestamp_ns": 10, "thread_id": 7, "name": f"L4/{op}"},
            {"kind": "marker", "marker_id": 1, "marker_flags": 1,
             "timestamp_ns": 30, "thread_id": 7},
            {"kind": "runtime", "correlation_id": 3, "thread_id": 7,
             "start_ns": 15, "end_ns": 20},
            {"kind": "kernel", "name": "opaque_gemm", "correlation_id": 3,
             "start_ns": 50, "end_ns": 60, "device_id": 0, "stream_id": 1,
             "grid": [1, 1, 1], "block": [32, 1, 1]},
        ]
        base.with_name(f"trace.jsonl.{pid}").write_text(
            "\n".join(json.dumps(r) for r in rows))
    events = injection.read_shards()
    assert {e.pid: e.range_op for e in events} == {100: "attn_q_a", 200: "attn_kv_a"}


def test_phase_option_does_not_discard_graph_or_json(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"kind": "kernel", "name": "opaque_gemm",
                    "range_op": "attn_q_a", "pid": 100, "device_id": 0,
                    "start_ns": 1, "end_ns": 101}))
    assert deviate([str(path), "--model", "glm-5.2-fp8", "--gpu", "H200",
                    "--tp", "8", "--ep", "8", "--steps", "2",
                    "--pid", "100", "--by-phase", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ops"]["attn_q_a"]["floor_s"] > 0
    assert report["by_phase"]["unknown"]


def test_audit_exposes_invalid_tags_and_worker_selection(tmp_path):
    graph, _ = predict("glm-5.2-fp8")
    path = tmp_path / "trace.jsonl"
    rows = [{"kind": "kernel", "name": "opaque_gemm", "range_op": "attn_index_proj",
             "range_layer": 3, "pid": pid, "device_id": 0,
             "start_ns": 10, "end_ns": 110} for pid in (100, 200)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    report = audit_trace(path, graph, pid=100)
    assert report["selected_workers"] == 1
    assert len(report["workers"]) == 2
    assert report["device_time_ns"] == 100
    # IndexShare layer 3 must not launch an indexer projection.
    assert report["unexpected_layer_tags"] == 1


def test_capture_model_reaches_preflight_without_default_experiment(tmp_path, monkeypatch):
    from gitm.cli import main
    from gitm.serve import vllm

    commands = []
    monkeypatch.setattr(vllm, "preflight", lambda cmd, *a, **kw: commands.append(cmd) or [])
    assert main(["capture", "serve", "--model", "zai-org/GLM-5.2-FP8",
                 "--tp", "8", "--dry-run", "--out", str(tmp_path)]) == 0
    assert commands[0][:3] == ["vllm", "serve", "zai-org/GLM-5.2-FP8"]


def test_glm_live_sidecar_uses_glm_predictor(tmp_path, monkeypatch):
    from gitm.serve.attach import _emit_predicted_graph

    (tmp_path / "config.json").write_text(json.dumps(GLM_CONFIG))
    target = discover.Target(pid=1, cmdline=["vllm", "serve", str(tmp_path)])
    _emit_predicted_graph(target, tmp_path)
    payload = json.loads((tmp_path / "predicted_moe_graph.json").read_text())
    assert payload["family"] == "glm_moe_dsa"
    assert "attn_q_a" in {n["op"] for n in payload["nodes"]}


def test_merged_workers_are_refused_by_deviation(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps({"kind": "kernel", "name": "rms_norm",
                    "pid": p, "device_id": 0, "start_ns": 1, "end_ns": 100})
                    for p in (100, 200)))
    assert deviate([str(path), "--model", "glm-5.2-fp8"]) == 2
    assert "per-rank" in capsys.readouterr().out


def test_collective_and_sparse_attention_taxonomies_agree():
    from gitm.optimizer.deviation import classify_op
    from gitm.tracer.kernel_taxonomy import classify_kernel

    assert classify_kernel("moe_all_to_all") == "collective"
    assert classify_op("moe_all_to_all") == "moe_all_to_all"
    assert classify_kernel("sparse_fwd_kernel") == "attention"
    assert classify_op("sparse_fwd_kernel") == "attn_score_value"


@pytest.mark.parametrize("tag,code", [("attn_q_a", 0), ("unknown_module", 1)])
def test_validation_command_gates_attribution(tmp_path, capsys, tag, code):
    from gitm.optimizer.validate_trace import main

    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"kind": "kernel", "name": "opaque_gemm",
                    "range_op": tag, "range_layer": 0, "pid": 100, "device_id": 0,
                    "start_ns": 1, "end_ns": 100}))
    assert main([str(path), "--pid", "100", "--expected-workers", "1"]) == code
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] == (code == 0)
    # Coverage passing does not hide unobserved graph terms.
    assert report["missing_ops"]
