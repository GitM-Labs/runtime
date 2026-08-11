"""Tests for attaching to an already-running vLLM server (gitm/serve/attach.py).

GPU-free, and /proc-free: every test drives a fake process tree on disk, so the
resolution and classification logic is exercised on a mac dev box exactly as it runs
on a pod. What is deliberately covered is the set of ways attach can go wrong
*quietly* — a server that was never started under the collector, another profiler
holding the injection hook, two servers on one box, an engine cohort that never
joined — because each of those otherwise ends in an empty trace an hour later with no
error anywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gitm.serve import attach as att
from gitm.serve import discover
from gitm.tracer import injection


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("duration_s", 0.0, "duration_s"),
        ("duration_s", float("nan"), "duration_s"),
        ("requests", -1, "requests"),
        ("concurrency", 0, "concurrency"),
        ("input_tokens", 0, "input_tokens"),
        ("output_tokens", -1, "output_tokens"),
        ("request_timeout", float("inf"), "request_timeout"),
        ("metrics_interval", 0.0, "metrics_interval"),
        ("port", 70000, "port"),
    ],
)
def test_attach_options_refuse_invalid_window_inputs(field, value, message):
    with pytest.raises(ValueError, match=message):
        att.AttachOptions(**{field: value})


def test_attach_options_accept_normal_observe_and_drive_windows():
    assert att.AttachOptions().mode == "observe"
    assert att.AttachOptions(requests=1, concurrency=1).mode == "drive"

LIB = str(injection.lib_path())


def _mkproc(tmp_path, pid: int, cmdline: list[str], env: dict[str, str] | None):
    d = tmp_path / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cmdline").write_bytes(("\0".join(cmdline) + "\0").encode())
    if env is not None:
        (d / "environ").write_bytes(
            ("\0".join(f"{k}={v}" for k, v in env.items()) + "\0").encode()
        )
    return d


def _traced_env(trace_out, lib: str = LIB) -> dict[str, str]:
    return {"PATH": "/usr/bin", injection.ENV_LIB: lib, injection.ENV_OUT: str(trace_out)}


SERVE = ["/usr/bin/vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000"]


# --- process identification --------------------------------------------------


def test_recognizes_the_server_and_not_its_own_workers():
    assert discover.is_vllm_server(SERVE)
    assert discover.is_vllm_server(
        ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "m"]
    )
    # EngineCore and the TP workers inherit the same environment and are traced by the
    # same collector, but they do not hold the port and are not what "the server" means.
    assert not discover.is_vllm_server(["VLLM::EngineCore"])
    assert not discover.is_vllm_server(["python", "-c", "from vllm import LLM"])
    assert not discover.is_vllm_server([])


# --- traceability verdicts ---------------------------------------------------


def test_server_launched_under_the_collector_is_traceable(tmp_path):
    proc = tmp_path / "proc"
    trace_out = tmp_path / "run" / "trace.jsonl"
    trace_out.parent.mkdir(parents=True)
    _mkproc(proc, 100, SERVE, _traced_env(trace_out))
    # The cohort the collector actually opened: frontend, EngineCore, two TP workers.
    for pid in (100, 101, 102, 103):
        trace_out.with_name(trace_out.name + f".{pid}").write_text("")

    t = discover.classify(100, proc)
    assert t.traceable is True
    assert t.trace_out == str(trace_out)
    assert t.shard_pids == [100, 101, 102, 103]


def test_server_without_the_injection_variable_is_refused_with_a_remedy(tmp_path):
    """The case this whole module exists for: the driver reads CUDA_INJECTION64_PATH
    once at CUDA init, so there is no way to make this process traceable in place. It
    has to be said now, not discovered as an empty trace after the window closes."""
    proc = tmp_path / "proc"
    _mkproc(proc, 200, SERVE, {"PATH": "/usr/bin"})

    t = discover.classify(200, proc)
    assert t.traceable is False
    assert injection.ENV_LIB in t.reason
    remedy = t.remedy()
    assert injection.ENV_LIB in remedy and injection.ENV_OUT in remedy
    assert "gitm capture serve" in remedy


def test_another_profiler_holding_the_hook_is_not_mistaken_for_ours(tmp_path):
    """nsys sets CUDA_INJECTION64_PATH too. Arming a window against it produces no
    shards, which is indistinguishable from a broken install unless we say so."""
    proc = tmp_path / "proc"
    _mkproc(proc, 300, SERVE, _traced_env(tmp_path / "t.jsonl", lib="/opt/nsight/libToolsInjection64.so"))

    t = discover.classify(300, proc)
    assert t.traceable is False
    assert "another profiler" in t.reason


def test_injection_lib_without_an_output_path_is_inert(tmp_path):
    proc = tmp_path / "proc"
    _mkproc(proc, 400, SERVE, {injection.ENV_LIB: LIB})

    t = discover.classify(400, proc)
    assert t.traceable is False
    assert injection.ENV_OUT in t.reason


def test_process_owned_by_another_user_is_declined_not_escalated(tmp_path):
    """An unreadable /proc/<pid>/environ is the kernel telling us this is not our
    process. gitm attaches user-space only; the answer is to decline, not to find a
    way around it."""
    proc = tmp_path / "proc"
    _mkproc(proc, 500, SERVE, env=None)  # no environ file == unreadable

    t = discover.classify(500, proc)
    assert t.traceable is False
    assert "another user" in t.reason


def test_dead_pid_reports_as_dead(tmp_path):
    t = discover.classify(999, tmp_path / "proc")
    assert t.traceable is False and "not live" in t.reason


# --- target resolution -------------------------------------------------------


def test_single_server_is_discovered_without_a_hint(tmp_path):
    proc = tmp_path / "proc"
    trace_out = tmp_path / "trace.jsonl"
    _mkproc(proc, 100, SERVE, _traced_env(trace_out))

    target, how = discover.resolve_target(proc=proc)
    assert target is not None and target.pid == 100
    assert "discovered" in how


def test_two_servers_refuse_to_be_guessed_between(tmp_path):
    """Guessing here means tracing the wrong model and never finding out."""
    proc = tmp_path / "proc"
    _mkproc(proc, 100, SERVE, _traced_env(tmp_path / "a.jsonl"))
    _mkproc(proc, 200, ["/usr/bin/vllm", "serve", "other/model"], _traced_env(tmp_path / "b.jsonl"))

    target, how = discover.resolve_target(proc=proc)
    assert target is None
    assert "refusing to guess" in how and "100" in how and "200" in how


def test_no_server_at_all_says_how_to_get_one(tmp_path):
    target, how = discover.resolve_target(proc=tmp_path / "proc")
    assert target is None and "gitm capture serve" in how


def test_explicit_pid_skips_discovery_entirely(tmp_path):
    proc = tmp_path / "proc"
    _mkproc(proc, 100, SERVE, _traced_env(tmp_path / "a.jsonl"))
    _mkproc(proc, 200, SERVE, _traced_env(tmp_path / "b.jsonl"))

    target, how = discover.resolve_target(pid=200, proc=proc)
    assert target is not None and target.pid == 200 and "--pid" in how


# --- port -> pid via /proc/net/tcp ------------------------------------------

# A real listening row. Field 1 is hex ip:port, field 3 is the state (0A == LISTEN),
# field 9 is the socket inode — the handle that ties it back to a process's fd.
NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 555001 1 0000 100 0
   1: 0100007F:D431 0100007F:1F40 01 00000000:00000000 00:00000000 00000000  1000        0 555002 1 0000 100 0
"""


def _mknet(proc, body: str = NET_TCP):
    (proc / "net").mkdir(parents=True, exist_ok=True)
    (proc / "net" / "tcp").write_text(body)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_port_resolves_to_the_process_holding_the_listening_socket(tmp_path):
    proc = tmp_path / "proc"
    d = _mkproc(proc, 100, SERVE, _traced_env(tmp_path / "t.jsonl"))
    _mknet(proc)
    fd = d / "fd"
    fd.mkdir()
    os.symlink("socket:[555001]", fd / "3")

    assert discover.pid_listening_on(0x1F40, proc) == 100
    # The established connection on the same port (state 01) must not match: a client
    # of the server is not the server.
    assert discover.pid_listening_on(0xD431, proc) is None


def test_unresolvable_port_explains_the_same_host_constraint(tmp_path):
    proc = tmp_path / "proc"
    _mknet(proc, "  sl  local_address rem_address   st\n")
    target, how = discover.resolve_target(port=9999, proc=proc)
    assert target is None and "same-host" in how


# --- the port the server told us about --------------------------------------


def test_port_is_read_off_the_servers_own_command_line():
    assert att._port_from_cmdline(SERVE) == 8000
    assert att._port_from_cmdline(["vllm", "serve", "m", "--port=9001"]) == 9001
    assert att._port_from_cmdline(["vllm", "serve", "m"]) == 8000


def test_base_url_prefers_explicit_then_flag_then_cmdline():
    target = discover.Target(pid=1, cmdline=["vllm", "serve", "m", "--port", "9100"])
    assert att._base_url_for(target, att.AttachOptions()) == "http://127.0.0.1:9100"
    assert att._base_url_for(target, att.AttachOptions(port=7000)) == "http://127.0.0.1:7000"
    assert (
        att._base_url_for(target, att.AttachOptions(base_url="http://127.0.0.1:1234/"))
        == "http://127.0.0.1:1234"
    )


def test_server_metric_lines_surface_missing_token_and_tpot_values():
    from gitm.serve.metrics import ServerWindow

    lines = att._server_metric_lines(
        ServerWindow(requests_finished=3, generation_tokens=None, ttft_mean_s=0.012)
    )

    assert "output-token count unavailable" in lines[0]
    assert "TPOT mean unavailable" in lines[1]
    assert "0 output tokens" not in "\n".join(lines)


def test_server_metric_lines_render_measured_values_without_caveats():
    from gitm.serve.metrics import ServerWindow

    lines = att._server_metric_lines(
        ServerWindow(
            requests_finished=3,
            generation_tokens=24,
            output_tokens_per_s=12,
            ttft_mean_s=0.012,
            tpot_mean_s=0.0035,
        )
    )

    assert lines == [
        "    server: 3 requests, 24 output tokens, 12 tok/s",
        "    server TTFT mean 12 ms   TPOT mean 3.5 ms",
    ]


def test_predicted_graph_surfaces_resolved_warnings_and_bytes_fallback(
    tmp_path, monkeypatch, capsys
):
    from gitm.planner.context import peak_for_sku
    from gitm.planner.moe_graph import spec_from_hf_config
    from gitm.planner.roofline import BatchConfig, ShardingConfig
    from gitm.serve import model_config as mc

    cfg = {
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 0,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 1,
        "vocab_size": 128,
        "n_routed_experts": 2,
        "n_shared_experts": 0,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "index_n_heads": 1,
        "index_head_dim": 8,
        "index_topk": 8,
        "sliding_window": 128,
        "compress_ratios": [0],
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
        "torch_dtype": "bfloat16",
    }
    spec = replace(spec_from_hf_config(cfg), expert_dtype="future_fp3")
    resolved = mc.LiveSpec(
        spec=spec,
        sharding=ShardingConfig(),
        batch=BatchConfig(batch=1, kv_cache_len=128),
        source_path=tmp_path / "config.json",
        model_ref="org/model",
        warnings=["decode batch was not observed; using batch=1 single-sequence floor"],
    )
    monkeypatch.setattr(mc, "live_moe_spec", lambda _target, **_kwargs: resolved)

    import gitm.planner.context as planner_context

    monkeypatch.setattr(
        planner_context,
        "build_planner_context",
        lambda: SimpleNamespace(
            peak=peak_for_sku("NVIDIA B200"),
            sku="NVIDIA B200",
            num_gpus=1,
            num_gpus_is_fallback=True,
        ),
    )

    att._emit_predicted_graph(discover.Target(pid=1, cmdline=[]), tmp_path)

    payload = json.loads((tmp_path / "predicted_moe_graph.json").read_text())
    assert payload["has_fallback_bytes"] is True
    assert payload["resident_weight_bytes_per_rank"] > 0
    assert payload["resident_weight_bytes_is_lower_bound"] is False
    assert payload["kv_bytes_per_token_per_sequence"] == 0.0
    assert payload["kv_fixed_bytes_per_sequence"] > 0
    assert payload["num_gpus_is_fallback"] is True
    assert any("GPU count was unavailable" in warning for warning in payload["warnings"])
    assert any(node["bytes_are_fallback"] for node in payload["nodes"])
    assert payload["warnings"]
    stdout = capsys.readouterr().out
    assert "single-sequence floor" in stdout
    assert "unknown-dtype bf16 fallback" in stdout


def test_predicted_graph_known_dtypes_leave_bytes_fallback_clean(tmp_path, monkeypatch):
    from gitm.planner.context import peak_for_sku
    from gitm.planner.moe_graph import spec_from_hf_config
    from gitm.planner.roofline import BatchConfig, ShardingConfig
    from gitm.serve import model_config as mc

    cfg = {
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 0,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 1,
        "vocab_size": 128,
        "n_routed_experts": 2,
        "n_shared_experts": 0,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "index_n_heads": 1,
        "index_head_dim": 8,
        "index_topk": 8,
        "sliding_window": 128,
        "compress_ratios": [0],
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
        "torch_dtype": "bfloat16",
    }
    resolved = mc.LiveSpec(
        spec=spec_from_hf_config(cfg),
        sharding=ShardingConfig(),
        batch=BatchConfig(batch=1, kv_cache_len=128),
        source_path=tmp_path / "config.json",
        model_ref="org/model",
    )
    monkeypatch.setattr(mc, "live_moe_spec", lambda _target, **_kwargs: resolved)

    import gitm.planner.context as planner_context

    monkeypatch.setattr(
        planner_context,
        "build_planner_context",
        lambda: SimpleNamespace(
            peak=peak_for_sku("NVIDIA B200"),
            sku="NVIDIA B200",
            num_gpus=1,
            num_gpus_is_fallback=False,
        ),
    )

    att._emit_predicted_graph(discover.Target(pid=1, cmdline=[]), tmp_path)
    payload = json.loads((tmp_path / "predicted_moe_graph.json").read_text())
    assert payload["has_fallback_bytes"] is False
    assert payload["resident_weight_bytes_per_rank"] > 0
    assert payload["resident_weight_bytes_is_lower_bound"] is False
    assert payload["kv_bytes_per_token_per_sequence"] == 0.0
    assert payload["kv_fixed_bytes_per_sequence"] > 0
    assert payload["num_gpus_is_fallback"] is False
    assert not any("GPU count was unavailable" in warning for warning in payload["warnings"])
    assert not any(node["bytes_are_fallback"] for node in payload["nodes"])


# --- preflight ---------------------------------------------------------------


def test_untraceable_target_fails_preflight_before_anything_is_touched(tmp_path):
    proc = tmp_path / "proc"
    _mkproc(proc, 200, SERVE, {"PATH": "/usr/bin"})
    checks = att.attach_checks(discover.classify(200, proc), "http://127.0.0.1:8000")

    assert checks[0].name == "target" and checks[0].status == "fail"
    # It stops there: no point checking a clock or a metrics endpoint for a window
    # that will never be opened.
    assert len(checks) == 1


def test_lone_process_cohort_is_flagged_as_probably_missing_the_engine(tmp_path):
    """One shard means only the frontend is collecting — the engine processes were
    spawned before the injection variable was exported. That traces cleanly and
    contains no model kernels, which is the most expensive way to learn nothing."""
    proc = tmp_path / "proc"
    trace_out = tmp_path / "trace.jsonl"
    _mkproc(proc, 100, SERVE, _traced_env(trace_out))
    trace_out.with_name(trace_out.name + ".100").write_text("")

    checks = att.attach_checks(discover.classify(100, proc), "http://127.0.0.1:8000")
    cohort = next(c for c in checks if c.name == "collector-cohort")
    assert cohort.status == "warn" and "only one CUDA process" in cohort.detail


def test_an_already_open_window_blocks_a_second_one(tmp_path, monkeypatch):
    """Two armed windows on one server is not a partial failure — it is two traces
    that each contain both windows' kernels."""
    proc = tmp_path / "proc"
    trace_out = tmp_path / "trace.jsonl"
    _mkproc(proc, 100, SERVE, _traced_env(trace_out))
    trace_out.with_name(trace_out.name + ".arm").touch()

    checks = att.attach_checks(discover.classify(100, proc), "http://127.0.0.1:8000")
    window = next(c for c in checks if c.name == "window")
    assert window.status == "fail" and "already open" in window.detail
    # ENV_OUT was only borrowed for the check, never left behind.
    assert injection.ENV_OUT not in os.environ or os.environ[injection.ENV_OUT] != str(trace_out)


def test_remote_server_is_refused_because_shards_are_local(tmp_path):
    proc = tmp_path / "proc"
    trace_out = tmp_path / "trace.jsonl"
    _mkproc(proc, 100, SERVE, _traced_env(trace_out))

    checks = att.attach_checks(discover.classify(100, proc), "http://10.0.0.5:8000")
    host = next(c for c in checks if c.name == "server-host")
    assert host.status == "fail" and "same host" in host.detail


# --- mode selection ----------------------------------------------------------


def test_observe_is_the_default_and_requests_switches_to_drive():
    assert att.AttachOptions().mode == "observe"
    assert att.AttachOptions(requests=32).mode == "drive"


# --- result classification ---------------------------------------------------


class _FakeTrace:
    run_id = "abc"
    source = "cupti"
    device_count = 4
    duration_ns = 10_000_000

    def __init__(self, events):
        self.events = events


class _FakeKernel:
    kind = "kernel"

    def __init__(self, name="sm90_gemm", start_ns=0, end_ns=1000, device=0):
        self.name, self.start_ns, self.end_ns, self.device = name, start_ns, end_ns, device
        self.duration_ns = end_ns - start_ns
        self.stream = 0
        self.correlation_id = 1


def _write(tmp_path, events, *, had_traffic):
    from gitm.serve.artifacts import write_capture_artifacts

    return write_capture_artifacts(
        tmp_path,
        trace=_FakeTrace(events),
        trace_path=tmp_path / "trace.jsonl",
        manifest={"workload_id": "vllm-attach"},
        had_traffic=had_traffic,
    )


def test_empty_trace_and_idle_window_are_different_failures(tmp_path):
    """They need different fixes: no kernels means the collector never saw the engine,
    no traffic means the window was pointed at an idle server. Collapsing them into
    one 'failed' sends the operator to the wrong place."""
    assert _write(tmp_path / "a", [], had_traffic=True).status == "no_kernels"
    assert _write(tmp_path / "b", [_FakeKernel()], had_traffic=False).status == "no_traffic"
    assert _write(tmp_path / "c", [_FakeKernel()], had_traffic=True).status == "ok"


def test_zero_duration_kernel_records_do_not_make_capture_successful(tmp_path):
    result = _write(
        tmp_path / "zero-duration",
        [_FakeKernel(start_ns=10, end_ns=10)],
        had_traffic=True,
    )

    assert result.status == "no_kernels"
    assert result.n_kernels == 0
    assert result.breakdown.n_invalid_duration == 1
    assert any("non-positive duration" in warning for warning in result.warnings)


def test_both_paths_write_the_same_artifact_set(tmp_path):
    """A driven benchmark and a production observation of the same server are only
    comparable if they leave the same files behind."""
    result = _write(tmp_path, [_FakeKernel()], had_traffic=True)
    assert result.ok
    written = {p.name for p in tmp_path.iterdir()}
    assert {"kernel_breakdown.json", "run_manifest.json"} <= written

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["trace"]["kernels"] == 1
    assert manifest["trace"]["run_id"] == "abc"


# --- CLI wiring --------------------------------------------------------------


def test_cli_parses_attach_flags():
    from gitm.cli import _parser

    args = _parser().parse_args(["capture", "attach", "--pid", "42", "--duration", "5"])
    assert args.cmd == "capture" and args.capture_mode == "attach"
    assert args.pid == 42 and args.duration == 5.0


def test_cli_rejects_pid_and_port_together():
    from gitm.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["capture", "attach", "--pid", "1", "--port", "8000"])


def test_cli_splits_the_serve_command_off_before_argparse(monkeypatch):
    """`vllm serve M --port 9000` shares flag names with gitm's own parser. If
    argparse claimed them, the capture would target a port the server was never told
    about — and the run would fail with a health-check timeout, not a flag error."""
    from gitm import cli

    seen = {}

    def fake_launch(args, serve_argv):
        seen["port"] = args.port
        seen["serve_argv"] = serve_argv
        return 0, None

    monkeypatch.setattr("gitm.serve.vllm.launch_and_capture", fake_launch)
    rc = cli.main(["capture", "serve", "--", "vllm", "serve", "M", "--port", "9000"])

    assert rc == 0
    assert seen["port"] == 8000  # gitm's own default, untouched by the serve command
    assert seen["serve_argv"] == ["vllm", "serve", "M", "--port", "9000"]


def test_the_dash_dash_split_is_scoped_to_capture(monkeypatch):
    """argparse's own ``--`` (end-of-flags) still has to work for every other command:
    `gitm analyze -- ./-oddly-named.json` must reach analyze with its path intact."""
    from gitm import cli

    seen = {}
    monkeypatch.setattr(
        "gitm.importers.analyze.analyze_paths",
        lambda paths, **kw: seen.update(paths=list(paths)) or _FakeAnalysis(),
    )
    cli.main(["analyze", "--out", "/tmp/r.md", "--", "a.json", "-oddly-named.json"])

    # Nothing was stripped: `--` reached argparse, which read the rest as positionals
    # — including the one that would otherwise look like a flag.
    assert [p.name for p in seen["paths"]] == ["a.json", "-oddly-named.json"]


class _FakeAnalysis:
    summary = {"n_workloads": 1, "n_failures": 0}
    workloads = [object()]


def test_bare_capture_prints_help_and_exits_non_zero(capsys):
    """An incomplete command must not read as success to a script."""
    from gitm.cli import main

    rc = main(["capture"])
    assert rc == 2
    assert "{serve,attach}" in capsys.readouterr().out


def test_cli_list_reports_every_server_with_its_verdict(tmp_path, monkeypatch, capsys):
    proc = tmp_path / "proc"
    _mkproc(proc, 100, SERVE, _traced_env(tmp_path / "a.jsonl"))
    _mkproc(proc, 200, SERVE, {"PATH": "/usr/bin"})
    monkeypatch.setattr(discover, "PROC", proc)

    rc = att.print_targets(proc)
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert [t["pid"] for t in listed] == [100, 200]
    assert [t["traceable"] for t in listed] == [True, False]
