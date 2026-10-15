"""Cross-process trace collection (gitm.tracer.injection).
The C half can only be exercised on a GPU box. Everything here is the Python half:
mode detection, shard merging, and the window filter — which is where the subtle
bugs live, because a wrong window silently turns a vLLM trace into 80 seconds of
weight loading and CUDA-graph capture.
"""

from __future__ import annotations

import importlib
import json
import os

import pytest

from gitm.tracer import injection
from gitm.tracer.capture import capture

# gitm.tracer re-exports the capture() function as gitm.tracer.capture, so the
# module has to be fetched explicitly to patch its internals.
capture_mod = importlib.import_module("gitm.tracer.capture")


def _kernel(name: str, start: int, end: int) -> str:
    return json.dumps({
        "kind": "kernel", "name": name, "start_ns": start, "end_ns": end,
        "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": 3,
        "grid": [1, 1, 1], "block": [32, 1, 1],
        "static_shared_mem": 0, "dynamic_shared_mem": 0, "registers_per_thread": 32,
    })


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """A run that looks injected, without needing the .so to exist."""
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv(injection.ENV_LIB, f"/opt/gitm/{injection.LIB_NAME}")
    monkeypatch.setenv(injection.ENV_OUT, str(out))
    monkeypatch.setenv(injection.ENV_SETTLE, "0")  # no real sleep in tests
    return out


# --------------------------------------------------------------------------- #
# mode detection                                                              #
# --------------------------------------------------------------------------- #
def test_inactive_without_env(monkeypatch):
    monkeypatch.delenv(injection.ENV_LIB, raising=False)
    monkeypatch.delenv(injection.ENV_OUT, raising=False)
    assert not injection.active()


def test_inactive_when_another_profiler_owns_the_injection_hook(run_env, monkeypatch):
    """nsys sets CUDA_INJECTION64_PATH too. Those records are not ours to merge."""
    monkeypatch.setenv(injection.ENV_LIB, "/opt/nsight/libToolsInjection64.so")
    assert not injection.active()


def test_active_with_our_lib_and_an_output(run_env):
    assert injection.active()
    assert injection.active_vendor() == "nvidia"


def test_active_vendor_amd_via_rocp_tool_libraries(tmp_path, monkeypatch):
    """ROCP_TOOL_LIBRARIES is the AMD hook — colon-separated, and ours may ride
    alongside another tool's entry without disowning the run."""
    from gitm.tracer import _rocm

    monkeypatch.delenv(injection.ENV_LIB, raising=False)
    monkeypatch.setenv(injection.ENV_OUT, str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(
        injection.ENV_ROCP, f"/opt/other/libtool.so:/opt/gitm/{_rocm.LIB_NAME}"
    )
    assert injection.active_vendor() == "amd"


def test_inactive_when_only_another_rocm_tool_is_listed(tmp_path, monkeypatch):
    """rocprofv3 sets ROCP_TOOL_LIBRARIES too. Those records are not ours."""
    monkeypatch.delenv(injection.ENV_LIB, raising=False)
    monkeypatch.setenv(injection.ENV_OUT, str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv(injection.ENV_ROCP, "/opt/rocm/lib/librocprofv3-tool.so")
    assert injection.active_vendor() is None


def test_run_env_amd_sets_the_rocm_hook_and_no_nvtx_injection_path(tmp_path):
    """On AMD, nvtx=True must NOT render NVTX_INJECTION64_PATH: rocTX reaches
    the injected tool through rocprofiler's own marker service, and exporting
    the NVIDIA variable would only mislead whoever reads the env."""
    env = injection.run_env(tmp_path / "t.jsonl", nvtx=True, vendor="amd")
    assert injection.ENV_ROCP in env
    assert env[injection.ENV_NVTX] == "1"
    assert injection.ENV_LIB not in env
    assert injection.ENV_NVTX_INJECT not in env


def test_run_env_nvidia_shape_is_unchanged(tmp_path):
    env = injection.run_env(tmp_path / "t.jsonl", vendor="nvidia")
    assert injection.ENV_LIB in env
    assert injection.ENV_ROCP not in env


def test_run_env_amd_starts_vllm_workers_with_spawn(tmp_path):
    """A forked EngineCore on ROCm keeps the parent's HIP runtime but not the
    rocprofiler tool, so it records nothing (P1-1). Spawn reloads the tool."""
    env = injection.run_env(tmp_path / "t.jsonl", vendor="amd")
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_run_env_nvidia_leaves_the_start_method_alone(tmp_path):
    env = injection.run_env(tmp_path / "t.jsonl", vendor="nvidia")
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in env


def _factory_env(monkeypatch, vendor):
    """What the vLLM factory leaves in the environment before importing vLLM.

    vLLM is blocked from importing, so the factory stops right after the point
    under test instead of building an engine.
    """
    import sys

    from gitm import workloads

    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)
    monkeypatch.setattr(injection, "active_vendor", lambda: vendor)
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError):
        workloads._vllm_decode_factory(None)
    return os.environ


def test_vllm_factory_never_switches_an_embedded_caller_to_spawn(monkeypatch):
    """Spawn re-imports __main__ in every worker. A script that exists as a file
    can still start the workload at top level, and each worker would start it
    again, so the factory cannot tell spawn is safe. It warns instead."""
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    with pytest.warns(RuntimeWarning, match="fork"):
        env = _factory_env(monkeypatch, "amd")
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in env


def test_vllm_factory_keeps_an_explicit_start_method(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    assert _factory_env(monkeypatch, "amd")["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_vllm_factory_leaves_the_start_method_alone_on_nvidia(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in _factory_env(monkeypatch, "nvidia")


# --------------------------------------------------------------------------- #
# shard merge                                                                 #
# --------------------------------------------------------------------------- #
def test_merges_shards_from_every_pid_sorted_by_start(run_env):
    """The whole point: the child process's kernels must appear in the trace."""
    run_env.with_name(run_env.name + ".100").write_text(_kernel("parent_memset", 30, 40) + "\n")
    run_env.with_name(run_env.name + ".9335").write_text(
        _kernel("flash_fwd_kernel", 10, 20) + "\n" + _kernel("rms_norm", 50, 60) + "\n"
    )

    events = injection.read_shards()

    assert [e.name for e in events] == ["flash_fwd_kernel", "parent_memset", "rms_norm"]


def test_window_filter_drops_records_outside_the_capture_window(run_env):
    """Model load and CUDA-graph capture run before the window and must not count."""
    run_env.with_name(run_env.name + ".9335").write_text(
        "\n".join([
            _kernel("graph_capture_warmup", 50, 60),   # before window
            _kernel("decode_step", 150, 160),          # inside
            _kernel("teardown", 500, 510),             # after window
        ]) + "\n"
    )

    events = injection.read_shards(start_ns=100, end_ns=200)

    assert [e.name for e in events] == ["decode_step"]


def test_collector_drop_report_warns_and_is_not_an_event(run_env):
    """A collector writes {"kind":"meta","dropped_records":N} for every armed
    buffer. A non-zero count must surface as a loss warning, not decode as an
    event and not count as a malformed line."""
    shard = run_env.with_name(run_env.name + ".111")
    shard.write_text(
        json.dumps({"kind": "meta", "dropped_records": 7}) + "\n"
        + _kernel("k", 100, 200) + "\n"
    )
    with pytest.warns(RuntimeWarning, match="7 record"):
        events = injection.read_shards()
    assert [e.name for e in events] == ["k"]


def test_loss_count_is_known_only_when_every_shard_reports_one(run_env, monkeypatch):
    """Zero must mean a collector counted and found none, never that it could not count."""
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: True)
    meta = lambda n, ts=None: json.dumps(  # noqa: E731
        {"kind": "meta", "dropped_records": n, **({"timestamp_ns": ts} if ts else {})})
    a = run_env.with_name(run_env.name + ".111")
    b = run_env.with_name(run_env.name + ".222")
    a.write_text(meta(0, 60) + "\n" + _kernel("k", 100, 200) + "\n")
    assert injection.read_shards_with_loss()[1] == 0

    # A second shard from a library that writes no count: unknown, not zero.
    b.write_text(_kernel("old", 300, 400) + "\n")
    assert injection.read_shards_with_loss()[1] is None
    # Even when none of its events fall in the window: its buffers may not have landed.
    assert injection.read_shards_with_loss(50, 250)[1] is None
    # An empty shard, and one holding only an earlier capture's count, are as silent.
    for silent in ("", meta(0, 10) + "\n" + _kernel("old", 20, 30) + "\n"):
        b.write_text(silent)
        assert injection.read_shards_with_loss(50, 250)[1] is None, silent

    # Collector drops and unreadable lines add up; a count from before the window is
    # an earlier capture's, and a collector that could not say leaves it unknown.
    b.write_text(meta(4, 10) + "\n" + meta(2, 150) + "\n" + _kernel("k2", 120, 130)
                 + "\n" + '{"kind":"kernel","na')
    with pytest.warns(RuntimeWarning):
        assert injection.read_shards_with_loss(50, 250)[1] == 3
    b.write_text(meta(-1, 150) + "\n" + _kernel("k2", 120, 130) + "\n")
    assert injection.read_shards_with_loss(50, 250)[1] == -1  # lost, and not counted


def test_a_process_that_died_without_its_final_line_leaves_the_loss_unknown(run_env, monkeypatch):
    """A killed worker's last buffers never reach its shard; an orderly exit says so."""
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: False)
    shard = run_env.with_name(run_env.name + ".111")
    body = json.dumps({"kind": "meta", "dropped_records": 0}) + "\n" + _kernel("k", 1, 2) + "\n"
    shard.write_text(body)
    assert injection.read_shards_with_loss()[1] == -1
    shard.write_text(body + json.dumps({"kind": "meta", "final": True, "timestamp_ns": 3}) + "\n")
    assert injection.read_shards_with_loss()[1] == 0


def test_partial_trailing_line_from_a_killed_process_is_tolerated(run_env):
    """A SIGKILLed child leaves a half-written record; losing it must not fail the run."""
    shard = run_env.with_name(run_env.name + ".9335")
    shard.write_text(_kernel("decode_step", 10, 20) + "\n" + '{"kind":"kernel","na')

    events = injection.read_shards()

    assert [e.name for e in events] == ["decode_step"]


def test_arm_marker_is_not_mistaken_for_a_shard(run_env):
    injection.arm()
    run_env.with_name(run_env.name + ".9335").write_text(_kernel("decode_step", 10, 20) + "\n")

    assert injection.arm_path().exists()
    assert injection.shard_paths() == [run_env.with_name(run_env.name + ".9335")]
    assert [e.name for e in injection.read_shards()] == ["decode_step"]

    injection.disarm()
    assert not injection.arm_path().exists()


def test_clear_shards_prevents_a_stale_run_bleeding_into_this_one(run_env):
    run_env.with_name(run_env.name + ".111").write_text(_kernel("last_run", 10, 20) + "\n")

    injection.clear_shards()

    assert injection.read_shards() == []


# --------------------------------------------------------------------------- #
# capture() mode switch                                                       #
# --------------------------------------------------------------------------- #
def test_capture_merges_injected_shards_and_never_starts_the_local_backend(
    run_env, tmp_path, monkeypatch
):
    """Under injection, capture() must NOT call backend.start().
    CUPTI allows one activity-callback registration per process. The injected library
    already holds it; registering again from the in-process shim would clobber it and
    we would collect nothing.
    """
    def _boom():
        raise AssertionError("capture() started the in-process backend under injection")

    monkeypatch.setattr(capture_mod, "_backend", _boom)
    monkeypatch.setattr(injection, "cupti_now", iter([100, 200]).__next__)

    out = tmp_path / "merged.jsonl"
    with capture(out, workload_id="vllm-decode") as trace:
        # Stand in for the injected library writing from the EngineCore child.
        run_env.with_name(run_env.name + ".9335").write_text(
            "\n".join([
                _kernel("model_load", 10, 20),       # before the window opened
                _kernel("flash_fwd_kernel", 150, 160),
            ]) + "\n"
        )

    assert [e.name for e in trace.events] == ["flash_fwd_kernel"]
    assert trace.vendor == "nvidia"
    assert not injection.arm_path().exists()  # window closed

    written = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert written[0]["_header"]["workload_id"] == "vllm-decode"
    assert [e["name"] for e in written[1:]] == ["flash_fwd_kernel"]


def test_capture_falls_back_to_the_local_backend_when_not_injected(tmp_path, monkeypatch):
    monkeypatch.delenv(injection.ENV_LIB, raising=False)
    monkeypatch.delenv(injection.ENV_OUT, raising=False)

    started = []

    class FakeBackend:
        vendor = "nvidia"

        def device_count(self):
            return 1

        def start(self):
            started.append(True)

        def stop(self):
            return []

    monkeypatch.setattr(capture_mod, "_backend", FakeBackend)

    with capture(tmp_path / "t.jsonl"):
        pass

    assert started == [True]


def test_stale_shard_cleanup_never_deletes_a_live_process_shard(run_env):
    """The bug that made a working injection look like a dead one.
    The injected library opens its shard at CUDA init — for vLLM that is during the
    engine build, BEFORE capture() is entered. Unlinking it then leaves the live
    EngineCore writing into a deleted inode: no records on disk, no error, and a merge
    that returns empty exactly as if the driver had never loaded the library.
    """
    import os

    live = run_env.with_name(f"{run_env.name}.{os.getpid()}")   # us: definitely alive
    dead = run_env.with_name(f"{run_env.name}.999999")          # no such process
    live.write_text(_kernel("engine_core_decode", 10, 20) + "\n")
    dead.write_text(_kernel("last_run", 10, 20) + "\n")

    injection.clear_stale_shards()

    assert live.exists(), "deleted a shard a live process is holding open"
    assert not dead.exists()


def test_clear_shards_still_wipes_everything_when_nothing_is_collecting(run_env):
    import os

    run_env.with_name(f"{run_env.name}.{os.getpid()}").write_text(_kernel("k", 1, 2) + "\n")

    injection.clear_shards()

    assert injection.shard_paths() == []


def test_set_decode_run_defaults_fills_env_and_respects_exports(tmp_path, monkeypatch):
    """scripts/fp8_ab.py must Just Work with no manual exports — but anything the
    user did export has to win."""
    from gitm.workloads import set_decode_run_defaults

    for k in ("CUDA_INJECTION64_PATH", "GITM_TRACE_OUT", "GITM_VLLM_MODEL",
              "GITM_VLLM_GPU_MEM", "GITM_VLLM_PROMPTS", "GITM_VLLM_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GITM_VLLM_GPU_MEM", "0.30")   # a user override
    monkeypatch.setenv("GITM_TRACE_OUT", str(tmp_path / "t.jsonl"))

    env = set_decode_run_defaults()

    assert env["GITM_VLLM_GPU_MEM"] == "0.30"                    # export preserved
    assert env["GITM_VLLM_PROMPTS"] == "512"                    # default filled
    assert env["GITM_VLLM_MAX_TOKENS"] == "2048"
    assert env["CUDA_INJECTION64_PATH"].endswith(injection.LIB_NAME)
    assert (tmp_path / "t.jsonl").parent.exists()


# ── correlation records must survive read_shards ────────────────────────────
#
# The bug this guards: a marker carries `timestamp_ns`, not `start_ns`. The
# window filter required `start_ns`, so every marker was counted as malformed and
# discarded before correlation ever saw one. The capture still succeeded, the
# trace still decoded, and every kernel came back with `range_op` null — no error
# anywhere. On a real H200 run that silently threw away 502,491 markers.


def _corr_shard(tmp_path, monkeypatch, lines):
    import json as _json

    base = tmp_path / "trace.jsonl"
    (tmp_path / "trace.jsonl.999").write_text(
        "\n".join(_json.dumps(r) for r in lines) + "\n"
    )
    monkeypatch.setenv("GITM_TRACE_OUT", str(base))
    return base


def _corr_kernel(t, cid, name="nvjet_sm90_tst_128x8_TNT"):
    return {"kind": "kernel", "name": name, "start_ns": t, "end_ns": t + 100,
            "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": cid,
            "grid": [1, 1, 1], "block": [1, 1, 1], "static_shared_mem": 0,
            "dynamic_shared_mem": 0, "registers_per_thread": 32}


def test_markers_are_not_discarded_as_malformed(tmp_path, monkeypatch):
    from gitm.tracer import injection

    _corr_shard(tmp_path, monkeypatch, [
        {"kind": "marker", "name": "L3/qkv_proj", "timestamp_ns": 1000,
         "marker_id": 1, "marker_flags": 0, "thread_id": 7},
        {"kind": "runtime", "start_ns": 1010, "end_ns": 1020,
         "correlation_id": 42, "thread_id": 7},
        {"kind": "marker", "name": "", "timestamp_ns": 1100,
         "marker_id": 1, "marker_flags": 1, "thread_id": 7},
        _corr_kernel(2000, 42),
    ])
    events = injection.read_shards()
    kernels = [e for e in events if e.kind == "kernel"]
    assert len(kernels) == 1
    assert (kernels[0].range_op, kernels[0].range_layer) == ("qkv_proj", 3)


def test_a_range_opening_before_the_window_still_attributes(tmp_path, monkeypatch):
    """Windowing correlation records breaks pairing and un-attributes silently.

    A range pushed before the capture window opened still encloses launches
    inside it — vLLM pushes a layer range and the kernels arrive later. Dropping
    the start half leaves an end with nothing to pair to, so the range vanishes
    and every kernel it covered comes back unattributed.
    """
    from gitm.tracer import injection

    _corr_shard(tmp_path, monkeypatch, [
        {"kind": "marker", "name": "L1/moe_routed", "timestamp_ns": 10,
         "marker_id": 1, "marker_flags": 0, "thread_id": 7},      # before window
        {"kind": "runtime", "start_ns": 20, "end_ns": 30,
         "correlation_id": 42, "thread_id": 7},                    # before window
        {"kind": "marker", "name": "", "timestamp_ns": 9000,
         "marker_id": 1, "marker_flags": 1, "thread_id": 7},
        _corr_kernel(5000, 42),                                         # inside window
    ])
    events = injection.read_shards(1000, 8000)
    (kernel,) = [e for e in events if e.kind == "kernel"]
    assert (kernel.range_op, kernel.range_layer) == ("moe_routed", 1)


def test_kernels_outside_the_window_are_still_dropped(tmp_path, monkeypatch):
    """Exempting correlation records must not exempt the events themselves —
    the window filter exists because the collector runs for the process's whole
    life, and weight loading would otherwise dominate the trace."""
    from gitm.tracer import injection

    _corr_shard(tmp_path, monkeypatch, [
        _corr_kernel(10, 1), _corr_kernel(5000, 2), _corr_kernel(90000, 3),
    ])
    events = injection.read_shards(1000, 8000)
    assert [e.start_ns for e in events if e.kind == "kernel"] == [5000]


def test_a_truncated_final_line_is_still_reported(tmp_path, monkeypatch):
    """The genuine malformed case must not be masked by the marker exemption."""
    import warnings as _w

    from gitm.tracer import injection

    base = tmp_path / "trace.jsonl"
    (tmp_path / "trace.jsonl.999").write_text(
        '{"kind":"kernel","name":"a","start_ns":1,"end_ns":2,"device_id":0,'
        '"context_id":1,"stream_id":7,"correlation_id":1,"grid":[1,1,1],'
        '"block":[1,1,1],"static_shared_mem":0,"dynamic_shared_mem":0,'
        '"registers_per_thread":1}\n{"kind":"kernel","name":"trunc\n'
    )
    monkeypatch.setenv("GITM_TRACE_OUT", str(base))
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        injection.read_shards()
    assert any("malformed or incomplete" in str(c.message) for c in caught)


def test_loss_count_edges(run_env, monkeypatch):
    """What must leave the loss unknown, and what counts as a lost record."""
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: True)
    meta = lambda n, **kw: json.dumps({"kind": "meta", "dropped_records": n, **kw})  # noqa: E731
    shard = run_env.with_name(run_env.name + ".111")
    good = meta(0, timestamp_ns=150) + "\n" + _kernel("k", 160, 170) + "\n"
    shard.write_text(good)
    assert injection.read_shards_with_loss(100, 300)[1] == 0
    # A count exactly at the window start is this window's.
    shard.write_text(meta(2, timestamp_ns=100) + "\n" + _kernel("k", 160, 170) + "\n")
    with pytest.warns(RuntimeWarning):
        assert injection.read_shards_with_loss(100, 300)[1] == 2
    # A stamp of 0 is a failed clock read, not an earlier capture; true is not a count.
    for bad in (meta(4096, timestamp_ns=0), meta(1, timestamp_ns=True), meta(True)):
        shard.write_text(good + bad + "\n")
        assert injection.read_shards_with_loss(100, 300)[1] is None, bad
    # Lines that are not records, and records the decoder has no model for, are lost.
    shard.write_text(good + "[1]\n" + json.dumps({"kind": "kernel", "start_ns": "x"}) + "\n"
                     + json.dumps({"kind": "novel", "start_ns": 165}) + "\n")
    with pytest.warns(RuntimeWarning):
        assert injection.read_shards_with_loss(100, 300)[1] == 3


def test_an_unreadable_shard_leaves_the_loss_unknown(run_env, monkeypatch):
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: True)
    ok = run_env.with_name(run_env.name + ".111")
    ok.write_text(json.dumps({"kind": "meta", "dropped_records": 0}) + "\n"
                  + _kernel("k", 100, 200) + "\n")
    bad = run_env.with_name(run_env.name + ".222")
    bad.write_text("x")
    bad.chmod(0)
    try:
        with pytest.warns(RuntimeWarning, match="unreadable"):
            assert injection.read_shards_with_loss()[1] == -1
    finally:
        bad.chmod(0o644)


def test_capture_writes_the_merged_loss_count_to_the_header(run_env, tmp_path, monkeypatch):
    monkeypatch.setattr(injection, "clock_now", lambda: None)
    monkeypatch.setattr(injection, "settle", lambda: None)
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: True)
    out = tmp_path / "trace.jsonl"
    with pytest.warns(Warning), capture(out) as trace:
        run_env.with_name(run_env.name + ".111").write_text(
            json.dumps({"kind": "meta", "dropped_records": 4}) + "\n" + _kernel("k", 1, 2) + "\n")
    assert trace.dropped_records == 4
    assert json.loads(out.read_text().splitlines()[0])["_header"]["dropped_records"] == 4


def test_what_cannot_be_placed_or_was_never_collected_is_unknown(run_env, monkeypatch):
    # No shard at all: the collector never loaded, which is not a capture with zero loss.
    assert injection.read_shards_with_loss(100, 300) == ([], None)
    shard = run_env.with_name(run_env.name + ".111")
    k = _kernel("k", 160, 170) + "\n"
    meta = lambda n, **kw: json.dumps({"kind": "meta", "dropped_records": n, **kw}) + "\n"  # noqa: E731
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: True)
    # A count with no stamp cannot be placed in a window; a float or text stamp is not one.
    for line in (meta(0), meta(0, timestamp_ns=150.0), meta(0, timestamp_ns="150")):
        shard.write_text(meta(0, timestamp_ns=150) + k + line)
        assert injection.read_shards_with_loss(100, 300)[1] is None, line
    # A count stamped after the window's end still counts: the last buffer lands during settle.
    shard.write_text(meta(2, timestamp_ns=400) + k)
    with pytest.warns(RuntimeWarning):
        assert injection.read_shards_with_loss(100, 300)[1] == 2
    # A dead process with a count and no final line is unknown, whatever else it wrote.
    monkeypatch.setattr(injection, "_pid_alive", lambda pid: False)
    shard.write_text(meta(0, timestamp_ns=150) + json.dumps(
        {"kind": "marker", "timestamp_ns": 160, "marker_id": 1, "marker_flags": 0, "name": "x",
         "thread_id": 1}) + "\n")
    assert injection.read_shards_with_loss(100, 300)[1] == -1


def _cli_run_env(monkeypatch, vendor, argv0="/venv/bin/gitm"):
    """The environment `gitm run` hands the loop, without running the loop.

    ``argv0`` is how the process was started: the ``gitm`` console script by
    default, which is the entry point that may choose spawn.
    """
    import sys

    import gitm
    from gitm.cli import main

    monkeypatch.setattr(sys, "argv", [argv0])

    seen: dict[str, str | None] = {}

    def fake_optimize(**kw):
        seen["method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        return {"summary": {"status": "ok"}, "report_md": "", "run_dir": None}

    monkeypatch.setattr(injection, "active_vendor", lambda: vendor)
    monkeypatch.setattr(gitm, "optimize", fake_optimize)
    assert main(["run", "--workload", "vllm-decode", "--no-history"]) == 0
    return seen["method"]


def test_gitm_run_starts_workers_with_spawn_on_amd(monkeypatch):
    """The gitm command is a console script a spawned worker can re-import, so
    this is the entry point that can choose spawn for the operator."""
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert _cli_run_env(monkeypatch, "amd") == "spawn"


def test_gitm_run_keeps_an_explicit_start_method(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    assert _cli_run_env(monkeypatch, "amd") == "fork"


def test_main_called_from_another_entry_point_does_not_choose_spawn(monkeypatch):
    """python -c, a notebook or an unguarded script can reach main() too, and a
    spawned worker re-importing them can fail or re-run their work."""
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert _cli_run_env(monkeypatch, "amd", argv0="-c") is None


def test_gitm_run_leaves_the_start_method_alone_on_nvidia(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert _cli_run_env(monkeypatch, "nvidia") is None
