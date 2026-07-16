"""Cross-process trace collection (gitm.tracer.injection).
The C half can only be exercised on a GPU box. Everything here is the Python half:
mode detection, shard merging, and the window filter — which is where the subtle
bugs live, because a wrong window silently turns a vLLM trace into 80 seconds of
weight loading and CUDA-graph capture.
"""

from __future__ import annotations

import importlib
import json

import pytest

from gitm.tracer import injection
from gitm.tracer.capture import capture

# gitm.tracer re-exports the capture() *function* as gitm.tracer.capture, so the
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
