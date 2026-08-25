"""Tests for the traced ``vllm serve`` driver (gitm/serve/vllm.py).

GPU-free. The parts that only ever run on a pod — CUPTI, the engine — are not
exercised, but everything that decides whether the pod run produces a usable
result is: the SSE stream parse that TTFT comes from (against a real socket, not a
mock), the prompt builder's cache-defeating property, the ``--`` command
passthrough, and the preflight's classification of a rejected serve flag.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gitm.serve import vllm as sc

# --- streaming request: the TTFT measurement --------------------------------


class _SSEHandler(BaseHTTPRequestHandler):
    """A minimal vLLM-shaped SSE responder. ``chunks`` is set per test."""

    chunks: list[dict] = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in self.chunks:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a):  # silence the per-request stderr line
        pass


@pytest.fixture
def sse_server():
    def serve(chunks):
        _SSEHandler.chunks = chunks
        httpd = HTTPServer(("127.0.0.1", 0), _SSEHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{httpd.server_port}", httpd

    servers = []
    yield lambda chunks: servers.append(serve(chunks)) or servers[-1][0]
    for _, httpd in servers:
        httpd.shutdown()


def _delta(**kw):
    return {"choices": [{"delta": kw}]}


def test_streamed_request_yields_ttft_and_token_count(sse_server):
    base = sse_server([
        _delta(role="assistant"),          # no text: must not count as the first token
        _delta(content="hello"),
        _delta(content=" world"),
        {"choices": [], "usage": {"completion_tokens": 7}},
    ])
    rec = sc.one_request(base, "m", "p", max_tokens=8, ignore_eos=True, timeout_s=30)

    assert rec is not None
    assert rec.ttft_s is not None and rec.ttft_s >= 0
    assert rec.finished_wall_s >= rec.first_token_wall_s
    # usage wins over the chunk count (2 text chunks, but 7 real tokens)
    assert rec.n_output_tokens == 7
    assert rec.tpot_s is not None


def test_reasoning_content_starts_the_ttft_clock(sse_server):
    """With --reasoning-parser qwen3 the first token arrives as reasoning_content.
    Counting only `content` would push TTFT out by the whole reasoning block."""
    base = sse_server([
        _delta(reasoning_content="thinking"),
        _delta(content="answer"),
        {"choices": [], "usage": {"completion_tokens": 2}},
    ])
    rec = sc.one_request(base, "m", "p", max_tokens=4, ignore_eos=True, timeout_s=30)

    assert rec is not None and rec.first_token_wall_s is not None
    assert rec.first_token_wall_s < rec.finished_wall_s


def test_chunk_count_is_the_fallback_when_usage_is_absent(sse_server):
    base = sse_server([_delta(content="a"), _delta(content="b"), _delta(content="c")])
    rec = sc.one_request(base, "m", "p", max_tokens=4, ignore_eos=True, timeout_s=30)

    assert rec is not None and rec.n_output_tokens == 3


def test_unreachable_server_returns_none_not_a_fake_record():
    # A failed request must never become a RequestRecord: an invented arrival with no
    # first token would silently drop out of the percentiles as "unmeasurable" rather
    # than being counted as the failure it is.
    assert sc.one_request("http://127.0.0.1:1", "m", "p", 4, True, 0.5) is None


def test_records_feed_the_existing_summary(sse_server):
    from gitm.tracer.vllm_stats import summarize_requests

    base = sse_server([_delta(content="a"), _delta(content="b"),
                       {"choices": [], "usage": {"completion_tokens": 4}}])
    recs, failures = sc.drive_load(base, "m", ["p1", "p2", "p3"], concurrency=3,
                                   max_tokens=4, ignore_eos=True, timeout_s=30)

    assert failures == 0 and len(recs) == 3
    summary = summarize_requests(recs)
    assert summary.n_requests == 3
    assert summary.n_ttft == 3
    assert summary.window_s is not None


# --- prompts ----------------------------------------------------------------


def test_prompts_are_deterministic_and_not_prefix_shareable():
    a = sc.build_prompts(16, 256, seed=42)
    assert a == sc.build_prompts(16, 256, seed=42)
    assert len(set(a)) == 16
    # No two prompts may share a long prefix, or vLLM's prefix cache serves the
    # prefill and the run stops measuring prefill at all.
    first_words = {p.split()[0] for p in a}
    assert len(first_words) > 1


def test_prompt_length_tracks_the_requested_token_count():
    short = sc.build_prompts(1, 128, seed=1)[0].split()
    long = sc.build_prompts(1, 1024, seed=1)[0].split()
    assert len(long) > len(short) * 3


# --- argv plumbing ----------------------------------------------------------


def test_default_serve_argv_is_the_pinned_experiment():
    assert sc.DEFAULT_SERVE_ARGV[:2] == ["vllm", "serve"]
    assert sc._arg_of(sc.DEFAULT_SERVE_ARGV, "--max-num-seqs") == "256"
    assert sc._arg_of(sc.DEFAULT_SERVE_ARGV, "--gpu-memory-utilization") == "0.95"


def test_arg_of_reads_both_spellings():
    assert sc._arg_of(["--port", "9000"], "--port") == "9000"
    assert sc._arg_of(["--port=9000"], "--port") == "9000"
    assert sc._arg_of(["--other", "1"], "--port") is None


def test_serve_args_check_is_a_warn_when_vllm_is_absent():
    """vLLM is not installed on a dev box, and a check that cannot run must not block
    the run — the server would reject bad flags itself."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("vllm installed — this test covers the unavailable path")

    checks = sc.check_serve_args(["vllm", "serve", "some/model", "--tensor-parallel-size", "2"])
    assert [c.status for c in checks] == ["warn"]
    assert "could not validate" in checks[0].detail


# --- topology ---------------------------------------------------------------

# Real `nvidia-smi topo -m` shapes. The leading tab on the header line and the
# " X " diagonal (spaces around it) are what a hand-rolled parser gets wrong.
TOPO_NVSWITCH_4 = """\t GPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV18\tNV18\tNV18\t0-51\t0
GPU1\tNV18\t X \tNV18\tNV18\t0-51\t0
GPU2\tNV18\tNV18\t X \tNV18\t0-51\t0
GPU3\tNV18\tNV18\tNV18\t X \t0-51\t0
"""

# Two NVLink-bridged pairs, PCIe between them — the topology that makes TP=4 look
# inexplicably worse than TP=2 while a GPU0<->GPU1 spot check reports "healthy".
TOPO_SPLIT_PAIRS = """\t GPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity
GPU0\t X \tNV4\tSYS\tSYS\t0-23
GPU1\tNV4\t X \tSYS\tSYS\t0-23
GPU2\tSYS\tSYS\t X \tNV4\t24-47
GPU3\tSYS\tSYS\tNV4\t X \t24-47
"""


def test_topo_parser_handles_the_real_matrix_shape():
    rows = sc.parse_topo(TOPO_NVSWITCH_4)
    assert sorted(rows) == [0, 1, 2, 3]
    assert rows[0][1] == "NV18"      # GPU0 -> GPU1
    assert rows[2][3] == "NV18"      # GPU2 -> GPU3
    assert rows[0][0] == "X"         # diagonal survives the split


def test_all_pairs_nvlink_passes(monkeypatch):
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_NVSWITCH_4))
    checks = sc.check_nvlink(4)
    assert [c.status for c in checks] == ["pass"]
    assert "6 pairs" in checks[0].detail   # 4 choose 2


def test_split_pairs_caught_at_tp4_but_not_tp2(monkeypatch):
    """The regression this test exists for: checking only GPU0<->GPU1 calls this
    box healthy, and a TP=4 run then measures PCIe while reporting NVLink."""
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_SPLIT_PAIRS))

    at_4 = sc.check_nvlink(4)
    assert at_4[0].status == "warn"
    assert "4 of 6 rank pairs" in at_4[0].detail
    assert "GPU0<->GPU2 = SYS" in at_4[0].detail

    # TP=2 uses only GPU0/GPU1, which really are bridged — no false alarm
    assert sc.check_nvlink(2)[0].status == "pass"


def test_masked_devices_check_the_physical_pairs_they_actually_use(monkeypatch):
    """CUDA_VISIBLE_DEVICES=2,3 means the pair that matters is (2,3), not (0,1).
    On the split box those two ARE bridged, so checking (0,1) by habit would be
    right by luck here and wrong the moment the mask is 1,2."""
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_SPLIT_PAIRS))

    masked_23 = sc.Devices(indices=[2, 3], count=2, source="CUDA_VISIBLE_DEVICES")
    assert sc.check_nvlink(2, masked_23)[0].status == "pass"

    masked_12 = sc.Devices(indices=[1, 2], count=2, source="CUDA_VISIBLE_DEVICES")
    checks = sc.check_nvlink(2, masked_12)
    assert checks[0].status == "warn"
    assert "GPU1<->GPU2 = SYS" in checks[0].detail


def test_uuid_mask_cannot_be_mapped_and_says_so(monkeypatch):
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_NVSWITCH_4))
    dev = sc.Devices(indices=None, count=2, source="CUDA_VISIBLE_DEVICES (UUIDs)")
    checks = sc.check_nvlink(2, dev)
    assert checks[0].status == "warn"
    assert "unverified" in checks[0].detail


def test_too_few_gpus_for_the_requested_tp(monkeypatch):
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_SPLIT_PAIRS))
    checks = sc.check_nvlink(8)
    assert checks[0].status == "warn"
    assert "4 GPU rows" in checks[0].detail


def test_single_gpu_run_skips_the_check(monkeypatch):
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, TOPO_NVSWITCH_4))
    assert sc.check_nvlink(1) == []


# --- device detection / TP sizing -------------------------------------------

SMI_4 = ("0, NVIDIA H100 80GB HBM3\n1, NVIDIA H100 80GB HBM3\n"
         "2, NVIDIA H100 80GB HBM3\n3, NVIDIA H100 80GB HBM3\n")


def test_detects_every_gpu_when_unmasked(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, SMI_4))
    dev = sc.visible_devices()
    assert dev.count == 4 and dev.indices == [0, 1, 2, 3]
    assert dev.names[0].startswith("NVIDIA H100")


def test_cuda_visible_devices_wins_over_nvidia_smi(monkeypatch):
    """nvidia-smi reports all 4 cards regardless of the mask, but vLLM only sees
    the subset — sizing TP off nvidia-smi would ask for 4 ranks on a 2-GPU run."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(sc, "_run", lambda *a, **k: (0, SMI_4))
    dev = sc.visible_devices()
    assert dev.count == 2 and dev.indices == [2, 3]


def test_uuid_mask_keeps_the_count_but_not_the_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123,GPU-def456")
    dev = sc.visible_devices()
    assert dev.count == 2 and dev.indices is None


def test_tp_defaults_to_the_whole_box():
    dev = sc.Devices(indices=[0, 1, 2, 3], count=4, source="nvidia-smi")
    assert sc.resolve_tp(["vllm", "serve", "m"], dev, None) == 4


def test_explicit_tp_in_the_serve_command_wins():
    dev = sc.Devices(indices=[0, 1, 2, 3], count=4, source="nvidia-smi")
    argv = ["vllm", "serve", "m", "--tensor-parallel-size", "2"]
    assert sc.resolve_tp(argv, dev, None) == 2
    assert sc.resolve_tp(argv, dev, 4) == 2      # the command still wins over --tp
    assert sc.resolve_tp(["vllm", "serve", "m"], dev, 2) == 2   # --tp when unset


def test_tp_accounts_for_data_parallel_size():
    """vLLM's world is TP x DP. On a 4-GPU box `--data-parallel-size 4` means TP=1;
    defaulting TP to the whole box would request 16 GPUs and die in NCCL bootstrap."""
    dev = sc.Devices(indices=[0, 1, 2, 3], count=4, source="nvidia-smi")
    assert sc.resolve_tp(["vllm", "serve", "m", "--data-parallel-size", "4"], dev, None) == 1
    assert sc.resolve_tp(["vllm", "serve", "m", "--data-parallel-size", "2"], dev, None) == 2
    assert sc.resolve_tp(["vllm", "serve", "m"], dev, None) == 4


def test_world_size_must_match_the_box():
    dev = sc.Devices(indices=[0, 1, 2, 3], count=4, source="nvidia-smi")
    assert sc.check_world(2, 2, dev)[0].status == "pass"
    assert sc.check_world(1, 4, dev)[0].status == "pass"

    over = sc.check_world(4, 4, dev)[0]
    assert over.status == "fail" and "needs 16 GPUs" in over.detail

    under = sc.check_world(2, 1, dev)[0]
    assert under.status == "warn" and "2 idle" in under.detail


def test_tp_never_resolves_to_zero_on_a_cpu_box():
    dev = sc.Devices(indices=[], count=0, source="nvidia-smi unavailable")
    assert sc.resolve_tp(["vllm", "serve", "m"], dev, None) == 1


def test_default_command_carries_no_tp_so_it_fits_any_box():
    assert sc._arg_of(sc.DEFAULT_SERVE_ARGV, "--tensor-parallel-size") is None


def test_extra_gpus_beyond_tp_are_reported_not_hidden():
    dev = sc.Devices(indices=[0, 1, 2, 3], count=4, source="nvidia-smi")
    check = sc.check_gpus(2, dev)[0]
    assert check.status == "pass"
    assert "leaving 2 GPU(s) idle" in check.detail


def test_preflight_fails_closed_without_a_gpu():
    """On a box with no nvidia-smi the driver must refuse to launch (exit 2), not
    start a server that cannot possibly work."""
    dev = sc.Devices(indices=[], count=0, source="nvidia-smi unavailable")
    assert sc.check_gpus(2, dev)[0].status == "fail"


# ── the port must be free before launching ──────────────────────────────────
#
# `--keep-server` makes a leftover server the expected state between captures,
# and the failure it produces is the most confusing on this path: the new server
# cannot bind and dies, /health answers instantly from the OLD one, every request
# succeeds against it, and the capture arms an output path nobody is writing to.
# The run reports success and returns an empty trace, with a warning blaming
# CUDA-graph replay. Observed twice on a real H200 pod, ~7 minutes wasted each.


def test_a_free_port_passes(monkeypatch):
    from gitm.serve import discover, vllm

    monkeypatch.setattr(discover, "pid_listening_on", lambda *a, **k: None)
    (check,) = vllm.check_port_free(8000)
    assert check.status == "pass"


def test_an_occupied_port_fails_and_names_the_process(monkeypatch):
    from gitm.serve import discover, vllm

    monkeypatch.setattr(discover, "pid_listening_on", lambda *a, **k: 17295)
    monkeypatch.setattr(discover, "read_cmdline",
                        lambda *a, **k: ["vllm", "serve", "Qwen/Qwen3.6-35B-A3B"])
    (check,) = vllm.check_port_free(8000)

    assert check.status == "fail"
    assert "17295" in check.detail
    # The remedy has to be in the message: the symptom points at CUDA graphs, so
    # nobody reading the failure would otherwise think to look for a live server.
    assert "kill -INT -17295" in check.detail
    assert "gitm capture attach" in check.detail


def test_the_check_is_wired_into_preflight(monkeypatch):
    from gitm.serve import discover, vllm

    monkeypatch.setattr(discover, "pid_listening_on", lambda *a, **k: 999)
    monkeypatch.setattr(discover, "read_cmdline", lambda *a, **k: ["vllm"])
    devices = vllm.Devices(indices=[0], count=1, source="test")

    names = {c.name for c in vllm.preflight([], devices, 1, skip_args=True, port=8000)}
    assert "port" in names
    # Omitting the port must keep the old behaviour rather than silently passing.
    assert "port" not in {
        c.name for c in vllm.preflight([], devices, 1, skip_args=True)
    }
