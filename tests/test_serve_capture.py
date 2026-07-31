"""Tests for the traced ``vllm serve`` driver (scripts/serve_capture.py).

GPU-free. The parts that only ever run on a pod — CUPTI, the engine — are not
exercised, but everything that decides whether the pod run produces a usable
result is: the SSE stream parse that TTFT comes from (against a real socket, not a
mock), the prompt builder's cache-defeating property, the ``--`` command
passthrough, and the preflight's classification of a rejected serve flag.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    """Import scripts/serve_capture.py by path — scripts/ is not a package.

    The module must be in sys.modules *before* it executes: @dataclass resolves
    annotations through sys.modules[cls.__module__], which is None otherwise.
    """
    spec = importlib.util.spec_from_file_location("serve_capture", REPO / "scripts" / "serve_capture.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load()


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
    assert sc._arg_of(sc.DEFAULT_SERVE_ARGV, "--tensor-parallel-size") == "2"
    assert sc._arg_of(sc.DEFAULT_SERVE_ARGV, "--max-num-seqs") == "256"


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


def test_preflight_fails_closed_without_a_gpu():
    """On a box with no nvidia-smi the driver must refuse to launch (exit 2), not
    start a server that cannot possibly work."""
    checks = sc.check_gpus(2)
    assert checks[0].status == "fail"
