"""Only engines that are traced carry the tracer (K-1).

On MI355X, rocprofiler-sdk's queue interposition can deadlock against hipGraph
replay (vLLM #56506, ROCm #11623), and it is loaded into every process the tool
is injected into. The first Kimi run hung that way on an A/B engine it never
traced. So the opening capture and each re-trace are traced, and every engine
that is only A/B-measured is built without the tracer, on both sides of the A/B.
"""

from __future__ import annotations

import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from gitm.tracer import injection


def test_untraced_env_removes_the_injection_and_puts_it_back(monkeypatch):
    monkeypatch.setenv(injection.ENV_LIB, "/x/libgitm_inject.so")
    monkeypatch.setenv(injection.ENV_ROCP, "/x/libgitm_rocm_inject.so")
    monkeypatch.setenv(injection.ENV_NVTX_INJECT, "/x/libcupti.so")
    monkeypatch.setenv("LD_PRELOAD", "/x/libother.so:/x/libgitm_roctx_shim.so")
    monkeypatch.setenv(injection.ENV_OUT, "/t/trace.jsonl")

    with injection.untraced_env():
        for var in (injection.ENV_LIB, injection.ENV_ROCP, injection.ENV_NVTX_INJECT):
            assert var not in os.environ
        assert os.environ["LD_PRELOAD"] == "/x/libother.so"   # someone else's preload stays
        assert os.environ[injection.ENV_OUT] == "/t/trace.jsonl"

    assert os.environ[injection.ENV_LIB] == "/x/libgitm_inject.so"
    assert os.environ["LD_PRELOAD"] == "/x/libother.so:/x/libgitm_roctx_shim.so"


def _fake_vllm(monkeypatch, seen):
    """A vllm module whose LLM records the tracer variable it was started with."""

    class _Out:
        def __init__(self, n):
            self.outputs = [types.SimpleNamespace(token_ids=list(range(n)))]

    class _LLM:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs
            seen.append((os.environ.get(injection.ENV_LIB), kwargs.get("enforce_eager")))

        def generate(self, prompts, params):
            return [_Out(params.max_tokens) for _ in prompts]

    class _SamplingParams:
        def __init__(self, max_tokens=0, temperature=0.0):
            self.max_tokens = max_tokens

    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = _LLM, _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake)


def test_only_the_opening_engine_and_retraces_carry_the_tracer(tmp_path, monkeypatch):
    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    seen: list = []
    _fake_vllm(monkeypatch, seen)
    monkeypatch.setenv(injection.ENV_LIB, "/x/libgitm_inject.so")
    monkeypatch.setenv(injection.ENV_OUT, str(tmp_path / "t.jsonl"))
    for var, val in (("GITM_VLLM_PROMPTS", "2"), ("GITM_VLLM_MAX_TOKENS", "3")):
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)

    engine = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode")).engine
    assert engine.gitm_traced and seen[-1][0] == "/x/libgitm_inject.so"

    candidate = engine.gitm_restart_fn(engine, {"max_num_seqs": 64})
    assert seen[-1][0] is None and not candidate.gitm_traced
    baseline = engine.gitm_baseline_restart_fn(engine)
    assert seen[-1][0] is None and not baseline.gitm_traced

    retrace = baseline.gitm_traced_rebuild_fn(baseline)
    assert seen[-1] == ("/x/libgitm_inject.so", True)          # traced, and eager
    assert "enforce_eager" not in retrace.gitm_llm_kwargs or \
        retrace.gitm_llm_kwargs["enforce_eager"] == engine.gitm_llm_kwargs.get("enforce_eager")
    # Every engine the factory builds carries the loop's hooks, not only the first.
    for e in (candidate, baseline, retrace):
        assert callable(e.gitm_throughput_fn) and callable(e.gitm_restart_fn)


def _run_on_a_traced_opening_engine(tmp_path, monkeypatch, rerank="off"):
    """Run the loop from a traced opening engine. Returns the tracing swaps,
    after checking no A/B was measured on a traced engine."""
    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace
    from .test_vllm_knobs_and_restart import _FullEngine

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100,
                               end_ns=i * 100 + 80) for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    measured_on: list[bool] = []

    class _Engine(_FullEngine):
        def __init__(self, traced=False, max_num_seqs=64):
            super().__init__(max_num_seqs=max_num_seqs)
            self.gitm_traced = traced
            self.gitm_throughput_fn = self._tps
            self.gitm_untraced_rebuild_fn = lambda old: _Engine(False, old.scheduler_config.max_num_seqs)
            self.gitm_traced_rebuild_fn = lambda old: _Engine(True, old.scheduler_config.max_num_seqs)

        def _tps(self, e):
            measured_on.append(e.gitm_traced)
            return float(e.scheduler_config.max_num_seqs)

        def _restart(self, _old, knob_values):
            return _Engine(False, int(knob_values.get("max_num_seqs", 64)))

    out = run_loop(LoopConfig(engine=_Engine(traced=True), workload="vllm-decode",
                              budget="24h", scratch=str(tmp_path), top_n_interventions=5,
                              rerank=rerank))

    assert measured_on, "no A/B was measured"
    assert not any(measured_on), "an A/B was measured on a traced engine"
    swaps = json.loads((Path(out["run_dir"]) / "tracing.json").read_text())["swaps"]
    assert swaps[0] == {"when": "before the first A/B", "to": "untraced", "error": None}
    return swaps


def test_every_ab_is_measured_on_untraced_engines(tmp_path, monkeypatch):
    """The opening engine is traced for the capture; the loop swaps in an
    untraced baseline before the first A/B, or that A/B would credit the
    candidate with the tracer's overhead."""
    _run_on_a_traced_opening_engine(tmp_path, monkeypatch)


def test_a_retrace_runs_on_a_traced_engine_and_swaps_back(tmp_path, monkeypatch):
    """--rerank recapture: the re-trace needs the tracer, the next A/B must not
    have it. Swapped to a traced build for the trace, then back."""
    monkeypatch.setattr(injection, "active_vendor", lambda: "nvidia")
    swaps = _run_on_a_traced_opening_engine(tmp_path, monkeypatch, rerank="recapture")
    retraces = [w for w in swaps if w["when"].startswith("re-trace")]
    backs = [w for w in swaps if w["when"].startswith("after re-trace")]
    assert retraces and len(retraces) == len(backs)
    assert all(w["to"] == "traced" for w in retraces) and all(w["to"] == "untraced" for w in backs)
