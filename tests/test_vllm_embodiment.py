"""Stream A — vLLM embodiment: gate-context, precondition gate, live-engine
applicator, scheduler-stats adapter, and deviation-only tracing.

These cover the new wiring added for the ``vllm-decode`` workload without needing
a GPU or a real vLLM install: a fake duck-typed engine stands in for the live
engine, and the loop is driven through a monkeypatched capture window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.metrics import HardwarePeak, hfu, mbu, mfu
from gitm.optimizer.preconditions import GateContext, check_gate

from .conftest import make_kernel, make_trace


# --------------------------------------------------------------------------- #
# metrics: HFU / MFU / MBU                                                     #
# --------------------------------------------------------------------------- #
def test_utilization_ratios_and_none_guards():
    peak = HardwarePeak("A100", peak_flops=312e12, peak_bw_bytes_s=2039e9)
    assert hfu(156e12, peak) == pytest.approx(0.5)
    assert mbu(2039e9, peak) == pytest.approx(1.0)
    assert mfu(312e12, 1.0, peak) == pytest.approx(1.0)
    # Unknown peak / bad inputs => None (never a misleading number).
    assert hfu(1e12, None) is None
    assert mbu(None, peak) is None
    assert mfu(1e12, 0.0, peak) is None  # zero elapsed
    assert hfu(-1.0, peak) is None  # negative achieved


# --------------------------------------------------------------------------- #
# preconditions: the applicability gate                                       #
# --------------------------------------------------------------------------- #
def _spec(**over) -> InterventionSpec:
    base = dict(
        name="x", summary="s", knob="k", value=1,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="test",
    )
    base.update(over)
    return InterventionSpec.model_validate(base)


def test_gate_workload_mismatch_rejects():
    spec = _spec(applicability={"workloads": ["vllm-decode"]})
    res = check_gate(spec, GateContext(workload="hft"))
    assert not res.ok and "workload" in res.reason


def test_gate_dtype_known_mismatch_rejects_but_unknown_is_lenient():
    spec = _spec(applicability={"workloads": ["vllm-decode"], "requires_dtype": ["fp16", "bf16"]})
    # Known fp32 box => rejected.
    assert not check_gate(spec, GateContext(workload="vllm-decode", dtype="fp32")).ok
    # Unknown dtype => not rejected (gate is conservative about unknowns).
    assert check_gate(spec, GateContext(workload="vllm-decode", dtype=None)).ok


def test_gate_hardware_substring_match():
    spec = _spec(applicability={"workloads": ["vllm-decode"], "requires_hardware": ["A100", "H100"]})
    assert check_gate(spec, GateContext(workload="vllm-decode", hardware="NVIDIA A100-SXM4-80GB")).ok
    assert not check_gate(spec, GateContext(workload="vllm-decode", hardware="NVIDIA L4")).ok


def test_gate_kv_cache_bounds():
    spec = _spec(applicability={"workloads": ["vllm-decode"], "max_kv_cache_len": 4096})
    assert check_gate(spec, GateContext(workload="vllm-decode", kv_cache_len=2048)).ok
    assert not check_gate(spec, GateContext(workload="vllm-decode", kv_cache_len=8192)).ok


def test_gate_multi_gpu_knob_on_single_gpu_rejected():
    spec = _spec(knob="tensor_parallel_size", value=2)
    assert not check_gate(spec, GateContext(workload="vllm-decode", num_gpus=1)).ok
    assert check_gate(spec, GateContext(workload="vllm-decode", num_gpus=2)).ok


# --------------------------------------------------------------------------- #
# planner context                                                             #
# --------------------------------------------------------------------------- #
def test_build_planner_context_sku_override(monkeypatch):
    from gitm.planner.context import build_planner_context

    monkeypatch.setenv("GITM_GPU_SKU", "NVIDIA A100-SXM4-80GB")
    pctx = build_planner_context(engine=None, workload="vllm-decode", num_gpus=1)
    assert pctx.peak is not None and pctx.peak.peak_flops == 312e12
    assert pctx.gate.workload == "vllm-decode"
    assert pctx.gate.hardware == "NVIDIA A100-SXM4-80GB"
    assert pctx.gate.dtype is None  # no engine attached


def test_unknown_sku_yields_no_peak(monkeypatch):
    from gitm.planner.context import build_planner_context

    monkeypatch.setenv("GITM_GPU_SKU", "SomeFutureGPU-9000")
    pctx = build_planner_context(engine=None, num_gpus=1)
    assert pctx.peak is None  # unknown => unreported, never wrong


# --------------------------------------------------------------------------- #
# library + policy: workload filter and gate-driven selection                 #
# --------------------------------------------------------------------------- #
def test_load_library_filters_by_workload():
    from gitm.kernels.library import load_library

    vllm = load_library(workload="vllm-decode")
    assert len(vllm) > 0
    other = load_library(workload="hft")  # no vLLM lever lists hft
    assert other == []


def test_select_interventions_gate_rejects_inapplicable_levers():
    from gitm.agents.policy import Policy, select_interventions
    from gitm.kernels.library import load_library

    library = load_library(workload="vllm-decode")
    trace = make_trace(events=[make_kernel("paged_attention", end_ns=500)])
    # Single fp32 GPU: fp8-KV (requires fp16/bf16) and TP=2 (multi-GPU) must be rejected.
    ctx = GateContext(workload="vllm-decode", dtype="fp32", hardware="NVIDIA L4", num_gpus=1)
    ranked = select_interventions(trace, library, Policy(), top_n=50, ctx=ctx)
    by_knob = {c.spec.knob: c for c in ranked}
    assert by_knob["tensor_parallel_size"].rejected_reason.startswith("precondition")
    assert by_knob["kv_cache_dtype"].rejected_reason.startswith("precondition")
    # Without a ctx the gate is skipped — those levers are no longer pre-rejected.
    ranked_no_ctx = select_interventions(trace, library, Policy(), top_n=50)
    knob_reasons = {c.spec.knob: c.rejected_reason for c in ranked_no_ctx}
    assert knob_reasons["tensor_parallel_size"] != "precondition: knob 'tensor_parallel_size' requires >1 GPU (num_gpus=1)"


# --------------------------------------------------------------------------- #
# scheduler-stats adapter (Task 2)                                            #
# --------------------------------------------------------------------------- #
class _FakeDeque(list):
    """A list that also stands in for vLLM's request deques (len() is what we read)."""


class _FakeScheduler:
    def __init__(self, running, waiting, swapped, preempt):
        self.running = _FakeDeque(range(running))
        self.waiting = _FakeDeque(range(waiting))
        self.swapped = _FakeDeque(range(swapped))
        self.num_cumulative_preemption = preempt


class _FakeSchedulerConfig:
    max_num_seqs = 256


class _FakeEngine:
    """Duck-typed stand-in for a vLLM LLMEngine (only the attrs the adapter reads)."""

    def __init__(self, running=4, waiting=10, swapped=2, preempt=3):
        self.scheduler = [_FakeScheduler(running, waiting, swapped, preempt)]
        self.scheduler_config = _FakeSchedulerConfig()

    def get_num_unfinished_requests(self):
        return 16


def test_read_scheduler_stats_duck_typed():
    from gitm.tracer.vllm_stats import read_scheduler_stats

    s = read_scheduler_stats(_FakeEngine(running=4, waiting=10))
    assert s is not None
    assert s.num_running == 4 and s.num_waiting == 10 and s.num_swapped == 2
    assert s.num_unfinished == 16
    assert s.preemptions_cumulative == 3
    assert s.batch_occupancy == pytest.approx(4 / 256)


def test_read_scheduler_stats_none_when_unreadable():
    from gitm.tracer.vllm_stats import read_scheduler_stats

    assert read_scheduler_stats(None) is None
    assert read_scheduler_stats(object()) is None  # nothing readable => None, not a crash


def test_sample_scheduler_stats_collects_and_summarizes():
    from gitm.tracer.vllm_stats import sample_scheduler_stats

    engine = _FakeEngine(running=8, waiting=5, swapped=0, preempt=0)
    with sample_scheduler_stats(engine, interval_s=0.002) as sampler:
        # Let a few samples land.
        for _ in range(2000):
            engine.get_num_unfinished_requests()
    summ = sampler.summary()
    assert summ.n_samples > 0
    assert summ.peak_running == 8
    assert summ.peak_queue_depth == 5
    assert summ.mean_batch_occupancy == pytest.approx(8 / 256, rel=0.01)


def test_sample_scheduler_stats_no_engine_is_noop():
    from gitm.tracer.vllm_stats import sample_scheduler_stats

    with sample_scheduler_stats(None) as sampler:
        pass
    assert sampler.samples == []
    assert sampler.summary().n_samples == 0


# --------------------------------------------------------------------------- #
# deviation-only tracing (Task 6)                                             #
# --------------------------------------------------------------------------- #
def _decode_graph():
    from gitm.planner.graph import predict_graph

    return predict_graph()


def test_deviation_trace_keeps_only_departures():
    from gitm.optimizer.deviation import deviation_summary, deviation_trace

    graph = _decode_graph()
    # Build observed kernels: most match prediction (in-band), a few are 100x slower.
    obs = []
    for i, node in enumerate(graph.nodes):
        dur = node.prediction.t_pred_s
        scale = 100.0 if i % 10 == 0 else 1.0  # every 10th kernel grossly over-runs
        ns = max(int(dur * scale * 1e9), 1)
        obs.append(make_kernel(node.op, start_ns=i * 10_000, end_ns=i * 10_000 + ns))
    trace = make_trace(events=obs)

    reduced = deviation_trace(trace, graph)
    summ = deviation_summary(trace, graph)
    assert summ["n_observed"] == len(obs)
    assert 0 < summ["n_kept"] < summ["n_observed"]  # some dropped, some kept
    assert summ["reduction"] > 0.0
    assert len(reduced.kernels()) == summ["n_kept"]
    # The reduced trace is still a well-formed Trace with the header preserved.
    assert reduced.workload_id == trace.workload_id and reduced.run_id == trace.run_id


def test_unpredicted_kernels_are_always_departures():
    from gitm.optimizer.deviation import deviating_kernel_indices

    graph = _decode_graph()
    n_pred = len(graph.nodes)
    # One in-band predicted kernel + 3 extra unmodeled kernels past the graph.
    obs = [make_kernel(graph.nodes[0].op,
                       end_ns=max(int(graph.nodes[0].prediction.t_pred_s * 1e9), 1))]
    obs += [make_kernel("mystery_kernel", start_ns=i, end_ns=i + 5) for i in range(3)]
    trace = make_trace(events=obs)
    dev = deviating_kernel_indices(trace, graph)
    # The 3 unpredicted kernels (indices 1,2,3) are kept regardless of timing.
    assert {1, 2, 3}.issubset(set(dev.kept_indices))
    assert dev.n_predicted == n_pred


# --------------------------------------------------------------------------- #
# end-to-end: live-engine applicator through run_loop (Task 4 + 5)            #
# --------------------------------------------------------------------------- #
class _ModelConfig:
    dtype = "torch.bfloat16"  # _engine_dtype reads model_config.dtype => "bf16"


class _LiveEngine:
    """Duck-typed vLLM engine with one hot-swappable knob and a deterministic
    throughput probe, so the rollback-gated A/B has a real (noise-free) signal."""

    def __init__(self):
        self.model_config = _ModelConfig()
        self.max_num_seqs = 32  # the one knob this fake engine can hot-swap
        # Decode throughput scales with batch width — so raising max_num_seqs is a
        # measurable win and is kept; knobs with no matching attribute raise on
        # apply and are rolled back (the "structural knob can't hot-swap" path).
        self.gitm_throughput_fn = lambda e: float(e.max_num_seqs)


def test_run_loop_live_engine_keeps_winning_knob_and_rolls_back_unswappable(
    tmp_path: Path, monkeypatch
):
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    # Populated nvidia trace so the guard passes and the vLLM library path runs.
    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [
            make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 80)
            for i in range(80)
        ]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    engine = _LiveEngine()
    cfg = LoopConfig(
        engine=engine,
        workload="vllm-decode",
        budget="5s",
        scratch=str(tmp_path),
        top_n_interventions=50,  # evaluate the whole library, not just top 5
        workload_runner=lambda: {"generated_tokens": 128},
    )
    out = run_loop(cfg)
    summary = out["summary"]
    assert summary["status"] == "ok" and summary["mode"] == "intervention"

    # The hot-swappable winning knob is kept with a measured positive delta.
    assert engine.max_num_seqs == 256
    import json

    claims = json.loads((Path(out["run_dir"]) / "ranked_candidates.json").read_text())
    assert any(c["name"] == "max_num_seqs_256" for c in claims)
    # report carries the measured live A/B verdict (not a prediction).
    assert "live A/B" in out["report_md"]
    # Knobs the engine can't hot-swap were rolled back, not silently kept.
    assert summary["n_rolled_back"] >= 1


def test_run_loop_no_engine_is_predict_only(tmp_path: Path, monkeypatch):
    """Without an engine, candidates are unverified (no measured delta), never won."""
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel("paged_attention", start_ns=i * 100, end_ns=i * 100 + 80)
                   for i in range(40)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    out = run_loop(LoopConfig(workload="vllm-decode", budget="5s", scratch=str(tmp_path)))
    assert out["summary"]["status"] == "ok"
    # No scheduler stats without an engine.
    assert out["summary"]["scheduler_stats"] is None
