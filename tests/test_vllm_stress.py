"""Stress + edge-case coverage for the vLLM embodiment surface.

Pushes the concurrency (sampler thread), the rollback gate across many
candidates (state must not leak between them), error paths (engines that raise,
empty/degenerate inputs), and the deviation filter's boundaries.
"""

from __future__ import annotations

import threading

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import LiveEngineApplicator, apply_intervention
from gitm.optimizer.deviation import deviating_kernel_indices, deviation_summary, deviation_trace
from gitm.optimizer.scheduler_attribution import scheduler_causes
from gitm.optimizer.vllm_knobs import get_knob, set_knob
from gitm.planner.graph import predict_graph
from gitm.tracer.vllm_stats import SchedulerStatsSampler, read_scheduler_stats, summarize

from .conftest import make_kernel, make_trace


def _spec(knob, value):
    return InterventionSpec.model_validate(
        dict(name=knob, summary=knob, knob=knob, value=value, expected_delta_mean=0.05,
             expected_delta_lo=0.0, expected_delta_hi=0.1, source="t")
    )


# --------------------------------------------------------------------------- #
# applicator state must not leak across candidates (the last_result reset bug) #
# --------------------------------------------------------------------------- #
class _SchedCfg:
    def __init__(self):
        self.max_num_seqs = 32


class _Engine:
    def __init__(self, max_num_seqs: int = 32):
        self.scheduler_config = _SchedCfg()
        self.scheduler_config.max_num_seqs = max_num_seqs

    def restart(self, _old_engine, knob_values):
        if set(knob_values) != {"max_num_seqs"}:
            raise AttributeError("unsupported fake-engine restart knob")
        return _Engine(max_num_seqs=int(knob_values["max_num_seqs"]))


def test_failed_apply_clears_previous_ab_result():
    """A failed restart candidate must not surface the previous A/B result."""
    e = _Engine()
    app = LiveEngineApplicator(
        e,
        throughput_fn=lambda x: float(x.scheduler_config.max_num_seqs),
        restart_fn=e.restart,
    )

    r1 = apply_intervention(_spec("max_num_seqs", 256), app, min_keep_delta=0.0)
    assert r1.applied and app.last_result is not None

    # Next: an unsupported structural knob raises during restart. measure never
    # runs, so last_result must be reset to None, not stale from r1.
    r2 = apply_intervention(_spec("quantization", "awq"), app, min_keep_delta=0.0)
    assert r2.rolled_back and app.last_result is None


def test_many_candidates_restore_is_clean():
    """Restart, rollback, restart leaves the applicator at the last kept engine."""
    e = _Engine()
    app = LiveEngineApplicator(
        e,
        throughput_fn=lambda x: float(x.scheduler_config.max_num_seqs),
        restart_fn=e.restart,
    )

    apply_intervention(_spec("max_num_seqs", 256), app, min_keep_delta=0.0)  # 32->256 kept
    assert app.engine.scheduler_config.max_num_seqs == 256
    apply_intervention(_spec("max_num_seqs", 64), app, min_keep_delta=0.0)  # 256->64 slower
    assert app.engine.scheduler_config.max_num_seqs == 256  # restored
    apply_intervention(_spec("max_num_seqs", 512), app, min_keep_delta=0.0)  # 256->512 kept
    assert app.engine.scheduler_config.max_num_seqs == 512


# --------------------------------------------------------------------------- #
# sampler concurrency + error resilience                                      #
# --------------------------------------------------------------------------- #
class _RaisingEngine:
    @property
    def scheduler(self):
        raise RuntimeError("engine busy")

    def get_num_unfinished_requests(self):
        raise RuntimeError("engine busy")


def test_sampler_swallows_engine_exceptions():
    sampler = SchedulerStatsSampler(_RaisingEngine(), interval_s=0.002)
    sampler.start()
    for _ in range(500):
        pass
    sampler.stop()  # must not raise
    # A raising engine yields no usable samples, summary is the empty shape.
    assert sampler.summary().n_samples == 0


def test_sampler_repeated_start_stop_is_safe():
    class _E:
        def __init__(self):
            self.scheduler_config = type("C", (), {"max_num_seqs": 8})()
            self.scheduler = [type("S", (), {"running": [0], "waiting": [], "swapped": []})()]

        def get_num_unfinished_requests(self):
            return 1

    sampler = SchedulerStatsSampler(_E(), interval_s=0.001)
    for _ in range(5):
        sampler.start()
        sampler.start()  # idempotent
        sampler.stop()
        sampler.stop()  # idempotent
    assert sampler.summary().n_samples >= 1  # synchronous first read each cycle... at least one
    assert not any(t.name == "gitm-vllm-stats" and t.is_alive() for t in threading.enumerate())


def test_read_scheduler_stats_partial_engine():
    """An engine exposing only *some* fields yields a partial sample, not None."""
    class _Partial:
        scheduler = [type("S", (), {"waiting": [0, 1, 2]})()]  # only waiting, no running/swapped
    s = read_scheduler_stats(_Partial())
    assert s is not None and s.num_waiting == 3
    assert s.num_running is None and s.batch_occupancy is None


def test_summarize_single_sample():
    s = read_scheduler_stats(
        type("E", (), {"scheduler": [type("S", (), {"running": [0, 1], "waiting": [], "swapped": []})()],
                       "get_num_unfinished_requests": lambda self: 2})()
    )
    summ = summarize([s])
    assert summ.n_samples == 1 and summ.duration_s == 0.0
    assert summ.peak_running == 2


def test_summarize_in_flight_ignores_the_idle_head_and_tail():
    """The sampler snapshots before the workload is submitted and after it
    drains. Averaging those zeros in would halve the batch the graph is priced
    for, so only the samples that had work in flight count."""
    from gitm.tracer.vllm_stats import SchedulerSample

    samples = [SchedulerSample(t_ns=0, num_unfinished=0),
               SchedulerSample(t_ns=1, num_unfinished=16),
               SchedulerSample(t_ns=2, num_unfinished=16),
               SchedulerSample(t_ns=3, num_unfinished=0)]
    summ = summarize(samples, max_num_seqs=256)
    assert summ.mean_unfinished == 16.0
    assert summ.peak_unfinished == 16
    assert summ.mean_bounded_inflight == 16.0  # nothing above capacity to bound
    assert summ.max_num_seqs == 256


def test_each_sample_is_bounded_before_averaging_not_after():
    """gitm submits every prompt at once, so in-flight starts above capacity and
    drains through it. Bounding the average instead of each sample reads such a
    window as if it sat at capacity throughout: 64 and 2 in flight at a cap of 32
    is a mean batch of 17, not 32."""
    from gitm.tracer.vllm_stats import SchedulerSample

    samples = [SchedulerSample(t_ns=0, num_unfinished=64),
               SchedulerSample(t_ns=1, num_unfinished=2)]
    summ = summarize(samples, max_num_seqs=32)
    assert summ.mean_unfinished == 33.0        # the raw observation, unbounded
    assert summ.mean_bounded_inflight == 17.0  # the batch
    assert min(summ.mean_unfinished, 32) == 32.0  # what bounding the mean gives


def test_bounded_inflight_needs_a_capacity():
    from gitm.tracer.vllm_stats import SchedulerSample

    summ = summarize([SchedulerSample(t_ns=0, num_unfinished=64)])
    assert summ.mean_unfinished == 64.0
    assert summ.mean_bounded_inflight is None


def test_summarize_in_flight_absent_when_never_exposed():
    from gitm.tracer.vllm_stats import SchedulerSample

    summ = summarize([SchedulerSample(t_ns=0, num_waiting=3)])
    assert summ.mean_unfinished is None and summ.peak_unfinished is None
    assert summ.mean_bounded_inflight is None
    assert summarize([]).mean_unfinished is None


# --------------------------------------------------------------------------- #
# scheduler_causes edge cases                                                 #
# --------------------------------------------------------------------------- #
def test_scheduler_causes_clamped_severity():
    from gitm.tracer.vllm_stats import SchedulerStatsSummary

    s = SchedulerStatsSummary(
        n_samples=5, duration_s=1.0, peak_queue_depth=10_000, mean_running=1.0,
        peak_running=1, mean_batch_occupancy=0.0, total_preemptions=10_000,
        peak_gpu_cache_usage=1.0, peak_swapped=0,
    )
    causes = scheduler_causes(s)
    assert causes and all(0.0 <= c.severity <= 1.0 for c in causes)


# --------------------------------------------------------------------------- #
# deviation filter boundaries                                                 #
# --------------------------------------------------------------------------- #
def test_deviation_all_in_band_keeps_nothing():
    graph = predict_graph()
    # Every observed kernel exactly matches its predicted time => 0 departures.
    obs = [make_kernel(n.op, start_ns=i * 1000,
                       end_ns=i * 1000 + max(int(n.prediction.t_pred_s * 1e9), 1))
           for i, n in enumerate(graph.nodes)]
    trace = make_trace(events=obs)
    summ = deviation_summary(trace, graph)
    assert summ["n_kept"] == 0 and summ["reduction"] == 1.0
    assert deviation_trace(trace, graph).kernels() == []


def test_deviation_empty_trace():
    graph = predict_graph()
    summ = deviation_summary(make_trace(events=[]), graph)
    assert summ["n_observed"] == 0 and summ["reduction"] == 0.0


def test_deviation_fewer_obs_than_pred():
    graph = predict_graph()
    obs = [make_kernel(graph.nodes[0].op, end_ns=10**9)]  # 1 grossly-slow kernel
    dev = deviating_kernel_indices(make_trace(events=obs), graph)
    assert dev.n_observed == 1 and 0 in dev.kept_indices


# --------------------------------------------------------------------------- #
# env-knob restore footgun                                                    #
# --------------------------------------------------------------------------- #
def test_admission_backlog_fires_when_nothing_admitted():
    """peak_running == 0 with a non-empty queue is the worst backlog, not silence."""
    from gitm.tracer.vllm_stats import SchedulerStatsSummary

    s = SchedulerStatsSummary(
        n_samples=5, duration_s=1.0, peak_queue_depth=50, mean_running=0.0,
        peak_running=0, mean_batch_occupancy=0.0, total_preemptions=0,
        peak_gpu_cache_usage=0.1, peak_swapped=0,
    )
    assert any(c.signal == "admission_backlog" for c in scheduler_causes(s))


def test_occupancy_clamped_under_multiple_schedulers():
    """Summed running across schedulers must not push occupancy above 1.0."""
    class _Sch:
        def __init__(self, r):
            self.running = list(range(r))
            self.waiting: list = []
            self.swapped: list = []

    class _Cfg:
        max_num_seqs = 100

    class _E:
        scheduler = [_Sch(80), _Sch(80)]  # 160 running across 2 schedulers
        scheduler_config = _Cfg()

    s = read_scheduler_stats(_E())
    assert s.num_running == 160
    assert s.batch_occupancy is not None and 0.0 <= s.batch_occupancy <= 1.0


def test_measure_rolls_back_on_nonpositive_baseline():
    """An unmeasurable (zero) baseline must roll the candidate back, not keep it."""
    class _Idle:
        scheduler_config = _SchedCfg()

    # Baseline probe returns 0 (idle engine), so measure() raises after apply.
    app = LiveEngineApplicator(_Idle(), throughput_fn=lambda e: 0.0, restart_fn=lambda _e, _kv: _Idle())
    res = apply_intervention(_spec("max_num_seqs", 256), app, min_keep_delta=0.0)
    assert not res.applied and res.rolled_back


def test_deviation_multistep_does_not_keep_everything():
    """A multi-step decode trace (more kernels than one predicted step) must
    compare cyclically, not label every post-step-1 kernel as unmodeled."""
    graph = predict_graph()
    step = graph.nodes
    # 3 full decode steps, every kernel exactly on its predicted time => 0 departures.
    obs = []
    for s in range(3):
        for j, n in enumerate(step):
            k = s * len(step) + j
            obs.append(make_kernel(n.op, start_ns=k * 1000,
                                   end_ns=k * 1000 + max(int(n.prediction.t_pred_s * 1e9), 1)))
    trace = make_trace(events=obs)
    summ = deviation_summary(trace, graph)
    assert summ["n_observed"] == 3 * len(step)
    assert summ["n_kept"] == 0  # cyclic pairing: all in-band across all 3 steps
    assert "<unpredicted>" not in summ["kept_ops"]


def test_env_knob_restore_to_none_unsets(monkeypatch):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    set_knob(object(), "VLLM_ATTENTION_BACKEND", "FLASHINFER")
    assert get_knob(object(), "VLLM_ATTENTION_BACKEND") == "FLASHINFER"
    # Restoring the original (unset) value must remove the var, not write "None".
    set_knob(object(), "VLLM_ATTENTION_BACKEND", None)
    import os
    assert "VLLM_ATTENTION_BACKEND" not in os.environ


# ── the capacity that turns an in-flight count into a batch ─────────────────


def test_max_num_seqs_is_found_on_the_shape_vllm_actually_presents():
    """The first full local run captured `mean_unfinished` of 243 and still
    priced its graph at batch=1, because none of the probed paths reached
    `max_num_seqs`. The bounded figure needs a capacity, so without one the
    whole in-flight fallback is dead and every residual reads +100%.

    vLLM 0.30: LLM -> LLMEngine.vllm_config -> SchedulerConfig. Both halves of
    that chain were listed separately and never joined.
    """
    from types import SimpleNamespace as N

    from gitm.tracer.vllm_stats import _max_num_seqs

    llm = N(llm_engine=N(vllm_config=N(scheduler_config=N(max_num_seqs=256))))
    assert _max_num_seqs(llm) == 256


def test_max_num_seqs_still_found_on_the_older_shapes():
    """Each path is a vLLM version that presented the config somewhere else.
    Adding one must not cost the others."""
    from types import SimpleNamespace as N

    from gitm.tracer.vllm_stats import _max_num_seqs

    assert _max_num_seqs(N(scheduler_config=N(max_num_seqs=64))) == 64
    assert _max_num_seqs(N(engine=N(scheduler_config=N(max_num_seqs=32)))) == 32
    assert _max_num_seqs(N(llm_engine=N(scheduler_config=N(max_num_seqs=16)))) == 16
    assert _max_num_seqs(N(vllm_config=N(scheduler_config=N(max_num_seqs=8)))) == 8


def test_an_engine_that_says_nothing_yields_none_rather_than_a_guess():
    """None keeps the in-flight count unused. A guessed capacity would clamp a
    real measurement to an invented ceiling."""
    from types import SimpleNamespace as N

    from gitm.tracer.vllm_stats import _max_num_seqs

    assert _max_num_seqs(N()) is None
    assert _max_num_seqs(N(scheduler_config=N(max_num_seqs=0))) is None
    assert _max_num_seqs(N(scheduler_config=N(max_num_seqs="many"))) is None


def test_the_capacity_reaches_the_batch_through_the_sampler_a_run_uses():
    """The whole chain, driven the way a live capture drives it.

    Calling ``summarize(max_num_seqs=_max_num_seqs(engine))`` by hand would
    prove the two pieces and not the join between them — and the join is
    exactly what failed. ``SchedulerStatsSampler.summary()`` is what a run
    actually calls, and it is the thing that has to carry the capacity from the
    engine into the summary.
    """
    from types import SimpleNamespace as N

    from gitm.scheduler.loop import _batch_config_from_stats
    from gitm.tracer.vllm_stats import SchedulerSample, SchedulerStatsSampler

    engine = N(llm_engine=N(vllm_config=N(scheduler_config=N(max_num_seqs=256))))
    sampler = SchedulerStatsSampler(engine)
    # What the collector would have gathered over the window.
    sampler.samples = [SchedulerSample(t_ns=i, num_unfinished=243) for i in range(5)]

    summ = sampler.summary()
    assert summ.max_num_seqs == 256, "the sampler did not carry the capacity through"
    assert summ.mean_bounded_inflight == 243.0

    cfg, source = _batch_config_from_stats(summ)
    assert cfg is not None and cfg.batch == 243, "still falling back to batch=1"
    assert source == "unfinished"


def test_an_engine_with_no_capacity_still_falls_back_through_the_sampler():
    """The failure this reproduces: in-flight captured, capacity absent, batch
    back to 1. It is what the first full run did."""
    from types import SimpleNamespace as N

    from gitm.scheduler.loop import _batch_config_from_stats
    from gitm.tracer.vllm_stats import SchedulerSample, SchedulerStatsSampler

    sampler = SchedulerStatsSampler(N())          # says nothing about capacity
    sampler.samples = [SchedulerSample(t_ns=i, num_unfinished=243) for i in range(5)]

    summ = sampler.summary()
    assert summ.mean_unfinished == 243.0          # captured
    assert summ.max_num_seqs is None              # but nothing to bound it
    assert summ.mean_bounded_inflight is None
    assert _batch_config_from_stats(summ) == (None, None)
