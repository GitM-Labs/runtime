"""Knob taxonomy, restart-apply for structural knobs, and
scheduler-stats causal attribution.

No GPU / no vLLM: fake engines stand in. These cover the two pieces that were
still in progress — #4 (hot-swap vs restart-apply) and #2 (stats → attribution).
"""

from __future__ import annotations

import pytest

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import (
    LiveEngineApplicator,
    StructuralKnobRequiresRestart,
    apply_intervention,
)
from gitm.optimizer.scheduler_attribution import scheduler_causes
from gitm.optimizer.vllm_knobs import (
    get_knob,
    knob_kind,
    resolve_knob,
    resolve_relative_value,
    set_knob,
    unmet_prerequisite,
)
from gitm.tracer.vllm_stats import SchedulerStatsSummary


def _spec(knob: str, value) -> InterventionSpec:
    return InterventionSpec.model_validate(
        dict(name=knob, summary=f"set {knob}", knob=knob, value=value,
             expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
             source="test")
    )


# --------------------------------------------------------------------------- #
# knob taxonomy                                                               #
# --------------------------------------------------------------------------- #
def test_knob_kind_classification():
    assert knob_kind("max_num_seqs") == "structural"
    assert knob_kind("max_num_batched_tokens") == "structural"
    assert knob_kind("scheduling_policy") == "structural"
    assert knob_kind("async_scheduling") == "structural"
    assert knob_kind("dbo_prefill_token_threshold") == "structural"
    assert knob_kind("kv_sharing_fast_prefill") == "structural"
    assert knob_kind("tensor_parallel_size") == "structural"
    assert knob_kind("block_size") == "structural"
    assert knob_kind("totally_unknown_knob") == "structural"  # safe default


class _SchedCfgFlags:
    def __init__(self, *, chunked_prefill_enabled=False, enable_dbo=False):
        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.enable_dbo = enable_dbo


class _EngineWithFlags:
    def __init__(self, **flags):
        self.scheduler_config = _SchedCfgFlags(**flags)


def test_unmet_prerequisite():
    gated = ("max_num_partial_prefills", "long_prefill_token_threshold",
             "dbo_prefill_token_threshold")

    # No prerequisite for this knob -> None regardless of engine.
    assert unmet_prerequisite(_EngineWithFlags(), "max_num_seqs") is None
    assert unmet_prerequisite(None, "max_num_seqs") is None

    # No live engine -> can't verify a prerequisite-gated knob -> reject.
    for knob in gated:
        assert "no live engine" in unmet_prerequisite(None, knob)

    # Live flag off -> reject; on -> allowed.
    off = _EngineWithFlags(chunked_prefill_enabled=False, enable_dbo=False)
    on = _EngineWithFlags(chunked_prefill_enabled=True, enable_dbo=True)
    for knob in gated:
        assert unmet_prerequisite(off, knob) is not None
        assert unmet_prerequisite(on, knob) is None

    # Engine exposes neither the taxonomy path nor a flat attr -> conservative reject.
    assert "unknown on this engine" in unmet_prerequisite(object(), "dbo_prefill_token_threshold")


def test_resolve_relative_value():
    relative = _spec("max_num_batched_tokens", 8192)
    relative.value_multiplier = 2.0
    absolute = _spec("max_num_seqs", 256)  # no multiplier -> untouched

    class _Sched:
        def __init__(self, tokens):
            self.max_num_batched_tokens = tokens

    class _Engine:
        def __init__(self, tokens):
            self.scheduler_config = _Sched(tokens)

    # No engine, or no multiplier -> the static YAML/literal value is unchanged.
    assert resolve_relative_value(relative, None).value == 8192
    assert resolve_relative_value(absolute, _Engine(512)) is absolute

    # A live engine's current value scales instead of the hardcoded literal —
    # a tiny model's 512 doubles to 1024, not always jumping to 8192.
    small = resolve_relative_value(relative, _Engine(512))
    assert small.value == 1024
    assert "512 -> 1024" in small.summary

    big = resolve_relative_value(relative, _Engine(4096))
    assert big.value == 8192

    # Can't read the current value -> falls back to the static value, not a crash.
    assert resolve_relative_value(relative, object()).value == 8192


def test_resolve_relative_value_respects_clamp():
    frac = _spec("gpu_memory_utilization", 0.92)
    frac.value_multiplier = 1.10
    frac.value_max = 0.97

    class _Engine:
        cache_config = type("C", (), {"gpu_memory_utilization": 0.95})()

    # 0.95 * 1.10 = 1.045, which would be an invalid fraction -> clamped.
    assert resolve_relative_value(frac, _Engine()).value == 0.97


def test_expand_relative_candidates():
    from gitm.optimizer.vllm_knobs import expand_relative_candidates

    swept = _spec("max_num_seqs", 256)
    swept.value_multiplier_grid = [0.5, 2.0, 4.0]
    plain = _spec("block_size", 16)  # no multiplier/grid -> untouched

    class _Sched:
        def __init__(self, seqs):
            self.max_num_seqs = seqs

    class _Engine:
        def __init__(self, seqs):
            self.scheduler_config = _Sched(seqs)

    # No grid at all -> single-item list, spec passed through as-is.
    assert expand_relative_candidates(plain, _Engine(32)) == [plain]

    # Each grid point becomes its own candidate, scaled off the SAME current
    # value, with a distinct name.
    out = expand_relative_candidates(swept, _Engine(64))
    assert {s.value for s in out} == {32, 128, 256}
    assert len({s.name for s in out}) == 3
    assert all(s.name.startswith("max_num_seqs") for s in out)

    # No live engine -> nothing to scale relative to; one candidate at the
    # static fallback, not 3 duplicates of it.
    offline = expand_relative_candidates(swept, None)
    assert len(offline) == 1 and offline[0].value == 256

    # Every grid point collapses to the same value (current == 0) -> dedup to
    # one candidate, not 3 identical ones.
    zeroed = expand_relative_candidates(swept, _Engine(0))
    assert len(zeroed) == 1 and zeroed[0].value == 256  # static fallback


class _SchedCfg:
    def __init__(self):
        self.max_num_seqs = 32


class _StructuredEngine:
    def __init__(self):
        self.scheduler_config = _SchedCfg()


class _FlatEngine:
    def __init__(self):
        self.max_num_seqs = 32  # no scheduler_config — older/test layout


def test_get_set_via_structured_path():
    e = _StructuredEngine()
    assert get_knob(e, "max_num_seqs") == 32
    set_knob(e, "max_num_seqs", 256)
    assert e.scheduler_config.max_num_seqs == 256


def test_get_set_flat_fallback():
    e = _FlatEngine()
    assert get_knob(e, "max_num_seqs") == 32
    set_knob(e, "max_num_seqs", 128)
    assert e.max_num_seqs == 128


def test_env_knob_round_trip(monkeypatch):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    assert resolve_knob("VLLM_ATTENTION_BACKEND").is_env
    set_knob(object(), "VLLM_ATTENTION_BACKEND", "FLASHINFER")
    assert get_knob(object(), "VLLM_ATTENTION_BACKEND") == "FLASHINFER"


def test_unresolvable_knob_raises():
    with pytest.raises(AttributeError):
        set_knob(object(), "max_num_seqs", 8)  # bare object has neither path nor flat attr


# --------------------------------------------------------------------------- #
# LiveEngineApplicator: restart-apply                                         #
# --------------------------------------------------------------------------- #
class _TpsEngine:
    """Engine whose throughput is a settable number; restart yields a new one."""

    def __init__(self, tps: float):
        self.scheduler_config = _SchedCfg()
        self._tps = tps


def _tps_of(e):
    return e._tps


def test_engine_arg_knob_uses_restart_not_hotswap():
    e = _TpsEngine(100.0)
    built: list[dict[str, object]] = []

    def restart_fn(old, knob_values):
        assert old is e
        built.append(dict(knob_values))
        return _TpsEngine(200.0)

    app = LiveEngineApplicator(e, throughput_fn=_tps_of, restart_fn=restart_fn)
    res = apply_intervention(_spec("max_num_seqs", 256), app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert built == [{"max_num_seqs": 256}]
    assert app.engine is not e and app.engine._tps == 200.0
    assert app.last_result.via == "restart"
    assert app.last_result.speedup == pytest.approx(2.0)


def test_restart_apply_structural_knob_kept_and_swaps_engine():
    e0 = _TpsEngine(100.0)

    def restart_fn(old, knob_values):
        assert knob_values == {"block_size": 16}
        return _TpsEngine(200.0)  # the "restarted" engine is faster

    app = LiveEngineApplicator(e0, throughput_fn=_tps_of, restart_fn=restart_fn)
    res = apply_intervention(_spec("block_size", 16), app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert app.engine is not e0 and app.engine._tps == 200.0  # live engine replaced
    assert app.last_result.via == "restart"
    assert app.last_result.speedup == pytest.approx(2.0)


def test_restart_apply_rolls_back_to_original_engine_on_regression():
    e0 = _TpsEngine(100.0)
    candidate = _TpsEngine(50.0)  # slower => must roll back

    app = LiveEngineApplicator(e0, throughput_fn=_tps_of, restart_fn=lambda *a: candidate)
    res = apply_intervention(_spec("tensor_parallel_size", 2), app, min_keep_delta=0.0)
    assert res.applied and res.rolled_back
    assert app.engine is e0  # original engine restored



def test_serial_restart_releases_baseline_before_building_candidate():
    class Engine(_TpsEngine):
        def __init__(self, tps: float):
            super().__init__(tps)
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    baseline = Engine(100.0)
    restored = Engine(100.0)
    events: list[str] = []

    def restart_fn(old, knob_values):
        assert old is baseline
        assert old.shutdown_called
        events.append("restart:" + ",".join(f"{k}={v}" for k, v in knob_values.items()))
        return Engine(50.0)

    def baseline_restart_fn(old):
        assert old is baseline
        events.append("restore-baseline")
        return restored

    app = LiveEngineApplicator(
        baseline,
        throughput_fn=_tps_of,
        restart_fn=restart_fn,
        baseline_restart_fn=baseline_restart_fn,
        restart_mode="serial",
    )

    res = apply_intervention(_spec("block_size", 16), app, min_keep_delta=0.0)

    assert res.applied and res.rolled_back
    assert events == ["restart:block_size=16", "restore-baseline"]
    assert app.engine is restored
    assert app.last_result.via == "restart"


def test_serial_restart_rebuilds_baseline_if_candidate_build_fails():
    baseline = _TpsEngine(100.0)
    restored = _TpsEngine(100.0)

    def restart_fn(_old, _knob_values):
        raise RuntimeError("candidate OOM")

    app = LiveEngineApplicator(
        baseline,
        throughput_fn=_tps_of,
        restart_fn=restart_fn,
        baseline_restart_fn=lambda _old: restored,
        restart_mode="serial",
    )

    res = apply_intervention(_spec("block_size", 16), app, min_keep_delta=0.0)

    assert not res.applied and res.rolled_back
    assert "candidate OOM" in res.error
    assert app.engine is restored

def test_structural_knob_without_restart_fn_rolls_back_cleanly():
    e0 = _TpsEngine(100.0)
    app = LiveEngineApplicator(e0, throughput_fn=_tps_of)  # no restart_fn
    res = apply_intervention(_spec("quantization", "awq"), app, min_keep_delta=0.0)
    assert not res.applied and res.rolled_back
    assert "structural" in res.error
    assert app.engine is e0  # nothing changed


def test_structural_raises_typed_exception_directly():
    app = LiveEngineApplicator(_TpsEngine(1.0), throughput_fn=_tps_of)
    with pytest.raises(StructuralKnobRequiresRestart):
        app.apply(_spec("pipeline_parallel_size", 2))


# --------------------------------------------------------------------------- #
# scheduler-stats causal attribution                                #
# --------------------------------------------------------------------------- #
def _summary(**over) -> SchedulerStatsSummary:
    base = dict(
        n_samples=10, duration_s=1.0, peak_queue_depth=0, mean_running=4.0,
        peak_running=4, mean_batch_occupancy=0.9, total_preemptions=0,
        peak_gpu_cache_usage=0.5, peak_swapped=0,
    )
    base.update(over)
    return SchedulerStatsSummary(**base)


def test_no_samples_yields_no_causes():
    assert scheduler_causes(None) == []
    assert scheduler_causes(_summary(n_samples=0)) == []


def test_preemption_cause():
    causes = scheduler_causes(_summary(total_preemptions=5))
    assert any(c.signal == "kv_cache_preemption" for c in causes)
    kv = next(c for c in causes if c.signal == "kv_cache_preemption")
    assert "gpu_memory_utilization" in kv.motivates_knobs


def test_under_filled_batch_cause():
    causes = scheduler_causes(_summary(mean_batch_occupancy=0.1))
    occ = next(c for c in causes if c.signal == "under_filled_batch")
    assert "max_num_seqs" in occ.motivates_knobs
    assert occ.severity > 0.5  # 0.1 vs 0.6 floor is a big deficit


def test_admission_backlog_cause():
    causes = scheduler_causes(_summary(peak_queue_depth=20, peak_running=4))
    assert any(c.signal == "admission_backlog" for c in causes)


def test_kv_cache_pressure_cause():
    causes = scheduler_causes(_summary(peak_gpu_cache_usage=0.98))
    assert any(c.signal == "kv_cache_pressure" for c in causes)


def test_causes_sorted_by_severity_desc():
    causes = scheduler_causes(
        _summary(total_preemptions=10, mean_batch_occupancy=0.55,
                 peak_queue_depth=20, peak_running=4, peak_gpu_cache_usage=0.95)
    )
    sevs = [c.severity for c in causes]
    assert sevs == sorted(sevs, reverse=True)
    assert len(causes) >= 3


# --------------------------------------------------------------------------- #
# end-to-end: scheduler stats feed attribution + claim evidence (Task 2)      #
# --------------------------------------------------------------------------- #
class _ModelCfgBf16:
    dtype = "torch.bfloat16"


class _LowOccScheduler:
    running = [0, 1]  # 2 running
    waiting: list = []
    swapped: list = []


class _SchedCfg64:
    max_num_seqs = 64  # 2/64 occupancy => under-filled-batch cause fires


class _FullEngine:
    """Fake engine exposing scheduler stats and a restartable max_num_seqs."""

    def __init__(self, max_num_seqs: int = 64):
        self.model_config = _ModelCfgBf16()
        self.scheduler = [_LowOccScheduler()]
        self.scheduler_config = _SchedCfg64()
        self.scheduler_config.max_num_seqs = max_num_seqs
        self.gitm_throughput_fn = lambda e: float(e.scheduler_config.max_num_seqs)
        self.gitm_restart_fn = self._restart

    def get_num_unfinished_requests(self):
        return 2

    def _restart(self, _old_engine, knob_values):
        return _FullEngine(max_num_seqs=int(knob_values.get("max_num_seqs", 64)))


def test_run_loop_scheduler_stats_feed_attribution_and_claims(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager
    from pathlib import Path

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 80)
                   for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    engine = _FullEngine()
    # Non-expiring budget: max_num_seqs_dynamic ranks low and this asserts it's
    # reached + kept; a short wall-clock would make that racy under load (loop
    # is budget-bounded).
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=50))

    run_dir = Path(out["run_dir"])
    residuals = json.loads((run_dir / "residuals.json").read_text())
    # The engine signal reached attribution: under-filled-batch cause is present.
    signals = {c["signal"] for c in residuals["scheduler_causes"]}
    assert "under_filled_batch" in signals
    # And it reached the claim that it motivates (max_num_seqs), in the report.
    assert "scheduler[under_filled_batch]" in out["report_md"]
    # The restart path won and was kept for the generated max_num_seqs lever.
    assert "via restart" in out["report_md"]
    assert "max_num_seqs" in out["report_md"]
    # Scheduler summary surfaced in the run summary (synchronous first sample).
    assert out["summary"]["scheduler_stats"] is not None

def test_report_kernel_time_residual_uses_weighted_total_and_clamps():
    from gitm.optimizer.monitor import KernelResidual, Residuals
    from gitm.scheduler.loop import _agg_kt_residual

    res = Residuals(
        per_kernel=[
            KernelResidual(op="tiny", layer=None, r_kt=9999.0, r_mt=None, t_obs_s=200e-6, t_pred_s=1e-8),
            KernelResidual(op="main", layer=None, r_kt=0.1, r_mt=None, t_obs_s=11e-6, t_pred_s=10e-6),
        ]
    )

    assert _agg_kt_residual(res) == 1.0

    sane = Residuals(
        per_kernel=[
            KernelResidual(op="a", layer=None, r_kt=9.0, r_mt=None, t_obs_s=12e-6, t_pred_s=10e-6),
            KernelResidual(op="b", layer=None, r_kt=0.2, r_mt=None, t_obs_s=12e-6, t_pred_s=10e-6),
        ]
    )
    assert _agg_kt_residual(sane) == pytest.approx(0.2)


def test_ar_target_residual_uses_the_search_target_not_a_hardcoded_zero():
    from gitm.agents.autoresearch import AutoresearchRun, ResidualTarget
    from gitm.scheduler.loop import _ar_target_residual

    # No target found (nothing exceeded its predicted ceiling) -> honest 0.0.
    empty = AutoresearchRun(bottleneck_class="idle_stall", results=[], target=None)
    assert _ar_target_residual(empty) == 0.0

    # A real target -> its residual surfaces, clamped like every other residual.
    modest = AutoresearchRun(
        bottleneck_class="idle_stall", results=[],
        target=ResidualTarget(op="attn_score_value", residual=0.42, n_kernels=8),
    )
    assert _ar_target_residual(modest) == pytest.approx(0.42)

    huge = AutoresearchRun(
        bottleneck_class="idle_stall", results=[],
        target=ResidualTarget(op="attn_score_value", residual=17.8, n_kernels=8),
    )
    assert _ar_target_residual(huge) == 1.0
