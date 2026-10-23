"""Knob taxonomy, restart-apply for structural knobs, and
scheduler-stats causal attribution.

No GPU / no vLLM: fake engines stand in. These cover the two pieces that were
still in progress — #4 (hot-swap vs restart-apply) and #2 (stats → attribution).
"""

from __future__ import annotations

import re

import numpy as np
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


def test_prerequisite_flag_is_not_its_own_prerequisite():
    # ("dbo", "enable_dbo") substring-matches enable_dbo itself; turning the flag
    # on must not demand that it already be on.
    off = _EngineWithFlags(enable_dbo=False)
    assert unmet_prerequisite(off, "enable_dbo") is None
    assert unmet_prerequisite(None, "enable_dbo") is None
    assert unmet_prerequisite(object(), "enable_dbo") is None
    # The knobs the flag gates are still gated by it.
    assert unmet_prerequisite(off, "dbo_decode_token_threshold") is not None


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


class _ShutEngine(_TpsEngine):
    def __init__(self, tps: float):
        super().__init__(tps)
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


def test_parallel_restart_kept_candidate_releases_old_engine():
    """A kept parallel-mode candidate must release the engine it replaced.

    Regression: restore() was the only release path, so on a keep the previous
    engine sat in _prev until the next snapshot() dropped it unreleased — every
    later candidate was built beside two resident engines, not one.
    """
    e0 = _ShutEngine(100.0)
    e1 = _ShutEngine(200.0)  # wins against e0
    e2 = _ShutEngine(150.0)  # loses against e1
    builds = iter([e1, e2])
    built_from: list = []

    def restart_fn(old, _knob_values):
        built_from.append(old)
        return next(builds)

    app = LiveEngineApplicator(e0, throughput_fn=_tps_of, restart_fn=restart_fn)

    res = apply_intervention(_spec("block_size", 16), app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    # Released at the keep itself — not deferred to a next candidate that a
    # last-candidate win would never get.
    assert e0.shutdown_calls == 1
    assert app.engine is e1 and e1.shutdown_calls == 0

    res = apply_intervention(_spec("block_size", 32), app, min_keep_delta=0.0)
    assert res.rolled_back
    assert built_from == [e0, e1]  # the winner is the next baseline
    assert e2.shutdown_calls == 1  # rollback drops the losing candidate...
    assert app.engine is e1 and e1.shutdown_calls == 0  # ...and keeps the winner
    assert e0.shutdown_calls == 1  # never released twice


def test_engine_release_failure_is_warned_not_swallowed():
    class Broken(_TpsEngine):
        def shutdown(self):
            raise RuntimeError("rank 3 did not exit")

    e0 = Broken(100.0)
    app = LiveEngineApplicator(e0, throughput_fn=_tps_of,
                               restart_fn=lambda *_: _TpsEngine(200.0))
    with pytest.warns(RuntimeWarning, match="rank 3 did not exit"):
        res = apply_intervention(_spec("block_size", 16), app, min_keep_delta=0.0)
    # A failed release still never aborts the A/B: the winner is kept.
    assert res.applied and not res.rolled_back


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
        self.shut_down = False

    def shutdown(self):
        self.shut_down = True

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
    cfg = LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                     scratch=str(tmp_path), top_n_interventions=50)
    out = run_loop(cfg)

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
    # The kept restart released the original engine, and the caller is handed
    # the live winner instead of that released handle.
    assert engine.shut_down
    assert cfg.engine is not engine and not cfg.engine.shut_down


def test_run_loop_vetoes_library_lever_with_unmet_prerequisite(tmp_path, monkeypatch):
    from contextlib import contextmanager

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

    # _FullEngine has no enable_expert_parallel and no enable_dbo.
    out = run_loop(LoopConfig(engine=_FullEngine(), workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=50))

    report = out["report_md"]
    # The library lever is vetoed before apply, same as an autoresearch candidate.
    assert "enable_eplb (prerequisite 'enable_expert_parallel' unknown on this engine)" in report
    # enable_dbo is its own prerequisite flag, so it is not vetoed for lacking itself.
    assert "enable_dbo (prerequisite" not in report


class _ParallelCfg:
    def __init__(self, *, enable_expert_parallel: bool, enable_eplb: bool):
        self.enable_expert_parallel = enable_expert_parallel
        self.enable_eplb = enable_eplb


class _ExpertParallelEngine(_FullEngine):
    """_FullEngine whose MoE flags only a restart can change. Each flag on adds
    throughput, so the A/B keeps whichever lever turns one on."""

    def __init__(self, *, enable_expert_parallel: bool = False, enable_eplb: bool = False):
        super().__init__()
        self.parallel_config = _ParallelCfg(
            enable_expert_parallel=enable_expert_parallel, enable_eplb=enable_eplb)
        self.gitm_throughput_fn = lambda e: (
            64.0 + 8.0 * e.parallel_config.enable_expert_parallel
            + 4.0 * e.parallel_config.enable_eplb)

    def _restart(self, old_engine, knob_values):
        pc = old_engine.parallel_config
        return _ExpertParallelEngine(
            enable_expert_parallel=bool(
                knob_values.get("enable_expert_parallel", pc.enable_expert_parallel)),
            enable_eplb=bool(knob_values.get("enable_eplb", pc.enable_eplb)))


def test_run_loop_prerequisite_reads_the_engine_a_kept_restart_left(tmp_path, monkeypatch):
    """cfg.engine has expert parallel off. The first lever turns it on through a
    restart that is kept, so enable_eplb's prerequisite holds only on the engine
    the applicator now holds. A check that read cfg.engine would veto it."""
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.kernels.spec import Applicability, SafetyGate
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace

    def lever(name, knob, mean):
        return InterventionSpec(
            name=name, summary=f"set {knob}", knob=knob, value=True,
            expected_delta_mean=mean, expected_delta_lo=0.0, expected_delta_hi=0.2,
            source="test", applies_to_kernels=["paged_attention"],
            applicability=Applicability(workloads=["vllm-decode"]),
            safety=SafetyGate(tier="moderate"))

    ep = lever("ep_on", "enable_expert_parallel", 0.08)
    eplb = lever("eplb_on", "enable_eplb", 0.04)

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 80)
                   for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    applied: list[str] = []
    real_apply = loop.apply_intervention

    def recording_apply(spec, *a, **kw):
        applied.append(spec.name)
        return real_apply(spec, *a, **kw)

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    monkeypatch.setattr(loop, "load_library", lambda workload=None: [ep, eplb])
    monkeypatch.setattr(loop, "apply_intervention", recording_apply)

    engine = _ExpertParallelEngine()
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path)))

    report = out["report_md"]
    assert applied == ["ep_on", "eplb_on"]
    assert "eplb_on (prerequisite" not in report
    # The restart behind ep_on was kept, and cfg.engine itself never changed.
    assert report.count("via restart") >= 1
    assert engine.parallel_config.enable_expert_parallel is False


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

    # No target found (nothing exceeded its predicted ceiling): the run-level
    # residual when there is one, else None. A 0.0 here read as "on target".
    empty = AutoresearchRun(bottleneck_class="idle_stall", results=[], target=None)
    assert _ar_target_residual(empty) is None
    assert _ar_target_residual(empty, 0.12) == 0.12

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


@pytest.mark.parametrize(
    ("error", "status"),
    [(ImportError("No module named 'statsmodels.api'"), "unavailable"),
     (np.linalg.LinAlgError("Singular matrix"), "failed")],
)
def test_run_loop_survives_dr_failure_and_records_granger_status(tmp_path, monkeypatch, error, status):
    """A DR failure (dr.py imports statsmodels.api on its own, and its numpy math
    can raise) must not lose the Granger record or the report."""
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

    def dr_without_statsmodels(*args, **kwargs):
        raise error

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    monkeypatch.setattr(loop, "attribute_dr", dr_without_statsmodels)

    out = run_loop(LoopConfig(engine=_FullEngine(), workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=3))

    residuals = json.loads((Path(out["run_dir"]) / "residuals.json").read_text())
    assert residuals["dr"]["status"] == status
    assert residuals["dr"]["first_errors"] == {type(error).__name__: str(error)}
    assert residuals["dr"]["pairs_attempted"] == 0  # same record shape as a DR that ran
    assert residuals["granger"]["status"] in {"ok", "partial", "failed", "insufficient_data"}
    assert "top_hypotheses_granger" in residuals  # legacy key kept for readers
    assert "no strong causal signal" not in out["report_md"]


_OP_PAIR = re.compile(r"\S+\s*(→|->)\s*\S+\s*\(p=")


def _two_op_capture(n_per_op: int = 40):
    """A capture whose kernels classify to two graph ops, enough for Granger."""
    from contextlib import contextmanager

    import numpy as np

    from .conftest import make_kernel, make_trace

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        rng = np.random.default_rng(0)
        kernels, t = [], 0
        for _ in range(n_per_op):
            for name, base in (("_ZN5flash24flash_fwd_splitkv_kernel", 30_000),
                               ("silu_and_mul_kernel", 8_000)):
                dur = int(base * (1 + rng.normal(0, 0.2)))
                kernels.append(make_kernel(name, start_ns=t, end_ns=t + dur))
                t += dur + 100
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    return fake_capture


def _run_two_op_loop(tmp_path, monkeypatch):
    """Predict-only loop run (no engine), so no claim has a live A/B and every
    catalog claim's evidence is the Granger text."""
    import json
    from pathlib import Path

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    monkeypatch.setattr(loop, "capture", _two_op_capture())
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    out = run_loop(LoopConfig(engine=None, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=5))
    residuals = json.loads((Path(out["run_dir"]) / "residuals.json").read_text())
    return out, residuals


def test_run_loop_granger_completes_on_installed_statsmodels(tmp_path, monkeypatch):
    """Through the real loop and the installed statsmodels, Granger runs to
    completion, and no claim shows an op pair."""
    pytest.importorskip("statsmodels.tsa.stattools")
    out, residuals = _run_two_op_loop(tmp_path, monkeypatch)

    granger = residuals["granger"]
    assert granger["status"] == "ok", granger
    assert granger["pairs_completed"] == granger["pairs_attempted"] == 2
    assert {p["exploratory"] for p in granger["top_pairs"]} == {True}
    assert len(residuals["top_hypotheses_granger"]) == 2
    # DR's record is its real outcome, not a blanket "ok" for "did not raise".
    dr = residuals["dr"]
    assert dr["status"] in {"ok", "partial", "failed", "insufficient_data"}
    assert {"pairs_attempted", "pairs_completed", "warnings", "reason"} <= dr.keys()
    if dr["status"] == "insufficient_data":
        assert dr["reason"]
    claim_rows = [line for line in out["report_md"].splitlines() if re.match(r"\| \d+ \|", line)]
    assert claim_rows
    assert all("Granger ran (2/2 op pairs) but is not used as evidence" in row for row in claim_rows)
    assert not _OP_PAIR.search(out["report_md"])
    assert "no strong causal signal" not in out["report_md"]


def test_run_loop_keeps_legacy_granger_key_on_partial(tmp_path, monkeypatch):
    """Readers of top_hypotheses_granger still get the pairs that completed."""
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    real = stattools.grangercausalitytests
    calls = {"n": 0}

    def fail_first_pair(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(stattools, "grangercausalitytests", fail_first_pair)
    out, residuals = _run_two_op_loop(tmp_path, monkeypatch)

    assert residuals["granger"]["status"] == "partial"
    assert residuals["granger"]["failures"] == {"ValueError": 1}
    assert len(residuals["top_hypotheses_granger"]) == 1
    assert "1 of 2 op pairs failed (ValueError: 1)" in out["report_md"]
def test_nested_knobs_reach_llm_as_the_argument_vllm_takes():
    """P2-1, which was marked fixed but was not: LLM() rejects
    num_speculative_tokens at the top level; it lives in speculative_config."""
    from gitm.optimizer.vllm_knobs import engine_kwargs, knob_from_server_arg, server_arg

    assert engine_kwargs({"num_speculative_tokens": 5, "max_num_seqs": 64}) == {
        "speculative_config": {"method": "ngram", "num_speculative_tokens": 5},
        "max_num_seqs": 64,
    }
    flag, value = server_arg("num_speculative_tokens", 5)
    assert flag == "--speculative-config"
    assert knob_from_server_arg("speculative_config", value) == ("num_speculative_tokens", 5)
    # Read back as the lever only when the token count is all that changed from
    # the baseline. Against no baseline, that is the n-gram config gitm emits.
    other = '{"method": "eagle", "model": "x", "num_speculative_tokens": 5}'
    assert knob_from_server_arg("speculative_config", other)[0] == "speculative_config"
    eagle_base = '{"method": "eagle", "model": "x", "num_speculative_tokens": 2}'
    assert knob_from_server_arg("speculative_config", other, eagle_base) == \
        ("num_speculative_tokens", 5)
    # A changed method is not the token-count lever, even at the lever's count.
    ngram_base = '{"method": "ngram", "num_speculative_tokens": 5}'
    assert knob_from_server_arg("speculative_config", other, ngram_base)[0] == \
        "speculative_config"
    assert knob_from_server_arg("speculative_config", "not json")[0] == "speculative_config"


def test_a_nested_knob_is_merged_into_the_baselines_config_not_put_in_its_place():
    """Replacing it would drop the baseline's method, draft model and lookup
    window, and the A/B would measure all of that, not the lever."""
    import json

    from gitm.optimizer.vllm_knobs import engine_kwargs, server_arg

    base = {"speculative_config": {"method": "eagle", "model": "x",
                                   "num_speculative_tokens": 2}}
    assert engine_kwargs({"num_speculative_tokens": 5}, base) == {
        "speculative_config": {"method": "eagle", "model": "x", "num_speculative_tokens": 5}}
    assert base["speculative_config"]["num_speculative_tokens"] == 2     # not mutated
    _, value = server_arg("num_speculative_tokens", 5,
                          '{"method":"ngram","prompt_lookup_max":4}')
    assert json.loads(value) == {"method": "ngram", "prompt_lookup_max": 4,
                                 "num_speculative_tokens": 5}
    assert server_arg("max_num_seqs", 64) == ("--max-num-seqs", 64)


# --------------------------------------------------------------------------- #
# GITM_RESTART_MODE=sleep_swap                                                 #
# --------------------------------------------------------------------------- #
class _SleepEngine(_FullEngine):
    """_FullEngine that can sleep and wake, counting every build it causes.

    Throughput is max_num_seqs, as for _FullEngine; measuring an engine that is
    asleep or shut down fails, as it would on a real one.
    """

    def __init__(self, max_num_seqs: int = 64, *, log: dict | None = None,
                 free: float | None = None, fail_builds: int = 0):
        super().__init__(max_num_seqs)
        self.log = log if log is not None else {"builds": 0, "fail_builds": fail_builds,
                                                "events": [], "engines": []}
        self.log["engines"].append(self)
        self.asleep = False
        self.gitm_llm_kwargs = {"enable_sleep_mode": True, "gpu_memory_utilization": 0.9}
        self.gitm_throughput_fn = self._measure
        self.gitm_baseline_restart_fn = self._rebuild_baseline
        self._free = free
        if free is not None:
            self.gitm_free_gpu_fraction_fn = lambda e, values=None: e._free

    def _measure(self, e):
        assert not e.asleep and not e.shut_down, "measured an engine that is not live"
        return float(e.scheduler_config.max_num_seqs)

    def sleep(self, level=1):
        assert level == 1
        self.log["events"].append(("sleep", self.scheduler_config.max_num_seqs))
        self.asleep = True

    def wake_up(self):
        self.log["events"].append(("wake", self.scheduler_config.max_num_seqs))
        self.asleep = False

    def shutdown(self):
        self.log["events"].append(("shutdown", self.scheduler_config.max_num_seqs))
        self.shut_down = True

    def _build(self, max_num_seqs):
        self.log["builds"] += 1
        if self.log["fail_builds"] > 0:
            self.log["fail_builds"] -= 1
            raise RuntimeError("Free memory on device is less than desired")
        return _SleepEngine(max_num_seqs, log=self.log, free=self._free)

    def _restart(self, old, knob_values):
        return self._build(int(knob_values.get("max_num_seqs", 64)))

    def _rebuild_baseline(self, old):
        return self._build(old.scheduler_config.max_num_seqs)

    def live(self):
        return [e for e in self.log["engines"] if not e.shut_down]


def _sleep_app(engine, mode="sleep_swap", **kw):
    return LiveEngineApplicator(
        engine, throughput_fn=engine.gitm_throughput_fn, restart_fn=engine.gitm_restart_fn,
        restart_mode=mode, force_restart=True, **kw)


def test_sleep_swap_rollback_wakes_the_baseline_without_a_rebuild():
    base = _SleepEngine(64)
    app = _sleep_app(base)
    res = apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert res.applied and res.rolled_back
    assert app.engine is base and not base.asleep and not base.shut_down
    assert base.log["builds"] == 1                     # the candidate only
    assert base.live() == [base]
    assert base.log["events"] == [("sleep", 64), ("shutdown", 32), ("wake", 64)]
    (swap,) = app.swap_log
    assert swap["via"] == "sleep_swap" and swap["fallback"] is None
    assert len(swap["sleep_s"]) == 1 and len(swap["wake_s"]) == 1


def test_sleep_swap_keep_shuts_the_old_baseline_and_promotes_the_candidate():
    base = _SleepEngine(64)
    app = _sleep_app(base)
    res = apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert res.applied and not res.rolled_back
    assert base.shut_down
    assert app.engine.scheduler_config.max_num_seqs == 128 and not app.engine.asleep
    assert base.live() == [app.engine]
    assert base.log["builds"] == 1
    # The next candidate sleeps the promoted engine, not the released one.
    apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)
    assert app.engine.scheduler_config.max_num_seqs == 128
    assert base.log["builds"] == 2 and len(base.live()) == 1


def test_sleep_swap_falls_back_to_serial_when_the_candidate_cannot_fit():
    # 5% free with the baseline asleep; the candidate asks for 90%.
    base = _SleepEngine(64, free=0.05)
    app = _sleep_app(base)
    res = apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert res.applied and res.rolled_back
    assert base.shut_down                               # released, not kept asleep
    assert base.log["builds"] == 2                      # candidate + serial rebuild
    assert app.engine is not base and base.live() == [app.engine]
    (swap,) = app.swap_log
    assert swap["via"] == "serial"
    assert "5% of the device is free" in swap["fallback"]
    assert "gpu_memory_utilization=0.90" in swap["fallback"]
    # Measured as a serial restart: the candidate kept its own fraction.
    assert app.last_result.via == "restart"


def test_sleep_swap_falls_back_to_serial_when_the_build_fails_beside_the_sleeper():
    base = _SleepEngine(64, fail_builds=1)
    app = _sleep_app(base)
    res = apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert res.applied and not res.rolled_back
    assert base.shut_down
    assert app.engine.scheduler_config.max_num_seqs == 128
    assert base.live() == [app.engine]
    assert "failed to build beside the sleeping baseline" in app.swap_log[0]["fallback"]


def test_sleep_swap_falls_back_when_the_baseline_was_built_without_sleep_mode():
    base = _SleepEngine(64)
    base.gitm_llm_kwargs["enable_sleep_mode"] = False
    app = _sleep_app(base)
    apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert not any(ev == "sleep" for ev, _ in base.log["events"])
    assert "enable_sleep_mode" in app.swap_log[0]["fallback"]
    assert base.live() == [app.engine]


def test_sleep_swap_rounds_swap_the_same_two_builds_and_use_factor_one():
    from gitm.optimizer.apply import NoiseFloor, mean_noise_factor

    base = _SleepEngine(64)
    noise = NoiseFloor()
    noise.samples.extend([0.02, -0.02, 0.02])
    app = _sleep_app(base, validation=3, noise_gate=True, noise=noise)
    res = apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert res.applied and not res.rolled_back
    assert len(app.last_result.rounds) == 3
    assert base.log["builds"] == 1                      # no rebuild across rounds
    assert app.last_result.threshold == pytest.approx(
        noise.threshold(1.0, minimum=app.min_effect,
                        without_floor=app.threshold_without_floor))
    # Rebuilt rounds would have earned a smaller bar; swapped rounds must not.
    assert mean_noise_factor(3, rebuilt=True) < 1.0
    # A woken baseline is not a fresh build, so it adds no A/A sample.
    assert noise.samples == [0.02, -0.02, 0.02]
    # Kept, and the candidate is the one awake.
    assert not app.engine.asleep and app.engine.scheduler_config.max_num_seqs == 128
    assert base.live() == [app.engine]


def test_sleep_swap_keep_after_a_baseline_round_wakes_the_candidate():
    base = _SleepEngine(64)
    app = _sleep_app(base, validation=2)
    res = apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert res.applied and not res.rolled_back
    # Round 2 ended on the baseline; the keep woke the candidate again.
    assert app.engine.scheduler_config.max_num_seqs == 128 and not app.engine.asleep
    assert base.shut_down and base.log["builds"] == 1


def test_sleep_swap_releases_the_candidate_when_its_measurement_fails():
    base = _SleepEngine(64)
    app = _sleep_app(base)
    real = base._measure

    def flaky(e):
        if e is not base:
            raise RuntimeError("decode hung")
        return real(e)

    app._tps = flaky
    res = apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert res.rolled_back and "decode hung" in res.error
    assert app.engine is base and not base.asleep
    assert base.live() == [base]


def test_sleep_swap_baseline_that_will_not_wake_is_lost_not_leaked():
    base = _SleepEngine(64)

    def broken_wake():
        raise RuntimeError("wake_up failed")

    base.wake_up = broken_wake
    app = _sleep_app(base)
    res = apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert res.restore_failed and "could not wake the baseline" in res.error
    assert app.engine is None
    assert base.live() == []


def test_sleep_swap_interrupted_build_still_wakes_the_baseline():
    base = _SleepEngine(64)

    def interrupted(_old, _values):
        raise KeyboardInterrupt

    app = _sleep_app(base)
    app._restart_fn = interrupted
    with pytest.raises(KeyboardInterrupt):
        apply_intervention(_spec("max_num_seqs", 128), app, min_keep_delta=0.0)

    assert app.engine is base and not base.asleep and base.live() == [base]


def test_close_releases_an_engine_left_asleep():
    base = _SleepEngine(64)
    app = _sleep_app(base)
    app.snapshot()
    app.apply(_spec("max_num_seqs", 128))     # baseline asleep, candidate live
    assert base.asleep
    app.close()

    assert base.shut_down and base.live() == [app.engine]
    app.close()                               # idempotent
    assert base.live() == [app.engine]


def test_flag_off_serial_is_unchanged():
    from gitm.optimizer.apply import resolve_restart_mode

    base = _SleepEngine(64)
    assert resolve_restart_mode(base, None)[0] == "serial"
    app = _sleep_app(base, mode="serial")
    res = apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert res.rolled_back
    assert not any(ev in {"sleep", "wake"} for ev, _ in base.log["events"])
    assert base.log["builds"] == 2                      # candidate + baseline rebuild
    assert app.swap_log == [] and app.engine is not base
    assert app.last_result.rounds and app.last_result.via == "restart"


def test_run_loop_sleep_swap_records_swaps_and_leaves_one_engine(tmp_path, monkeypatch):
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
    monkeypatch.setenv("GITM_RESTART_MODE", "sleep_swap")

    engine = _SleepEngine()
    cfg = LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                     scratch=str(tmp_path), top_n_interventions=50)
    out = run_loop(cfg)

    record = json.loads((Path(out["run_dir"]) / "restart_mode.json").read_text())
    assert record["mode"] == "sleep_swap"
    assert record["swaps"] and all(s["via"] == "sleep_swap" for s in record["swaps"])
    assert all(s["sleep_s"] and s["wake_s"] for s in record["swaps"])
    # Nothing is left asleep: the one engine alive is the one handed back.
    assert engine.live() == [cfg.engine] and not cfg.engine.asleep


def test_sleep_swap_env_builds_every_engine_with_sleep_mode(tmp_path, monkeypatch):
    import sys
    import types

    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    class _LLM:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs

    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = _LLM, lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)
    monkeypatch.setenv("GITM_RESTART_MODE", "sleep_swap")

    engine = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode")).engine
    assert engine.kwargs["enable_sleep_mode"] is True
    cand = engine.gitm_restart_fn(engine, {"block_size": 32})
    assert cand.kwargs["enable_sleep_mode"] is True
    assert callable(engine.gitm_free_gpu_fraction_fn)

    monkeypatch.setenv("GITM_RESTART_MODE", "serial")
    plain = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode")).engine
    assert "enable_sleep_mode" not in plain.kwargs


def test_sleep_swap_keep_below_measures_own_bar_still_leaves_the_candidate_live():
    # A +3% candidate against a 20% floor: round 1 is no clear loss, so round 2
    # runs and ends on the baseline, and measure() does not judge it a keep.
    # A caller's negative min_keep_delta keeps it anyway.
    from gitm.optimizer.apply import NoiseFloor

    base = _SleepEngine(64)
    noise = NoiseFloor()
    noise.samples.extend([0.2, -0.2, 0.2])
    app = _sleep_app(base, validation=2, noise_gate=True, noise=noise)
    res = apply_intervention(_spec("max_num_seqs", 66), app, min_keep_delta=-1.0)

    assert res.applied and not res.rolled_back
    assert len(app.last_result.rounds) == 2
    assert app.engine is not base and not app.engine.shut_down and not app.engine.asleep
    assert app.engine.scheduler_config.max_num_seqs == 66
    assert base.shut_down and base.live() == [app.engine]


def test_close_releases_the_running_engine_when_a_swap_stopped_halfway():
    base = _SleepEngine(64)
    app = _sleep_app(base)
    app.snapshot()
    app.apply(_spec("max_num_seqs", 128))
    cand = app.engine
    app._sleep(cand)                 # a swap interrupted after its sleep, before its wake
    app.close()

    assert base.live() == [] and app.engine is None
    assert cand.shut_down and base.shut_down


def test_sleep_swap_fit_check_is_asked_about_the_candidates_settings():
    asked = []
    base = _SleepEngine(64, free=0.95)
    base.gitm_free_gpu_fraction_fn = lambda e, values=None: asked.append(values) or 0.95
    app = _sleep_app(base)
    apply_intervention(_spec("max_num_seqs", 32), app, min_keep_delta=0.0)

    assert asked == [{"max_num_seqs": 32}]


def test_free_gpu_fraction_reads_only_the_gpus_the_candidate_would_use(monkeypatch):
    import sys
    import types

    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    class _LLM:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = _LLM, lambda **kw: types.SimpleNamespace(**kw)
    # Eight GPUs; the last four are nearly full (someone else's job).
    nvml = types.ModuleType("pynvml")
    nvml.nvmlInit = nvml.nvmlShutdown = lambda: None
    nvml.nvmlDeviceGetCount = lambda: 8
    nvml.nvmlDeviceGetHandleByIndex = lambda i: i
    nvml.nvmlDeviceGetMemoryInfo = lambda i: types.SimpleNamespace(
        free=95 if i < 4 else 5, total=100)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.delenv("GITM_VLLM_SYNTHETIC", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("GITM_RESTART_MODE", "sleep_swap")
    monkeypatch.setenv("GITM_VLLM_TP", "8")

    engine = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode")).engine
    free = engine.gitm_free_gpu_fraction_fn
    assert free(engine) == pytest.approx(0.05)
    assert free(engine, {"tensor_parallel_size": 4}) == pytest.approx(0.95)


def test_run_loop_checkpoints_swaps_before_the_run_ends(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 80)
                   for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    calls = []
    real_apply = loop.apply_intervention

    def dies_on_second(spec, *a, **kw):
        calls.append(spec.name)
        if len(calls) == 2:
            raise RuntimeError("node lost")
        return real_apply(spec, *a, **kw)

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    monkeypatch.setattr(loop, "apply_intervention", dies_on_second)
    monkeypatch.setenv("GITM_RESTART_MODE", "sleep_swap")

    engine = _SleepEngine()
    with pytest.raises(RuntimeError, match="node lost"):
        run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                            scratch=str(tmp_path), top_n_interventions=50))

    (record_path,) = tmp_path.rglob("restart_mode.json")
    swaps = json.loads(record_path.read_text())["swaps"]
    assert [s["candidate"] for s in swaps] == calls[:1]
    assert swaps[0]["sleep_s"] and swaps[0]["wake_s"]
    assert not any(e.asleep and not e.shut_down for e in engine.log["engines"])


def test_sleep_swap_candidate_that_will_not_wake_for_the_keep_is_rolled_back(monkeypatch):
    # The keep-time wake-up of the candidate fails. It happens inside measure(),
    # so the candidate is rolled back and the baseline woken, never a keep that
    # hands back an engine still asleep.
    from gitm.optimizer.apply import NoiseFloor

    real_wake = _SleepEngine.wake_up

    def wake(self):
        if self.scheduler_config.max_num_seqs == 66:
            raise RuntimeError("wake_up failed")
        real_wake(self)

    monkeypatch.setattr(_SleepEngine, "wake_up", wake)
    base = _SleepEngine(64)
    noise = NoiseFloor()
    noise.samples.extend([0.2, -0.2, 0.2])
    app = _sleep_app(base, validation=2, noise_gate=True, noise=noise)
    res = apply_intervention(_spec("max_num_seqs", 66), app, min_keep_delta=-1.0)

    assert res.rolled_back and not res.kept and "wake_up failed" in res.error
    assert app.engine is base and not base.asleep and base.live() == [base]


def test_run_loop_writes_the_last_swap_on_the_way_out(tmp_path, monkeypatch):
    # The second candidate's A/B completes, then the run dies before its
    # verification record (and so its checkpoint) is written. Cleanup still
    # writes its swap.
    import json
    from contextlib import contextmanager

    import gitm.scheduler.loop as loop
    from gitm.scheduler.loop import LoopConfig, run_loop

    from .conftest import make_kernel, make_trace

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        kernels = [make_kernel(f"paged_attention_{i % 4}", start_ns=i * 100, end_ns=i * 100 + 80)
                   for i in range(80)]
        yield make_trace(events=kernels, vendor="nvidia", run_id=run_id or "r")

    calls = []
    real_apply = loop.apply_intervention

    def dies_after_second(spec, *a, **kw):
        calls.append(spec.name)
        result = real_apply(spec, *a, **kw)
        if len(calls) == 2:
            raise RuntimeError("node lost")
        return result

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    monkeypatch.setattr(loop, "apply_intervention", dies_after_second)
    monkeypatch.setenv("GITM_RESTART_MODE", "sleep_swap")

    with pytest.raises(RuntimeError, match="node lost"):
        run_loop(LoopConfig(engine=_SleepEngine(), workload="vllm-decode", budget="24h",
                            scratch=str(tmp_path), top_n_interventions=50))

    (record_path,) = tmp_path.rglob("restart_mode.json")
    swaps = json.loads(record_path.read_text())["swaps"]
    assert [s["candidate"] for s in swaps] == calls
