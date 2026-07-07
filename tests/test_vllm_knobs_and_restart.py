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
from gitm.optimizer.vllm_knobs import get_knob, knob_kind, resolve_knob, set_knob
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
    assert knob_kind("max_num_seqs") == "scheduling"
    assert knob_kind("max_num_batched_tokens") == "scheduling"
    assert knob_kind("scheduling_policy") == "scheduling"
    assert knob_kind("tensor_parallel_size") == "structural"
    assert knob_kind("block_size") == "structural"
    assert knob_kind("totally_unknown_knob") == "structural"  # safe default


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
# LiveEngineApplicator: hot-swap vs restart-apply                             #
# --------------------------------------------------------------------------- #
class _TpsEngine:
    """Engine whose throughput is a settable number; restart yields a new one."""

    def __init__(self, tps: float):
        self.scheduler_config = _SchedCfg()
        self._tps = tps


def _tps_of(e):
    return e._tps


def test_hotswap_scheduling_knob_kept():
    e = _TpsEngine(100.0)
    # throughput rises with max_num_seqs so the hot-swap is a measurable win.
    app = LiveEngineApplicator(e, throughput_fn=lambda x: float(x.scheduler_config.max_num_seqs))
    res = apply_intervention(_spec("max_num_seqs", 256), app, min_keep_delta=0.0)
    assert res.applied and not res.rolled_back
    assert e.scheduler_config.max_num_seqs == 256
    assert app.last_result.via == "hot-swap"
    assert app.last_result.speedup == pytest.approx(256 / 32)


def test_restart_apply_structural_knob_kept_and_swaps_engine():
    e0 = _TpsEngine(100.0)

    def restart_fn(old, knob, value):
        assert knob == "block_size" and value == 16
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
    """Fake engine exposing scheduler stats AND a hot-swappable max_num_seqs."""

    def __init__(self):
        self.model_config = _ModelCfgBf16()
        self.scheduler = [_LowOccScheduler()]
        self.scheduler_config = _SchedCfg64()
        self.gitm_throughput_fn = lambda e: float(e.scheduler_config.max_num_seqs)

    def get_num_unfinished_requests(self):
        return 2


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
    # Non-expiring budget: max_num_seqs_256 ranks low and this asserts it's reached
    # + kept; a short wall-clock would make that racy under load (loop is budget-bounded).
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=50))

    run_dir = Path(out["run_dir"])
    residuals = json.loads((run_dir / "residuals.json").read_text())
    # The engine signal reached attribution: under-filled-batch cause is present.
    signals = {c["signal"] for c in residuals["scheduler_causes"]}
    assert "under_filled_batch" in signals
    # And it reached the claim that it motivates (max_num_seqs), in the report.
    assert "scheduler[under_filled_batch]" in out["report_md"]
    # The hot-swap won and was kept.
    assert engine.scheduler_config.max_num_seqs == 256
    # Scheduler summary surfaced in the run summary (synchronous first sample).
    assert out["summary"]["scheduler_stats"] is not None
