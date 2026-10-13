"""A lever an observed cause argues for is tried before one nothing points at (S-3).

A whole-step lever's predicted delta is a catalogue constant on any trace, so the
order was the catalogue's whatever the run showed. The scheduler and collective
causes already name the knobs they argue for; the ranking now reads them.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from gitm.agents.policy import Policy, select_interventions
from gitm.optimizer.history import History

from .test_policy_history import FP, SKU, _record, _spec, _trace


def _lib():
    # big_lever has the larger prior, so it ranks first unless a cause intervenes.
    return [_spec("big_lever", ["fused_moe_kernel"], mean=0.08),
            _spec("small_lever", ["gemm"], mean=0.02)]


def _rank(**kw):
    return select_interventions(_trace(), _lib(), kw.pop("policy", Policy()), top_n=5,
                                fingerprint=FP, **kw)


def test_no_cause_leaves_the_order_as_it_was():
    plain = [c.spec.name for c in _rank()]
    assert plain == [c.spec.name for c in _rank(motivated={})]
    assert plain[0] == "big_lever"
    assert all(c.motivated_by is None for c in _rank(motivated={}))


def test_a_motivated_lever_outranks_a_larger_prior():
    ranked = _rank(motivated={"small_lever": "kv_cache_preemption"})
    assert ranked[0].spec.name == "small_lever"
    assert ranked[0].motivated_by == "kv_cache_preemption"
    assert ranked[1].motivated_by is None


def test_a_cause_does_not_lift_a_known_loser_or_a_demoted_lever():
    """It is a precedence below the evidence of past runs: a lever measured as a
    loss, or whose record disagrees with itself, stays where that put it."""
    for record in (_record("small_lever", mean=-0.30),
                   _record("small_lever", mean=0.49, wins=2, losses=2)):
        h = History(records={(record.intervention_name, SKU, FP): record}, runs_read=1)
        ranked = _rank(policy=Policy(use_history=True), history=h, gpu_sku=SKU,
                       motivated={"small_lever": "kv_cache_preemption"})
        assert ranked[0].spec.name == "big_lever"


def test_a_rejected_lever_is_not_credited_with_a_cause():
    specs = [_spec("fix_moe", ["moe_routed"], kernel_time=True)]
    ranked = select_interventions(_trace(), specs, Policy(), top_n=5,
                                  recoverable={"moe_routed": 0.0},
                                  motivated={"fix_moe": "under_filled_batch"})
    assert ranked[0].rejected_reason is not None
    assert ranked[0].motivated_by is None


def test_the_loop_ranks_by_its_causes_and_records_which(tmp_path, monkeypatch):
    import gitm.scheduler.loop as loop
    from gitm.optimizer.scheduler_attribution import SchedulerCause
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
    monkeypatch.setattr(loop, "scheduler_causes", lambda _s: [SchedulerCause(
        signal="kv_cache_preemption", effect="e", severity=0.9, note="n",
        motivates_knobs=["max_num_seqs"])])

    engine = _FullEngine()
    engine.gitm_throughput_fn = lambda e: float(e.scheduler_config.max_num_seqs)
    out = run_loop(LoopConfig(engine=engine, workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=10))
    ranked = json.loads((Path(out["run_dir"]) / "ranked_candidates.json").read_text())

    motivated = [r for r in ranked if r["motivated_by"] == "kv_cache_preemption"]
    assert motivated, "no lever setting max_num_seqs was credited"
    live = [r for r in ranked if r["rejected_reason"] is None and r["predicted_delta"] > 0]
    # Every motivated live lever sits ahead of every unmotivated one.
    flags = [r["motivated_by"] is not None for r in live]
    assert flags == sorted(flags, reverse=True)
