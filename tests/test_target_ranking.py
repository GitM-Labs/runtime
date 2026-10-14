"""Levers aimed at the time a run measured above its floor are tried first.

Recoverable time used to be only a filter: a lever aimed at regions already at
their floor was dropped, and every survivor was ordered by the catalogue's
estimate. Now the survivors are ordered by it too, as a precedence: seconds
above floor against seconds above floor, ahead of the estimate.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from gitm.agents import policy as policy_mod
from gitm.agents import targeting
from gitm.agents.policy import Policy, select_interventions
from gitm.agents.targeting import targets_from_recoverable

from .test_policy_history import _spec, _trace

# _trace() holds 1000 ns of device time, so the default 1% threshold is 1e-8 s.


def _rank(specs, **kw):
    return select_interventions(_trace(), specs, Policy(), top_n=10, **kw)


def test_the_gate_and_the_ordering_share_one_join():
    """Three places made the lever↔op join and the policy did not use the shared
    one its docstring claimed. They agree by construction now."""
    assert policy_mod.ops_aimed_at is targeting.ops_aimed_at


def test_a_lever_aimed_at_more_lost_time_outranks_a_larger_estimate():
    big = _spec("big_estimate", ["gemm"], mean=0.08, kernel_time=True)
    small = _spec("small_estimate", ["moe_routed"], mean=0.02, kernel_time=True)
    ranked = _rank([big, small], recoverable={"gemm": 1e-7, "moe_routed": 5e-7})

    assert [c.spec.name for c in ranked] == ["small_estimate", "big_estimate"]
    assert ranked[0].targets == (("moe_routed", 5e-7),)
    assert ranked[0].targets_s == 5e-7


def test_without_a_recoverable_map_the_order_is_the_catalogues():
    big = _spec("big_estimate", ["gemm"], mean=0.08, kernel_time=True)
    small = _spec("small_estimate", ["moe_routed"], mean=0.02, kernel_time=True)
    ranked = _rank([big, small])
    assert ranked[0].spec.name == "big_estimate"
    assert all(c.targets == () for c in ranked)


def test_a_gap_below_the_threshold_buys_no_precedence():
    big = _spec("big_estimate", ["gemm"], mean=0.08, kernel_time=True)
    small = _spec("small_estimate", ["moe_routed"], mean=0.02, kernel_time=True)
    ranked = _rank([big, small], recoverable={"gemm": 1e-9, "moe_routed": 5e-9})
    assert ranked[0].spec.name == "big_estimate"
    assert all(c.targets_s is None for c in ranked)


def test_unjudgeable_and_absent_ops_add_nothing_and_reject_nothing():
    lever = _spec("multi", ["gemm", "moe_routed", "attn"], kernel_time=True)
    (c,) = _rank([lever], recoverable={"gemm": None, "moe_routed": 3e-7})
    assert c.rejected_reason is None
    assert c.targets == (("moe_routed", 3e-7),)


def test_a_lever_that_does_not_work_by_speeding_its_ops_is_not_targeted():
    """Attention running over its floor is no evidence for a lever that helps
    through cache capacity, so it gets no precedence from it."""
    cache = _spec("cache_lever", ["attn"], mean=0.02, kernel_time=False)
    other = _spec("other", ["gemm"], mean=0.05, kernel_time=False)
    ranked = _rank([cache, other], recoverable={"attn": 9e-7})
    assert ranked[0].spec.name == "other"
    assert all(c.targets == () for c in ranked)


def test_a_whole_step_lever_follows_targeted_ones_unless_a_cause_points_at_it():
    whole = _spec("whole", [], mean=0.20)
    whole = whole.model_copy(update={"whole_step": True})
    aimed = _spec("aimed", ["moe_routed"], mean=0.02, kernel_time=True)

    ranked = _rank([whole, aimed], recoverable={"moe_routed": 5e-7})
    assert [c.spec.name for c in ranked] == ["aimed", "whole"]

    ranked = _rank([whole, aimed], recoverable={"moe_routed": 5e-7},
                   motivated={"whole": "under_filled_batch"})
    assert ranked[0].spec.name == "whole"


def test_targets_json_names_the_time_nothing_is_aimed_at():
    lib = [_spec("fix_gemm", ["gemm"], kernel_time=True),
           _spec("cache", ["attn"], kernel_time=False)]
    doc = targets_from_recoverable(
        {"gemm": 2e-3, "attn": 1e-3, "rmsnorm": 4e-3, "moe": None, "rope": 0.0},
        lib, device_s=0.1)

    by_op = {r["op"]: r for r in doc["regions"]}
    assert [r["op"] for r in doc["regions"]] == ["rmsnorm", "gemm", "attn"]
    assert by_op["gemm"]["levers"] == ["fix_gemm"]
    assert by_op["attn"]["levers"] == [] and by_op["attn"]["also_named_by"] == ["cache"]
    assert doc["uncovered"] == ["rmsnorm"]          # time with nothing aimed at it
    assert doc["unjudgeable"] == ["moe"]
    assert "rope" not in by_op                      # at its floor
    assert abs(by_op["rmsnorm"]["share_of_device"] - 0.04) < 1e-12


def test_a_run_whose_floors_were_not_priced_says_so(tmp_path, monkeypatch):
    """The test loop runs on a default graph, so nothing is targeted and both
    files say why rather than leaving an empty list to read as 'no time lost'."""
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
    out = run_loop(LoopConfig(engine=_FullEngine(), workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=5,
                              rerank="recapture"))
    run_dir = Path(out["run_dir"])

    targets = json.loads((run_dir / "targets.json").read_text())
    assert targets["regions"] == [] and "not priced" in targets["not_targeted_because"]
    ranked = json.loads((run_dir / "ranked_candidates.json").read_text())
    assert all(r["targets"] == [] for r in ranked)
    steps = json.loads((run_dir / "rerank.json").read_text())["steps"]
    for step in (s for s in steps if s["recaptured"]):
        assert step["targeted"] is False
        assert "not priced" in step["not_targeted_because"]


def test_a_retrace_is_targeted_only_while_it_runs_at_the_priced_batch():
    """The floors were priced at the opening batch. Any kept lever can move it,
    not only a whole-step one: more KV cache admits more sequences. So the
    re-rank asks the batch the re-trace actually ran at."""
    from gitm.planner.roofline import BatchConfig
    from gitm.scheduler.loop import _floors_hold

    priced = BatchConfig(batch=251)
    assert _floors_hold(priced, BatchConfig(batch=245)) is None          # within 10%
    moved = _floors_hold(priced, BatchConfig(batch=512))
    assert moved is not None and "251 to 512" in moved
    assert "default batch" in _floors_hold(None, BatchConfig(batch=251))
    assert "no batch" in _floors_hold(priced, None)


def test_uncovered_lists_every_op_losing_time_not_only_the_top_ones():
    lib = [_spec("fix_gemm", ["gemm"], kernel_time=True)]
    doc = targets_from_recoverable({"gemm": 9e-3, "rmsnorm": 1e-3, "rope": 5e-4},
                                   lib, device_s=0.1, top=1)
    assert [r["op"] for r in doc["regions"]] == ["gemm"]
    assert doc["uncovered"] == ["rmsnorm", "rope"]      # past the cutoff, still listed


def test_a_retrace_is_not_targeted_once_the_engine_is_not_the_priced_one():
    """Pricing reads the model, the GPU and the batch, never engine settings,
    so a kept TP or KV-dtype change leaves the floors describing another engine
    even at the same batch. The engine's own settings are compared."""
    from gitm.planner.roofline import BatchConfig
    from gitm.scheduler.loop import _floors_hold

    b = BatchConfig(batch=251)
    opening = {"tensor_parallel_size": 8, "enforce_eager": True}
    assert _floors_hold(b, b, opening, dict(opening)) is None
    moved = _floors_hold(b, b, opening, {**opening, "tensor_parallel_size": 4})
    assert moved is not None and "tensor_parallel_size" in moved
    added = _floors_hold(b, b, opening, {**opening, "kv_cache_dtype": "fp8"})
    assert added is not None and "kv_cache_dtype" in added
