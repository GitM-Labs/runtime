"""Levers that would re-apply what the engine already runs are skipped, per re-rank."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.vllm_knobs import (
    baseline_noop,
    current_knob_values,
    noop_reason,
    set_knob,
)
from gitm.tracer.schema import KernelEvent, Trace

from .test_vllm_knobs_and_restart import _FullEngine


def _engine(**sched) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(**{"max_num_seqs": 256, "max_num_batched_tokens": 8192,
                                            **sched}),
        model_config=SimpleNamespace(enforce_eager=False),
    )


def _spec(name, knobs, *, mean=0.05) -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob="+".join(knobs), knobs=knobs,
        expected_delta_mean=mean, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applies_to_kernels=["k"],
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )


def _trace() -> Trace:
    return Trace(
        workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
        vendor="amd", captured_at_ns=0, duration_ns=1000,
        events=[KernelEvent(name="k", start_ns=0, end_ns=1000, stream_id=0, device_id=0,
                            correlation_id=1)],
    )


def _slots(ranked) -> list[str]:
    return [c.spec.name for c in ranked if c.baseline_noop is None]


def test_noop_only_when_every_knob_already_holds_its_target():
    e = _engine()
    assert "max_num_seqs=256" in baseline_noop(e, _spec("a", {"max_num_seqs": 256}))
    assert baseline_noop(e, _spec("b", {"max_num_seqs": 512})) is None
    both = {"max_num_seqs": 256, "max_num_batched_tokens": 8192}
    assert baseline_noop(e, _spec("c", both)) is not None
    half = {"max_num_seqs": 256, "max_num_batched_tokens": 16384}
    assert baseline_noop(e, _spec("d", half)) is None


def test_unreadable_knob_is_never_a_noop():
    e = _engine()
    # cache_config.block_size isn't on this engine at all.
    assert baseline_noop(e, _spec("a", {"block_size": 16})) is None
    assert baseline_noop(e, _spec("b", {"max_num_seqs": 256, "block_size": 16})) is None
    assert baseline_noop(None, _spec("c", {"max_num_seqs": 256})) is None
    assert "block_size" not in current_knob_values(e, ["block_size", "max_num_seqs"])


def test_bool_and_int_are_different_settings():
    assert noop_reason(_spec("a", {"enforce_eager": 0}), {"enforce_eager": False}) is None
    assert noop_reason(_spec("b", {"enforce_eager": False}), {"enforce_eager": False})


def test_noop_never_takes_a_slot_and_is_returned_after_them():
    e = _engine()
    lib = [_spec("already_on", {"enforce_eager": False}, mean=0.09),
           _spec("raise_seqs", {"max_num_seqs": 512}, mean=0.04),
           _spec("raise_tokens", {"max_num_batched_tokens": 16384}, mean=0.03)]
    knobs = {k for s in lib for k in s.knob_values}
    ranked = select_interventions(_trace(), lib, Policy(), top_n=2,
                                  current_values=current_knob_values(e, knobs))
    assert _slots(ranked) == ["raise_seqs", "raise_tokens"]
    assert [c.spec.name for c in ranked][-1] == "already_on"
    assert ranked[-1].predicted_delta == 0.0
    assert ranked[-1].rejected_reason is None


def test_without_current_values_ranking_is_unchanged():
    lib = [_spec("already_on", {"enforce_eager": False}, mean=0.09),
           _spec("raise_seqs", {"max_num_seqs": 512}, mean=0.04)]
    ranked = select_interventions(_trace(), lib, Policy(), top_n=5)
    assert [c.spec.name for c in ranked] == ["already_on", "raise_seqs"]
    assert all(c.baseline_noop is None for c in ranked)


def test_lever_that_becomes_a_noop_mid_run_is_dropped_on_rerank():
    engine = _engine(max_num_seqs=256)
    first_lever = _spec("raise_seqs", {"max_num_seqs": 512}, mean=0.08)
    same_value = _spec("seqs_512_from_sweep", {"max_num_seqs": 512}, mean=0.04)
    unrelated = _spec("raise_tokens", {"max_num_batched_tokens": 16384}, mean=0.03)
    library = [first_lever, same_value, unrelated]
    knobs = {k for s in library for k in s.knob_values}

    at_start = current_knob_values(engine, knobs)
    first = select_interventions(_trace(), library, Policy(), top_n=2, current_values=at_start)
    # Neither is a no-op against the baseline: 256 != 512.
    assert _slots(first) == ["raise_seqs", "seqs_512_from_sweep"]

    # The top lever is applied and kept; the engine now runs 512.
    set_knob(engine, "max_num_seqs", 512)
    remaining = [s for s in library if s.name != "raise_seqs"]

    rerank = select_interventions(_trace(), remaining, Policy(), top_n=2,
                                  current_values=current_knob_values(engine, knobs))
    assert _slots(rerank) == ["raise_tokens"]
    dropped = next(c for c in rerank if c.spec.name == "seqs_512_from_sweep")
    assert "max_num_seqs=512" in dropped.baseline_noop

    # Values computed once at the top of the run still say 256, so the same
    # re-rank would spend a slot re-applying 512 — the stale-skip bug.
    stale = select_interventions(_trace(), remaining, Policy(), top_n=2, current_values=at_start)
    assert "seqs_512_from_sweep" in _slots(stale)


class _EagerOffEngine(_FullEngine):
    """_FullEngine that also reports enforce_eager=False (CUDA graphs on)."""

    def __init__(self, max_num_seqs: int = 64):
        super().__init__(max_num_seqs)
        self.model_config.enforce_eager = False

    def _restart(self, _old_engine, knob_values):
        return _EagerOffEngine(max_num_seqs=int(knob_values.get("max_num_seqs", 64)))


def test_run_loop_records_baseline_noops_apart_from_rejections(tmp_path, monkeypatch):
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

    out = run_loop(LoopConfig(engine=_EagerOffEngine(), workload="vllm-decode", budget="24h",
                              scratch=str(tmp_path), top_n_interventions=50))

    noops = json.loads((Path(out["run_dir"]) / "baseline_noop.json").read_text())
    names = [n["name"] for n in noops]
    assert "cuda_graphs_enable" in names
    assert out["summary"]["n_baseline_noop"] == len(noops)
    # Skipped, not rejected, and never applied.
    assert "cuda_graphs_enable (" not in out["report_md"]
