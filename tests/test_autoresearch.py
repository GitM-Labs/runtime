"""Tests for the autoresearch agent — classify → propose → gate → apply/rollback."""

from __future__ import annotations

import gitm.agents.autoresearch as ar
from gitm.agents.autoresearch import (
    AutoresearchRun,
    autoresearch,
    autoresearch_v0,
    classify_bottleneck,
    propose,
)
from gitm.agents.policy import Policy
from gitm.kernels.library import load_library
from gitm.kernels.spec import InterventionSpec, SafetyGate
from gitm.optimizer.apply import DictApplicator
from gitm.tracer.schema import Trace

from .conftest import make_kernel, make_memcpy, make_trace


def test_module_imports() -> None:
    """Regression guard: the shipped stub imported a nonexistent module."""
    assert ar.propose is propose


# --- propose -----------------------------------------------------------------


def test_propose_known_class_returns_specs() -> None:
    specs = propose("idle_stall")
    assert specs, "known class should yield proposals"
    assert all(isinstance(s, InterventionSpec) for s in specs)
    # Never high-risk: unproven proposals stay at moderate and lean on the gate.
    assert all(s.safety.tier != "high_risk" for s in specs)


def test_propose_unknown_class_is_empty() -> None:
    assert propose("no_such_bottleneck") == []


def test_proposed_knobs_are_disjoint_from_catalog() -> None:
    """The whole point of autoresearch is proposing *outside* the library."""
    catalog_knobs = {spec.knob for spec in load_library()}
    assert catalog_knobs, "catalog should be non-empty"
    for cls in ("idle_stall", "memory_bound", "compute_bound"):
        for spec in propose(cls):
            assert spec.knob not in catalog_knobs, f"{spec.knob} duplicates the catalog"


# --- classify_bottleneck -----------------------------------------------------


def test_classify_idle_stall_from_serialized_kernels() -> None:
    # Back-to-back kernels on one stream, no overlap ⇒ serialized concurrency = 1.
    events = [
        make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90, stream_id=0)
        for i in range(6)
    ]
    assert classify_bottleneck(make_trace(events=events)) == "idle_stall"


def test_classify_memory_bound_from_memcpy_heavy_trace() -> None:
    # Overlapping kernels (no stall) but memcpys dominate the op mix.
    kernels = [make_kernel("k", start_ns=0, end_ns=1000, stream_id=0)]
    memcpys = [make_memcpy(start_ns=i * 10, end_ns=i * 10 + 5) for i in range(4)]
    assert classify_bottleneck(make_trace(events=kernels + memcpys)) == "memory_bound"


def test_classify_compute_bound_when_overlapped_and_no_memcpy() -> None:
    # Two kernels that temporally overlap ⇒ not serialized, no memcpy ⇒ compute.
    events = [
        make_kernel("a", start_ns=0, end_ns=100, stream_id=0),
        make_kernel("b", start_ns=50, end_ns=150, stream_id=1),
    ]
    assert classify_bottleneck(make_trace(events=events)) == "compute_bound"


def test_classify_empty_trace_defaults_to_compute() -> None:
    assert classify_bottleneck(make_trace(events=[])) == "compute_bound"


# --- autoresearch_v0: apply / rollback ---------------------------------------


def _trace() -> Trace:
    return make_trace(events=[make_kernel("paged_attention", start_ns=0, end_ns=500)])


def test_applies_and_keeps_on_measured_win() -> None:
    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: 0.10)  # +10% ⇒ keep

    results = autoresearch_v0(_trace(), "idle_stall", applicator=applicator)

    assert results, "idle_stall has proposals"
    kept = [r for r in results if r.applicable and not r.rolled_back]
    assert kept, "a positive delta must be kept"
    for r in kept:
        assert r.measured_delta == 0.10
        assert config.get(r.spec.knob) == r.spec.value  # the knob was actually set


def test_rolls_back_on_regression() -> None:
    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: -0.10)  # slower ⇒ revert

    results = autoresearch_v0(_trace(), "memory_bound", applicator=applicator)

    assert results
    assert all(r.applicable and r.rolled_back for r in results)
    assert config == {}, "a regressing proposal must leave the config untouched"


def test_gate_rejection_is_recorded_not_applied(monkeypatch) -> None:
    """A proposal the gate rejects is reported, never applied."""
    high_risk = InterventionSpec(
        name="autoresearch:test:danger",
        summary="hypothetical high-risk proposal",
        knob="some_risky_knob",
        value=1,
        expected_delta_mean=0.05,
        expected_delta_lo=0.0,
        expected_delta_hi=0.15,
        source="test",
        safety=SafetyGate(tier="high_risk"),
    )
    monkeypatch.setattr(ar, "propose", lambda _cls: [high_risk])

    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: 0.10)
    results = autoresearch_v0(
        _trace(), "idle_stall", applicator=applicator, policy=Policy(skip_high_risk=True)
    )

    assert len(results) == 1
    r = results[0]
    assert not r.applicable
    assert r.rejected_reason == "policy.skip_high_risk"
    assert config == {}, "a rejected proposal must never touch the config"


def test_unknown_class_yields_no_results() -> None:
    applicator = DictApplicator({}, measure_fn=lambda spec: 0.10)
    assert autoresearch_v0(_trace(), "no_such_class", applicator=applicator) == []


def test_autoresearch_forwards_audit_to_apply_gate(tmp_path) -> None:
    """A live proposal's apply must land on the safety trail when an audit is given."""
    from gitm.safety import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    applicator = DictApplicator({}, measure_fn=lambda spec: 0.10)
    results = autoresearch_v0(_trace(), "idle_stall", applicator=applicator, audit=log)

    applied = [r for r in results if r.applicable and not r.rolled_back]
    assert applied
    # Every applied proposal is recorded as an apply on the trail.
    apply_events = [e for e in log.entries() if e.event == "apply"]
    assert {e.intervention for e in apply_events} == {r.spec.name for r in applied}


# --- end-to-end entry point --------------------------------------------------


def test_autoresearch_end_to_end_classifies_and_runs() -> None:
    events = [
        make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90, stream_id=0)
        for i in range(6)
    ]
    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: 0.08)

    run = autoresearch(make_trace(events=events), applicator=applicator)

    assert isinstance(run, AutoresearchRun)
    assert run.bottleneck_class == "idle_stall"
    assert run.results
    assert all(r.bottleneck_class == "idle_stall" for r in run.results)
