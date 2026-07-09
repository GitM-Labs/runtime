"""Tests for the autoresearch agent — classify → propose → gate → apply/rollback."""

from __future__ import annotations

import gitm.agents.autoresearch as ar
from gitm.agents.autoresearch import (
    AutoresearchRun,
    autoresearch,
    autoresearch_v0,
    classify_bottleneck,
    largest_residual,
    propose,
)
from gitm.agents.policy import Policy
from gitm.kernels.library import load_library
from gitm.kernels.spec import InterventionSpec, SafetyGate
from gitm.optimizer.apply import DictApplicator
from gitm.optimizer.monitor import KernelResidual, Residuals
from gitm.tracer.schema import Trace

from .conftest import make_kernel, make_memcpy, make_sync, make_trace


def _residuals(pairs: list[tuple[str, float]]) -> Residuals:
    """Build a Residuals from (op, r_kt) pairs."""
    res = Residuals()
    res.per_kernel = [KernelResidual(op=op, layer=None, r_kt=r, r_mt=None) for op, r in pairs]
    return res


def test_module_imports() -> None:
    """Regression guard: the shipped stub imported a nonexistent module."""
    assert ar.propose is propose


def test_public_api_names_all_resolve() -> None:
    """Every name in __all__ is actually defined (no stale/typo'd exports)."""
    assert ar.__all__, "the module should declare a public surface"
    missing = [name for name in ar.__all__ if not hasattr(ar, name)]
    assert not missing, f"__all__ names undefined symbols: {missing}"


# --- propose -----------------------------------------------------------------


def test_propose_known_class_returns_specs() -> None:
    specs = propose("compute_bound")
    assert specs, "known class with safe standalone rules should yield proposals"
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
    # Overlapping kernels (no stall) but memcpys dominate GPU-op time.
    kernels = [
        make_kernel("a", start_ns=0, end_ns=100, stream_id=0),
        make_kernel("b", start_ns=0, end_ns=100, stream_id=1),
    ]
    memcpys = [make_memcpy(start_ns=i * 100, end_ns=i * 100 + 100) for i in range(4)]
    assert classify_bottleneck(make_trace(events=kernels + memcpys)) == "memory_bound"


def test_classify_compute_bound_when_transfers_do_not_dominate_gpu_time() -> None:
    # Counting events would overreact to tiny transfers; the classifier uses GPU-op time.
    kernels = [
        make_kernel("a", start_ns=0, end_ns=1000, stream_id=0),
        make_kernel("b", start_ns=0, end_ns=1000, stream_id=1),
    ]
    memcpys = [make_memcpy(start_ns=i * 10, end_ns=i * 10 + 5) for i in range(4)]
    syncs = [make_sync(start_ns=i * 20, end_ns=i * 20 + 1) for i in range(8)]
    assert classify_bottleneck(make_trace(events=kernels + memcpys + syncs)) == "compute_bound"


def test_classify_compute_bound_when_overlapped_and_no_memcpy() -> None:
    # Two kernels that temporally overlap ⇒ not serialized, no memcpy ⇒ compute.
    events = [
        make_kernel("a", start_ns=0, end_ns=100, stream_id=0),
        make_kernel("b", start_ns=50, end_ns=150, stream_id=1),
    ]
    assert classify_bottleneck(make_trace(events=events)) == "compute_bound"


def test_classify_empty_trace_defaults_to_compute() -> None:
    assert classify_bottleneck(make_trace(events=[])) == "compute_bound"


# --- classify_bottleneck: roofline-weighted memory signal ---------------------


def test_roofline_memory_fraction() -> None:
    from gitm.agents.autoresearch import _roofline_memory_fraction

    assert _roofline_memory_fraction(None) is None
    assert _roofline_memory_fraction(Residuals()) is None  # no per_kernel data
    res = Residuals(per_kernel=[
        KernelResidual(op="a", layer=None, r_kt=0.0, r_mt=None, t_obs_s=3.0, bound="memory"),
        KernelResidual(op="b", layer=None, r_kt=0.0, r_mt=None, t_obs_s=1.0, bound="compute"),
    ])
    assert _roofline_memory_fraction(res) == 0.75  # time-weighted, not a head count


def test_classify_catches_intrinsic_memory_boundedness_with_no_memcpy() -> None:
    """Roofline-flagged memory-boundedness flips a memcpy-blind compute_bound
    verdict (see test_classify_compute_bound_when_overlapped_and_no_memcpy)."""
    events = [
        make_kernel("a", start_ns=0, end_ns=100, stream_id=0),
        make_kernel("b", start_ns=50, end_ns=150, stream_id=1),
    ]
    trace = make_trace(events=events)
    assert classify_bottleneck(trace) == "compute_bound"  # unchanged, no residuals

    res = _residuals([("a", 0.0), ("b", 0.0)])
    for kr in res.per_kernel:
        kr.bound = "memory"
        kr.t_obs_s = 1.0
    assert classify_bottleneck(trace, res) == "memory_bound"


# --- autoresearch_v0: apply / rollback ---------------------------------------


def _trace() -> Trace:
    return make_trace(events=[make_kernel("paged_attention", start_ns=0, end_ns=500)])


def test_applies_and_keeps_on_measured_win() -> None:
    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: 0.10)  # +10% ⇒ keep

    # compute_bound: memory_bound's static candidates (cpu_offload_gb,
    # preemption_mode) graduated to the curated catalog, so the static table
    # is empty for it now (see _RULES) — same as idle_stall already was.
    results = autoresearch_v0(_trace(), "compute_bound", applicator=applicator)

    assert results, "compute_bound has safe standalone proposals"
    kept = [r for r in results if r.applicable and not r.rolled_back]
    assert kept, "a positive delta must be kept"
    for r in kept:
        assert r.measured_delta == 0.10
        assert config.get(r.spec.knob) == r.spec.value  # the knob was actually set


def test_rolls_back_on_regression() -> None:
    config: dict = {}
    applicator = DictApplicator(config, measure_fn=lambda spec: -0.10)  # slower ⇒ revert

    results = autoresearch_v0(_trace(), "compute_bound", applicator=applicator)

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
    monkeypatch.setattr(ar, "propose", lambda _cls, target_op=None: [high_risk])

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
    assert run.results == []



# --- repoint at the largest residual -----------------------------------------


def test_largest_residual_picks_biggest_gap_not_jitter() -> None:
    """The target is the op furthest over its ceiling (largest mean r_kt), not the
    jitteriest op (high variance, ~0 mean) or the smaller-gap op."""
    res = _residuals([
        ("A", 0.80), ("A", 0.60),   # mean +0.70 — the biggest gap
        ("B", 0.10), ("B", 0.20),   # mean +0.15 — smaller gap
        ("C", 1.50), ("C", -1.50),  # mean  0.00 — pure jitter, must NOT win
    ])
    target = largest_residual(res)
    assert target is not None
    assert target.op == "A"
    assert target.n_kernels == 2


def test_largest_residual_none_when_nothing_over_ceiling() -> None:
    # Everything runs at or under the predicted ceiling ⇒ no bottleneck to chase.
    assert largest_residual(_residuals([("A", -0.2), ("B", -0.1)])) is None
    assert largest_residual(Residuals()) is None


def test_autoresearch_targets_largest_residual_op() -> None:
    # Kernels named so the residual op "paged_attention" matches a real kernel.
    events = [
        make_kernel("paged_attention", start_ns=i * 100, end_ns=i * 100 + 90, stream_id=0)
        for i in range(6)
    ]
    res = _residuals([("paged_attention", 0.9), ("linear", 0.2)])
    applicator = DictApplicator({}, measure_fn=lambda spec: 0.05)

    run = autoresearch(make_trace(events=events), applicator=applicator, residuals=res)

    assert run.target is not None and run.target.op == "paged_attention"
    assert run.results == []

    # The idle generated partial-prefill knobs are filtered as unsupported/noisy.



def test_autoresearch_without_residuals_has_no_target() -> None:
    events = [make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90) for i in range(4)]
    run = autoresearch(make_trace(events=events), applicator=DictApplicator({}))
    assert run.target is None
    assert all(r.target_op is None for r in run.results)
    # Untargeted proposals stay unscoped (coverage 1.0 in predict_delta).
    assert all(r.spec.applies_to_kernels == [] for r in run.results)


def test_off_trace_residual_op_is_not_tagged() -> None:
    """If the largest-residual op doesn't match any kernel name, don't scope to it
    (that would zero predict_delta coverage) — but still record it as the target."""
    events = [make_kernel("gemm", start_ns=i * 100, end_ns=i * 100 + 90) for i in range(4)]
    res = _residuals([("paged_attention", 0.9)])  # op not present in the trace
    run = autoresearch(make_trace(events=events), applicator=DictApplicator({}), residuals=res)

    assert run.target is not None and run.target.op == "paged_attention"
    assert all(r.target_op == "paged_attention" for r in run.results)  # recorded
    assert all(r.spec.applies_to_kernels == [] for r in run.results)  # but not scoped


# --- EngineArgs-driven generative proposer -----------------------------------

from gitm.agents.autoresearch import (  # noqa: E402
    _CLASS_KEYWORDS,
    _RULES,
    BOTTLENECK_CLASSES,
    EngineArgsProposer,
    FallbackProposer,
    GenerativeProposer,
    Knob,
    StochasticProposer,
    TableProposer,
    VLLMKnobSource,
    _affinity_strength,
    _argparse_domains,
    _candidate_spec,
    _delta_mean_for,
    _field_kind_and_choices,
    _is_tunable,
    _joint_prerequisite_candidates,
    _knobs_from_engine_args,
    _requires_multi_gpu,
    _value_grid,
    _visible_gpu_count,
)


class _ListSource:
    """A minimal KnobSource for tests: yields a fixed knob list."""

    def __init__(self, knobs: list[Knob]) -> None:
        self._knobs = knobs

    def knobs(self) -> list[Knob]:
        return list(self._knobs)


def test_value_grid_flips_bool() -> None:
    assert _value_grid(Knob("enforce_eager", "bool", default=False)) == [True]


def test_value_grid_enum_returns_other_members() -> None:
    grid = _value_grid(
        Knob("preemption_mode", "enum", default="recompute", choices=("recompute", "swap"))
    )
    assert grid == ["swap"]


def test_value_grid_int_searches_multiple_distinct_positive_values() -> None:
    grid = _value_grid(Knob("max_num_partial_prefills", "int", default=1))
    assert len(grid) >= 2, "a value grid must actually search several values"
    assert len(set(grid)) == len(grid)  # distinct
    assert all(isinstance(v, int) and v >= 1 and v != 1 for v in grid)


def test_value_grid_explicit_grid_wins() -> None:
    grid = _value_grid(
        Knob("long_prefill_token_threshold", "int", default=0, grid=(2048, 4096))
    )
    assert grid == [2048, 4096]


def _stub_knobs() -> list[Knob]:
    return [
        Knob("max_num_partial_prefills", "int", default=1),  # idle_stall-affine
        Knob("cpu_offload_gb", "int", default=4),  # memory_bound-affine
        Knob("block_size", "int", default=16),  # memory-affine but IN the catalog
    ]


def test_engineargs_proposer_searches_multiple_values_per_knob() -> None:
    p = EngineArgsProposer(
        knobs=[Knob("max_num_partial_prefills", "int", default=1)], catalog_knobs=set()
    )
    specs = p.propose("idle_stall")
    assert len(specs) >= 2, "value-grid search should try several values"
    assert len({s.value for s in specs}) == len(specs)  # one spec per distinct value
    assert all(s.knob == "max_num_partial_prefills" for s in specs)
    assert all(isinstance(s, InterventionSpec) for s in specs)
    # Generated candidates are never high-risk; they lean on the rollback gate.
    assert all(s.safety.tier == "moderate" for s in specs)
    # The value-grid naming (knob=value) distinguishes them from table proposals.
    assert all("=" in s.name for s in specs)


def test_engineargs_proposer_excludes_catalog_knobs() -> None:
    """block_size is a real EngineArgs knob, but it's in the curated library —
    autoresearch proposes *outside* the catalog, so it must be dropped."""
    p = EngineArgsProposer(knobs=_stub_knobs(), catalog_knobs={"block_size"})
    specs = p.propose("memory_bound")
    assert specs, "cpu_offload_gb is affine to memory_bound"
    assert all(s.knob != "block_size" for s in specs)
    assert {s.knob for s in specs} == {"cpu_offload_gb"}


def test_engineargs_proposer_scopes_candidates_to_bottleneck_class() -> None:
    p = EngineArgsProposer(knobs=_stub_knobs(), catalog_knobs=set())
    idle = {s.knob for s in p.propose("idle_stall")}
    mem = {s.knob for s in p.propose("memory_bound")}
    assert idle == {"max_num_partial_prefills"}
    assert mem == {"cpu_offload_gb", "block_size"}  # both memory-affine by name


def test_engineargs_proposer_unknown_class_is_empty() -> None:
    assert EngineArgsProposer(knobs=_stub_knobs()).propose("no_such_class") == []


def test_engineargs_proposer_scopes_specs_to_target_op() -> None:
    p = EngineArgsProposer(
        knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()
    )
    specs = p.propose("memory_bound", target_op="paged_attention")
    assert specs
    assert all(s.applies_to_kernels == ["paged_attention"] for s in specs)


def test_engineargs_offline_fallback_runs_without_vllm() -> None:
    """vLLM isn't importable in CI; the frozen fallback catalog still yields
    candidates for compute_bound, and every candidate stays outside the
    curated library. memory_bound's fallback knobs (cpu_offload_gb,
    preemption_mode) graduated to the curated catalog, so the offline path
    has nothing left for that class — same gap idle_stall already had."""
    catalog = {s.knob for s in load_library()}
    p = EngineArgsProposer()  # default knobs → _engine_arg_knobs() → frozen fallback
    assert p.propose("memory_bound") == []
    specs = p.propose("compute_bound")
    assert specs, "compute_bound should yield fallback candidates offline"
    assert all(s.safety.tier == "moderate" for s in specs)
    assert all(s.knob not in catalog for s in specs)
    assert p.propose("idle_stall") == []  # disjoint from the library


def test_engineargs_candidates_route_through_gate_and_rollback() -> None:
    """A generated candidate is kept on a measured win and reverted on a regression —
    the same gate the catalog goes through, nothing special for generated specs."""
    proposer = EngineArgsProposer(
        knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()
    )
    grid = _value_grid(Knob("cpu_offload_gb", "int", default=4))

    kept_cfg: dict = {}
    kept = autoresearch_v0(
        _trace(),
        "memory_bound",
        applicator=DictApplicator(kept_cfg, measure_fn=lambda s: 0.10),
        proposer=proposer,
    )
    assert kept and all(r.applicable and not r.rolled_back for r in kept)
    assert kept_cfg.get("cpu_offload_gb") in grid  # the generated knob was set

    revert_cfg: dict = {}
    reverted = autoresearch_v0(
        _trace(),
        "memory_bound",
        applicator=DictApplicator(revert_cfg, measure_fn=lambda s: -0.10),
        proposer=proposer,
    )
    assert reverted and all(r.rolled_back for r in reverted)
    assert revert_cfg == {}, "a regressing candidate must leave the config untouched"


class _FailingApplicator:
    """An applicator whose apply() always raises — simulates a live engine
    build/restart failure (e.g. an unsatisfiable structural knob)."""

    def snapshot(self) -> dict:
        return {}

    def apply(self, spec) -> None:
        raise RuntimeError("engine build failed: two-engine distributed clash")

    def restore(self, snapshot) -> None:
        return None

    def measure(self, spec) -> float | None:
        return None


def test_apply_failure_surfaces_the_error_not_just_a_bare_none() -> None:
    """A candidate whose live apply raises is rolled back with measured_delta=None
    — the same shape as 'measured and lost'. apply_error must distinguish the two
    so a report can say *why* it failed instead of a bare unexplained '-'."""
    proposer = EngineArgsProposer(
        knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()
    )
    results = autoresearch_v0(
        _trace(), "memory_bound", applicator=_FailingApplicator(), proposer=proposer
    )
    assert results and all(r.rolled_back and r.measured_delta is None for r in results)
    assert all(r.apply_error and "distributed clash" in r.apply_error for r in results)


def test_unmet_prerequisite_vetoes_partial_prefill_without_chunked_prefill() -> None:
    """max_num_partial_prefills is no longer denylisted (see _is_tunable), but
    unmet_prerequisite as the reject hook still stops it on an engine where
    enable_chunked_prefill is off — without forbidding the knob everywhere."""
    from gitm.optimizer.vllm_knobs import unmet_prerequisite

    class _Sched:
        def __init__(self, enabled: bool):
            self.chunked_prefill_enabled = enabled

    class _Engine:
        def __init__(self, enabled: bool):
            self.scheduler_config = _Sched(enabled)

    proposer = EngineArgsProposer(
        knobs=[Knob("max_num_partial_prefills", "int", default=1)], catalog_knobs=set()
    )

    off = _Engine(enabled=False)
    rejected = autoresearch_v0(
        _trace(), "idle_stall",
        applicator=DictApplicator({}, measure_fn=lambda s: 0.10),
        proposer=proposer,
        reject=lambda spec: unmet_prerequisite(off, spec.knob),
    )
    assert rejected and all(not r.applicable for r in rejected)
    assert all("enable_chunked_prefill" in (r.rejected_reason or "") for r in rejected)

    on = _Engine(enabled=True)
    kept = autoresearch_v0(
        _trace(), "idle_stall",
        applicator=DictApplicator({}, measure_fn=lambda s: 0.10),
        proposer=proposer,
        reject=lambda spec: unmet_prerequisite(on, spec.knob),
    )
    assert kept and all(r.applicable for r in kept)


def test_apply_error_is_none_when_not_applicable_or_measured() -> None:
    # A candidate vetoed by the caller's reject hook never reaches apply — no
    # apply_error to report (the same shape as a plain gate rejection).
    rejected = autoresearch_v0(
        _trace(), "idle_stall",
        applicator=DictApplicator({}, measure_fn=lambda s: 0.10),
        proposer=EngineArgsProposer(
            knobs=[Knob("max_num_partial_prefills", "int", default=1)], catalog_knobs=set()
        ),
        reject=lambda spec: "vetoed",
    )
    assert rejected and all(not r.applicable and r.apply_error is None for r in rejected)
    # A successfully measured-and-kept candidate also carries no apply_error.
    kept = autoresearch_v0(
        _trace(), "memory_bound",
        applicator=DictApplicator({}, measure_fn=lambda s: 0.10),
        proposer=EngineArgsProposer(knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()),
    )
    assert kept and all(r.apply_error is None for r in kept)


def test_fallback_proposer_uses_table_only_when_primary_is_empty() -> None:
    table = TableProposer()
    # Primary has no compute_bound knob here → falls back to the table's compute lever.
    fb = FallbackProposer(
        EngineArgsProposer(knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()),
        table,
    )
    # memory_bound: primary yields cpu_offload_gb candidates → table NOT consulted.
    mem = fb.propose("memory_bound")
    assert {s.knob for s in mem} == {"cpu_offload_gb"}
    assert all("=" in s.name for s in mem)  # generated, not table
    # compute_bound: primary empty → table lever surfaces.
    compute = fb.propose("compute_bound")
    assert compute == table.propose("compute_bound")
    assert compute, "the table has a compute_bound lever"


def test_autoresearch_end_to_end_with_engineargs_proposer() -> None:
    events = [
        make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90, stream_id=0) for i in range(6)
    ]  # serialized → idle_stall
    proposer = EngineArgsProposer(
        knobs=[Knob("max_num_partial_prefills", "int", default=1)], catalog_knobs=set()
    )
    run = autoresearch(
        make_trace(events=events),
        applicator=DictApplicator({}, measure_fn=lambda s: 0.05),
        proposer=proposer,
    )
    assert run.bottleneck_class == "idle_stall"
    assert run.results and all(r.spec.knob == "max_num_partial_prefills" for r in run.results)
    assert all("=" in r.spec.name for r in run.results)  # generated value-grid candidates


def test_table_proposer_matches_module_propose() -> None:
    assert [s.knob for s in TableProposer().propose("idle_stall")] == [
        s.knob for s in propose("idle_stall")
    ]


# --- workload-agnostic generative proposer (KnobSource seam) ------------------


def test_generative_proposer_serves_a_non_vllm_workload() -> None:
    """A non-vLLM workload plugs in via a KnobSource + workload label — no table.
    Knobs declare their class affinity explicitly, so the vLLM keyword table is
    irrelevant here."""
    source = _ListSource([Knob("cache_prefetch_depth", "int", default=2, classes=("memory_bound",))])
    p = GenerativeProposer(source, workload="triton-serve", catalog_knobs=set())

    specs = p.propose("memory_bound")
    assert specs and all(s.knob == "cache_prefetch_depth" for s in specs)
    # The candidate is labelled for the caller's workload, not hardcoded vLLM.
    assert all(s.applicability.workloads == ["triton-serve"] for s in specs)
    assert p.propose("idle_stall") == []  # not tagged for idle_stall


def test_knob_explicit_classes_override_keyword_affinity() -> None:
    # A knob whose NAME wouldn't keyword-match idle_stall, but is tagged for it.
    source = _ListSource([Knob("weird_lever", "int", default=1, classes=("idle_stall",))])
    p = GenerativeProposer(source, catalog_knobs=set())
    assert {s.knob for s in p.propose("idle_stall")} == {"weird_lever"}
    assert p.propose("memory_bound") == []


def test_engineargs_proposer_is_a_vllm_bound_generative_proposer() -> None:
    """EngineArgsProposer is just GenerativeProposer bound to the vLLM surface."""
    p = EngineArgsProposer(knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set())
    assert isinstance(p, GenerativeProposer)
    specs = p.propose("memory_bound")
    assert specs and all(s.applicability.workloads == ["vllm-decode"] for s in specs)


def test_vllm_knob_source_yields_offline_fallback_without_vllm() -> None:
    # vLLM isn't importable in CI → the source yields the frozen fallback catalog.
    knobs = VLLMKnobSource().knobs()
    assert knobs and all(isinstance(k, Knob) for k in knobs)
    names = {k.name for k in knobs}
    assert "cpu_offload_gb" in names
    assert "max_num_partial_prefills" not in names


def test_is_tunable_excludes_non_perf_engine_args() -> None:
    # Identity / IO / logging / RNG fields are not performance knobs.
    for name in ("model", "served_model_name", "tokenizer", "seed",
                 "disable_log_stats", "download_dir", "revision"):
        assert not _is_tunable(name), f"{name} should be excluded"
    # Real standalone optimization knobs pass.
    for name in ("max_num_seqs", "gpu_memory_utilization", "compilation_config",
                 "cpu_offload_gb"):
        assert _is_tunable(name), f"{name} should be kept"
    # kv_sharing_fast_prefill is WIP/no-op per vLLM docs -> denylisted outright.
    assert not _is_tunable("kv_sharing_fast_prefill")
    # Prerequisite-gated (enable_chunked_prefill/enable_dbo), not denylisted —
    # vetoed live via unmet_prerequisite instead (see test_vllm_knobs).
    for name in ("max_num_partial_prefills", "max_long_partial_prefills",
                 "long_prefill_token_threshold", "dbo_prefill_token_threshold"):
        assert _is_tunable(name), f"{name} should be kept (prerequisite-gated, not denylisted)"


def test_field_kind_and_choices_extracts_literal_enum() -> None:
    import typing

    kind, choices = _field_kind_and_choices(typing.Literal["recompute", "swap"])
    assert kind == "enum"
    assert choices == ("recompute", "swap")
    # Plain scalars fall back to kind-only (no choices) and stay robust.
    assert _field_kind_and_choices(bool) == ("bool", ())
    assert _field_kind_and_choices(int) == ("int", ())


# --- valid domains sourced from EngineArgs' CLI args -------------------------

import argparse  # noqa: E402
import dataclasses  # noqa: E402


@dataclasses.dataclass
class _FakeEngineArgs:
    """An EngineArgs-like dataclass with a CLI builder — exercises the argparse
    domain extraction without importing vLLM (absent in CI)."""

    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 256
    gpu_frac: float = 0.9
    enforce_eager: bool = False
    served_model_name: str = "m"  # non-perf (model/name) → excluded
    middleware: tuple = ()  # list-valued (nargs=+) → skipped
    tensor_parallel_size: int = 1  # multi-GPU topology → skipped on a 1-GPU box

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--kv-cache-dtype", dest="kv_cache_dtype",
            choices=["auto", "fp8", "fp8_e5m2"], default="auto",
        )
        parser.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=256)
        parser.add_argument("--gpu-frac", dest="gpu_frac", type=float, default=0.9)
        parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
        parser.add_argument("--served-model-name", dest="served_model_name", type=str, default="m")
        parser.add_argument("--middleware", dest="middleware", nargs="+", type=str, default=[])
        parser.add_argument("--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
        return parser


def test_knobs_from_engine_args_sources_valid_enum_domain() -> None:
    """The enum domain comes from argparse ``choices`` — so the value grid only
    contains values that can actually apply, never an invented ladder point."""
    knobs = {k.name: k for k in _knobs_from_engine_args(_FakeEngineArgs, gpu_count=1)}
    kv = knobs["kv_cache_dtype"]
    assert kv.kind == "enum"
    assert set(kv.choices) == {"auto", "fp8", "fp8_e5m2"}
    assert set(_value_grid(kv)) == {"fp8", "fp8_e5m2"}  # valid non-default choices only


def test_knobs_from_engine_args_types_from_argparse() -> None:
    knobs = {k.name: k for k in _knobs_from_engine_args(_FakeEngineArgs, gpu_count=1)}
    assert knobs["max_num_seqs"].kind == "int"
    assert knobs["gpu_frac"].kind == "float"
    assert knobs["enforce_eager"].kind == "bool"


def test_knobs_from_engine_args_skips_nonperf_and_list_args() -> None:
    names = {k.name for k in _knobs_from_engine_args(_FakeEngineArgs, gpu_count=1)}
    assert "served_model_name" not in names  # non-performance field
    assert "middleware" not in names  # list-valued arg (nargs=+): not a scalar knob


def test_argparse_domains_reads_choices_type_and_nargs() -> None:
    d = _argparse_domains(_FakeEngineArgs)
    assert d["kv_cache_dtype"].choices == ("auto", "fp8", "fp8_e5m2")
    assert d["max_num_seqs"].type is int
    assert d["middleware"].is_list is True
    assert d["kv_cache_dtype"].is_list is False


def test_argparse_domains_empty_when_no_cli_builder() -> None:
    class _Bare:
        pass

    assert _argparse_domains(_Bare) == {}


# --- hardware-applicability: skip multi-GPU knobs on a single-GPU box --------


def test_requires_multi_gpu_flags_known_topology_knobs() -> None:
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "data_parallel_size", "prefill_context_parallel_size",
                 "decode_context_parallel_size", "expert_parallel_size"):
        assert _requires_multi_gpu(name), f"{name} should be flagged as multi-GPU-only"
    # Ordinary knobs (including ones with "size" in the name) aren't caught up in it.
    for name in ("max_num_seqs", "cpu_offload_gb", "block_size", "max_num_batched_tokens"):
        assert not _requires_multi_gpu(name), f"{name} should NOT be flagged"


def test_knobs_from_engine_args_skips_multi_gpu_knob_on_single_gpu() -> None:
    names = {k.name for k in _knobs_from_engine_args(_FakeEngineArgs, gpu_count=1)}
    assert "tensor_parallel_size" not in names, (
        "a 1-GPU box can't satisfy a multi-GPU topology knob — proposing it "
        "would only waste a restart-A/B on a build that can't succeed"
    )


def test_knobs_from_engine_args_includes_multi_gpu_knob_with_multiple_gpus() -> None:
    names = {k.name for k in _knobs_from_engine_args(_FakeEngineArgs, gpu_count=2)}
    assert "tensor_parallel_size" in names


def test_visible_gpu_count_is_a_positive_int() -> None:
    # Environment-dependent (torch/CUDA may or may not be present) — the only
    # invariant that must hold everywhere is "a usable, positive count".
    n = _visible_gpu_count()
    assert isinstance(n, int) and n >= 1


def test_vllm_knob_source_gpu_count_override_is_accepted_offline() -> None:
    # vLLM isn't importable in CI, so the offline fallback catalog is returned
    # regardless of gpu_count — this just proves the parameter doesn't crash
    # the offline path (the fallback catalog has no multi-GPU knobs to filter).
    assert VLLMKnobSource(gpu_count=1).knobs() == VLLMKnobSource(gpu_count=4).knobs()


def test_engineargs_proposer_accepts_gpu_count_offline() -> None:
    # Same offline-safety guarantee at the EngineArgsProposer entry point.
    p = EngineArgsProposer(gpu_count=1)
    assert p.propose("compute_bound")


def test_candidate_specs_share_the_forced_fields() -> None:
    """DRY guard: safety tier, delta bounds, applicability match. Excludes
    expected_delta_mean, which now legitimately varies by affinity strength
    (see test_generated_delta_mean_scales_with_affinity_strength)."""
    table = propose("compute_bound")[0]
    generated = EngineArgsProposer(
        knobs=[Knob("cpu_offload_gb", "int", default=4)], catalog_knobs=set()
    ).propose("memory_bound")[0]
    for s in (table, generated):
        assert s.safety.tier == "moderate"
        assert (s.expected_delta_lo, s.expected_delta_hi) == (0.0, 0.15)
        assert s.applicability.workloads == ["vllm-decode"]


def test_bottleneck_vocabulary_is_single_sourced() -> None:
    """classify emits only known classes; the fallback table and keyword-affinity
    map key off the same authoritative vocabulary, so the three can't drift."""
    assert set(_RULES) == set(BOTTLENECK_CLASSES)
    assert set(_CLASS_KEYWORDS) == set(BOTTLENECK_CLASSES)


def test_classify_always_returns_a_known_class() -> None:
    traces = [
        make_trace(events=[]),  # empty → compute default
        make_trace(events=[
            make_kernel("k", start_ns=i * 100, end_ns=i * 100 + 90) for i in range(6)
        ]),  # serialized → idle
        make_trace(events=[make_kernel("k", start_ns=0, end_ns=1000)]
                   + [make_memcpy(start_ns=i * 10, end_ns=i * 10 + 5) for i in range(4)]),
    ]
    assert all(classify_bottleneck(t) in BOTTLENECK_CLASSES for t in traces)


def test_generative_proposer_accepts_custom_affinity_keywords() -> None:
    """A workload whose knobs use their own naming supplies its own affinity
    keywords — no reliance on the vLLM-flavoured defaults, and still no table."""
    source = _ListSource([Knob("shard_rebalance_interval", "int", default=4)])
    p = GenerativeProposer(
        source,
        workload="mesh-serve",
        catalog_knobs=set(),
        affinity_keywords={"memory_bound": ("shard", "rebalance")},
    )
    assert {s.knob for s in p.propose("memory_bound")} == {"shard_rebalance_interval"}
    # The default vLLM keyword vocabulary wouldn't have matched this name.
    assert GenerativeProposer(source, catalog_knobs=set()).propose("memory_bound") == []


def test_generative_proposer_is_uncapped_by_default() -> None:
    knobs = [Knob(f"prefill_{i}", "bool", default=False) for i in range(30)]
    p = GenerativeProposer(_ListSource(knobs), catalog_knobs=set())
    assert len(p.propose("idle_stall")) == 30  # all idle-affine, one value each


def test_generative_proposer_caps_candidate_count() -> None:
    knobs = [Knob(f"prefill_{i}", "bool", default=False) for i in range(50)]
    p = GenerativeProposer(_ListSource(knobs), catalog_knobs=set(), max_candidates=10)
    assert len(p.propose("idle_stall")) == 10  # bounded search over a large surface


def test_engineargs_proposer_bounds_a_large_surface_by_default() -> None:
    # The vLLM binding caps by default so the real ~100-field surface can't flood.
    knobs = [Knob(f"prefill_{i}", "bool", default=False) for i in range(100)]
    specs = EngineArgsProposer(knobs=knobs, catalog_knobs=set()).propose("idle_stall")
    assert 0 < len(specs) <= 24


def test_candidate_spec_helper_forces_safety_and_delta() -> None:
    spec = _candidate_spec(
        name="autoresearch:test:x=1",
        summary="s",
        knob="x",
        value=1,
        applies_to_kernels=[],
        bottleneck_class="idle_stall",
        workload="some-workload",
        source="test",
    )
    assert spec.safety.tier == "moderate"
    assert spec.expected_delta_hi == 0.15
    assert spec.applicability.workloads == ["some-workload"]


# --- ranking: delta_mean scales with a real, computable signal ---------------


def test_affinity_strength_and_delta_mean_for() -> None:
    keywords = ("cache", "swap", "offload", "cpu")
    assert _affinity_strength(Knob("cpu_offload_gb", "int", default=4), keywords) == 2
    assert _affinity_strength(Knob("swap_space", "int", default=0), keywords) == 1
    assert _affinity_strength(Knob("unrelated_thing", "int", default=0), keywords) == 0
    # An explicit tag is authored ground truth: it scores as matching every
    # keyword (the max), so it can never be outranked by a coincidental
    # multi-keyword name match.
    assert _affinity_strength(Knob("x", "int", default=0, classes=("idle_stall",)), keywords) == len(keywords)
    assert _affinity_strength(Knob("x", "int", default=0, classes=("idle_stall",)), ()) == 1  # no keywords -> floor at 1

    assert _delta_mean_for(0) == _delta_mean_for(1)  # zero matches floors at 1
    assert abs(_delta_mean_for(2) - 2 * _delta_mean_for(1)) < 1e-9
    assert _delta_mean_for(100) == 0.15  # capped, never exceeds expected_delta_hi


def test_generated_delta_mean_scales_with_affinity_strength() -> None:
    """A 2-keyword match must rank above a 1-match (previously identical)."""
    p = EngineArgsProposer(
        knobs=[
            Knob("cpu_offload_gb", "int", default=4),  # matches "cpu" + "offload"
            Knob("preemption_mode", "enum", default="recompute",
                 choices=("recompute", "swap")),  # matches "preempt" only
        ],
        catalog_knobs=set(),
    )
    specs = {s.knob: s for s in p.propose("memory_bound")}
    assert specs["cpu_offload_gb"].expected_delta_mean > specs["preemption_mode"].expected_delta_mean


# --- joint candidates: prerequisite + dependent knob together ----------------


def test_joint_prerequisite_candidates() -> None:
    dbo_knobs = [
        Knob("enable_dbo", "bool", default=False),
        Knob("dbo_prefill_token_threshold", "int", default=512),
    ]
    specs = _joint_prerequisite_candidates(
        dbo_knobs, bottleneck_class="idle_stall", workload="vllm-decode", target_op=None,
        keywords=_CLASS_KEYWORDS["idle_stall"],
    )
    assert specs  # at least one value-grid point for the dependent knob
    for s in specs:
        assert set(s.knobs) == {"enable_dbo", "dbo_prefill_token_threshold"}
        assert s.knobs["enable_dbo"] is True
        assert s.value is None  # display-only label lives in .knob

    # No enable_dbo in the surface (e.g. offline fallback catalog) -> nothing.
    assert _joint_prerequisite_candidates(
        [Knob("dbo_prefill_token_threshold", "int", default=512)],
        bottleneck_class="idle_stall", workload="vllm-decode", target_op=None,
        keywords=_CLASS_KEYWORDS["idle_stall"],
    ) == []

    # compute_bound's keywords don't match "prefill" -> no joint candidate.
    assert _joint_prerequisite_candidates(
        dbo_knobs, bottleneck_class="compute_bound", workload="vllm-decode", target_op=None,
        keywords=_CLASS_KEYWORDS["compute_bound"],
    ) == []

    # End-to-end through the proposer a real EngineArgs surface would yield.
    joint = [s for s in EngineArgsProposer(knobs=dbo_knobs, catalog_knobs=set()).propose("idle_stall")
             if len(s.knobs) > 1]
    assert joint and all(set(s.knobs) == {"enable_dbo", "dbo_prefill_token_threshold"} for s in joint)


def test_joint_candidates_survive_max_candidates_even_when_grid_alone_fills_it() -> None:
    """Joint candidates go first, so a large single-knob value grid can't
    silently truncate them off the end of the list."""
    dbo_knobs = [
        Knob("enable_dbo", "bool", default=False),
        Knob("dbo_prefill_token_threshold", "int", default=512),
    ]
    # A single-knob grid big enough to fill the whole cap on its own.
    filler = [Knob(f"prefill_filler_{i}", "bool", default=False) for i in range(50)]
    p = EngineArgsProposer(knobs=dbo_knobs + filler, catalog_knobs=set(), max_candidates=10)
    specs = p.propose("idle_stall")
    assert len(specs) == 10
    assert any(len(s.knobs) > 1 for s in specs), "joint candidates were crowded out by the cap"


def test_intervention_spec_knob_values_property() -> None:
    from gitm.kernels.spec import InterventionSpec

    def _spec(**kw):
        return InterventionSpec(name="n", summary="s", expected_delta_mean=0.05,
                                 expected_delta_lo=0.0, expected_delta_hi=0.1, source="test", **kw)

    assert _spec(knob="max_num_seqs", value=256).knob_values == {"max_num_seqs": 256}
    assert _spec(knob="a=1,b=2", knobs={"a": 1, "b": 2}).knob_values == {"a": 1, "b": 2}


# --- stochastic (entropy-guided) proposer ------------------------------------


def _mixed_source() -> _ListSource:
    # One idle-affine knob (name contains "prefill") and one off-class knob.
    return _ListSource(
        [Knob("prefill_slots", "int", default=1), Knob("zzz_widget", "int", default=1)]
    )


def test_stochastic_is_reproducible_for_a_given_seed() -> None:
    src = _mixed_source()
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=6, seed=42, epsilon=0.4)
    first = [s.name for s in p.propose("idle_stall")]
    second = [s.name for s in p.propose("idle_stall")]
    assert first and first == second  # same seed → identical draws


def test_stochastic_seed_actually_varies_exploration() -> None:
    src = _mixed_source()
    variants = {
        tuple(s.name for s in StochasticProposer(
            src, catalog_knobs=set(), n_samples=6, seed=k, epsilon=0.5
        ).propose("idle_stall"))
        for k in range(6)
    }
    assert len(variants) > 1  # the seed is a real exploration control


def test_stochastic_epsilon_zero_is_pure_heuristic() -> None:
    # No entropy floor ⇒ only class-affine knobs are ever sampled, any seed.
    src = _mixed_source()
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=20, seed=7, epsilon=0.0)
    specs = p.propose("idle_stall")
    assert specs and all(s.knob == "prefill_slots" for s in specs)


def test_stochastic_epsilon_lets_the_search_wander_off_class() -> None:
    # A nonzero floor ⇒ the off-class knob is reachable (the entropy source).
    src = _mixed_source()
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=20, seed=0, epsilon=1.0)
    knobs = {s.knob for s in p.propose("idle_stall")}
    assert "prefill_slots" in knobs and "zzz_widget" in knobs


def test_stochastic_empty_when_no_signal_and_no_entropy() -> None:
    # Off-class knob only, epsilon=0 ⇒ no heuristic signal and no floor ⇒ nothing.
    src = _ListSource([Knob("zzz_widget", "int", default=1)])
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=8, seed=0, epsilon=0.0)
    assert p.propose("idle_stall") == []


def test_stochastic_retries_duplicates_but_returns_unique_candidates() -> None:
    src = _ListSource([Knob("prefill_enabled", "bool", default=False)])
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=6, seed=0, epsilon=0.0)

    specs = p.propose("idle_stall")

    assert len(specs) == 1
    assert specs[0].knob == "prefill_enabled"
    assert specs[0].value is True


def test_stochastic_skips_knobs_without_value_grid() -> None:
    src = _ListSource([
        Knob("prefill_freeform", "str", default="auto"),
        Knob("prefill_enabled", "bool", default=False),
    ])
    p = StochasticProposer(src, catalog_knobs=set(), n_samples=3, seed=0, epsilon=0.0)

    specs = p.propose("idle_stall")

    assert specs
    assert {s.knob for s in specs} == {"prefill_enabled"}


def test_stochastic_candidates_route_through_gate_and_rollback() -> None:
    proposer = StochasticProposer(
        _mixed_source(), catalog_knobs=set(), n_samples=6, seed=1, epsilon=0.3
    )
    cfg: dict = {}
    results = autoresearch_v0(
        _trace(),
        "idle_stall",
        applicator=DictApplicator(cfg, measure_fn=lambda s: 0.10),
        proposer=proposer,
    )
    assert results and all(r.applicable and not r.rolled_back for r in results)
    assert all(s.safety.tier == "moderate" for s in (r.spec for r in results))
