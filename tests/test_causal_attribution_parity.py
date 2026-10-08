"""A known cause must survive decode identically to ground truth, on both vendors.

truth = the fixture's own trace; decoded = the same execution through each
collector's records and the real decoder; naive = the pre-graph-identity
decode, which proves the comparison can fail.
"""

from __future__ import annotations

import warnings

import pytest

from gitm.optimizer.attribution import attribute
from gitm.optimizer.dr import attribute_dr
from gitm.optimizer.mechanism_fixtures import (
    Observation,
    RegionSlowdown,
    Scenario,
    generate,
    observe,
    same_residuals,
)
from gitm.optimizer.monitor import check_invariants, recoverable_by, recoverable_by_op, residuals
from gitm.tracer._cupti_decode import decode_records_with_report
from gitm.tracer.emulate import EmulationConfig, emulate, launches_from_fixture

SLOW_LAYERS = frozenset(range(8))
CAUSE = "mlp_down"


@pytest.fixture(scope="module")
def fx():
    return generate(Scenario(n_steps=6, noise_cv=0.05, seed=3, side_stream=True,
                             mechanisms=(RegionSlowdown(1.0, ops={CAUSE},
                                                        layers=SLOW_LAYERS),)))[0]


def _observe(fx, records) -> Observation:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, _ = decode_records_with_report(records)
    trace = fx.trace.model_copy(update={"events": events})
    res = residuals(trace, fx.graph)
    return Observation(res, check_invariants(res), fx.traced_step_ns, 0, fx.true_tpot_s)


def _ranking(hyps, n=10):
    return [(h.cause_op, h.effect_op, round(h.p_value, 12), h.direction) for h in hyps.top(n)]


def _cause_found(obs) -> set[int]:
    return {v.layer for v in obs.violations if v.node_op == CAUSE}


EXACT = [
    pytest.param(EmulationConfig("nvidia"), id="nvidia-eager"),
    pytest.param(EmulationConfig("amd"), id="amd-eager"),
    pytest.param(EmulationConfig("amd", graphs=True, graph_launch_range="L0/qkv_proj"),
                 id="amd-graphs"),
    pytest.param(EmulationConfig("nvidia", graphs=True, cupti_node_map=True,
                                 graph_launch_range="L0/qkv_proj"),
                 id="nvidia-graphs-with-node-map"),
]


def test_truth_finds_the_injected_cause(fx):
    found = _cause_found(observe(fx))
    assert found and found <= SLOW_LAYERS


@pytest.mark.parametrize("cfg", EXACT)
def test_decoded_causal_products_equal_truth(fx, cfg):
    truth = observe(fx)
    got = _observe(fx, emulate(launches_from_fixture(fx), cfg).records)
    assert same_residuals(truth, got, atol=0.0)
    assert got.violation_signature() == truth.violation_signature()
    assert _cause_found(got) == _cause_found(truth)
    assert recoverable_by_op(got.residuals) == recoverable_by_op(truth.residuals)
    assert _ranking(attribute(got.residuals, fx.graph)) == \
        _ranking(attribute(truth.residuals, fx.graph))
    assert _ranking(attribute_dr(got.residuals, fx.graph)) == \
        _ranking(attribute_dr(truth.residuals, fx.graph))


def test_both_vendors_agree_with_each_other_in_graph_mode(fx):
    la = launches_from_fixture(fx)
    amd = _observe(fx, emulate(la, EmulationConfig("amd", graphs=True)).records)
    nv = _observe(fx, emulate(la, EmulationConfig("nvidia", graphs=True,
                                                  cupti_node_map=True)).records)
    assert same_residuals(amd, nv, atol=0.0)
    assert amd.violation_signature() == nv.violation_signature()


def test_attribution_is_not_vacuous(fx):
    """The rankings compared above are real rankings, not two empty lists."""
    truth = observe(fx)
    assert attribute(truth.residuals, fx.graph).hypotheses
    assert attribute_dr(truth.residuals, fx.graph).hypotheses


def test_the_naive_decode_loses_the_cause(fx):
    """Before graph identity, every replayed kernel took the range around its launch."""
    em = emulate(launches_from_fixture(fx),
                 EmulationConfig("amd", graphs=True, graph_launch_range="L0/qkv_proj"))
    old = [{k: v for k, v in r.items() if k not in ("graph_id", "graph_node_id", "graph_launch")}
           for r in em.records]
    naive = _observe(fx, old)
    assert _cause_found(naive) == set()
    assert not same_residuals(naive, observe(fx))
    assert set(recoverable_by_op(naive.residuals)) == {"qkv_proj"}
    assert {(k.op, k.layer) for k in naive.residuals.per_kernel} == {("qkv_proj", 0)}
    assert attribute(naive.residuals, fx.graph).hypotheses == []


@pytest.mark.parametrize("hazard", [dict(stamp_per_dispatch=False),
                                    dict(swap_replay_pair=(2, 3)),
                                    dict(exec_untracked=True),
                                    dict(stamp_inherits_launch=True),
                                    dict(capture_unranged=True),
                                    dict(worker_dispatch=True),
                                    dict(capture_compiled=True)])
def test_a_refused_replay_never_invents_a_violation(fx, hazard):
    """Refusing identity loses evidence; it must not create any."""
    truth = {(v.invariant, v.node_op, v.layer) for v in observe(fx).violations}
    got = _observe(fx, emulate(launches_from_fixture(fx),
                               EmulationConfig("amd", graphs=True,
                                               graph_launch_range="L0/qkv_proj",
                                               **hazard)).records)
    assert {(v.invariant, v.node_op, v.layer) for v in got.violations} <= truth


def test_nvidia_graphs_without_the_node_map_miss_the_cause_but_invent_nothing(fx):
    truth = {(v.invariant, v.node_op, v.layer) for v in observe(fx).violations}
    got = _observe(fx, emulate(launches_from_fixture(fx),
                               EmulationConfig("nvidia", graphs=True)).records)
    assert _cause_found(got) == set()
    assert {(v.invariant, v.node_op, v.layer) for v in got.violations} <= truth


def test_stratifying_by_a_range_annotation_localises_the_cause():
    slow = frozenset(layer for layer in range(32) if layer % 4 == 0)
    fx = generate(Scenario(n_steps=6, noise_cv=0.03, seed=1,
                           mechanisms=(RegionSlowdown(0.6, ops={CAUSE}, layers=slow),)))[0]
    cfg = EmulationConfig("amd", graphs=True, annotate_launch=lambda la: (
        {"wave": str(la.layer % 4)} if la.op == CAUSE else None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, _ = decode_records_with_report(emulate(launches_from_fixture(fx), cfg).records)
    res = residuals(fx.trace.model_copy(update={"events": events}), fx.graph,
                    with_attributes=True)

    by_wave = {k: v for k, v in recoverable_by(res, ("wave",)).items() if k.startswith(CAUSE)}
    assert set(by_wave) == {f"{CAUSE}[wave={w}]" for w in "0123"}
    hot = by_wave[f"{CAUSE}[wave=0]"]
    assert all(hot > 3 * v for k, v in by_wave.items() if not k.endswith("=0]"))
    # The op-level number is the sum; stratifying splits it, it does not move it.
    assert sum(by_wave.values()) == pytest.approx(recoverable_by_op(res)[CAUSE])
    # Violations still name the op, at exactly the slowed layers.
    assert {v.layer for v in check_invariants(res, stratify=("wave",))
            if v.node_op == CAUSE} == set(slow)
    # And attribution, stratified, speaks in strata.
    hyps = attribute(res, fx.graph, stratify=("wave",)).hypotheses
    assert any(h.cause_op == f"{CAUSE}[wave=0]" for h in hyps)


def test_static_attributes_come_from_the_graph_itself():
    from gitm.tracer.kernel_attributes import AttributeIndex

    fx = generate(Scenario(n_steps=1))[0]
    idx = AttributeIndex.from_graph(fx.graph)
    # A dense model: one structural class for every layer.
    assert {a["layer_class"] for a in idx.by_layer.values()} == {"c0"}
    k = {"range_layer": 3, "range_op": "moe_routed", "stream_id": 7, "graph_id": 2,
         "identity": "graph_node", "range_attrs": {"wave": "1", "layer_class": "override"}}
    attrs = idx.attributes(k)
    assert attrs["moe_phase"] == "expert" and attrs["replay"] == "graph"
    assert attrs["wave"] == "1" and attrs["layer_class"] == "override"


def _window(events, t0, t1):
    return [e for e in events if t0 <= e.start_ns <= t1]


def test_attribution_survives_a_real_window_and_a_refused_replay(fx):
    starts = sorted(e.start_ns for e in fx.trace.events)
    t0, t1 = starts[len(starts) // 7], starts[(6 * len(starts)) // 7]
    la = launches_from_fixture(fx)

    def observed(records):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events, report = decode_records_with_report(records)
        trace = fx.trace.model_copy(update={"events": _window(events, t0, t1)})
        return residuals(trace, fx.graph), report

    truth = residuals(fx.trace.model_copy(update={"events": _window(fx.trace.events, t0, t1)}),
                      fx.graph)
    clean, _ = observed(emulate(la, EmulationConfig("amd", graphs=True)).records)
    assert _ranking(attribute(clean, fx.graph)) == _ranking(attribute(truth, fx.graph))
    assert _ranking(attribute_dr(clean, fx.graph)) == _ranking(attribute_dr(truth, fx.graph))

    # Refuse one replay: mismatch one of its kernels' symbol.
    records = emulate(la, EmulationConfig("amd", graphs=True)).records
    victim = next(r for r in records if r.get("kind") == "kernel" and r.get("graph_id"))
    victim["kernel_id"] ^= 1
    refused, report = observed(records)
    assert sum(report.graph_refused.values()) > 0
    counts = {}
    for k in refused.per_kernel:
        counts[k.op] = counts.get(k.op, 0) + 1
    assert len(set(counts.values())) > 1  # same-cardinality ops now differ in length
    assert attribute(refused, fx.graph).hypotheses
    assert attribute_dr(refused, fx.graph).hypotheses


def test_once_per_step_ops_are_never_aligned_with_per_layer_ops(fx):
    from gitm.optimizer.attribution import comparable

    truth = observe(fx)
    n_layer = sum(1 for k in truth.residuals.per_kernel if k.op == CAUSE)
    n_step = sum(1 for k in truth.residuals.per_kernel if k.op == "lm_head")
    assert not comparable(n_layer, n_step)
    hyps = attribute(truth.residuals, fx.graph).hypotheses +         attribute_dr(truth.residuals, fx.graph).hypotheses
    assert not any("lm_head" in (h.cause_op, h.effect_op) for h in hyps)
