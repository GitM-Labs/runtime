"""Correlation changes the causal answer, and the ROCm port gives the same answer.

A fixture with a known cause — a slowdown injected into one op at chosen
layers — is observed three ways:

* **truth**: the fixture's own trace, where every kernel carries the identity
  it was launched with (``mechanism_fixtures.observe``);
* **decoded**: the same execution rendered as a vendor's collector records it
  and decoded by the real pipeline (gitm.tracer.emulate);
* **naive**: decoded the way the collector worked before graph identity
  existed — a replayed kernel takes the range around its launch.

The requirement is not that decoded identity looks plausible but that every
downstream causal product — residuals, invariant violations, Granger and
doubly-robust rankings, recoverable time — is *identical* to truth on every
mechanism that claims exact identity, on both vendors; that the injected
cause is among the violations; and that the naive decode loses it. The last
is what proves the test can fail.
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
    """Not necessarily at every slowed layer — multi-basis confirmation keeps
    only the anomalies it can corroborate — but only at slowed layers. Truth's
    own finding is then the reference every decode must reproduce."""
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
    """Before graph identity, every replayed kernel took the range around its
    launch. With the launch inside an op range, every kernel of the step is
    filed as that one op: the mixture's median sits inside the band, so the
    monitor reports nothing, and the only op attribution can name is the wrong
    one."""
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
                                    dict(exec_untracked=True)])
def test_a_refused_replay_never_invents_a_violation(fx, hazard):
    """Refusing identity loses evidence; it must not create any. Every
    violation a hazard-hit decode reports is one truth reports too."""
    truth = {(v.invariant, v.node_op, v.layer) for v in observe(fx).violations}
    got = _observe(fx, emulate(launches_from_fixture(fx),
                               EmulationConfig("amd", graphs=True,
                                               graph_launch_range="L0/qkv_proj",
                                               **hazard)).records)
    assert {(v.invariant, v.node_op, v.layer) for v in got.violations} <= truth


def test_nvidia_graphs_today_miss_the_cause_but_invent_nothing(fx):
    """The gap the capture-time node map closes, measured: the per-layer cause
    is invisible, and nothing false is reported in its place."""
    truth = {(v.invariant, v.node_op, v.layer) for v in observe(fx).violations}
    got = _observe(fx, emulate(launches_from_fixture(fx),
                               EmulationConfig("nvidia", graphs=True)).records)
    assert _cause_found(got) == set()
    assert {(v.invariant, v.node_op, v.layer) for v in got.violations} <= truth


def test_stratifying_by_a_range_annotation_localises_the_cause():
    """Expert-parallel waves are dynamic — they cannot come from the layer — so
    they ride on the range as an annotation. A slowdown confined to wave 0 is a
    quarter of mlp_down's launches: op-level attribution can only say
    "mlp_down"; stratified by wave it says which wave, and only that one."""
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
