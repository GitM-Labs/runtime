"""Kernel identity accuracy against ground truth, per vendor, mode and hazard.

Each case starts from a mechanism fixture — what the host launched, in order,
for which op and layer — renders it as one vendor's collector would record it
(gitm.tracer.emulate), runs the records through the real decoder, and scores
every decoded kernel against the launch truth.

Two numbers per case: accuracy, and **wrong** — kernels named as a different op
or layer than the one launched. Wrong must be zero everywhere: a missing
identity falls back to the name and is visible in the coverage report; a wrong
one is trusted downstream and silently mis-prices the op. Where a mechanism
cannot name a kernel (NVIDIA graph replay before the capture-time node map
lands), the requirement is that it says nothing, not something false.
"""

from __future__ import annotations

import pytest

from gitm.optimizer.mechanism_fixtures import Scenario, generate
from gitm.planner.roofline import BatchConfig
from gitm.tracer._cupti_decode import decode_records_with_report
from gitm.tracer.emulate import EmulationConfig, emulate, launches_from_fixture, score

SCENARIOS = {
    "dense": Scenario(n_steps=3),
    "side_stream": Scenario(n_steps=3, side_stream=True),
    "noisy_large_batch": Scenario(n_steps=4, noise_cv=0.1, seed=7,
                                  points=(BatchConfig(batch=64, kv_cache_len=8192),)),
}


def _launches(name):
    return launches_from_fixture(generate(SCENARIOS[name])[0])


def _run(launches, cfg):
    em = emulate(launches, cfg)
    events, report = decode_records_with_report(em.records)
    return score(events, em.truth), report, events


EXACT = [
    pytest.param(EmulationConfig("nvidia"), id="nvidia-eager"),
    pytest.param(EmulationConfig("amd"), id="amd-eager"),
    pytest.param(EmulationConfig("amd", graphs=True), id="amd-graphs"),
    pytest.param(EmulationConfig("nvidia", graphs=True, cupti_node_map=True),
                 id="nvidia-graphs-with-node-map"),
    pytest.param(EmulationConfig("amd", graphs=True, blit_memset_node=True), id="amd-graphs-blit"),
    pytest.param(EmulationConfig("amd", helper_thread_ops=frozenset({"mlp_down", "qkv_proj"}),
                                 device_offset_ns=10**12), id="amd-eager-helper-thread-skew"),
    pytest.param(EmulationConfig("nvidia", helper_thread_ops=frozenset({"mlp_down"}),
                                 device_offset_ns=-(10**9)), id="nvidia-eager-helper-thread-skew"),
    pytest.param(EmulationConfig("amd", graphs=True, graph_launch_range="L0/qkv_proj"),
                 id="amd-graphs-launched-inside-an-op-range"),
]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("cfg", EXACT)
def test_identity_is_exact(cfg, scenario):
    s, report, _ = _run(_launches(scenario), cfg)
    assert s.n > 0
    assert (s.accuracy, s.wrong, s.missing) == (1.0, 0, 0), s.wrong_examples
    assert not report.graph_refused and not report.stamp_containment_disagree


def test_amd_eager_identity_comes_from_the_stamp_not_containment():
    _, report, _ = _run(_launches("dense"), EmulationConfig("amd"))
    assert set(report.identity) == {"range_id"}


def test_amd_graph_identity_comes_from_the_capture_projection():
    _, report, _ = _run(_launches("dense"), EmulationConfig("amd", graphs=True))
    assert set(report.identity) == {"graph_node"}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_nvidia_graphs_today_say_nothing_rather_than_something_wrong(scenario):
    """No capture-time node map yet: replayed GEMMs cannot be named, and must
    not borrow the step's range. Only name-identifiable kernels resolve."""
    s, report, _ = _run(_launches(scenario), EmulationConfig("nvidia", graphs=True))
    assert s.wrong == 0
    # A layer is needed for every per-layer op, and no name carries one, so
    # nothing per-layer resolves; only the step-level and unranged kernels do.
    assert s.missing > 0 and s.accuracy < 1
    assert report.graph_refused["no_node"] == report.graph_kernels


HAZARDS = [
    pytest.param(dict(stamp_per_dispatch=False), "duplicate_ordinal",
                 id="stamp-does-not-advance"),
    pytest.param(dict(swap_replay_pair=(2, 3)), "signature_",
                 id="replay-reordered-across-different-kernels"),
    pytest.param(dict(exec_untracked=True), "untracked_launch", id="instantiate-never-seen"),
    # The likeliest 7.2.3 failure: no per-dispatch request inside the launch,
    # so every dispatch inherits the hipGraphLaunch call's stamp — the id of
    # the live launch range — and no graph identity. The graph_launch guard
    # must run before the stamped join, or the launch range becomes the op.
    pytest.param(dict(stamp_inherits_launch=True), "untracked_launch",
                 id="dispatch-inherits-the-launch-stamp"),
]


@pytest.mark.parametrize("kw,reason", HAZARDS)
def test_amd_graph_hazards_refuse_instead_of_guessing(kw, reason):
    s, report, events = _run(_launches("side_stream"),
                             EmulationConfig("amd", graphs=True,
                                             graph_launch_range="L0/qkv_proj", **kw))
    assert s.wrong == 0, s.wrong_examples
    assert any(r.startswith(reason) for r in report.graph_refused), report.graph_refused
    # The refused kernels are still known to be replays: none carries the
    # range around the launch as its op.
    assert not any(e.range_op == "qkv_proj" and e.range_layer == 0 and e.identity != "graph_node"
                   for e in events if e.kind == "kernel")


def test_known_limit_reordering_two_identical_nodes_is_undetectable():
    """Validation compares a replayed kernel with its node's signature (symbol,
    geometry, kind). Two nodes with the same signature — the same projection in
    two layers — cannot be told apart if the runtime swaps them. A single-stream
    capture (vLLM's) replays in capture order, so this needs a forked capture
    and a reordering runtime; it is pinned here so the limit is a known one."""
    launches = _launches("dense")
    step0 = [i for i, la in enumerate(launches) if la.step == 0]
    same = [i for i in step0 if launches[i].op == "qkv_proj"][:2]
    s, report, _ = _run(launches, EmulationConfig("amd", graphs=True,
                                                  swap_replay_pair=tuple(same)))
    assert not report.graph_refused
    assert s.wrong == 2 * len({la.step for la in launches})


def test_the_old_graph_behaviour_is_what_these_tests_exist_to_catch():
    """Strip graph identity (the collector before #160) and decode: every
    replayed kernel takes the launch's range as its op."""
    launches = _launches("dense")
    em = emulate(launches, EmulationConfig("amd", graphs=True, graph_launch_range="L0/qkv_proj"))
    old = [{k: v for k, v in r.items() if k not in ("graph_id", "graph_node_id", "graph_launch")}
           for r in em.records]
    events, _ = decode_records_with_report(old)
    s = score(events, em.truth)
    assert s.wrong > 0.9 * s.n


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shard_record_order_does_not_matter(seed):
    launches = _launches("side_stream")
    for cfg in (EmulationConfig("nvidia", seed=seed), EmulationConfig("amd", graphs=True,
                                                                       seed=seed)):
        s, _, _ = _run(launches, cfg)
        assert (s.accuracy, s.wrong) == (1.0, 0)


def test_ranks_with_colliding_ids_are_scored_independently():
    """Two ranks issue identical launch sequences, so correlation ids, marker
    ids and graph node ids collide across processes; decoding must partition."""
    launches = _launches("dense")
    a = emulate(launches, EmulationConfig("amd", graphs=True, pid=100))
    b = emulate(launches, EmulationConfig("amd", pid=200, device_offset_ns=7))
    events, report = decode_records_with_report(a.records + b.records)
    sa = score([e for e in events if e.pid == 100], a.truth)
    sb = score([e for e in events if e.pid == 200], b.truth)
    assert (sa.accuracy, sa.wrong, sb.accuracy, sb.wrong) == (1.0, 0, 1.0, 0)
    assert report.identity["graph_node"] and report.identity["range_id"]


def test_memcpys_label_every_step_on_both_vendors_in_graph_mode():
    launches = _launches("dense")
    for vendor in ("amd", "nvidia"):
        em = emulate(launches, EmulationConfig(vendor, graphs=True))
        events, report = decode_records_with_report(em.records)
        copies = [e for e in events if e.kind == "memcpy"]
        assert copies and all(c.launch_range == "decode_step" for c in copies)
        assert report.memcpys_labelled == report.memcpys


def test_a_capture_without_ranges_is_reported_not_silent():
    """torch.compile can trace the instrumentation away during capture: every
    node exists, none is named, nothing is refused — and without a report the
    capture would look like a model with no identifiable ops."""
    s, report, _ = _run(_launches("dense"), EmulationConfig("amd", graphs=True,
                                                            capture_unranged=True))
    assert s.wrong == 0 and not report.graph_refused
    assert report.graph_unnamed == report.graph_kernels
    assert any("captured outside every range" in p for p in report.problems())


def test_a_few_unranged_nodes_are_normal():
    _, report, _ = _run(_launches("side_stream"), EmulationConfig("amd", graphs=True))
    assert 0 < report.graph_unnamed < report.graph_kernels / 2
    assert report.problems() == []
