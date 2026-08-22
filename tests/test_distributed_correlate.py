"""Correlation must be scoped to the process that produced the identifiers.

CUPTI assigns ``correlation_id`` per process, from a low origin in each. A
tensor-parallel capture therefore contains, for each identifier, one record per
rank. Correlating a merged sequence resolves every identifier to whichever
record happened to be indexed last, producing range attributions drawn from an
arbitrary rank while remaining syntactically well-formed.

These tests construct two ranks whose identifiers collide by construction,
establish that the merged path mis-attributes, and establish that the
partitioned path does not.
"""

from __future__ import annotations

import pytest

from gitm.distributed.correlate import (
    correlate_by_rank,
    correlate_kernels_to_ranges,
    correlation_id_collisions,
)
from gitm.distributed.topology import Topology, topology_from_records


def _kernel(cid: int, device: int, t0: int = 100) -> dict:
    return {
        "kind": "kernel", "name": "gemm", "start_ns": t0, "end_ns": t0 + 10,
        "device_id": device, "context_id": 1, "stream_id": 7, "correlation_id": cid,
    }


def _runtime(cid: int, thread: int, t0: int = 100) -> dict:
    return {
        "kind": "runtime", "name": "cudaLaunchKernel", "start_ns": t0, "end_ns": t0 + 2,
        "correlation_id": cid, "thread_id": thread,
    }


def _marker(name: str, thread: int, t0: int = 90, t1: int = 200) -> dict:
    return {"kind": "marker", "name": name, "start_ns": t0, "end_ns": t1, "thread_id": thread}


def _rank_records(device: int, thread: int, op: str, cids: list[int]) -> list[dict]:
    """One process's records: a marker enclosing a launch/kernel pair per cid."""
    recs: list[dict] = [_marker(f"L{device}/{op}", thread)]
    for cid in cids:
        recs.append(_runtime(cid, thread))
        recs.append(_kernel(cid, device))
    return recs


# Two ranks, identical correlation identifiers, distinguishable range names.
BY_PID = {
    1000: _rank_records(device=0, thread=10, op="attn_score_value", cids=[1, 2, 3]),
    2000: _rank_records(device=1, thread=20, op="moe_routed", cids=[1, 2, 3]),
}


# ── the collision ───────────────────────────────────────────────────────────


def test_identifiers_collide_across_processes():
    """The precondition. Without collisions the two paths would agree trivially."""
    collisions = correlation_id_collisions(BY_PID)
    assert set(collisions) == {1, 2, 3}
    assert all(pids == [1000, 2000] for pids in collisions.values())


def test_no_collisions_reported_for_a_single_process():
    assert correlation_id_collisions({1000: BY_PID[1000]}) == {}


# ── merged correlation is wrong ─────────────────────────────────────────────


def test_merged_correlation_attributes_kernels_to_the_wrong_rank():
    """Establishes the defect this module exists to remove.

    Every kernel resolves to a single range name, because one process's runtime
    records displaced the other's in the identifier index. The attribution is
    well-formed and describes only one of the two ranks.
    """
    merged = BY_PID[1000] + BY_PID[2000]
    enriched = correlate_kernels_to_ranges(merged)

    assert len(enriched) == 6
    ops = {e["range_op"] for e in enriched}
    assert len(ops) == 1, "merged correlation should collapse both ranks onto one range"


# ── partitioned correlation is correct ──────────────────────────────────────


def test_partitioned_correlation_preserves_per_rank_attribution():
    """Each rank's kernels resolve to that rank's own range."""
    enriched = correlate_by_rank(BY_PID)
    assert len(enriched) == 6

    by_pid: dict[int, set[str]] = {}
    for e in enriched:
        by_pid.setdefault(e["pid"], set()).add(e["range_op"])

    assert by_pid[1000] == {"attn_score_value"}
    assert by_pid[2000] == {"moe_routed"}


def test_layer_index_survives_partitioning():
    enriched = correlate_by_rank(BY_PID)
    layers = {e["pid"]: {e2["range_layer"] for e2 in enriched if e2["pid"] == e["pid"]}
              for e in enriched}
    assert layers[1000] == {0}
    assert layers[2000] == {1}


def test_every_record_carries_its_rank():
    enriched = correlate_by_rank(BY_PID)
    assert {e["local_rank"] for e in enriched} == {0, 1}
    assert all(isinstance(e["pid"], int) for e in enriched)


def test_output_is_ordered_by_start_time():
    recs = {
        1000: _rank_records(0, 10, "a", [1]),
        2000: [_marker("L1/b", 20, t0=290, t1=400), _runtime(1, 20, t0=300),
               _kernel(1, 1, t0=300)],
    }
    enriched = correlate_by_rank(recs)
    assert [e["start_ns"] for e in enriched] == sorted(e["start_ns"] for e in enriched)


def test_single_process_capture_is_unchanged_by_partitioning():
    """The common case must not regress: one process, one partition."""
    single = {1000: BY_PID[1000]}
    partitioned = correlate_by_rank(single)
    merged = correlate_kernels_to_ranges(BY_PID[1000])

    assert len(partitioned) == len(merged)
    for p, m in zip(partitioned, merged, strict=True):
        assert p["range_op"] == m["range_op"]
        assert p["range_layer"] == m["range_layer"]


def test_unknown_process_is_labelled_rather_than_dropped():
    """A pid absent from the topology still yields records, marked rank -1."""
    enriched = correlate_by_rank(BY_PID, Topology())
    assert len(enriched) == 6
    assert {e["local_rank"] for e in enriched} == {-1}


# ── topology ────────────────────────────────────────────────────────────────


def test_device_attribution_uses_the_modal_device():
    """Incidental peer activity must not relabel a rank.

    A worker emits records on peer devices through peer-to-peer copies and
    collectives; a minority of such records must not determine its device.
    """
    recs = [_kernel(i, device=0) for i in range(9)] + [_kernel(99, device=1)]
    topo = topology_from_records({1000: recs})
    assert topo.ranks[0].device_id == 0


def test_rank_ordinals_follow_device_then_pid():
    """Stability across captures is a precondition for comparing a rank between them."""
    topo = topology_from_records({
        9000: [_kernel(1, device=1)],
        1000: [_kernel(1, device=0)],
        5000: [_kernel(1, device=0)],
    })
    assert [(r.local_rank, r.device_id, r.pid) for r in topo.ranks] == [
        (0, 0, 1000), (1, 0, 5000), (2, 1, 9000),
    ]


def test_a_process_without_device_activity_is_recorded_but_ordered_last():
    """A frontend holding a CUDA context without launching kernels is a real
    collector with no device to attribute to. Assigning it device 0 would
    assert an association the records do not support."""
    topo = topology_from_records({
        1000: [{"kind": "marker", "name": "x", "start_ns": 1, "end_ns": 2}],
        2000: [_kernel(1, device=0)],
    })
    assert topo.ranks[-1].pid == 1000
    assert topo.ranks[-1].device_id is None
    assert topo.ranks[-1].collected_device_work is False
    assert topo.world_size == 1


def test_partial_capture_is_detectable():
    """One rank collecting where two devices are visible understates the job."""
    topo = topology_from_records({1000: [_kernel(1, device=0)]})
    complete = Topology(ranks=topo.ranks, visible_gpus=1, gpu_source="test")
    partial = Topology(ranks=topo.ranks, visible_gpus=2, gpu_source="test")

    assert complete.covers_all_visible_gpus
    assert not partial.covers_all_visible_gpus
    assert "PARTIAL" in partial.summary()


def test_empty_capture_summarises_without_raising():
    assert Topology().summary() == "no collecting processes found"
    assert Topology().world_size == 0
    assert Topology().devices == ()


# ── layering ────────────────────────────────────────────────────────────────


def test_distributed_package_has_no_outward_gitm_dependencies():
    """The package must remain importable without the rest of GITM.

    Shard discovery, the shard-filename convention and the trace-output
    environment are owned by ``gitm.tracer``, which passes shard locations
    inward. An import in the opposite direction would create a cycle and would
    place multi-process semantics behind a single-process module.
    """
    import pathlib
    import re

    pkg = pathlib.Path(__file__).resolve().parents[1] / "gitm" / "distributed"
    pattern = re.compile(r"^\s*(?:from|import)\s+(gitm\.[\w.]+)", re.MULTILINE)

    offenders: dict[str, set[str]] = {}
    for path in pkg.glob("*.py"):
        found = {
            m for m in pattern.findall(path.read_text())
            if not m.startswith("gitm.distributed")
        }
        if found:
            offenders[path.name] = found

    assert offenders == {}, f"outward gitm imports in gitm/distributed: {offenders}"


# ── the correlation index scales ────────────────────────────────────────────
#
# Containment used to be a nested scan: every marker examined for every kernel.
# At the shapes a real capture produces — 9.5M kernels against ~800k markers when
# instrumented per layer per op — that is 7.8e12 comparisons, roughly a day of
# CPU, so correlation was unusable exactly where it was most needed. The sweep
# below is O((n+m) log(n+m)).
#
# A faster algorithm that answers differently is worse than a slow one, so the
# first test here is differential against the original definition.


def _brute_force_enclosing(markers: list[dict], rt: dict) -> dict | None:
    """The original semantics, kept as the oracle: smallest-span marker on the
    same thread whose window fully contains the runtime window."""
    best, best_span = None, None
    for m in markers:
        if m.get("thread_id") != rt.get("thread_id"):
            continue
        if m["start_ns"] <= rt["start_ns"] and rt["end_ns"] <= m["end_ns"]:
            span = m["end_ns"] - m["start_ns"]
            if best_span is None or span < best_span:
                best, best_span = m, span
    return best


def _nested_capture(seed: int, n_threads: int = 3, n_steps: int = 12):
    """A properly nested push/pop capture, the shape real NVTX instrumentation
    produces: an outer layer range with op ranges inside it."""
    import random

    rng = random.Random(seed)
    markers: list[dict] = []
    runtimes: list[dict] = []
    cid = 0
    for thread in range(n_threads):
        t = rng.randint(0, 1000)
        for step in range(n_steps):
            layer = step % 4
            outer_start = t
            inner: list[dict] = []
            for op in ("qkv_proj", "attn_score_value", "moe_routed"):
                op_start = t
                t += rng.randint(5, 40)
                # Launches inside this op range. Some deliberately outlive it —
                # an async launch can end after its range is popped, and those
                # must fall through to the enclosing layer range.
                for _ in range(rng.randint(1, 3)):
                    cid += 1
                    s = rng.randint(op_start, t)
                    # >= 1 ns: a cudaLaunchKernel call is never instantaneous,
                    # and a zero-width window landing exactly on a range boundary
                    # is contained by both the closing and the opening range. That
                    # ambiguity is resolved deliberately, and separately, by
                    # test_a_launch_on_a_range_boundary_belongs_to_the_opening_range.
                    e = s + rng.randint(1, 30)
                    runtimes.append({"kind": "runtime", "correlation_id": cid,
                                     "start_ns": s, "end_ns": e, "thread_id": thread})
                inner.append({"kind": "marker", "name": f"L{layer}/{op}",
                              "start_ns": op_start, "end_ns": t, "thread_id": thread})
                t += rng.randint(0, 5)
            markers.extend(inner)
            markers.append({"kind": "marker", "name": f"L{layer}/layer",
                            "start_ns": outer_start, "end_ns": t, "thread_id": thread})
            t += rng.randint(1, 20)
    return markers, runtimes


@pytest.mark.parametrize("seed", range(8))
def test_sweep_matches_the_brute_force_definition(seed):
    """Differential: same answer as the nested scan it replaces, on nested
    ranges, multiple threads, and launches that outlive their range."""
    from gitm.distributed.correlate import _innermost_enclosing

    markers, runtimes = _nested_capture(seed)
    got = _innermost_enclosing(markers, runtimes)

    for rt in runtimes:
        expected = _brute_force_enclosing(markers, rt)
        actual = got.get(id(rt))
        if expected is None:
            assert actual is None, rt
        else:
            # Ties on span are legitimate; compare the resolved identity.
            assert actual is not None, rt
            assert (actual["name"], actual["start_ns"], actual["end_ns"]) == (
                expected["name"], expected["start_ns"], expected["end_ns"]
            ), rt


def test_a_launch_outliving_its_range_falls_through_to_the_enclosing_one():
    """An asynchronous launch can end after the range that issued it is popped.
    The innermost range that *fully contains* it is then the outer one, and
    stopping at the top of the stack would return no range at all."""
    from gitm.distributed.correlate import _innermost_enclosing

    outer = {"kind": "marker", "name": "L0/layer", "start_ns": 0, "end_ns": 100,
             "thread_id": 1}
    inner = {"kind": "marker", "name": "L0/qkv_proj", "start_ns": 10, "end_ns": 20,
             "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 12, "end_ns": 60,
          "thread_id": 1}

    got = _innermost_enclosing([outer, inner], [rt])
    assert got[id(rt)]["name"] == "L0/layer"


def test_threads_never_borrow_each_others_ranges():
    from gitm.distributed.correlate import _innermost_enclosing

    m = {"kind": "marker", "name": "L0/moe_routed", "start_ns": 0, "end_ns": 100,
         "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 10, "end_ns": 20,
          "thread_id": 2}
    assert _innermost_enclosing([m], [rt]) == {}


def test_a_range_closing_exactly_at_a_launch_does_not_contain_it():
    """Tie-breaking at equal timestamps is semantic. A range whose end coincides
    with a launch's start cannot contain a launch of non-zero duration."""
    from gitm.distributed.correlate import _innermost_enclosing

    m = {"kind": "marker", "name": "L0/a", "start_ns": 0, "end_ns": 10, "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 10, "end_ns": 20,
          "thread_id": 1}
    assert _innermost_enclosing([m], [rt]) == {}


def test_a_range_opening_exactly_at_a_launch_does_contain_it():
    from gitm.distributed.correlate import _innermost_enclosing

    m = {"kind": "marker", "name": "L0/a", "start_ns": 10, "end_ns": 30, "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 10, "end_ns": 20,
          "thread_id": 1}
    assert _innermost_enclosing([m], [rt])[id(rt)] is m


def test_crossed_ranges_do_not_desynchronise_the_stack():
    """A malformed capture — ranges that overlap without nesting — must degrade
    to a wrong answer for the crossed pair only, not corrupt every attribution
    after it. Popping by identity rather than from the top is what bounds it."""
    from gitm.distributed.correlate import _innermost_enclosing

    a = {"kind": "marker", "name": "L0/a", "start_ns": 0, "end_ns": 50, "thread_id": 1}
    b = {"kind": "marker", "name": "L0/b", "start_ns": 20, "end_ns": 80, "thread_id": 1}
    later = {"kind": "marker", "name": "L1/c", "start_ns": 100, "end_ns": 200,
             "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 120, "end_ns": 130,
          "thread_id": 1}

    got = _innermost_enclosing([a, b, later], [rt])
    assert got[id(rt)] is later


def test_correlation_is_linearithmic_not_quadratic():
    """Doubling both inputs must not quadruple the work. Measured in comparisons
    rather than wall time so the assertion does not depend on machine load."""
    from gitm.distributed.correlate import _innermost_enclosing

    def call_count(steps):
        markers, runtimes = _nested_capture(seed=1, n_threads=1, n_steps=steps)
        n = [0]
        orig = dict.get

        class Counting(dict):
            def __getitem__(self, k):
                n[0] += 1
                return super().__getitem__(k)

        markers = [Counting(m) for m in markers]
        runtimes = [Counting(r) for r in runtimes]
        _innermost_enclosing(markers, runtimes)
        del orig
        return n[0]

    small = call_count(50)
    large = call_count(200)
    # 4x the input. Quadratic would be ~16x; linearithmic is a little over 4x.
    assert large < 6 * small, f"{small} -> {large} looks superlinear"


def test_a_launch_on_a_range_boundary_belongs_to_the_opening_range():
    """The one place the sweep deliberately differs from "smallest enclosing span".

    When one range closes and the next opens at the same instant, a launch whose
    window touches that instant is contained by both. Span alone would pick
    whichever happened to be shorter, which carries no meaning. The sweep picks
    the range that just opened, because NVTX pops before it pushes: a call
    observed at that timestamp was issued after the pop.
    """
    from gitm.distributed.correlate import _innermost_enclosing

    closing = {"kind": "marker", "name": "L1/attn_score_value",
               "start_ns": 1577, "end_ns": 1601, "thread_id": 1}
    opening = {"kind": "marker", "name": "L1/moe_routed",
               "start_ns": 1601, "end_ns": 1638, "thread_id": 1}
    rt = {"kind": "runtime", "correlation_id": 1, "start_ns": 1601, "end_ns": 1601,
          "thread_id": 1}

    got = _innermost_enclosing([closing, opening], [rt])
    assert got[id(rt)]["name"] == "L1/moe_routed"
