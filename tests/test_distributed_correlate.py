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
