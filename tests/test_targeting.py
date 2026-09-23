"""Joining the region that is losing time to the levers that aim at it."""

from __future__ import annotations

import json

from gitm.agents.targeting import levers_for, render_targets, targets
from gitm.kernels.library import load_library
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.deviation_table import UNMODELED, from_trace
from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, RooflinePrediction


def _node(op, t_pred_s, bound="memory"):
    return PredictedNode(op=op, layer=None, prediction=RooflinePrediction(
        op=op, flops=0.0, bytes=0.0, t_compute_s=0.0, t_memory_s=0.0,
        t_pred_s=t_pred_s, bound=bound))


def _graph(*nodes):
    return Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(),
                 nodes=list(nodes))


def _trace(path, rows):
    """rows: (kernel_name, duration_ns, repeat)."""
    with open(path, "w", encoding="utf-8") as fh:
        t = 0
        for name, dur, n in rows:
            for _ in range(n):
                fh.write(json.dumps({"kind": "kernel", "name": name, "start_ns": t,
                                     "end_ns": t + dur, "stream_id": 7,
                                     "device_id": 0}) + "\n")
                t += dur + 100
    return path


def _spec(name, kernels=(), *, whole_step=False):
    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=0.08, expected_delta_lo=0.02, expected_delta_hi=0.14,
        source="t", applies_to_kernels=list(kernels), whole_step=whole_step,
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="low_risk"))


def _one_row(tmp_path, kernel, op, *, floor_s=100e-9):
    """A table with a single modeled row, well over its floor."""
    p = _trace(tmp_path / "t.jsonl", [(kernel, 5000, 10)])
    return from_trace(p, _graph(_node(op, floor_s)), steps=1)


# --------------------------------------------------------------------------- #
# the join                                                                     #
# --------------------------------------------------------------------------- #
def test_a_region_is_matched_to_the_levers_that_name_its_op(tmp_path):
    table = _one_row(tmp_path, "flash_fwd_kernel", "attn_score_value")
    lib = [_spec("kv_layout", ["attn_score_value"]),
           _spec("mlp_only", ["mlp_down"])]

    found = targets(table, lib, top=3)

    assert [s.name for s in found[0].levers] == ["kv_layout"]
    assert found[0].row.op == "attn_score_value"


def test_the_match_is_op_identity_not_a_substring(tmp_path):
    """A coincidental substring is how an untargeted lever gets tagged as
    targeted, and then ranks as though it addressed the region."""
    table = _one_row(tmp_path, "fused_moe_kernel", "moe_routed")

    assert levers_for(table.rows[0], [_spec("looks_close", ["moe"])]) == []
    assert levers_for(table.rows[0], [_spec("exact", ["moe_routed"])]) != []


def test_a_whole_step_lever_is_listed_apart_from_the_ones_aimed_here(tmp_path):
    """"Something applies here" and "something targets this" are different
    claims, and only the second justifies spending the run on this region."""
    table = _one_row(tmp_path, "flash_fwd_kernel", "attn_score_value")
    lib = [_spec("batch_shape", whole_step=True), _spec("kv_layout", ["attn_score_value"])]

    t = targets(table, lib, top=3)[0]

    assert [s.name for s in t.levers] == ["kv_layout"]
    assert [s.name for s in t.whole_step] == ["batch_shape"]
    assert t.uncovered is False


def test_unmodeled_work_is_never_a_target(tmp_path):
    """Kernels the graph does not model are its coverage gap, not a region a
    lever could be aimed at. Naming the marker must not conjure a match."""
    p = _trace(tmp_path / "u.jsonl", [("some_kernel_nobody_models", 5000, 10)])
    table = from_trace(p, _graph(_node("attn_score_value", 100e-9)), steps=1)
    unmodeled = next(r for r in table.rows if r.op == UNMODELED)

    assert levers_for(unmodeled, [_spec("anything", [UNMODELED])]) == []
    # and it never reaches the ranking in the first place
    assert all(t.row.op != UNMODELED for t in targets(table, [], top=5))


# --------------------------------------------------------------------------- #
# the region nothing can address                                               #
# --------------------------------------------------------------------------- #
def test_a_region_no_lever_names_is_reported_not_dropped(tmp_path):
    """The whole point. A row with no candidates says the catalog cannot address
    the place the time is going — dropping it would hide exactly that."""
    table = _one_row(tmp_path, "fused_moe_kernel", "moe_routed")
    lib = [_spec("kv_layout", ["attn_score_value"]), _spec("batch_shape", whole_step=True)]

    found = targets(table, lib, top=3)

    assert len(found) == 1
    assert found[0].row.op == "moe_routed"
    assert found[0].uncovered is True
    assert found[0].levers == ()
    assert "none target this" in render_targets(found)
    assert "cannot address" in render_targets(found)


def test_the_real_catalog_now_addresses_the_dominant_moe_region(tmp_path):
    """``moe_routed`` is about half the sparse-MoE graph's predicted step. When
    this join was written no catalog entry named it, so the row came back
    uncovered — the loop could rank perfectly and still have nothing to try
    where the time actually went.

    #124 scoped the MoE levers to ``moe_routed`` and named the expert GEMMs
    after what ``fused_moe_kernel`` classifies to, which closes that. The
    assertion is inverted rather than deleted: it was a tripwire for the gap
    and is worth keeping as one against a regression that reopens it."""
    table = _one_row(tmp_path, "fused_moe_kernel", "moe_routed")

    found = targets(table, load_library(workload="vllm-decode"), top=3)

    assert found[0].row.op == "moe_routed"
    assert found[0].uncovered is False
    assert found[0].levers, "the MoE levers should name this region after #124"


def test_a_dense_region_is_covered_by_the_real_catalog(tmp_path):
    """The same query on an architecture the catalog was written for."""
    table = _one_row(tmp_path, "flash_fwd_kernel", "attn_score_value")

    found = targets(table, load_library(workload="vllm-decode"), top=3)

    assert found[0].uncovered is False
    assert len(found[0].levers) >= 3


# --------------------------------------------------------------------------- #
# it inherits the table's filters                                              #
# --------------------------------------------------------------------------- #
def test_filters_and_top_pass_through_to_the_ranking(tmp_path):
    p = _trace(tmp_path / "t.jsonl", [("flash_fwd_kernel", 5000, 10),
                                      ("fused_moe_kernel", 4000, 10)])
    table = from_trace(p, _graph(_node("attn_score_value", 100e-9),
                                 _node("moe_routed", 100e-9, "compute")), steps=1)
    lib = load_library(workload="vllm-decode")

    assert len(targets(table, lib, top=1)) == 1
    assert targets(table, lib, bound="idle_stall") == []
    assert all(t.row.bound == "compute_bound"
               for t in targets(table, lib, bound="compute_bound"))


def test_nothing_recoverable_renders_as_nothing_rather_than_an_empty_table():
    assert render_targets([]) == "no recoverable time found"
