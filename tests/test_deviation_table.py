"""The typed deviation table: one row per (region, phase), ranked by recoverable time."""

from __future__ import annotations

import json

from gitm.agents.autoresearch import BOTTLENECK_CLASSES
from gitm.optimizer.bound_classes import (
    BOUND_CLASSES,
    COMPUTE_BOUND,
    IDLE_STALL,
    MEMORY_BOUND,
    normalize_bound,
)
from gitm.optimizer.deviation_table import (
    from_deviate_json,
    from_trace,
    rank_by_recoverable,
    render_table,
)
from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, RooflinePrediction


def _node(op, t_pred_s, bound="memory", layer=None):
    return PredictedNode(op=op, layer=layer, prediction=RooflinePrediction(
        op=op, flops=0.0, bytes=0.0, t_compute_s=0.0, t_memory_s=0.0,
        t_pred_s=t_pred_s, bound=bound))


def _graph(*nodes):
    return Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(), nodes=list(nodes))


def _trace(path, rows):
    """rows: (name, duration_ns, repeat)."""
    with open(path, "w", encoding="utf-8") as fh:
        t = 0
        for name, dur, n in rows:
            for _ in range(n):
                fh.write(json.dumps({
                    "kind": "kernel", "name": name, "start_ns": t, "end_ns": t + dur,
                    "stream_id": 7, "device_id": 0,
                }) + "\n")
                t += dur + 100
    return path


def _interleaved(path, steps, prefill=False):
    gdn = "_causal_conv1d_fwd_kernel" if prefill else "_causal_conv1d_update_kernel"
    with open(path, "w", encoding="utf-8") as fh:
        t = 0
        for _ in range(steps):
            for name, dur in ((gdn, 300), ("fused_moe_kernel", 2000),
                              ("nvjet_sm90_tst_128x8_TNT", 400)):
                fh.write(json.dumps({
                    "kind": "kernel", "name": name, "start_ns": t, "end_ns": t + dur,
                    "stream_id": 7, "device_id": 0,
                }) + "\n")
                t += dur + 100
    return path


# --------------------------------------------------------------------------- #
# ranking — the reason this module exists                                     #
# --------------------------------------------------------------------------- #
def test_a_dominant_op_slightly_over_outranks_a_rare_op_hugely_over(tmp_path):
    """The divergence from ``largest_residual``, pinned rather than left to be
    discovered. Ranking by mean fractional overshoot puts a kernel that ran twice
    at 10x above one holding most of the step at 20% over — and sends the next
    experiment at the op with almost no time behind it."""
    p = _trace(tmp_path / "t.jsonl", [
        ("fused_moe_kernel", 1200, 100),   # 120,000 ns observed, floor 100,000 -> +20,000
        ("ncclDevKernel_AllReduce", 1000, 2),   # 2,000 ns observed, floor 200 -> +1,800
    ])
    g = _graph(_node("moe_routed", 100_000e-9), _node("tp_all_reduce", 200e-9))

    ranked = rank_by_recoverable(from_trace(p, g, steps=1).rows)

    assert ranked[0].op == "moe_routed"
    assert ranked[0].recoverable_ms > ranked[1].recoverable_ms
    # ... and the ratio-based view disagrees, which is the point.
    assert ranked[1].observed_ms / ranked[1].predicted_ms > \
           ranked[0].observed_ms / ranked[0].predicted_ms


def test_unmodeled_work_is_counted_but_never_ranked(tmp_path):
    """Unmodeled kernels are the graph's coverage gap, not headroom. Reading them
    as time to recover is the error the modeled split exists to prevent."""
    p = _trace(tmp_path / "t.jsonl", [
        ("fused_moe_kernel", 1000, 10),
        ("some_kernel_nobody_models", 5000, 10),
    ])
    t = from_trace(p, _graph(_node("moe_routed", 1000e-9)), steps=1)

    assert any(not r.modeled for r in t.rows)
    assert all(r.modeled for r in rank_by_recoverable(t.rows))
    unmodeled = next(r for r in t.rows if not r.modeled)
    assert unmodeled.share_of_device > 0  # still counted in the denominator
    assert unmodeled.recoverable_ms == 0.0
    assert unmodeled.verdict == "unmodeled"


def test_observed_under_the_floor_is_a_defect_not_headroom(tmp_path):
    p = _trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 100, 10)])
    t = from_trace(p, _graph(_node("moe_routed", 10_000e-9)), steps=1)

    row = next(r for r in t.rows if r.op == "moe_routed")
    assert row.gap_ms < 0                 # the sign survives
    assert row.recoverable_ms == 0.0      # but it is not recoverable time
    assert row.verdict == "below_floor"
    assert row not in rank_by_recoverable(t.rows)


def test_filters_select_one_cell_for_a_spread_batch(tmp_path):
    p = _interleaved(tmp_path / "t.jsonl", 20, prefill=True)
    g = _graph(_node("moe_routed", 1e-6, "memory"),
               _node("linattn_conv", 1e-7, "compute"))
    rows = from_trace(p, g, steps=20).rows

    assert all(r.bound == MEMORY_BOUND for r in rank_by_recoverable(rows, bound=MEMORY_BOUND))
    assert all(r.phase == "prefill" for r in rank_by_recoverable(rows, phase="prefill"))
    assert rank_by_recoverable(rows, phase="decode") == []


# --------------------------------------------------------------------------- #
# phase                                                                        #
# --------------------------------------------------------------------------- #
def test_phase_confidence_separates_observed_from_inherited(tmp_path):
    """Roughly three quarters of device time is byte-identical across phases, so
    its phase is inherited from a neighbour. A row that says 'prefill' at 0
    confidence is a guess, and the type has to say so."""
    p = _interleaved(tmp_path / "t.jsonl", 20, prefill=True)
    g = _graph(_node("moe_routed", 1e-6), _node("linattn_conv", 1e-7))
    rows = {r.op: r for r in from_trace(p, g, steps=20).rows}

    assert rows["linattn_conv"].phase == "prefill"
    assert rows["linattn_conv"].phase_confidence == 1.0   # named its own phase
    assert rows["moe_routed"].phase == "prefill"
    assert rows["moe_routed"].phase_confidence == 0.0     # inherited from anchors


def test_one_op_spanning_two_phases_splits_into_two_rows(tmp_path):
    with open(tmp_path / "t.jsonl", "w", encoding="utf-8") as fh:
        t = 0
        for gdn in ("_causal_conv1d_fwd_kernel", "_causal_conv1d_update_kernel"):
            for name, dur in ((gdn, 300), ("fused_moe_kernel", 2000)):
                fh.write(json.dumps({"kind": "kernel", "name": name, "start_ns": t,
                                     "end_ns": t + dur, "stream_id": 7,
                                     "device_id": 0}) + "\n")
                t += dur + 100

    rows = from_trace(tmp_path / "t.jsonl", _graph(_node("moe_routed", 1e-6)), steps=1).rows
    moe = [r for r in rows if r.op == "moe_routed"]

    assert {r.phase for r in moe} == {"prefill", "decode"}
    assert sum(r.observed_ms for r in moe) == 2 * 2000 / 1e6


def test_phase_is_unknown_rather_than_guessed_without_anchors(tmp_path):
    p = _trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 10)])
    rows = from_trace(p, _graph(_node("moe_routed", 100e-9)), steps=1).rows

    assert all(r.phase == "unknown" for r in rows)


# --------------------------------------------------------------------------- #
# floors                                                                       #
# --------------------------------------------------------------------------- #
def test_a_graph_without_steps_refuses_rather_than_comparing_one_step_to_a_window(tmp_path):
    """`gitm deviate --as-json` has this bug live: pred * (steps or 1), no warning."""
    import pytest

    p = _trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 10)])
    with pytest.raises(ValueError, match="steps is required"):
        from_trace(p, _graph(_node("moe_routed", 100e-9)), steps=None)


def test_steps_scales_the_floor(tmp_path):
    p = _trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 10)])
    g = _graph(_node("moe_routed", 100e-9))

    one = next(r for r in from_trace(p, g, steps=1).rows if r.op == "moe_routed")
    ten = next(r for r in from_trace(p, g, steps=10).rows if r.op == "moe_routed")

    assert ten.predicted_ms == one.predicted_ms * 10
    assert ten.observed_ms == one.observed_ms  # the trace did not change


def test_two_graphs_measure_each_phase_against_its_own_prediction(tmp_path):
    p = _interleaved(tmp_path / "t.jsonl", 20, prefill=True)
    both = {"prefill": _graph(_node("moe_routed", 1e-6)),
            "decode": _graph(_node("moe_routed", 5e-7))}

    rows = from_trace(p, both, steps=20).rows
    moe = next(r for r in rows if r.op == "moe_routed")

    assert moe.floor_attribution == "graph"
    # one graph for both phases is an assumption, and says so
    single = next(r for r in from_trace(p, both["prefill"], steps=20).rows
                  if r.op == "moe_routed")
    assert single.floor_attribution == "prorata"


def test_the_bound_comes_through_and_flags_disagreeing_layers(tmp_path):
    p = _trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 10)])
    mixed = _graph(_node("moe_routed", 100e-9, "memory", layer=0),
                   _node("moe_routed", 50e-9, "compute", layer=1))

    row = next(r for r in from_trace(p, mixed, steps=1).rows if r.op == "moe_routed")

    assert row.roofline_bound == "memory"       # holds the most predicted time
    assert row.bound == MEMORY_BOUND
    assert row.bound_mixed is True


# --------------------------------------------------------------------------- #
# the existing artifact                                                        #
# --------------------------------------------------------------------------- #
def test_deviate_json_rehydrates_without_inventing_phase_or_bound():
    """The payload carries neither, so the rows say so rather than guessing."""
    t = from_deviate_json({
        "n_kernels": 14, "device_time_s": 0.001, "window_s": 1.0,
        "steps": 100, "band_width": 0.4,
        "ops": {"moe_routed": {"kernels": 10, "observed_s": 0.0008, "floor_s": 0.0004},
                "<unmodeled>": {"kernels": 4, "observed_s": 0.0002, "floor_s": None}},
    })
    row = next(r for r in t.rows if r.op == "moe_routed")

    assert row.phase == "unknown"
    assert row.bound is None
    assert abs(row.recoverable_ms - 0.4) < 1e-9
    assert any(not r.modeled for r in t.rows)


# --------------------------------------------------------------------------- #
# the vocabulary                                                               #
# --------------------------------------------------------------------------- #
def test_launch_normalizes_to_idle_stall():
    """Both name time the GPU spent not doing the op's work."""
    assert normalize_bound("launch") == IDLE_STALL
    assert normalize_bound("compute") == COMPUTE_BOUND
    assert normalize_bound("memory") == MEMORY_BOUND


def test_an_unknown_bound_is_none_rather_than_a_default():
    """A new roofline bound must surface as missing, not land silently in
    whichever class happened to be the fallback."""
    assert normalize_bound("quantum") is None
    assert normalize_bound(None) is None


def test_the_two_vocabularies_cannot_drift_apart():
    assert set(BOUND_CLASSES) == set(BOTTLENECK_CLASSES)


def test_render_names_the_rows_and_admits_inferred_phase(tmp_path):
    p = _interleaved(tmp_path / "t.jsonl", 20, prefill=True)
    out = render_table(from_trace(p, _graph(_node("moe_routed", 1e-6)), steps=20))

    assert "moe_routed" in out
    assert "inferred" in out
