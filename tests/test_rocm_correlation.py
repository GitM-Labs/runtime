"""ROCm identity: collector-stamped ranges and validated HIP-graph projection.

Contract in gitm/distributed/correlate.py ("Stamped ranges", "HIP graphs").
Each test pins one rule with hand-written records; the end-to-end accuracy of
the rules together is measured against emulated executions in
tests/test_identity_accuracy.py.
"""

from __future__ import annotations

import json
import warnings

import pytest

from gitm.distributed.correlate import (
    GRAPH_UNTRACKED,
    capture_node_id,
    correlate_records,
    exec_node_id,
    split_range_annotations,
)
from gitm.tracer import injection
from gitm.tracer._cupti_decode import decode_records_with_report, pair_markers


def _marker(mid, name, start, end, thread=1):
    return {"kind": "marker", "marker_id": mid, "name": name, "start_ns": start,
            "end_ns": end, "thread_id": thread}


def _rt(corr, start, end, thread=1, **kw):
    return {"kind": "runtime", "correlation_id": corr, "start_ns": start, "end_ns": end,
            "thread_id": thread, **kw}


def _k(corr, t=0, name="Cijk_Alik_Bljk_BBS", **kw):
    return {"kind": "kernel", "name": name, "correlation_id": corr, "start_ns": t,
            "end_ns": t + 10, "stream_id": 1, "device_id": 0, "grid": [8, 1, 1],
            "block": [256, 1, 1], **kw}


def _ops(out):
    return [(o["range_op"], o["range_layer"]) for o in out]


# ── stamped ranges ──────────────────────────────────────────────────────────


def test_a_stamp_names_the_kernel_without_any_runtime_record():
    """The join needs no host window and no thread: the id is the range."""
    out, _, rep = correlate_records([_marker(5, "L3/mlp_down", 0, 100), _k(1, range_id=5)])
    assert _ops(out) == [("mlp_down", 3)]
    assert out[0]["identity"] == "range_id" and rep.identity["range_id"] == 1


def test_a_stamp_survives_a_launch_from_another_thread():
    """Containment matches on thread_id; a helper-thread launch whose range was
    pushed on that same helper thread is what the stamp records."""
    recs = [_marker(5, "L0/qkv_proj", 0, 100, thread=2), _rt(1, 10, 20, thread=2),
            _k(1, range_id=5)]
    assert _ops(correlate_records(recs)[0]) == [("qkv_proj", 0)]


def test_stamp_wins_over_containment_and_the_disagreement_is_counted():
    recs = [_marker(5, "L1/attn_out_proj", 0, 100), _marker(6, "L9/mlp_down", 0, 100, thread=1),
            _rt(1, 10, 20), _k(1, range_id=5)]
    # Containment picks the innermost (equal span -> either); make it unambiguous.
    recs[1]["start_ns"], recs[1]["end_ns"] = 5, 50
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [("attn_out_proj", 1)]
    assert rep.stamp_containment_disagree == 1
    assert any("disagree" in p for p in rep.problems())


def test_an_unresolved_stamp_falls_back_to_containment():
    """Range pushed before the window armed: its id reached the kernel but its
    START half never reached the shard."""
    recs = [_marker(7, "L2/qkv_proj", 0, 100), _rt(1, 10, 20), _k(1, range_id=99)]
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [("qkv_proj", 2)] and out[0]["identity"] == "containment"
    assert rep.stamp_unresolved == 1


def test_an_end_only_range_names_nothing():
    """pair_markers drops an END without its START; the stamp then resolves to
    nothing rather than to the next range that happens to enclose it."""
    halves = [{"kind": "marker", "marker_id": 5, "marker_flags": 1, "timestamp_ns": 90,
               "thread_id": 1, "name": None},
              _rt(1, 10, 20), _k(1, range_id=5)]
    out, _, _ = correlate_records(pair_markers(halves))
    assert _ops(out) == [(None, None)] and out[0]["identity"] is None


def test_marker_halves_pair_whatever_order_buffers_arrive_in():
    halves = [
        {"kind": "marker", "marker_id": 1, "marker_flags": 1, "timestamp_ns": 90, "thread_id": 1},
        {"kind": "marker", "marker_id": 1, "marker_flags": 0, "timestamp_ns": 10, "thread_id": 1,
         "name": "L0/qkv_proj"},
        {"kind": "marker", "marker_id": 2, "marker_flags": 0, "timestamp_ns": 50, "thread_id": 1,
         "name": "point"},
        {"kind": "marker", "marker_id": 2, "marker_flags": 1, "timestamp_ns": 50, "thread_id": 1},
    ]
    got = {m["marker_id"]: (m["name"], m["start_ns"], m["end_ns"]) for m in pair_markers(halves)}
    assert got == {1: ("L0/qkv_proj", 10, 90), 2: ("point", 50, 50)}


# ── annotations ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,base,attrs", [
    ("L3/moe_routed#phase=expert,wave=2", "L3/moe_routed", {"phase": "expert", "wave": "2"}),
    ("L3/moe_routed", "L3/moe_routed", None),
    ("L3/moe_routed#", "L3/moe_routed", None),
    ("L3/moe_routed#bad pair,wave=1,=x,y=", "L3/moe_routed", {"wave": "1"}),
    ("{'Module': 'model.layers.0#1'}", "{'Module': 'model.layers.0#1'}", None),
])
def test_split_range_annotations(raw, base, attrs):
    assert split_range_annotations(raw) == (base, attrs)


def test_an_annotation_never_reaches_the_op():
    recs = pair_markers([
        {"kind": "marker", "marker_id": 1, "marker_flags": 0, "timestamp_ns": 0,
         "thread_id": 1, "name": "L4/moe_routed#wave=1"},
        {"kind": "marker", "marker_id": 1, "marker_flags": 1, "timestamp_ns": 100,
         "thread_id": 1},
        _k(1, range_id=1),
    ])
    out, _, _ = correlate_records(recs)
    assert _ops(out) == [("moe_routed", 4)] and out[0]["range_attrs"] == {"wave": "1"}


# ── HIP graphs ──────────────────────────────────────────────────────────────


def _node(cap, k, name, kind="kernel", **kw):
    return {"kind": "graph_node", "graph_node_id": capture_node_id(cap, k), "capture_id": cap,
            "node_kind": kind, "name": name, **kw}


def _gk(exec_seq, k, corr=50, t=None, **kw):
    return _k(corr, t=1000 + 20 * k if t is None else t, graph_id=exec_seq,
              graph_node_id=exec_node_id(exec_seq, k), range_id=0, **kw)


def _graph(n=3, cap=4, ex=2, names=("L0/qkv_proj", "L0/attn_out_proj", "L0/mlp_down"), **nkw):
    nodes = [_node(cap, i, names[i], kernel_id=100 + i, grid=[8 + i, 1, 1],
                   block=[256, 1, 1], **nkw) for i in range(n)]
    return [*nodes, {"kind": "graph_exec", "graph_id": ex, "capture_id": cap, "n_nodes": n},
            _marker(1, "decode_step", 0, 10_000), _rt(50, 100, 120, graph_launch=1, range_id=1)]


def _gks(ex=2, n=3, **kw):
    return [_gk(ex, i, kernel_id=100 + i, grid=[8 + i, 1, 1], **kw) for i in range(n)]


def test_replayed_kernels_take_the_range_their_node_was_captured_under():
    out, _, rep = correlate_records([*_graph(), *_gks()])
    assert _ops(out) == [("qkv_proj", 0), ("attn_out_proj", 0), ("mlp_down", 0)]
    assert {o["launch_range"] for o in out} == {"decode_step"}
    assert rep.identity["graph_node"] == 3 and not rep.graph_refused


def test_capture_node_names_are_normalized_like_markers():
    """vLLM's layerwise ranges arrive as dict reprs; a capture node named by one
    must resolve exactly as a marker of that name does."""
    vllm = "{'Module': 'model.layers.7.mlp.down_proj', 'shape': [1, 7168]}"
    recs = [_node(1, 0, vllm, kernel_id=9, grid=[8, 1, 1], block=[256, 1, 1]),
            {"kind": "graph_exec", "graph_id": 3, "capture_id": 1, "n_nodes": 1},
            _gk(3, 0, kernel_id=9)]
    ev, _ = decode_records_with_report(recs)
    assert (ev[0].range_op, ev[0].range_layer) == ("mlp_down", 7)


@pytest.mark.parametrize("mutate,reason", [
    (lambda ks: ks[1].update(kernel_id=999), "signature_kernel_id"),
    (lambda ks: ks[2].update(grid=[1, 1, 1]), "signature_grid"),
    (lambda ks: [k.update(graph_node_id=exec_node_id(2, 0)) for k in ks], "duplicate_ordinal"),
    (lambda ks: ks[2].update(graph_node_id=exec_node_id(2, 7)), "ordinal_out_of_range"),
])
def test_a_replay_that_fails_validation_is_refused_whole(mutate, reason):
    ks = _gks()
    mutate(ks)
    out, _, rep = correlate_records([*_graph(), *ks])
    assert _ops(out) == [(None, None)] * 3
    assert rep.graph_refused[reason] == 3
    assert {o["launch_range"] for o in out} == {"decode_step"}


def test_one_bad_replay_does_not_refuse_the_others():
    good = _gks()
    bad = [_gk(2, i, corr=51, t=5000 + i, kernel_id=100 + i, grid=[8 + i, 1, 1])
           for i in range(3)]
    bad[0]["kernel_id"] = 7
    out, _, rep = correlate_records([*_graph(), _rt(51, 200, 220, graph_launch=1),
                                     *good, *bad])
    assert _ops(out)[:3] == [("qkv_proj", 0), ("attn_out_proj", 0), ("mlp_down", 0)]
    assert _ops(out)[3:] == [(None, None)] * 3


def test_a_memset_node_may_replay_as_a_rocclr_blit_or_a_copy():
    recs = [*_graph(n=2), _node(4, 2, "", kind="memset", kernel_id=0)]
    recs[2]["n_nodes"] = 3
    blit = _gk(2, 2, name="__amd_rocclr_fillBufferAligned", kernel_id=555, grid=[1, 1, 1])
    out, _, rep = correlate_records([*recs, *_gks(n=2), blit])
    assert _ops(out)[:2] == [("qkv_proj", 0), ("attn_out_proj", 0)] and not rep.graph_refused


def test_a_memset_node_replayed_as_an_ordinary_kernel_means_the_ordinals_shifted():
    recs = [*_graph(n=2), _node(4, 2, "", kind="memset", kernel_id=0)]
    recs[2]["n_nodes"] = 3
    out, _, rep = correlate_records([*recs, *_gks(n=2), _gk(2, 2, kernel_id=1)])
    assert rep.graph_refused["signature_copy_node_ran_kernel"] == 3


def test_module_launch_nodes_validate_on_geometry_alone():
    """hipModuleLaunchKernel nodes carry kernel_id 0 (no symbol map exists for
    a hipFunction_t); geometry is still compared."""
    recs = _graph()
    for r in recs[:3]:
        r["kernel_id"] = 0
    ks = _gks()
    assert not correlate_records([*recs, *ks])[2].graph_refused
    ks[1]["grid"] = [99, 1, 1]
    assert correlate_records([*recs, *ks])[2].graph_refused["signature_grid"] == 3


def test_a_graph_launch_the_stamp_missed_never_takes_the_launch_range():
    """The kernel arrived unstamped, but its launch record is a hipGraphLaunch:
    taking the range around it as the op is the bug #160 fixed on NVIDIA."""
    recs = [_marker(1, "L0/qkv_proj", 0, 10_000), _rt(50, 100, 120, graph_launch=1),
            _k(50, t=1000, range_id=0)]
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [(None, None)]
    assert out[0]["graph_id"] == GRAPH_UNTRACKED and out[0]["launch_range"] == "L0/qkv_proj"
    assert rep.graph_refused["untracked_launch"] == 1


def test_the_guard_runs_before_the_stamped_join():
    """A replayed dispatch that inherited the hipGraphLaunch call's stamp
    carries the launch range's real marker id. Joining it would name every
    replayed kernel after the range around the launch."""
    recs = [_marker(1, "L0/qkv_proj", 0, 10_000), _rt(50, 100, 120, graph_launch=1, range_id=1),
            _k(50, t=1000, range_id=1), _k(50, t=1020, range_id=1)]
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [(None, None)] * 2
    assert {o["launch_range"] for o in out} == {"L0/qkv_proj"}
    assert rep.graph_refused["untracked_launch"] == 2 and not rep.identity["range_id"]


def test_an_exec_built_without_capture_names_nothing():
    recs = [{"kind": "graph_exec", "graph_id": 2, "capture_id": 0}, _rt(50, 1, 2, graph_launch=1),
            *_gks()]
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [(None, None)] * 3 and rep.graph_refused["no_node"] == 3


def test_a_node_captured_outside_every_range_is_not_a_refusal():
    recs = _graph()
    recs[1]["name"] = ""
    out, _, rep = correlate_records([*recs, *_gks()])
    assert _ops(out)[1] == (None, None) and not rep.graph_refused


def test_reinstantiation_gets_a_new_exec_and_both_resolve():
    recs = [*_graph(), {"kind": "graph_exec", "graph_id": 3, "capture_id": 4, "n_nodes": 3}]
    ks = [*_gks(ex=2), *[_gk(3, i, corr=60, t=9000 + i, kernel_id=100 + i, grid=[8 + i, 1, 1])
                         for i in range(3)]]
    out, _, rep = correlate_records([*recs, _rt(60, 300, 320, graph_launch=1), *ks])
    assert _ops(out)[:3] == _ops(out)[3:] == [("qkv_proj", 0), ("attn_out_proj", 0),
                                              ("mlp_down", 0)]


def test_cupti_replays_without_signatures_are_not_refused():
    """The CUPTI collector's nodes carry no signature; validating on fields a
    record never had would refuse every NVIDIA replay."""
    recs = [{"kind": "graph_node", "graph_node_id": 41, "name": "L0/qkv_proj"},
            _rt(9, 1, 2), _k(9, graph_id=3, graph_node_id=41)]
    out, _, rep = correlate_records(recs)
    assert _ops(out) == [("qkv_proj", 0)] and not rep.graph_refused


# ── memcpy labels ───────────────────────────────────────────────────────────


def test_memcpys_get_the_step_range_by_stamp_or_containment():
    recs = [_marker(1, "decode_step", 0, 100), _rt(1, 10, 20),
            {"kind": "memcpy", "correlation_id": 1, "start_ns": 5, "end_ns": 6, "range_id": 0},
            {"kind": "memcpy", "correlation_id": 77, "start_ns": 7, "end_ns": 8, "range_id": 1}]
    _, copies, rep = correlate_records(recs)
    assert [c["launch_range"] for c in copies] == ["decode_step", "decode_step"]
    assert rep.memcpys_labelled == 2


def test_decoded_memcpy_events_carry_the_label():
    recs = [{"kind": "marker", "marker_id": 1, "marker_flags": 0, "timestamp_ns": 0,
             "thread_id": 1, "name": "decode_step"},
            {"kind": "marker", "marker_id": 1, "marker_flags": 1, "timestamp_ns": 100,
             "thread_id": 1},
            {"kind": "memcpy", "copy_kind": 1, "bytes": 8, "correlation_id": 3, "start_ns": 5,
             "end_ns": 6, "device_id": 0, "stream_id": 0, "range_id": 1}]
    (ev,), _ = decode_records_with_report(recs)
    assert ev.kind == "memcpy" and ev.launch_range == "decode_step"


# ── through the shard reader ────────────────────────────────────────────────


def test_read_shards_keeps_pre_arm_graph_structure(tmp_path, monkeypatch):
    """Graph nodes are written at engine start, before any window. A windowed
    reader would drop them and leave every replay anonymous."""
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv(injection.ENV_OUT, str(out))
    early = [*_graph()]
    # Structural records carry no start_ns at all; the window is far later.
    window_kernels = _gks()
    shard = tmp_path / "trace.jsonl.4242"
    shard.write_text("\n".join(json.dumps(r) for r in [*early, *window_kernels]) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        events = injection.read_shards(start_ns=500, end_ns=50_000)
    assert [(e.range_op, e.range_layer) for e in events] == [
        ("qkv_proj", 0), ("attn_out_proj", 0), ("mlp_down", 0)]


def test_read_shards_warns_on_refused_replays_and_stamp_faults(tmp_path, monkeypatch):
    out = tmp_path / "trace.jsonl"
    monkeypatch.setenv(injection.ENV_OUT, str(out))
    ks = _gks()
    ks[0]["kernel_id"] = 1
    recs = [*_graph(), *ks, {"kind": "meta", "graph_stamp_overflow": 0,
                             "graph_untracked_launch": 2}]
    (tmp_path / "trace.jsonl.7").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    with pytest.warns(RuntimeWarning) as caught:
        injection.read_shards()
    msgs = " ".join(str(w.message) for w in caught)
    assert "refused node identity" in msgs and "graph_untracked_launch" in msgs


def test_read_shards_warns_when_two_collectors_share_a_directory(tmp_path, monkeypatch):
    monkeypatch.setenv(injection.ENV_OUT, str(tmp_path / "trace.jsonl"))
    (tmp_path / "trace.jsonl.1").write_text(json.dumps(
        {"kind": "meta", "collector": "rocprofiler-sdk"}) + "\n")
    (tmp_path / "trace.jsonl.2").write_text(json.dumps(
        {"kind": "meta", "collector": "cupti"}) + "\n")
    with pytest.warns(RuntimeWarning, match="more than one collector"):
        injection.read_shards()
