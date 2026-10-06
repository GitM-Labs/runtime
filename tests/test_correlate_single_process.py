"""correlate_kernels_to_ranges: kernel -> runtime (by correlation_id) -> NVTX
range (by host-timestamp containment on the runtime record, never the
kernel's own device-clock window). See docs/kernel_identity.md.
"""

from __future__ import annotations

from gitm.distributed.correlate import correlate_kernels_to_ranges, parse_range_name


def _kernel(name: str, corr_id: int, dev_start: int, dev_end: int) -> dict:
    return {
        "kind": "kernel", "name": name, "correlation_id": corr_id,
        "start_ns": dev_start, "end_ns": dev_end,
    }


def _runtime(corr_id: int, thread_id: int, host_start: int, host_end: int) -> dict:
    return {
        "kind": "runtime", "correlation_id": corr_id, "thread_id": thread_id,
        "start_ns": host_start, "end_ns": host_end,
    }


def _marker(name: str, thread_id: int, start: int, end: int) -> dict:
    return {"kind": "marker", "name": name, "thread_id": thread_id, "start_ns": start, "end_ns": end}


def test_parse_range_name():
    assert parse_range_name("L3/qkv_proj") == ("qkv_proj", 3)
    assert parse_range_name("L0/attn_out_proj") == ("attn_out_proj", 0)
    assert parse_range_name("lm_head") == ("lm_head", None)


def test_full_chain_correlates_by_correlation_id_and_host_containment():
    # Device-clock kernel window (100..900) is *outside* the host-clock range
    # window (1000..1100) on purpose -- async execution. Containment must go
    # through the runtime record's host window (1000..1050), not the kernel's.
    records = [
        _marker("L2/qkv_proj", thread_id=1, start=1000, end=1100),
        _runtime(corr_id=7, thread_id=1, host_start=1000, host_end=1050),
        _kernel("ampere_fp16_s16816gemm_128x128", corr_id=7, dev_start=100, dev_end=900),
    ]
    out = correlate_kernels_to_ranges(records)
    assert len(out) == 1
    assert out[0]["range_op"] == "qkv_proj"
    assert out[0]["range_layer"] == 2
    # original kernel fields preserved
    assert out[0]["name"] == "ampere_fp16_s16816gemm_128x128"


def test_no_runtime_record_leaves_range_unset():
    records = [_kernel("mystery_kernel", corr_id=1, dev_start=0, dev_end=10)]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None
    assert out[0]["range_layer"] is None


def test_runtime_outside_every_marker_leaves_range_unset():
    records = [
        _marker("L0/qkv_proj", thread_id=1, start=0, end=100),
        _runtime(corr_id=1, thread_id=1, host_start=200, host_end=250),  # after the range closed
        _kernel("k", corr_id=1, dev_start=200, dev_end=300),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None


def test_different_thread_marker_is_not_matched():
    records = [
        _marker("L0/qkv_proj", thread_id=2, start=0, end=1000),  # different thread
        _runtime(corr_id=1, thread_id=1, host_start=100, host_end=150),
        _kernel("k", corr_id=1, dev_start=100, dev_end=150),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] is None


def test_nested_ranges_pick_innermost():
    records = [
        _marker("L1/mlp_gate_up", thread_id=1, start=0, end=1000),   # outer
        _marker("silu_and_mul", thread_id=1, start=100, end=200),     # inner, no layer prefix
        _runtime(corr_id=1, thread_id=1, host_start=120, host_end=150),
        _kernel("k", corr_id=1, dev_start=120, dev_end=150),
    ]
    out = correlate_kernels_to_ranges(records)
    assert out[0]["range_op"] == "silu_and_mul"
    assert out[0]["range_layer"] is None


def test_multiple_kernels_preserve_order_and_independent_matches():
    records = [
        _marker("L0/qkv_proj", thread_id=1, start=0, end=100),
        _runtime(corr_id=1, thread_id=1, host_start=10, host_end=20),
        _kernel("first", corr_id=1, dev_start=10, dev_end=20),
        _marker("L0/attn_out_proj", thread_id=1, start=100, end=200),
        _runtime(corr_id=2, thread_id=1, host_start=110, host_end=120),
        _kernel("second", corr_id=2, dev_start=110, dev_end=120),
    ]
    out = correlate_kernels_to_ranges(records)
    assert [o["name"] for o in out] == ["first", "second"]
    assert out[0]["range_op"] == "qkv_proj"
    assert out[1]["range_op"] == "attn_out_proj"


# ── NVTX marker pairing ─────────────────────────────────────────────────────
#
# CUpti_ActivityMarker2 carries one timestamp, so a range arrives as two records
# sharing a marker_id, and the header states the name is NULL on the end. The C
# side stays stateless and emits both halves; pair_markers is where they become
# the {name, start_ns, end_ns, thread_id} range correlate.py documents.


def _start(mid, name, t, thread=7):
    return {"kind": "marker", "name": name, "timestamp_ns": t,
            "marker_id": mid, "marker_flags": 0, "thread_id": thread}


def _end(mid, t, thread=7):
    return {"kind": "marker", "name": "", "timestamp_ns": t,
            "marker_id": mid, "marker_flags": 1, "thread_id": thread}


def test_a_push_and_pop_become_one_range():
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([_start(1, "L0/qkv_proj", 100), _end(1, 120)])
    # marker_id survives pairing: it is the join key for collector-stamped
    # kernels (correlate.py, "Stamped ranges"). attrs is the split-off
    # annotation, None when the name carries none.
    assert got == [{"kind": "marker", "name": "L0/qkv_proj", "attrs": None,
                    "start_ns": 100, "end_ns": 120, "thread_id": 7, "marker_id": 1}]


def test_the_name_comes_from_the_start_half():
    """The end record's name is NULL by CUPTI's contract. Taking the name from
    whichever half arrived last would produce anonymous ranges."""
    from gitm.tracer._cupti_decode import pair_markers

    (range_,) = pair_markers([_start(1, "L3/moe_routed", 10), _end(1, 20)])
    assert range_["name"] == "L3/moe_routed"


def test_an_unclosed_range_is_dropped_not_extended():
    """A capture killed mid-step leaves a start with no end — the common case,
    not a corner one. Closing it at the capture's end would produce a range that
    appears to contain every launch after it and would silently claim them all."""
    from gitm.tracer._cupti_decode import pair_markers

    assert pair_markers([_start(1, "L0/a", 10)]) == []


def test_an_unopened_end_is_dropped():
    from gitm.tracer._cupti_decode import pair_markers

    assert pair_markers([_end(1, 10)]) == []


def test_ranges_pair_within_a_thread_not_across_them():
    """A marker id reused on another thread would otherwise splice two ranges
    into one window spanning both."""
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([
        _start(1, "L0/a", 10, thread=1), _start(1, "L0/b", 12, thread=2),
        _end(1, 20, thread=2), _end(1, 30, thread=1),
    ])
    by_thread = {r["thread_id"]: (r["name"], r["start_ns"], r["end_ns"]) for r in got}
    assert by_thread == {1: ("L0/a", 10, 30), 2: ("L0/b", 12, 20)}


def test_nested_ranges_survive_pairing():
    from gitm.tracer._cupti_decode import pair_markers

    got = pair_markers([
        _start(1, "L0/layer", 0), _start(2, "L0/qkv_proj", 10),
        _end(2, 20), _end(1, 100),
    ])
    assert {(r["name"], r["start_ns"], r["end_ns"]) for r in got} == {
        ("L0/layer", 0, 100), ("L0/qkv_proj", 10, 20),
    }


def test_non_marker_records_pass_through_in_order():
    from gitm.tracer._cupti_decode import pair_markers

    k = {"kind": "kernel", "name": "x", "start_ns": 1, "end_ns": 2}
    rt = {"kind": "runtime", "start_ns": 1, "end_ns": 2, "correlation_id": 9,
          "thread_id": 7}
    assert pair_markers([k, rt]) == [k, rt]


def test_an_anonymous_gemm_resolves_to_layer_and_op_through_the_full_chain():
    """The acceptance criterion for the whole NVTX path.

    `nvjet_sm90_tst_*` is cuBLAS's JIT GEMM family; its name carries no
    projection, so classify_op declines it and always will. Correlation recovers
    the identity that name matching cannot.
    """
    from gitm.optimizer.deviation import classify_op
    from gitm.tracer._cupti_decode import decode_records

    name = "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT"
    assert classify_op(name) is None  # unattributable by name, by construction

    events = decode_records([
        _start(1, "L0/qkv_proj", 100),
        {"kind": "runtime", "start_ns": 105, "end_ns": 108,
         "correlation_id": 42, "thread_id": 7},
        _end(1, 120),
        {"kind": "kernel", "name": name, "start_ns": 200, "end_ns": 900,
         "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": 42},
    ])
    (kernel,) = [e for e in events if e.kind == "kernel"]
    assert (kernel.range_op, kernel.range_layer) == ("qkv_proj", 0)


# ── vLLM's own layerwise NVTX instrumentation ───────────────────────────────
#
# `--enable-layerwise-nvtx-tracing` emits a range per module, named with the
# repr of a dict rather than a bare string:
#
#   {'Module': 'Qwen3_5Moe...model.layers.1.mlp.experts', 'Outputs': [[7, 2048]]}
#
# Captured from a real H200 run. This is strictly better than instrumenting the
# model ourselves — it runs inside EngineCore with no plugin, and it covers every
# module rather than the ones our table happens to name. All we owe it is a
# translation into the range vocabulary correlation already speaks.

_QWEN = "Qwen3_5MoeForConditionalGeneration.language_model.model.layers"


def _vllm_range(path: str, extra: str = "'Outputs': [[7, 2048]]") -> str:
    return f"{{'Module': '{path}', {extra}}}"


def test_vllm_module_paths_map_to_graph_ops():
    from gitm.tracer._cupti_decode import normalize_range_name

    assert normalize_range_name(_vllm_range(f"{_QWEN}.1.mlp.experts")) == "L1/moe_routed"
    assert normalize_range_name(_vllm_range(f"{_QWEN}.3.self_attn.qkv_proj")) == "L3/qkv_proj"
    assert normalize_range_name(_vllm_range(f"{_QWEN}.0.linear_attn")) == "L0/linattn_recurrent"


def test_an_unmapped_module_keeps_its_own_name_rather_than_being_dropped():
    """Dropping it would leave every kernel inside that module unattributed.
    Keeping it surfaces the op as observed-but-not-predicted, which is a visible
    gap rather than a silent one."""
    from gitm.tracer._cupti_decode import normalize_range_name

    assert normalize_range_name(_vllm_range(f"{_QWEN}.2.unmapped_module")) == "L2/unmapped_module"


def test_the_block_range_is_named_layer_not_its_index():
    """`layers.1` has the index as its leaf, so the naive reading is `L1/1`."""
    from gitm.tracer._cupti_decode import normalize_range_name

    assert normalize_range_name(_vllm_range(f"{_QWEN}.1")) == "L1/layer"


def test_ranges_from_our_own_hooks_pass_through_untouched():
    """Both instrumentation sources must feed one correlation path."""
    from gitm.tracer._cupti_decode import normalize_range_name

    assert normalize_range_name("L3/qkv_proj") == "L3/qkv_proj"
    assert normalize_range_name("lm_head") == "lm_head"
    assert normalize_range_name("") == ""


def test_a_malformed_dict_name_is_left_alone():
    from gitm.tracer._cupti_decode import normalize_range_name

    assert normalize_range_name("{'Outputs': [[7, 2048]]}") == "{'Outputs': [[7, 2048]]}"


def test_truncation_loses_the_shapes_not_the_module_path():
    """These names run long and the collector cuts at GITM_NAME_MAX. The module
    path arrives first, so a truncated name still identifies its op."""
    from gitm.tracer._cupti_decode import normalize_range_name

    full = _vllm_range(f"{_QWEN}.7.mlp.experts", "'Outputs': [[7, 2048]], 'Extra': [[1")
    assert normalize_range_name(full[:120]) == "L7/moe_routed"


def test_an_anonymous_gemm_resolves_through_vllms_own_ranges():
    """The acceptance criterion, using vLLM's instrumentation rather than ours.

    Nested exactly as the H200 capture showed: the block range encloses the mlp
    range, which encloses experts. Innermost wins.
    """
    from gitm.optimizer.deviation import classify_op
    from gitm.tracer._cupti_decode import decode_records

    def mk(mid, path, t, flags=0):
        return {"kind": "marker", "name": _vllm_range(path) if path else "",
                "timestamp_ns": t, "marker_id": mid, "marker_flags": flags,
                "thread_id": 7}

    name = "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT"
    assert classify_op(name) is None

    events = decode_records([
        mk(1, f"{_QWEN}.1", 100),
        mk(2, f"{_QWEN}.1.mlp", 105),
        mk(3, f"{_QWEN}.1.mlp.experts", 110),
        {"kind": "runtime", "start_ns": 112, "end_ns": 114,
         "correlation_id": 9, "thread_id": 7},
        mk(3, None, 120, 1), mk(2, None, 130, 1), mk(1, None, 200, 1),
        {"kind": "kernel", "name": name, "start_ns": 300, "end_ns": 900,
         "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": 9,
         "grid": [1, 1, 1], "block": [1, 1, 1]},
    ])
    (kernel,) = [e for e in events if e.kind == "kernel"]
    assert (kernel.range_op, kernel.range_layer) == ("moe_routed", 1)


# ── CUDA graphs ─────────────────────────────────────────────────────────────
#
# A graph replay is one cudaGraphLaunch, and every kernel it runs carries that
# launch's correlation_id. The range around the launch names the replay, not
# the op, so it must not become range_op. See correlate.py, "CUDA graphs".


def _graph_kernel(name, corr_id, graph_id, node_id, t=0):
    k = _kernel(name, corr_id=corr_id, dev_start=t, dev_end=t + 10)
    k["graph_id"] = graph_id
    k["graph_node_id"] = node_id
    return k


def _graph_node(node_id, name="", cloned_from=None):
    return {"kind": "graph_node", "graph_node_id": node_id, "name": name,
            "cloned_from": cloned_from}


def test_graph_replay_does_not_take_the_launch_range_as_op():
    """Two kernels from one replay share the launch's correlation_id. Before this,
    both got L5/qkv_proj because that was the range open around the launch."""
    records = [
        _marker("L5/qkv_proj", thread_id=1, start=0, end=1000),
        _runtime(corr_id=9, thread_id=1, host_start=100, host_end=120),  # cudaGraphLaunch
        _graph_kernel("nvjet_gemm", corr_id=9, graph_id=3, node_id=41),
        _graph_kernel("fused_moe_kernel", corr_id=9, graph_id=3, node_id=42, t=20),
    ]
    out = correlate_kernels_to_ranges(records)
    assert [o["range_op"] for o in out] == [None, None]
    assert [o["range_layer"] for o in out] == [None, None]
    assert [o["launch_range"] for o in out] == ["L5/qkv_proj", "L5/qkv_proj"]


def test_graph_node_map_names_each_replayed_kernel():
    records = [
        _marker("decode_step", thread_id=1, start=0, end=1000),
        _runtime(corr_id=9, thread_id=1, host_start=100, host_end=120),
        _graph_node(41, "L0/qkv_proj"),
        _graph_node(42, "L0/moe_routed"),
        _graph_kernel("nvjet_gemm", corr_id=9, graph_id=3, node_id=41),
        _graph_kernel("fused_moe_kernel", corr_id=9, graph_id=3, node_id=42, t=20),
    ]
    out = correlate_kernels_to_ranges(records)
    assert [(o["range_op"], o["range_layer"]) for o in out] == [
        ("qkv_proj", 0), ("moe_routed", 0)]
    assert {o["launch_range"] for o in out} == {"decode_step"}


def test_graph_node_resolves_through_clones():
    """Whether a kernel reports the captured node or its instantiated clone is
    not assumed; both resolve to the range the original was captured under."""
    records = [
        _graph_node(41, "L2/attn_out_proj"),
        _graph_node(900, cloned_from=41),
        _graph_node(901, cloned_from=900),
        _graph_kernel("k_orig", corr_id=1, graph_id=3, node_id=41),
        _graph_kernel("k_clone", corr_id=2, graph_id=4, node_id=901),
    ]
    out = correlate_kernels_to_ranges(records)
    assert [(o["range_op"], o["range_layer"]) for o in out] == [
        ("attn_out_proj", 2), ("attn_out_proj", 2)]


def test_graph_node_clone_cycle_resolves_to_none():
    records = [
        _graph_node(1, cloned_from=2),
        _graph_node(2, cloned_from=1),
        _graph_kernel("k", corr_id=1, graph_id=3, node_id=1),
    ]
    assert correlate_kernels_to_ranges(records)[0]["range_op"] is None


def test_graph_id_zero_is_an_eager_launch():
    """CUPTI reports graphId 0 for a kernel launched outside a graph."""
    records = [
        _marker("L1/qkv_proj", thread_id=1, start=0, end=100),
        _runtime(corr_id=1, thread_id=1, host_start=10, host_end=20),
        _graph_kernel("k", corr_id=1, graph_id=0, node_id=0),
    ]
    out = correlate_kernels_to_ranges(records)
    assert (out[0]["range_op"], out[0]["range_layer"]) == ("qkv_proj", 1)
    assert "launch_range" not in out[0]


def test_decode_carries_graph_identity_and_maps_zero_to_none():
    from gitm.tracer._cupti_decode import decode_records

    base = {"kind": "kernel", "start_ns": 0, "end_ns": 10, "device_id": 0,
            "stream_id": 7, "grid": [1, 1, 1], "block": [1, 1, 1]}
    records = [
        _start(1, "decode_step", 0, thread=1),
        _end(1, 1000, thread=1),
        _runtime(corr_id=9, thread_id=1, host_start=100, host_end=120),
        {**base, "name": "g", "correlation_id": 9, "graph_id": 3, "graph_node_id": 41},
        {**base, "name": "e", "correlation_id": 10, "graph_id": 0, "graph_node_id": 0,
         "start_ns": 20, "end_ns": 30},
        {**base, "name": "old", "correlation_id": 11, "start_ns": 40, "end_ns": 50},
    ]
    g, e, old = decode_records(records)
    assert (g.graph_id, g.graph_node_id, g.launch_range) == (3, 41, "decode_step")
    assert g.range_op is None
    assert (e.graph_id, e.graph_node_id, e.launch_range) == (None, None, None)
    assert (old.graph_id, old.graph_node_id) == (None, None)
