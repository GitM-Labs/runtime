"""Deviation matches kernels to predicted ops by identity, not position.

The old `i % len(pred)` pairing flagged ~everything uniformly under CUDA graphs.
Now each kernel is classified by name to its op and compared to that op's node;
unclassifiable kernels are unmodeled work and kept.
"""

from __future__ import annotations

from gitm.optimizer.deviation import (
    classify_op,
    deviating_kernel_indices,
    deviation_summary,
)
from gitm.planner.graph import predict_graph
from gitm.tracer.schema import KernelEvent, Trace


def _k(
    name: str, dur_s: float, t0: int = 0,
    range_op: str | None = None, range_layer: int | None = None,
) -> KernelEvent:
    return KernelEvent(
        name=name, start_ns=t0, end_ns=t0 + int(dur_s * 1e9), stream_id=0, device_id=0,
        range_op=range_op, range_layer=range_layer,
    )


def _trace(events: list[KernelEvent]) -> Trace:
    return Trace(
        workload_id="w", fingerprint="f", run_id="r", device_count=1,
        vendor="nvidia", captured_at_ns=0, duration_ns=10**9, events=events,
    )


def test_classify_op():
    assert classify_op("void flash_attn_fwd_kernel<>") == "attn_score_value"
    assert classify_op("triton_qkv_proj_gemm") == "qkv_proj"
    assert classify_op("cutlass_down_proj_kernel") == "mlp_down"
    assert classify_op("lm_head_logits") == "lm_head"
    assert classify_op("triton_rms_norm") is None  # not a modeled op


def test_classify_op_matches_real_vllm_kernel_names():
    """Confirmed against a real vLLM decode trace (L4, CUPTI) — these exact
    mangled kernel names came back from a live run. FlashAttention's real
    kernel is flash_fwd_*, NOT flash_attn_* (that needle alone misses it);
    vLLM's own KV-cache write/bookkeeping kernels weren't covered at all."""
    assert classify_op(
        "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64E"
        "Li64ELi256ELi4ELb0ELb0EN7cutlass6half_tE19Flash_kernel_traitsILi64E"
    ) == "attn_score_value"
    assert classify_op(
        "_ZN4vllm30reshape_and_cache_flash_kernelIttLNS_18Fp8KVCacheDataTypeE0EEE"
    ) == "attn_score_value"
    assert classify_op("_compute_slot_mapping_kernel") == "attn_score_value"
    # The dominant real kernel type (~35% of launches on that trace) is a bare
    # cuBLAS/cutlass GEMM shared across every projection — genuinely
    # unattributable by name alone, not a bug to chase with more substrings.
    assert classify_op("ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_stages_32x5_tn") is None
    assert classify_op(
        "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8EEE"
    ) is None


def test_in_band_op_not_kept_out_of_band_and_unmodeled_kept():
    g = predict_graph()
    t_attn = next(n.prediction.t_pred_s for n in g.nodes if n.op == "attn_score_value")
    tr = _trace([
        _k("flash_attn_kernel", t_attn),        # in band  -> NOT kept
        _k("flash_attn_kernel", t_attn * 8),    # 8x slow  -> kept (departure)
        _k("triton_rms_norm_kernel", 1e-6),     # unclassified -> kept (unmodeled)
    ])
    dev = deviating_kernel_indices(tr, g)
    assert dev.kept_indices == [1, 2]


def test_summary_keys_by_the_observed_kernels_op():
    g = predict_graph()
    t_attn = next(n.prediction.t_pred_s for n in g.nodes if n.op == "attn_score_value")
    tr = _trace([
        _k("flash_attn_kernel", t_attn * 8),    # departing attention
        _k("mystery_kernel", 1e-6),             # unmodeled
    ])
    summary = deviation_summary(tr, g)
    assert summary["kept_ops"] == {"attn_score_value": 1, "<unmodeled>": 1}


def test_range_identity_classifies_a_bare_gemm_that_name_matching_cannot():
    """The dominant real-world gap: bare cuBLAS/cutlass GEMMs carry no
    projection tag in their name (test_classify_op_matches_real_vllm_kernel_names
    above), so classify_op alone always calls them unmodeled. An NVTX range
    identity sidesteps the name entirely."""
    g = predict_graph()
    t_qkv = next(n.prediction.t_pred_s for n in g.nodes if n.op == "qkv_proj")
    bare_gemm = "ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_stages_32x5_tn"
    assert classify_op(bare_gemm) is None  # unattributable by name alone

    tr = _trace([_k(bare_gemm, t_qkv * 8, range_op="qkv_proj", range_layer=1)])
    dev = deviating_kernel_indices(tr, g)
    assert dev.kept_indices == [0]  # correctly identified and flagged as a departure

    summary = deviation_summary(tr, g)
    assert summary["kept_ops"] == {"qkv_proj": 1}  # not "<unmodeled>"


def test_range_identity_takes_priority_over_name_classification():
    """Name says attention (would classify to attn_score_value); the range
    identity says qkv_proj -- range wins, and recovers the real layer, which
    name-based classification can never do."""
    from gitm.optimizer.monitor import residuals

    g = predict_graph()
    t_qkv = next(n.prediction.t_pred_s for n in g.nodes if n.op == "qkv_proj")
    tr = _trace([_k("flash_attn_kernel", t_qkv, range_op="qkv_proj", range_layer=5)])
    res = residuals(tr, g)
    assert len(res.per_kernel) == 1
    assert res.per_kernel[0].op == "qkv_proj"
    assert res.per_kernel[0].layer == 5


def test_no_predicted_graph_keeps_everything():
    from gitm.planner.graph import Graph
    from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec

    empty = Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(), nodes=[])
    tr = _trace([_k("anything", 1e-6), _k("else", 1e-6)])
    assert deviating_kernel_indices(tr, empty).kept_indices == [0, 1]


# ── streaming subtraction: ``gitm deviate`` ─────────────────────────────────
#
# A real capture is millions of kernels and gigabytes of JSONL, so the path that
# materialises a validated Trace cannot be used on one. These cover the streaming
# path, whose whole reason for existing is that it never holds the file.


def _write_trace(path, rows):
    """rows: (name, duration_ns, repeat)."""
    import json

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": {"run_id": "t", "duration_ns": 10**9}}) + "\n")
        t = 0
        for name, dur, n in rows:
            for _ in range(n):
                fh.write(json.dumps({
                    "kind": "kernel", "name": name, "start_ns": t, "end_ns": t + dur,
                    "stream_id": 7, "device_id": 0,
                }) + "\n")
                t += dur + 1000
    return path


def test_stream_observed_aggregates_by_predicted_op(tmp_path):
    from gitm.optimizer.deviation import stream_observed

    p = _write_trace(tmp_path / "t.jsonl", [
        ("fused_moe_kernel", 1000, 10),
        ("moe_align_block_size", 500, 4),
        ("_causal_conv1d_update_kernel", 200, 6),
    ])
    per_op, n, total, span = stream_observed(p)

    assert n == 20
    assert per_op["moe_routed"] == [10, 10_000]
    assert per_op["moe_router"] == [4, 2_000]
    assert per_op["linattn_conv"] == [6, 1_200]
    assert total == 13_200
    assert span > total  # gaps between kernels are real and must not be absorbed


def test_unclassifiable_kernels_are_collected_separately(tmp_path):
    """Unmodeled work is the graph's coverage gap, not a deviation. Folding it
    into any modeled op would make that op look arbitrarily slow."""
    from gitm.optimizer.deviation import stream_observed

    p = _write_trace(tmp_path / "t.jsonl", [
        ("fused_moe_kernel", 1000, 2),
        ("nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT", 900, 3),
        ("unrolled_elementwise_kernel", 100, 5),
    ])
    per_op, _, _, _ = stream_observed(p)
    assert per_op["moe_routed"] == [2, 2_000]
    # A bare GEMM carries no projection in its name, so classify_op declines it —
    # deliberately different from the coarse taxonomy, which calls it a GEMM.
    assert per_op["<unmodeled>"] == [8, 3_200]


def test_an_nvtx_range_overrides_the_name_guess(tmp_path):
    """When a capture carries ranges they are the identity; the name is a guess."""
    import json

    p = tmp_path / "t.jsonl"
    with p.open("w") as fh:
        fh.write(json.dumps({"_header": {"run_id": "t"}}) + "\n")
        fh.write(json.dumps({
            "kind": "kernel", "name": "some_unrecognisable_kernel",
            "start_ns": 0, "end_ns": 500, "stream_id": 1, "device_id": 0,
            "range_op": "moe_routed", "range_layer": 3,
        }) + "\n")

    from gitm.optimizer.deviation import stream_observed

    per_op, _, _, _ = stream_observed(p)
    assert per_op == {"moe_routed": [1, 500]}


def test_non_kernel_events_do_not_enter_the_subtraction(tmp_path):
    import json

    p = tmp_path / "t.jsonl"
    with p.open("w") as fh:
        fh.write(json.dumps({"_header": {}}) + "\n")
        fh.write(json.dumps({"kind": "memcpy", "start_ns": 0, "end_ns": 900,
                             "stream_id": 1, "device_id": 0, "bytes": 4096,
                             "src": "host", "dst": "device"}) + "\n")
        fh.write(json.dumps({"kind": "kernel", "name": "fused_moe_kernel",
                             "start_ns": 1000, "end_ns": 2000, "stream_id": 1,
                             "device_id": 0}) + "\n")

    from gitm.optimizer.deviation import stream_observed

    per_op, n, total, _ = stream_observed(p)
    assert n == 1 and total == 1000
    assert set(per_op) == {"moe_routed"}


def test_a_truncated_line_is_skipped_rather_than_fatal(tmp_path):
    """Captures get killed mid-flush. A half-written final line must not lose the
    rest of the trace."""
    p = _write_trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 3)])
    with open(p, "a") as fh:
        fh.write('{"kind": "kernel", "name": "trunc')

    from gitm.optimizer.deviation import stream_observed

    _, n, _, _ = stream_observed(p)
    assert n == 3


def test_predicted_per_op_sums_over_layers():
    from gitm.optimizer.deviation import predicted_per_op
    from gitm.planner.hybrid_graph import predict_hybrid_graph
    from gitm.planner.model_catalogue import load_spec
    from gitm.planner.roofline import BatchConfig, HardwareSpec

    g = predict_hybrid_graph(load_spec("qwen3.6-35b-a3b"), HardwareSpec(),
                             BatchConfig(batch=2))
    import pytest

    per_op = predicted_per_op(g)
    # approx, not equality: summing 40 layers in a different order lands a few
    # ULPs apart, and pinning that would be a test of float addition.
    assert per_op["moe_routed"] == pytest.approx(
        sum(n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed")
    )
    assert per_op["lm_head"] > 0


def test_the_floor_scales_with_the_step_count(tmp_path, capsys):
    """A per-step floor against a whole window's observation is off by the number
    of steps — thousands. --steps is what makes the two comparable."""
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 30_000, 40)])
    main([str(p), "--model", "qwen3.6-35b-a3b", "--gpu", "H200", "--steps", "10",
          "--batch", "8", "--kv-len", "1024"])
    scaled = capsys.readouterr().out
    main([str(p), "--model", "qwen3.6-35b-a3b", "--gpu", "H200",
          "--batch", "8", "--kv-len", "1024"])
    unscaled = capsys.readouterr().out
    assert scaled != unscaled
    assert "10 steps" in scaled


def test_no_graph_reports_observation_alone(tmp_path, capsys):
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 4)])
    assert main([str(p), "--no-graph"]) == 0
    out = capsys.readouterr().out
    assert "moe_routed" in out
    assert "floor" not in out


def test_ops_with_no_observed_kernel_are_named(tmp_path, capsys):
    """Predicted-but-never-observed is ambiguous between "did not run" and "no
    kernel name classifies to it". The report must not resolve it silently."""
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 4)])
    main([str(p), "--model", "qwen3.6-35b-a3b", "--gpu", "H200"])
    out = capsys.readouterr().out
    assert "predicted but never observed" in out
    assert "taxonomy gap" in out


def test_unmodeled_share_is_labelled_as_coverage_not_headroom(tmp_path, capsys):
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [
        ("fused_moe_kernel", 1000, 2), ("unrolled_elementwise_kernel", 1000, 8),
    ])
    main([str(p), "--model", "qwen3.6-35b-a3b", "--gpu", "H200"])
    out = capsys.readouterr().out
    assert "coverage gap, not headroom" in out


def test_a_missing_trace_is_reported(capsys):
    from gitm.optimizer.deviation import main

    assert main(["/nonexistent/trace.jsonl", "--no-graph"]) == 2
    assert "cannot read trace" in capsys.readouterr().out


def test_a_trace_with_no_kernels_is_reported(tmp_path, capsys):
    import json

    from gitm.optimizer.deviation import main

    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"_header": {}}) + "\n")
    assert main([str(p), "--no-graph"]) == 1
    assert "nothing to subtract" in capsys.readouterr().out


def test_reachable_through_the_top_level_cli(tmp_path, capsys):
    from gitm.cli import main as cli_main

    p = _write_trace(tmp_path / "t.jsonl", [("fused_moe_kernel", 1000, 4)])
    assert cli_main(["deviate", str(p), "--model", "qwen3.6-35b-a3b",
                     "--gpu", "H200", "--steps", "2", "--json"]) == 0
    import json as _json

    payload = _json.loads(capsys.readouterr().out)
    assert payload["n_kernels"] == 4
    assert payload["ops"]["moe_routed"]["floor_s"] is not None
    assert payload["band_width"] == 0.4


# ── the rule table is a dict, and its insertion order is semantic ───────────


def test_op_rules_is_keyed_by_op_with_needle_tuples():
    """One entry per op, so a rule is looked up by the thing it produces rather
    than by scanning tuples for a second element."""
    from gitm.optimizer.deviation import _OP_RULES

    assert isinstance(_OP_RULES, dict)
    for op, needles in _OP_RULES.items():
        assert isinstance(op, str) and op
        assert isinstance(needles, tuple) and needles
        assert all(isinstance(k, str) and k == k.lower() for k in needles), op


def test_op_rule_order_is_load_bearing():
    """A dict reads like an unordered lookup. This one is not.

    Vocabularies overlap, so the narrow entries must be tried before the broad
    ones. Each pair below is a real collision: reordering them changes what a
    kernel classifies as without changing a single needle, which is exactly the
    edit someone makes while "tidying" a mapping.
    """
    from gitm.optimizer.deviation import _OP_RULES

    order = list(_OP_RULES)

    def before(a, b):
        assert order.index(a) < order.index(b), f"{a} must precede {b}"

    # "moe" appears in the routing kernels' names, so the broad expert-GEMM
    # entry would swallow routing and the shared expert.
    before("moe_router", "moe_routed")
    before("moe_shared", "moe_routed")
    # NCCL's all-to-all is named like every other NCCL kernel.
    before("moe_all_to_all", "tp_all_reduce")
    # GDN names must not reach the softmax-attention needles: a recurrent layer's
    # traffic is flat in context, the attention prediction is not.
    before("linattn_recurrent", "attn_score_value")
    before("linattn_conv", "attn_score_value")
    # "qkvz" contains "qkv".
    before("linattn_in_proj", "qkv_proj")
    # The indexer must not fall through to a bare index/gather rule.
    before("attn_index_score", "moe_routed")


def test_the_collisions_the_order_protects_against_are_real():
    """If a needle stops colliding, the ordering constraint above becomes
    vacuous and the test would keep passing while protecting nothing."""
    from gitm.optimizer.deviation import _OP_RULES

    def collides(op_a, op_b, sample):
        """`sample` must match both entries' needles — that is what makes order
        decide the outcome."""
        assert any(k in sample for k in _OP_RULES[op_a]), (op_a, sample)
        assert any(k in sample for k in _OP_RULES[op_b]), (op_b, sample)

    collides("moe_router", "moe_routed", "moe_align_block_size")
    collides("moe_shared", "moe_routed", "shared_expert_gemm")
    collides("linattn_in_proj", "qkv_proj", "in_proj_qkvz_kernel")


def test_every_needle_is_reachable():
    """A needle that can never fire is dead weight that reads as coverage — the
    `chunk_o`/`chunk_h` failure, where two entries matched nothing for months
    while the table looked complete."""
    from gitm.optimizer.deviation import _OP_RULES, classify_op

    unreachable = []
    for op, needles in _OP_RULES.items():
        for needle in needles:
            # A synthetic name containing only this needle must reach this op.
            if classify_op(f"kernel_{needle}_fwd") != op:
                unreachable.append((op, needle, classify_op(f"kernel_{needle}_fwd")))
    assert unreachable == [], f"needles shadowed by an earlier entry: {unreachable}"


# ── prefill vs decode ───────────────────────────────────────────────────────
#
# The two phases run *different kernels* for the ops whose algorithm differs, and
# identical kernels for everything else. So a name splits some of a trace and
# none of the rest, and the report has to say which — on a real H200 capture the
# unsplittable majority is about three quarters of device time.


def test_the_gdn_convolution_names_its_own_phase():
    """`_fwd` processes a whole sequence, `_update` one token. The cleanest
    discriminator available, and it survives backend changes because it is the
    same kernel family either way."""
    from gitm.tracer.kernel_taxonomy import classify_phase

    assert classify_phase("_causal_conv1d_fwd_kernel") == "prefill"
    assert classify_phase("_causal_conv1d_update_kernel") == "decode"


def test_both_gdn_prefill_backends_are_recognised():
    """vLLM picks per SKU: the FlashInfer CuTeDSL kernel on this H200, the Triton
    `chunk_*` family elsewhere. Listing only one silently drops prefill on the
    other machine — which is exactly what a first pass did, reporting zero
    prefill kernels on a trace that plainly contained them."""
    from gitm.tracer.kernel_taxonomy import classify_phase

    assert classify_phase(
        "kernel_cutlass_flashinfergdn_delta_rule_sm90_FullyFusedDeltaRuleSm90_obj"
    ) == "prefill"
    assert classify_phase("chunk_gated_delta_rule_fwd_kernel_h_blockdim64") == "prefill"
    assert classify_phase("solve_tril_16x16_kernel") == "prefill"


def test_flashinfer_ships_separate_prefill_and_decode_attention():
    from gitm.tracer.kernel_taxonomy import classify_phase

    assert classify_phase("flashinfer::BatchPrefillWithPagedKVCacheKernel") == "prefill"
    assert classify_phase("flashinfer::BatchDecodeWithPagedKVCacheKernel") == "decode"


def test_the_recurrent_decode_kernel_is_not_claimed_by_a_prefill_needle():
    """`fused_recurrent_gated_delta_rule_packed_decode_kernel` contains
    `delta_rule`, which the chunked prefill needles are close to. Decode is
    tested first for exactly this reason."""
    from gitm.tracer.kernel_taxonomy import classify_phase

    assert classify_phase(
        "fused_recurrent_gated_delta_rule_packed_decode_kernel"
    ) == "decode"


def test_shared_kernels_report_no_phase_rather_than_a_guess():
    """MoE and the GEMMs are byte-identical in both phases. Attributing them to
    either side would misattribute the majority of a trace."""
    from gitm.tracer.kernel_taxonomy import classify_phase

    assert classify_phase("fused_moe_kernel") is None
    assert classify_phase("nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT") is None
    assert classify_phase("") is None


def test_by_phase_reports_the_share_it_cannot_split(tmp_path, capsys):
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [
        ("_causal_conv1d_fwd_kernel", 20_000, 4),
        ("_causal_conv1d_update_kernel", 3_000, 300),
        ("fused_moe_kernel", 27_000, 400),
    ])
    assert main([str(p), "--by-phase"]) == 0
    out = capsys.readouterr().out

    assert "prefill" in out and "decode" in out and "unknown" in out
    # The unsplittable share must be stated, not folded into one side.
    assert "cannot be split by name" in out


def test_by_phase_needs_no_predicted_graph(tmp_path):
    """It reports what ran, not how it compares to a floor. Requiring --model
    would make the cheapest view the most awkward to reach."""
    from gitm.optimizer.deviation import main

    p = _write_trace(tmp_path / "t.jsonl", [("_causal_conv1d_fwd_kernel", 1000, 2)])
    assert main([str(p), "--by-phase"]) == 0
