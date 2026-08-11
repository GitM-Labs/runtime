"""The sparse-MoE decode graph, pinned against hand-derived numbers.

Every assertion here guards a term the dense graph does not have. The dense
model is not merely *imprecise* on this architecture — it is wrong in specific,
nameable directions, and each test below pins the correction:

* expert weight traffic saturates (union, not multiplication),
* the attention core's read stops growing with context,
* the KV latent is shared across query heads,
* fp4/fp8 tensors price against fp4/fp8 peaks, or say they didn't.

A regression on any of these produces a confidently wrong ceiling, which the
headroom report would then bill against.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gitm.optimizer.deviation import classify_op
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.moe_graph import (
    effective_kv_tokens,
    index_candidates,
    kv_bytes_per_token,
    kv_fixed_bytes_per_sequence,
    model_weight_bytes,
    predict_moe_graph,
    spec_from_hf_config,
)
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ModelSpec,
    ShardingConfig,
    resolve_peak,
    roofline,
    weight_bytes,
    weight_bytes_is_fallback,
)

# The shape of DeepSeek-V4-Flash-0731, trimmed to the keys the planner reads.
V4_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "sliding_window": 128,
    # Ships 46 long for 43 layers — the tail is MTP/padding, not layers.
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0, 0, 0],
    "num_nextn_predict_layers": 1,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
    "expert_dtype": "fp4",
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    "torch_dtype": "bfloat16",
}


# The base DeepSeek-V4-Flash checkpoint. Two real differences from -0731: no
# DSpark at all, and 44 compress_ratios rather than 46. Both variants ship under
# the same architecture, so the parser has to absorb the difference silently.
V4_BASE_CONFIG = {
    **{k: v for k, v in V4_CONFIG.items() if not k.startswith("dspark")},
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
}

# Published safetensors total for DeepSeek-V4-Flash, from the model repo.
V4_BASE_PUBLISHED_BYTES = 160e9


@pytest.fixture
def spec():
    return spec_from_hf_config(V4_CONFIG, name="DeepSeek-V4-Flash-0731")


@pytest.fixture
def base_spec():
    return spec_from_hf_config(V4_BASE_CONFIG, name="DeepSeek-V4-Flash")


@pytest.mark.parametrize(
    "missing_key",
    ["hidden_size", "n_routed_experts", "compress_ratios", "expert_dtype", "torch_dtype"],
)
def test_public_hf_builder_refuses_missing_shape_or_precision(missing_key):
    cfg = dict(V4_CONFIG)
    cfg.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        spec_from_hf_config(cfg)


def test_public_hf_builder_accepts_complete_declared_config():
    parsed = spec_from_hf_config(V4_CONFIG)

    assert parsed.hidden == V4_CONFIG["hidden_size"]
    assert parsed.expert_dtype == V4_CONFIG["expert_dtype"]


@pytest.fixture
def b200():
    return hardware_spec_for(peak_for_sku("NVIDIA B200"))


# ── the union term ──────────────────────────────────────────────────────────
#
# The term itself (saturation, monotonicity, clamps, zero/negative guards) is
# `distinct_experts` in roofline.py and is covered by test_moe_roofline.py. Only
# this graph's *use* of it is tested here.


def test_expert_fetch_is_driven_by_positions_not_sequences(spec, b200):
    """Speculative drafts widen the expert set the step has to fetch.

    Every drafted position routes independently, so eight sequences drafting four
    positions each wake the experts of 32 tokens, not 8. This is also the guard on
    the argument order into ``distinct_experts(batch, num_experts, top_k)`` — a
    swap there is silent, since all three are plain ints, and would put expert
    weight traffic somewhere between absurd and zero.
    """
    def routed_bytes(spec_tokens):
        g = predict_moe_graph(
            spec, b200,
            BatchConfig(batch=8, kv_cache_len=4096, speculative_tokens=spec_tokens),
        )
        return sum(n.prediction.bytes for n in g.nodes if n.op == "moe_routed")

    plain, drafted = routed_bytes(0), routed_bytes(3)
    assert drafted > plain
    # 8 positions wake ~44 of 256 experts; 32 wake ~140. Sublinear because the
    # union saturates — 4x the positions is nowhere near 4x the traffic.
    assert drafted < 4 * plain


# ── compressed, selected attention ──────────────────────────────────────────


def test_ratio_zero_layers_are_sliding_window_not_global(spec):
    """Layers 0-1 carry ratio 0, which means sliding-window — *not* global.

    Reading 0 as "uncompressed, therefore attends to everything" is the natural
    mistake and it is wrong by ``kv_len / sliding_window``: 512x at 64K. It is
    also self-refuting — if any layer read the whole cache, the 1M context this
    checkpoint advertises could not be served.
    """
    assert spec.compress_ratio(0) == spec.compress_ratio(1) == 0
    assert effective_kv_tokens(spec, 0, 65536) == spec.sliding_window
    assert effective_kv_tokens(spec, 1, 1_048_576) == spec.sliding_window


def test_no_layer_read_grows_with_context(spec):
    """The property that makes a million-token deployment possible at all.

    Sliding-window layers are bounded from the start; compressed layers stop
    growing once selection saturates. So the whole attention path is flat in
    context length, and any observed growth is a real deviation rather than
    something the architecture explains away.
    """
    at_64k = sum(effective_kv_tokens(spec, i, 65536) for i in range(spec.n_layers))
    at_1m = sum(effective_kv_tokens(spec, i, 1_048_576) for i in range(spec.n_layers))
    assert at_64k == at_1m


def test_sliding_window_layers_run_no_indexer(spec, b200):
    """A fixed recent window needs no selection, so no indexer node is emitted.

    Emitting one would put predicted work into the graph that no kernel can align
    to, which downstream reads as "predicted work that never ran".
    """
    assert index_candidates(spec, 0, 65536) == 0
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=65536))
    indexer_layers = {n.layer for n in g.nodes if n.op.startswith("attn_index")}
    assert 0 not in indexer_layers and 1 not in indexer_layers
    assert 2 in indexer_layers


def test_compressed_layer_read_is_bounded_by_window_plus_topk(spec):
    """sliding_window + index_topk is the ceiling, whatever the context."""
    assert spec.compress_ratio(2) == 4
    assert effective_kv_tokens(spec, 2, 65536) == 128 + 512


def test_attention_core_read_is_constant_in_context_once_selection_saturates(spec):
    """The headline consequence: context length leaves the attention core alone.

    A residual on `attn_score_value` that scales with sequence length is
    therefore *not* explained by "longer context" — it is a real deviation. That
    inference is only available because this is constant.
    """
    at_64k = effective_kv_tokens(spec, 2, 65536)
    at_1m = effective_kv_tokens(spec, 2, 1_048_576)
    assert at_64k == at_1m


def test_compress_ratio_moves_indexer_cost_not_attention_cost(spec):
    """Ratios 4 and 128 read identically at the core but differ 32x at the scan.

    This is why they are separate nodes. Folding them together would average away
    the only signal that distinguishes the two layer types.
    """
    assert effective_kv_tokens(spec, 2, 65536) == effective_kv_tokens(spec, 3, 65536)
    assert index_candidates(spec, 2, 65536) == 32 * index_candidates(spec, 3, 65536)


def test_indexer_candidates_grow_with_context(spec):
    """Unbounded in context — the node that decides if 1M tokens is viable."""
    assert index_candidates(spec, 2, 1_048_576) > index_candidates(spec, 2, 65536)


def test_effective_tokens_never_exceed_the_cache(spec):
    """A short sequence cannot read more than it holds."""
    assert effective_kv_tokens(spec, 2, 64) == 64


# ── dtype-aware peaks ───────────────────────────────────────────────────────


def test_b200_prices_fp4_against_the_fp4_peak(b200):
    peak, used = resolve_peak(b200, "fp4")
    assert used == "fp4"
    assert peak == pytest.approx(9000e12)


def test_fp4_on_a100_falls_back_and_says_so():
    """A100 has no fp4 path. Falling back is fine; hiding it is not."""
    peak, used = resolve_peak(HardwareSpec(), "fp4")
    assert used == "fp16"
    assert peak == pytest.approx(312e12)


def test_fallback_direction_understates_the_ceiling_rather_than_inflating_it():
    """Falling back *up* the precision ladder is the safe direction.

    A lower peak predicts a slower floor, so the observed run looks closer to its
    ceiling than it is — under-reporting headroom. The opposite error would
    invent headroom, which is the one the refund clause cannot survive.
    """
    hopper = hardware_spec_for(peak_for_sku("NVIDIA H100 80GB HBM3"))
    fp4_peak, _ = resolve_peak(hopper, "fp4")
    fp8_peak, _ = resolve_peak(hopper, "fp8")
    assert fp4_peak <= fp8_peak


def test_graph_flags_when_any_node_ran_on_a_fallback_peak(spec):
    """A100 cannot price this checkpoint; the graph must not pretend otherwise."""
    g = predict_moe_graph(spec, HardwareSpec(), BatchConfig(batch=1, kv_cache_len=1024))
    assert g.has_fallback_peaks


def test_b200_graph_needs_no_fallback(spec, b200):
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    assert not g.has_fallback_peaks
    assert not g.hardware_is_fallback


def test_default_hardware_graph_flags_catalogue_fallback(spec):
    g = predict_moe_graph(spec, HardwareSpec(), BatchConfig(batch=1, kv_cache_len=1024))
    assert g.hardware_is_fallback


def test_fp4_weight_bytes_include_the_block_scales():
    """MXFP4 is 0.5 bytes plus a 1-byte scale per 32 values, not 0.5 flat.

    6% of expert traffic on this checkpoint — larger than several effects the
    monitor is expected to resolve, so dropping it is not a rounding choice.
    """
    assert weight_bytes("fp4") == pytest.approx(0.5 + 1 / 32)
    assert weight_bytes("fp8") > 1.0  # block scales ride along
    assert weight_bytes("bf16") == 2.0


def test_unknown_weight_dtype_is_fail_open_but_explicitly_flagged(spec, b200):
    """The bf16 byte-width fallback must ride with the memory term it changes."""
    unknown = replace(spec, expert_dtype="future_fp3")
    g = predict_moe_graph(unknown, b200, BatchConfig(batch=1, kv_cache_len=1024))

    assert weight_bytes("future_fp3") == weight_bytes("bf16")
    assert weight_bytes_is_fallback("future_fp3")
    assert g.has_fallback_bytes
    assert {
        n.op for n in g.nodes if n.prediction.bytes_are_fallback
    } == {"moe_shared", "moe_routed"}


def test_known_weight_dtypes_leave_bytes_fallback_clean(spec, b200):
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=1024))

    assert not weight_bytes_is_fallback("fp4")
    assert not g.has_fallback_bytes
    assert not any(n.prediction.bytes_are_fallback for n in g.nodes)


def test_mixed_byte_node_flags_unknown_kv_dtype(spec, b200):
    """A node is flagged when any byte contributor is unknown, not just compute dtype."""
    unknown = replace(spec, kv_dtype="future_kv3")
    g = predict_moe_graph(unknown, b200, BatchConfig(batch=1, kv_cache_len=1024))

    flagged = {n.op for n in g.nodes if n.prediction.bytes_are_fallback}
    assert {"attn_kv_a", "attn_score_value"} <= flagged
    assert "moe_router" not in flagged


# ── config parsing ──────────────────────────────────────────────────────────


def test_config_parse_keeps_expert_and_linear_dtypes_distinct(spec):
    """The whole point of the checkpoint: experts fp4, everything else fp8."""
    assert spec.expert_dtype == "fp4"
    assert spec.weight_dtype == "fp8"
    assert spec.act_dtype == "bf16"


def test_compress_ratios_truncate_to_layer_count(spec):
    """The config ships 46 ratios for 43 layers.

    Keeping the tail would not error — it would silently leave layer indices
    correct but suggest 3 layers that don't exist. Truncating is the check.
    """
    assert len(V4_CONFIG["compress_ratios"]) == 46
    assert len(spec.compress_ratios) == 43
    assert spec.compress_ratio(42) == 4


def test_kv_latent_is_shared_across_query_heads(spec):
    """num_kv_heads == 1: one latent of 512+64, not 64 heads of it.

    Modelling this as GQA would inflate decode KV traffic 64x and make every
    long-context run look catastrophically memory-bound.
    """
    assert spec.kv_latent_dim == 512 + 64
    assert spec.kv_latent_dim < spec.n_heads * spec.head_dim


# ── the assembled graph ─────────────────────────────────────────────────────


def test_expert_weight_traffic_saturates_at_the_full_expert_set(spec, b200):
    """At large batch a step reads essentially every expert weight, once.

    Pins the absolute scale, not just the shape: 43 layers x 256 experts x 3
    matrices x 4096 x 2048 at ~0.53 bytes is ~145 GB, and the predicted bytes
    must land there rather than at some multiple of it.
    """
    g = predict_moe_graph(spec, b200, BatchConfig(batch=512, kv_cache_len=4096))
    routed = sum(n.prediction.bytes for n in g.nodes if n.op == "moe_routed")
    full_set = 43 * 256 * 3 * 4096 * 2048 * weight_bytes("fp4")
    assert routed == pytest.approx(full_set, rel=0.05)


def test_step_time_is_strongly_sublinear_in_batch(spec, b200):
    """64x the batch for well under 64x the time — because weights saturate.

    This curve is the reason batch sizing is a lever at all on this model, and a
    graph that predicted linear growth would rank that lever as worthless.
    """
    def step_s(b):
        return predict_moe_graph(
            spec, b200, BatchConfig(batch=b, kv_cache_len=32768)
        ).total_pred_s

    assert step_s(64) < 20 * step_s(1)


def test_low_batch_decode_is_memory_bound_on_the_experts(spec, b200):
    """Batch 1 moves 6 experts of weights to do 6 experts of arithmetic."""
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=4096))
    routed = [n for n in g.nodes if n.op == "moe_routed"]
    assert routed and all(n.prediction.bound == "memory" for n in routed)


def test_speculative_positions_amortise_cache_reads_but_not_flops(spec, b200):
    """MTP's mechanism, made visible.

    Four drafted positions verify against one KV read per sequence, so cache
    traffic is flat while attention FLOPs scale. Collapsing positions and
    sequences into one number would erase this and predict MTP as pure overhead.
    """
    plain = predict_moe_graph(spec, b200, BatchConfig(batch=8, kv_cache_len=32768))
    spec_dec = predict_moe_graph(
        spec, b200, BatchConfig(batch=8, kv_cache_len=32768, speculative_tokens=3)
    )
    def attn(g):
        n = next(x for x in g.nodes if x.op == "attn_score_value")
        return n.prediction.bytes, n.prediction.flops

    b_plain, f_plain = attn(plain)
    b_spec, f_spec = attn(spec_dec)
    assert b_spec == pytest.approx(b_plain)  # one cache read per sequence
    assert f_spec == pytest.approx(4 * f_plain)  # four positions of arithmetic


def test_mtp_head_emits_layers_beyond_the_stack(spec, b200):
    """The draft head is layer 43, not a renamed op — layer index is the identity."""
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    layers = {n.layer for n in g.nodes if n.layer is not None}
    assert max(layers) == spec.n_layers  # 43 == one MTP layer past 0..42


def test_dspark_only_on_its_declared_layers(spec, b200):
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    assert {n.layer for n in g.nodes if n.op == "dspark"} == {40, 41, 42}


def test_estimated_nodes_are_flagged(spec, b200):
    """Approximations must be visible to the report, never silent in the total."""
    g = predict_moe_graph(spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    estimated = {n.op for n in g.nodes if n.prediction.estimated}
    assert estimated == {"attn_out_proj", "dspark"}


def test_tokens_per_step_accounts_for_acceptance():
    """Drafts are paid for always and counted only when kept."""
    b = BatchConfig(batch=4, speculative_tokens=3, acceptance_rate=0.5)
    assert b.positions_per_step == 16  # all drafted work is computed
    assert b.tokens_per_step == pytest.approx(4 * (1 + 3 * 0.5))


# ── the observed side lines up with the predicted side ──────────────────────


@pytest.mark.parametrize(
    "kernel,op",
    [
        ("flash_mla_sparse_fwd_kernel", "attn_score_value"),
        ("lightning_indexer_topk_kernel", "attn_index_score"),
        ("cutlass_moe_mm_sm100", "moe_routed"),
        ("sm100_xmma_fp4_grouped_gemm", "moe_routed"),
        ("moe_align_block_size_kernel", "moe_router"),
        ("topk_softmax_kernel", "moe_router"),
        ("shared_expert_fused_kernel", "moe_shared"),
        ("dspark_markov_update_kernel", "dspark"),
        # NCCL names every kernel ncclDevKernel_<Op>; the all-to-all must not be
        # swallowed by a generic "nccl" needle, because EP dispatch is the one
        # collective an MoE deployment exists to trade against.
        ("ncclDevKernel_AllToAll_Simple", "moe_all_to_all"),
        ("ncclDevKernel_AllReduce_Sum_f32", "tp_all_reduce"),
        ("cross_device_reduce_2stage", "tp_all_reduce"),
    ],
)
def test_v4_kernel_names_align_to_predicted_ops(kernel, op):
    """Before this, every one of these returned None and the graph was unusable.

    A predicted graph nothing aligns to produces no residuals, and no residuals
    means no attribution — the whole loop downstream of Phase 1 goes quiet.
    """
    assert classify_op(kernel) == op


# ── sharding across ranks ───────────────────────────────────────────────────


@pytest.fixture
def b300():
    return hardware_spec_for(peak_for_sku("NVIDIA B300"))


def test_blackwell_ultra_is_in_the_catalogue(b300):
    """Regression: an unknown SKU silently resolves to A100 defaults.

    That failure is invisible in the worst way — the fp8/fp4 peaks flag
    themselves as fallbacks, but memory bandwidth is off by 3.9x with nothing
    raised, and on a memory-bound workload bandwidth is the entire prediction.
    """
    assert b300.name != "A100-SXM4-80GB"
    assert b300.peak_mem_bw_bytes_per_s == pytest.approx(8000e9)
    assert b300.peak_flops_fp4_per_s == pytest.approx(15000e12)


def test_b300_and_b200_predict_the_same_memory_bound_step(spec, b200, b300):
    """Blackwell Ultra's 1.67x fp4 uplift buys nothing on a memory-bound decode.

    Same 8 TB/s, and every node is memory-bound, so the extra compute never
    binds. Any prediction that shows B300 faster at decode has a bug — most
    likely a compute term that should have been a memory term.
    """
    bc = BatchConfig(batch=256, kv_cache_len=32768)
    sh = ShardingConfig(tp=8)
    assert predict_moe_graph(spec, b300, bc, sh).total_pred_s == pytest.approx(
        predict_moe_graph(spec, b200, bc, sh).total_pred_s
    )


def test_tensor_parallel_does_not_divide_kv_traffic(spec, b200):
    """With one KV head the latent cannot be split, so every rank reads all of it.

    The GQA intuition says TP shards the KV cache. Here it does not, and a graph
    that divided would rank TP as a KV-bandwidth lever that it simply is not.
    """
    bc = BatchConfig(batch=64, kv_cache_len=65536)

    def kv_bytes(tp):
        g = predict_moe_graph(spec, b200, bc, ShardingConfig(tp=tp))
        return sum(n.prediction.bytes for n in g.nodes if n.op == "attn_score_value")

    assert kv_bytes(8) == pytest.approx(kv_bytes(1))


def test_tensor_parallel_does_divide_expert_traffic(spec, b200):
    """The dominant term does shard — eight ranks, an eighth of the weights each."""
    bc = BatchConfig(batch=256, kv_cache_len=4096)

    def expert_bytes(sh):
        g = predict_moe_graph(spec, b200, bc, sh)
        return sum(n.prediction.bytes for n in g.nodes if n.op == "moe_routed")

    assert expert_bytes(ShardingConfig(tp=8)) == pytest.approx(
        expert_bytes(ShardingConfig(tp=1)) / 8, rel=0.02
    )


def test_ep_and_tp_move_identical_expert_bytes(spec, b200):
    """EP versus TP is a *collective* trade, not a memory trade.

    Whole experts on a few ranks and slices of every expert on all ranks move the
    same bytes per rank. Only the cross-rank traffic differs. A catalogue that
    believed EP saved HBM traffic would rank it for the wrong reason and be
    unable to explain why it didn't help.
    """
    bc = BatchConfig(batch=256, kv_cache_len=4096)

    def expert_bytes(sh):
        g = predict_moe_graph(spec, b200, bc, sh)
        return sum(n.prediction.bytes for n in g.nodes if n.op == "moe_routed")

    assert expert_bytes(ShardingConfig(tp=8, ep=8)) == pytest.approx(
        expert_bytes(ShardingConfig(tp=8))
    )


def test_expert_parallel_emits_all_to_all_and_tp_emits_all_reduce(spec, b200):
    bc = BatchConfig(batch=64, kv_cache_len=4096)
    ep_ops = {n.op for n in predict_moe_graph(spec, b200, bc, ShardingConfig(tp=8, ep=8)).nodes}
    tp_ops = {n.op for n in predict_moe_graph(spec, b200, bc, ShardingConfig(tp=8)).nodes}
    assert "moe_all_to_all" in ep_ops and "tp_all_reduce" in ep_ops
    assert "moe_all_to_all" not in tp_ops and "tp_all_reduce" in tp_ops


def test_single_rank_emits_no_collectives(spec, b200):
    """The unsharded graph must be unchanged — no phantom collective at TP=1."""
    g = predict_moe_graph(spec, b200, BatchConfig(batch=8, kv_cache_len=4096))
    assert not {n.op for n in g.nodes} & {"moe_all_to_all", "tp_all_reduce"}


def test_ep_costs_more_cross_rank_traffic_than_tp(spec, b200):
    """All-to-all ships every routed token; all-reduce ships one hidden state."""
    bc = BatchConfig(batch=256, kv_cache_len=4096)

    def link_s(sh):
        g = predict_moe_graph(spec, b200, bc, sh)
        return sum(
            n.prediction.t_pred_s
            for n in g.nodes
            if n.op in ("moe_all_to_all", "tp_all_reduce")
        )

    assert link_s(ShardingConfig(tp=8, ep=8)) > link_s(ShardingConfig(tp=8))


def test_unknown_interconnect_surfaces_rather_than_pricing_collectives_free(spec):
    """A SKU with no NVLink figure must not be credited with a free all-to-all."""
    no_link = HardwareSpec(peak_mem_bw_bytes_per_s=8000e9)  # interconnect stays 0.0
    g = predict_moe_graph(
        spec, no_link, BatchConfig(batch=64, kv_cache_len=4096), ShardingConfig(tp=8, ep=8)
    )
    assert g.has_unpriced_collectives
    assert g.has_unpriced_nodes


def test_priced_interconnect_is_not_flagged(spec, b200):
    g = predict_moe_graph(
        spec, b200, BatchConfig(batch=64, kv_cache_len=4096), ShardingConfig(tp=8, ep=8)
    )
    assert not g.has_unpriced_collectives
    assert not g.has_unpriced_nodes


def test_general_unpriced_node_net_is_not_collective_specific(spec):
    hw = HardwareSpec(peak_mem_bw_bytes_per_s=0.0)
    node = PredictedNode("memory_only", None, roofline("memory_only", 0.0, 1024.0, hw))
    g = Graph(model=spec, hw=hw, batch=BatchConfig(), nodes=[node])

    assert g.has_unpriced_nodes
    assert g.has_unpriced_memory
    assert not g.has_unpriced_compute
    assert not g.has_unpriced_collectives


def test_graph_preserves_each_missing_roofline_denominator(spec):
    hw = HardwareSpec(peak_flops_fp16_per_s=0.0, peak_mem_bw_bytes_per_s=1e9)
    node = PredictedNode("partial", None, roofline("partial", 1e12, 1024.0, hw))
    g = Graph(model=spec, hw=hw, batch=BatchConfig(), nodes=[node])

    assert g.has_unpriced_nodes
    assert g.has_unpriced_compute
    assert not g.has_unpriced_memory


def test_expert_imbalance_only_applies_under_expert_parallelism(spec, b200):
    """TP is balanced by construction; EP waits for the unluckiest rank."""
    bc = BatchConfig(batch=64, kv_cache_len=4096)

    def routed_s(sh):
        g = predict_moe_graph(spec, b200, bc, sh)
        return sum(n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed")

    assert routed_s(ShardingConfig(tp=8, ep=8, ep_imbalance=1.5)) > routed_s(
        ShardingConfig(tp=8, ep=8)
    )
    assert routed_s(ShardingConfig(tp=8, ep_imbalance=1.5)) == pytest.approx(
        routed_s(ShardingConfig(tp=8))
    )


# ── does it fit ─────────────────────────────────────────────────────────────


def test_weight_estimate_validates_against_the_published_checkpoint(base_spec):
    """The one place this model can be checked against ground truth.

    DeepSeek-V4-Flash publishes at 160 GB of safetensors. Predicting within a few
    percent of that means the dtype split, the expert count and the attention
    shapes are all right at once — a model that priced experts at bf16 would come
    out near 600 GB, and one that missed the fp4 block scales would be 6% low on
    the dominant term alone.
    """
    predicted = model_weight_bytes(base_spec)
    assert predicted == pytest.approx(V4_BASE_PUBLISHED_BYTES, rel=0.05)


def test_dspark_variant_is_a_lower_bound_not_an_estimate(spec, base_spec):
    """-0731 publishes 7 GB above the base checkpoint; the modelled DSpark is ~6 MB.

    The shape isn't public, so the graph carries a low-rank placeholder that is
    known to be orders of magnitude small. Pinning the gap here stops anyone
    reading that checkpoint's footprint as an estimate — and fails loudly if
    someone later "fixes" it by fitting a number to this one observation.
    """
    delta = model_weight_bytes(spec) - model_weight_bytes(base_spec)
    published_delta = 167e9 - V4_BASE_PUBLISHED_BYTES
    assert delta < published_delta / 100

    g = predict_moe_graph(spec, HardwareSpec(), BatchConfig(), ShardingConfig(tp=8))
    assert g.resident_weight_bytes_per_rank == pytest.approx(
        model_weight_bytes(spec, ShardingConfig(tp=8))
    )
    assert g.resident_weight_bytes_is_lower_bound is True


def test_base_checkpoint_has_no_dspark_nodes(base_spec, b200):
    """No dspark keys in the config means no dspark work in the graph.

    Which makes the base checkpoint the better pilot target: its only
    shape-estimated node is the grouped output projection.
    """
    g = predict_moe_graph(base_spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    assert not [n for n in g.nodes if n.op == "dspark"]
    assert {n.op for n in g.nodes if n.prediction.estimated} == {"attn_out_proj"}
    assert g.resident_weight_bytes_per_rank == pytest.approx(model_weight_bytes(base_spec))
    assert g.resident_weight_bytes_is_lower_bound is False


def test_both_checkpoint_variants_yield_identical_per_layer_ratios(spec, base_spec):
    """44 entries and 46 entries describe the same 43 layers.

    The trailing entries are MTP/padding slots. Keeping them would not error — it
    would silently suggest layers that don't exist — so both must truncate to the
    same thing.
    """
    assert len(V4_BASE_CONFIG["compress_ratios"]) == 44
    assert len(V4_CONFIG["compress_ratios"]) == 46
    assert base_spec.compress_ratios == spec.compress_ratios


def test_tensor_parallel_shrinks_the_resident_footprint(spec):
    """Eight ranks hold roughly an eighth each — this is what makes TP fit."""
    whole = model_weight_bytes(spec)
    sharded = model_weight_bytes(spec, ShardingConfig(tp=8))
    assert sharded < whole / 6  # replicated q_a/kv_a/router keep it above exactly 1/8


def test_kv_footprint_splits_growing_from_fixed(spec):
    """Sliding-window layers cost per *sequence*; compressed layers cost per token.

    Folding the window layers into the per-token rate inflates it ~37% and caps
    concurrency well below what the hardware allows. Applying any single ratio to
    all 41 compressed layers is separately wrong by up to 128x.
    """
    per_token = kv_bytes_per_token(spec)
    fixed = kv_fixed_bytes_per_sequence(spec)

    # Growth comes only from the 41 compressed layers, at 1/4 or 1/128 of a latent.
    naive_all_layers = spec.n_layers * (spec.kv_latent_dim + spec.index_head_dim)
    assert per_token < naive_all_layers / 5

    # The two window layers are real, bounded, and paid once per sequence.
    assert fixed == 2 * spec.sliding_window * spec.kv_latent_dim * weight_bytes(spec.kv_dtype)

    graph = predict_moe_graph(spec, HardwareSpec(), BatchConfig())
    assert graph.kv_bytes_per_token_per_sequence == pytest.approx(per_token)
    assert graph.kv_fixed_bytes_per_sequence == pytest.approx(fixed)
    # In magnitude the fixed term is tiny — worth ~40 tokens of context, so it
    # never drives a sizing decision. What mattered was excluding these layers
    # from the *rate*, which is a 37% error on every sequence at every length.
    assert fixed < 50 * per_token
    naive = per_token + 2 * (spec.kv_latent_dim + spec.index_head_dim) * weight_bytes(
        spec.kv_dtype
    )
    assert naive == pytest.approx(1.37 * per_token, rel=0.02)


def test_dense_graph_does_not_invent_sparse_kv_footprint():
    graph = Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig())
    assert graph.kv_bytes_per_token_per_sequence is None
    assert graph.kv_fixed_bytes_per_sequence is None


def test_b300_headroom_admits_a_full_replica_where_b200_is_marginal(spec):
    """The actual B300 value on this checkpoint: memory, not compute.

    156 GB of weights leaves ~21 GB on a 192 GB B200 and ~109 GB on a 288 GB
    B300. Asserted against a concrete serving point rather than a magic token
    threshold, so the test says what it means: a data-parallel replica holding
    batch 256 at 32K context fits on one and not the other.
    """
    resident = model_weight_bytes(spec)
    batch, ctx = 256, 32768
    need = batch * (kv_fixed_bytes_per_sequence(spec) + ctx * kv_bytes_per_token(spec))

    assert 192e9 * 0.92 - resident < need  # B200: does not fit
    assert 288e9 * 0.92 - resident > need  # B300: fits, with room


def test_indexer_is_not_misfiled_as_elementwise():
    """`index` is an elementwise needle in the coarse taxonomy.

    A misfiled kernel is worse than an unrecognised one: `other` is a visible
    finding, while this would make attention look cheap and elementwise look
    inexplicably expensive.
    """
    from gitm.tracer.kernel_taxonomy import classify_kernel

    assert classify_kernel("lightning_indexer_topk_kernel") == "attention"
    assert classify_kernel("flash_mla_sparse_fwd_kernel") == "attention"
