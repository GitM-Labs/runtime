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

import textwrap
from dataclasses import replace

import pytest

from gitm.optimizer.deviation import classify_op
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.hybrid_graph import (
    HybridMoEModelSpec,
    attention_page_bytes,
    is_hybrid_moe_config,
    mamba_page_bytes,
    predict_hybrid_graph,
)
from gitm.planner.hybrid_graph import kv_bytes_per_token as hybrid_kv_bytes_per_token
from gitm.planner.hybrid_graph import (
    kv_fixed_bytes_per_sequence as hybrid_kv_fixed_bytes,
)
from gitm.planner.hybrid_graph import model_weight_bytes as hybrid_model_weight_bytes
from gitm.planner.hybrid_graph import spec_from_hf_config as hybrid_spec_from_hf_config
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.moe_graph import (
    effective_kv_tokens,
    index_candidates,
    kv_bytes_per_token,
    kv_entry_bytes,
    kv_fixed_bytes_per_sequence,
    model_weight_bytes,
    predict_moe_graph,
    spec_from_hf_config,
)
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    resolve_peak,
    weight_bytes,
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
    # Manifold-Constrained Hyper-Connections + hash-routed leading layers.
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "num_hash_layers": 3,
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


def test_csa_is_flat_in_context_but_hca_is_not(spec):
    """The two compressed mechanisms scale differently, and that is the point.

    CSA selects ``index_topk`` entries, so once selection saturates its core read
    is constant however long the context grows. HCA attends densely over its
    compressed entries, so its read scales as ``kv_len / m'`` without bound. They
    happen to coincide at 64K, which is exactly why a model that assumed one
    mechanism looked correct there and was 13x wrong at 1M.
    """
    assert effective_kv_tokens(spec, 2, 65536) == effective_kv_tokens(spec, 2, 1_048_576)

    hca_64k = effective_kv_tokens(spec, 3, 65536)
    hca_1m = effective_kv_tokens(spec, 3, 1_048_576)
    assert hca_1m > 10 * hca_64k

    # So the stack as a whole is not flat, and any claim that it is comes from
    # having modelled only one of the two.
    at_64k = sum(effective_kv_tokens(spec, i, 65536) for i in range(spec.n_layers))
    at_1m = sum(effective_kv_tokens(spec, i, 1_048_576) for i in range(spec.n_layers))
    assert at_1m > 5 * at_64k


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


def test_csa_and_hca_differ_in_kind_not_degree(spec):
    """They are two mechanisms, not one mechanism at two settings.

    CSA compresses lightly and *selects*; HCA compresses heavily and attends
    densely with no indexer at all. Treating the compression ratio as a single
    dial — which the config's flat list of numbers invites — produces an indexer
    on every compressed layer, half of which never run one.
    """
    assert spec.attention_kind(0) == "swa"
    assert spec.attention_kind(2) == "csa"
    assert spec.attention_kind(3) == "hca"

    assert index_candidates(spec, 2, 65536) > 0
    assert index_candidates(spec, 3, 65536) == 0  # HCA runs no indexer

    # And at long context the dense HCA read dwarfs the selected CSA one.
    assert effective_kv_tokens(spec, 3, 1_048_576) > 10 * effective_kv_tokens(
        spec, 2, 1_048_576
    )


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


def test_fp4_weight_bytes_include_the_block_scales():
    """MXFP4 is 0.5 bytes plus a 1-byte scale per 32 values, not 0.5 flat.

    6% of expert traffic on this checkpoint — larger than several effects the
    monitor is expected to resolve, so dropping it is not a rounding choice.
    """
    assert weight_bytes("fp4") == pytest.approx(0.5 + 1 / 32)
    assert weight_bytes("fp8") > 1.0  # block scales ride along
    assert weight_bytes("bf16") == 2.0


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


def test_priced_interconnect_is_not_flagged(spec, b200):
    g = predict_moe_graph(
        spec, b200, BatchConfig(batch=64, kv_cache_len=4096), ShardingConfig(tp=8, ep=8)
    )
    assert not g.has_unpriced_collectives


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


def test_base_checkpoint_has_no_dspark_nodes(base_spec, b200):
    """No dspark keys in the config means no dspark work in the graph.

    Which makes the base checkpoint the better pilot target: its only
    shape-estimated node is the grouped output projection.
    """
    g = predict_moe_graph(base_spec, b200, BatchConfig(batch=1, kv_cache_len=1024))
    assert not [n for n in g.nodes if n.op == "dspark"]
    assert {n.op for n in g.nodes if n.prediction.estimated} == {"attn_out_proj"}


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
    assert fixed == 2 * spec.sliding_window * kv_entry_bytes(spec)
    # In magnitude the fixed term is tiny — worth ~40 tokens of context, so it
    # never drives a sizing decision. What mattered was excluding these layers
    # from the *rate*, which is a 37% error on every sequence at every length.
    assert fixed < 50 * per_token
    naive = per_token + 2 * (
        kv_entry_bytes(spec) + spec.index_head_dim * weight_bytes(spec.kv_dtype)
    )
    assert naive == pytest.approx(1.37 * per_token, rel=0.02)


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

# ── The hybrid linear-attention MoE graph family ──────────────────────────────
#
# The hybrid linear-attention MoE graph.
#
# Two layer types with different asymptotics is the whole point of this family, so
# most of what follows is about keeping them apart: a graph that prices thirty
# gated-DeltaNet layers as though they held KV caches produces a plausible total
# and mis-attributes every context-dependent residual.
#
# Where a figure can be checked against something outside this repository it is —
# vLLM's own page-size arithmetic, and the published checkpoint size.

# Aliased on import: ``kv_bytes_per_token``, ``model_weight_bytes`` and
# ``spec_from_hf_config`` all exist in BOTH families with the same names and
# different meanings. A bare import here rebinds the sparse-MoE ones for the
# whole module, and every DeepSeek-V4 test above then silently runs against a
# Qwen reader — which is exactly what happened when these files were merged.

# Qwen/Qwen3.6-35B-A3B, trimmed to the fields the graph reads.
QWEN36 = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_size": 2048,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
    },
    "vision_config": {"depth": 27, "hidden_size": 1152},
}

H200 = HardwareSpec(
    name="H200",
    peak_flops_bf16_per_s=989e12,
    peak_flops_fp16_per_s=989e12,
    peak_mem_bw_bytes_per_s=4.8e12,
    interconnect_bw_bytes_per_s=900e9,
)


@pytest.fixture
def hybrid_spec() -> HybridMoEModelSpec:
    """Read from the checkpoint's own config, as the attach path does.

    Named distinctly from the sparse-MoE ``spec`` fixture above: a second
    module-level ``def hybrid_spec`` would rebind the first, silently handing every
    DeepSeek-V4 test a Qwen shape.
    """
    return hybrid_spec_from_hf_config(QWEN36, name="Qwen/Qwen3.6-35B-A3B")


@pytest.fixture
def catalogued() -> HybridMoEModelSpec:
    """Read from ``gitm/planner/models/*.yaml``, as an offline caller does."""
    from gitm.planner.model_catalogue import load_spec

    return load_spec("qwen3.6-35b-a3b")


# ── the two readers must agree ─────────────────────────────────────────────


def test_catalogue_entry_matches_the_checkpoint_config(hybrid_spec, catalogued):
    """Two independent paths to the same spec: a transcribed YAML file and the
    checkpoint's own ``config.json``.

    They can drift — a catalogue entry is hand-written, and a checkpoint can be
    re-uploaded with revised shapes. Drift is exactly what makes a stale
    catalogue dangerous, because it keeps producing confident predictions for a
    model that has changed underneath it. Comparing every field except the two
    that are legitimately different (``name``, and ``conv_dim``, which the YAML
    states explicitly and the config reader derives) pins that.
    """
    from dataclasses import fields

    differing = {
        f.name: (getattr(hybrid_spec, f.name), getattr(catalogued, f.name))
        for f in fields(HybridMoEModelSpec)
        if f.name != "name" and getattr(hybrid_spec, f.name) != getattr(catalogued, f.name)
    }
    assert differing == {}


# ── reading the checkpoint ──────────────────────────────────────────────────


def test_config_is_read_through_the_multimodal_wrapper(hybrid_spec):
    """Every shape sits under ``text_config``; the top level carries only the
    vision tower and token ids. A reader looking at the top level finds nothing
    and falls back to defaults that describe a different model."""
    assert hybrid_spec.n_layers == 40
    assert hybrid_spec.hidden == 2048
    assert hybrid_spec.vocab == 248320


def test_the_layer_schedule_is_read_not_inferred(hybrid_spec):
    """Phase matters and an interval does not carry it.

    Qwen places full attention at layers 3, 7 ... 39. A modulo rule guessed from
    ``full_attention_interval`` alone puts them at 0, 4 ... 36 — the right
    *count* in the wrong *places*, which produces a believable total while every
    per-layer residual is compared against the other layer type.
    """
    assert hybrid_spec.n_full_attention_layers == 10
    assert hybrid_spec.n_linear_attention_layers == 30
    assert [i for i in range(40) if hybrid_spec.is_full_attention(i)] == list(range(3, 40, 4))


def test_interval_fallback_is_phased_to_end_on_a_full_attention_layer():
    """Without an explicit schedule the interval must still land layer 39."""
    s = HybridMoEModelSpec(n_layers=40, layer_types=(), full_attention_interval=4)
    assert [i for i in range(40) if s.is_full_attention(i)] == list(range(3, 40, 4))


def test_head_dim_is_read_rather_than_divided(hybrid_spec):
    """``head_dim`` is 256 against a 2048 hidden size, so the query projection
    *widens* to 4096. Deriving it as ``hidden / n_heads`` gives 128 and halves
    every attention projection."""
    assert hybrid_spec.head_dim == 256
    assert hybrid_spec.q_dim == 4096
    assert hybrid_spec.q_dim > hybrid_spec.hidden


def test_partial_rotary_factor_is_honoured(hybrid_spec):
    """Only a quarter of each head rotates; charging the whole head is 4x."""
    assert hybrid_spec.rope_dim == 64


def test_state_dtype_is_independent_of_the_model_dtype(hybrid_spec):
    """fp32 state under a bf16 model. Collapsing them halves the term that
    dominates the linear layers at low batch."""
    assert hybrid_spec.weight_dtype == "bf16"
    assert hybrid_spec.ssm_state_dtype == "fp32"


# ── the observable cross-check ─────────────────────────────────────────────


def test_page_arithmetic_reproduces_the_engines_own_padding(hybrid_spec):
    """vLLM equalises the attention and mamba page sizes and logs what it did:

        Setting attention block size to 1056 tokens ...
        Padding mamba page size by 0.76%

    Both numbers follow from the config, so reproducing them checks the KV and
    state arithmetic against something outside this repository. This is the
    tightest external check the family has.
    """
    attn = attention_page_bytes(hybrid_spec, 1056)
    mamba = mamba_page_bytes(hybrid_spec)
    assert attn == pytest.approx(2_162_688)
    assert mamba == pytest.approx(2_146_304)
    assert attn / mamba - 1 == pytest.approx(0.0076, abs=5e-5)


def test_predicted_weights_land_near_the_published_checkpoint(hybrid_spec):
    """66.97 GiB on disk (71.9 GB). The residual is norms, biases and per-tensor
    metadata this does not enumerate — the same class of gap as the sparse-MoE
    graph's -2.4%."""
    assert hybrid_model_weight_bytes(hybrid_spec) / 71.9e9 == pytest.approx(1.0, abs=0.06)


# ── the asymptotics that separate the two layer types ──────────────────────


def test_only_full_attention_layers_contribute_to_per_token_kv(hybrid_spec):
    """Thirty of forty layers keep no KV cache. Counting them would inflate the
    per-token rate 4x and cap concurrency far below what the hardware allows."""
    per_layer = 2 * hybrid_spec.kv_dim * 2  # K and V, bf16
    assert hybrid_kv_bytes_per_token(hybrid_spec) == pytest.approx(10 * per_layer)
    assert hybrid_kv_bytes_per_token(hybrid_spec) == pytest.approx(20 * 1024)


def test_linear_attention_state_is_flat_in_context():
    """The defining property. If this ever scales with ``kv_cache_len`` the
    family has collapsed back into a KV-cached model and every long-context
    conclusion drawn from it is wrong."""
    hybrid_spec = HybridMoEModelSpec()
    short = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=4, kv_cache_len=1024))
    long = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=4, kv_cache_len=131072))

    def lin_bytes(g):
        return sum(n.prediction.bytes for n in g.nodes if n.op == "linattn_recurrent")

    assert lin_bytes(short) == pytest.approx(lin_bytes(long))


def test_full_attention_traffic_does_grow_with_context():
    """The counterpart: if the ten attention layers were also flat, the graph
    would predict a model with no context cost at all."""
    hybrid_spec = HybridMoEModelSpec()
    short = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=4, kv_cache_len=1024))
    long = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=4, kv_cache_len=131072))

    def attn_bytes(g):
        return sum(n.prediction.bytes for n in g.nodes if n.op == "attn_score_value")

    assert attn_bytes(long) > 100 * attn_bytes(short)


def test_expert_weight_traffic_saturates_with_batch(catalogued):
    """The set-union term. FLOPs scale linearly with batch; weight traffic
    saturates at ``num_experts``, which is why MoE decode is bandwidth-bound at
    low batch and only becomes compute-bound well past the knee.

    Uses the catalogued checkpoint rather than the reference default: the
    numbers below (a knee near 32, a 32x ceiling) are properties of top-8-of-256
    routing, and asserting them against a spec that happens to be top-2-of-8
    would be checking arithmetic rather than the model.
    """
    hybrid_spec = catalogued

    def routed(b):
        g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=b, kv_cache_len=1024))
        ns = [n for n in g.nodes if n.op == "moe_routed"]
        return sum(n.prediction.flops for n in ns), sum(n.prediction.bytes for n in ns)

    f1, b1 = routed(1)
    f64, b64 = routed(64)
    f1024, b1024 = routed(1024)

    # FLOPs are exactly linear in batch.
    assert f64 / f1 == pytest.approx(64, rel=0.01)
    assert f1024 / f1 == pytest.approx(1024, rel=0.01)

    # Bytes are strictly sublinear, and the gap widens with batch. The knee sits
    # near ``num_experts / top_k`` == 32, so batch 64 is only just past it and
    # still grows fast; by 1024 every expert is awake and the term is pinned.
    assert b64 / b1 < 0.6 * (f64 / f1)
    assert b1024 / b1 < 0.05 * (f1024 / f1)

    # Saturation ceiling. The *weight* component caps at 32x its batch-1 value
    # (all 256 experts awake against 8), but the node's byte total also carries
    # activation traffic, which is honestly linear in batch and does not
    # saturate. So the node total may exceed 32x while the term this test is
    # about does not — checked against ``distinct_experts`` directly, since that
    # is where the ceiling actually lives.
    from gitm.planner.roofline import distinct_experts

    assert distinct_experts(1024, 256, 8) / distinct_experts(1, 256, 8) <= 256 / 8
    assert b1024 / b1 < 40


# ── graph shape ────────────────────────────────────────────────────────────


def test_each_layer_emits_the_nodes_for_its_own_kind(hybrid_spec):
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=2, kv_cache_len=512))
    by_layer: dict[int, set[str]] = {}
    for n in g.nodes:
        if n.layer is not None:
            by_layer.setdefault(n.layer, set()).add(n.op)

    assert "attn_score_value" in by_layer[3]
    assert "linattn_recurrent" not in by_layer[3]
    assert "linattn_recurrent" in by_layer[0]
    assert "attn_score_value" not in by_layer[0]
    # The MoE runs on every layer regardless of attention kind.
    assert all("moe_routed" in ops for ops in by_layer.values())


def test_shared_expert_is_priced_when_declared(hybrid_spec):
    """Every token pays it on top of its eight routed experts; omitting it
    under-counts activated FFN width by an eighth."""
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=1))
    assert sum(1 for n in g.nodes if n.op == "moe_shared") == hybrid_spec.n_layers


def test_shared_expert_absent_when_the_config_does_not_declare_one():
    s = HybridMoEModelSpec(shared_expert_intermediate_size=0)
    g = predict_hybrid_graph(s, H200, BatchConfig(batch=1))
    assert not any(n.op == "moe_shared" for n in g.nodes)


def test_mtp_head_is_opt_in(hybrid_spec):
    """The checkpoint declares one, but vLLM builds it only under a speculative
    config. A node no kernel can match reads as a permanently negative residual
    for an op that never ran — worse than an absent node."""
    default = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=1))
    with_mtp = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=1), with_mtp=True)
    assert max(n.layer for n in default.nodes if n.layer is not None) == 39
    assert max(n.layer for n in with_mtp.nodes if n.layer is not None) == 40


def test_lm_head_is_emitted_once_and_unlayered(hybrid_spec):
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=1))
    heads = [n for n in g.nodes if n.op == "lm_head"]
    assert len(heads) == 1
    assert heads[0].layer is None


# ── sharding ───────────────────────────────────────────────────────────────


def test_tensor_parallelism_divides_the_projections(hybrid_spec):
    b = BatchConfig(batch=8, kv_cache_len=4096)
    one = predict_hybrid_graph(hybrid_spec, H200, b, ShardingConfig(tp=1))
    two = predict_hybrid_graph(hybrid_spec, H200, b, ShardingConfig(tp=2))

    def proj(g):
        return sum(n.prediction.flops for n in g.nodes if n.op == "linattn_in_proj")

    assert proj(two) == pytest.approx(proj(one) / 2)


def test_sharded_graph_emits_the_collective_it_pays_for(hybrid_spec):
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=8), ShardingConfig(tp=2))
    assert any(n.op == "tp_all_reduce" for n in g.nodes)
    assert not g.has_unpriced_collectives


def test_indivisible_sharding_is_refused_rather_than_floored(hybrid_spec):
    """Head counts floor-divide throughout, so an indivisible split prices a
    whole path at zero work — a cheap, confident, completely wrong graph."""
    with pytest.raises(ValueError, match="does not divide"):
        predict_hybrid_graph(hybrid_spec, H200, BatchConfig(), ShardingConfig(tp=3))


def test_indivisible_linear_head_split_is_refused_too():
    s = HybridMoEModelSpec(n_heads=16, linear_num_value_heads=6)
    with pytest.raises(ValueError, match="linear-attention value heads"):
        predict_hybrid_graph(s, H200, BatchConfig(), ShardingConfig(tp=4))


# ── degenerate inputs ──────────────────────────────────────────────────────


def test_zero_layers_is_refused():
    with pytest.raises(ValueError, match="n_layers"):
        predict_hybrid_graph(HybridMoEModelSpec(n_layers=0), H200, BatchConfig())


def test_rotary_wider_than_the_head_is_refused():
    with pytest.raises(ValueError, match="rotary"):
        predict_hybrid_graph(
            HybridMoEModelSpec(partial_rotary_factor=2.0), H200, BatchConfig()
        )


def test_empty_context_does_not_collapse_the_graph(hybrid_spec):
    """A step with nothing cached still runs every projection and every expert.
    A zero total here would mean the graph silently priced the model at nothing."""
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=1, kv_cache_len=0))
    assert g.total_pred_s > 0
    assert all(n.prediction.t_pred_s >= 0 for n in g.nodes)


def test_no_node_predicts_zero_time(hybrid_spec):
    """Every emitted node represents a kernel that runs. A zero-time node is a
    term that silently dropped out of the model — the failure mode that only
    shows up as an unexplained residual much later."""
    g = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=8, kv_cache_len=4096))
    zero = {n.op for n in g.nodes if n.prediction.t_pred_s <= 0}
    assert zero == set()


# ── dispatch ───────────────────────────────────────────────────────────────


def test_detector_accepts_a_mixed_layer_schedule():
    assert is_hybrid_moe_config(QWEN36)


def test_detector_rejects_a_uniform_moe():
    """Routed experts alone describe a Mixtral, which the dense graph prices."""
    cfg = {"num_experts": 8, "num_experts_per_tok": 2, "num_hidden_layers": 32}
    assert not is_hybrid_moe_config(cfg)


def test_detector_rejects_a_pure_linear_model():
    """Linear attention alone describes a Mamba — no experts, no dispatch here."""
    cfg = {"layer_types": ["linear_attention"] * 48, "num_hidden_layers": 48}
    assert not is_hybrid_moe_config(cfg)


def test_registry_routes_qwen_to_the_hybrid_family():
    from gitm.planner.registry import detect_family, predict_for_config

    assert detect_family(QWEN36) == "hybrid"
    g, family = predict_for_config(QWEN36, H200, BatchConfig(batch=4))
    assert family == "hybrid"
    assert any(n.op == "linattn_recurrent" for n in g.nodes)


def test_registry_still_routes_deepseek_to_the_sparse_moe_family():
    """The narrowing order must not have stolen the family it was added beside."""
    from gitm.planner.registry import detect_family

    cfg = {
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "index_topk": 512,
        "num_hidden_layers": 43,
    }
    assert detect_family(cfg) == "sparse_moe"


def test_registry_refuses_the_dense_family_rather_than_guessing():
    from gitm.planner.registry import predict_for_config

    with pytest.raises(NotImplementedError, match="dense graph"):
        predict_for_config({"num_hidden_layers": 32, "hidden_size": 4096})

# ── Checkpoint shapes as data: gitm/planner/models/*.yaml ─────────────────────
#
# Checkpoint shapes loaded from YAML rather than baked into dataclass defaults.
#
# The defect this replaces: the spec's defaults *were* one particular checkpoint,
# so constructing a spec with no arguments silently described that model and every
# derived figure came out plausible while answering a question nobody asked. The
# same class of error as a hardware catalogue miss resolving to A100 defaults —
# not an exception, just a confident wrong answer.
#
# So the tests here are mostly about the ways a catalogue can lie: a typo that
# loads as a default, a layer schedule that does not cover the model, an entry
# that has drifted from the checkpoint it claims to describe.



# ── the defaults must not be a real checkpoint ─────────────────────────────


def test_bare_construction_is_obviously_not_a_deployment():
    """The regression this file exists for.

    A bare spec must be small enough that anyone reading a derived figure sees
    immediately that it describes nothing real. A plausible-looking default is
    worse than an absurd one: it survives review.
    """
    from gitm.planner.hybrid_graph import model_weight_bytes as hybrid_model_weight_bytes

    d = HybridMoEModelSpec()
    assert hybrid_model_weight_bytes(d) < 1e9        # under a gigabyte
    assert d.n_layers < 10
    assert "reference" in d.name


def test_defaults_do_not_match_any_catalogued_checkpoint():
    """Stronger than a size bound: no field-by-field match with a real entry."""
    d = HybridMoEModelSpec()
    for entry in available():
        hybrid_spec = load_spec(entry)
        assert (hybrid_spec.hidden, hybrid_spec.n_layers, hybrid_spec.num_experts) != (
            d.hidden, d.n_layers, d.num_experts
        ), f"reference defaults have drifted onto catalogue entry {entry!r}"


# ── loading ────────────────────────────────────────────────────────────────


def test_the_qwen_entry_is_present_and_declares_its_family():
    assert "qwen3.6-35b-a3b" in available()
    entry = load_entry("qwen3.6-35b-a3b")
    assert entry["family"] == "hybrid"
    assert entry["name"] == "Qwen/Qwen3.6-35B-A3B"


def test_an_entry_carries_its_own_provenance():
    """A fitted constant that reads like a transcribed one is how an estimate
    becomes a fact. The entry has to say which is which."""
    entry = load_entry("qwen3.6-35b-a3b")
    prov = entry["provenance"]
    assert any(e["field"] == "conv_dim" for e in prov["estimated"])
    assert prov["verified"] and prov["unmodelled"]


def test_predict_returns_a_graph_and_its_family():
    g, family = predict("qwen3.6-35b-a3b", H200, BatchConfig(batch=4, kv_cache_len=512))
    assert family == "hybrid"
    assert any(n.op == "linattn_recurrent" for n in g.nodes)


def test_an_unknown_name_lists_what_is_available():
    with pytest.raises(FileNotFoundError, match="qwen3.6-35b-a3b"):
        load_spec("no-such-model")


def test_a_path_to_a_yaml_file_is_accepted(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(textwrap.dedent("""
        name: tiny
        family: hybrid
        spec:
          hidden: 128
          n_layers: 2
    """))
    assert load_spec(p).hidden == 128


# ── validation ─────────────────────────────────────────────────────────────


def test_an_unknown_field_is_an_error_not_a_shrug(tmp_path):
    """A mistyped key would otherwise be dropped silently, leaving the spec
    holding a reference default while the file appears to set it — which is the
    original defect wearing a different hat."""
    p = tmp_path / "typo.yaml"
    p.write_text(textwrap.dedent("""
        name: typo
        family: hybrid
        spec:
          hidden: 2048
          num_expert: 256
    """))
    with pytest.raises(ValueError, match="unknown spec field"):
        load_spec(p)


def test_an_unknown_family_is_refused(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nfamily: transformer\nspec: {hidden: 8}\n")
    with pytest.raises(ValueError, match="family must be one of"):
        load_spec(p)


def test_a_missing_spec_block_is_refused(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("name: x\nfamily: hybrid\n")
    with pytest.raises(ValueError, match="missing a 'spec'"):
        load_spec(p)


# ── the layer schedule ─────────────────────────────────────────────────────


def test_the_compact_pattern_expands_to_the_full_schedule():
    """Forty entries written out is unambiguous but unreadable, and an
    unreadable schedule is one nobody checks."""
    hybrid_spec = load_spec("qwen3.6-35b-a3b")
    assert len(hybrid_spec.layer_types) == 40
    assert [i for i in range(40) if hybrid_spec.is_full_attention(i)] == list(range(3, 40, 4))


def test_a_pattern_that_does_not_tile_the_model_is_refused(tmp_path):
    """Silently truncating would leave the trailing layers taking the last
    entry's kind — a schedule that covers the model in appearance only."""
    p = tmp_path / "short.yaml"
    p.write_text(textwrap.dedent("""
        name: short
        family: hybrid
        spec:
          n_layers: 40
          layer_types:
            pattern: [linear_attention, full_attention]
            repeat: 3
    """))
    with pytest.raises(ValueError, match="expands to 6 entries but n_layers is 40"):
        load_spec(p)


def test_a_plain_list_is_still_accepted(tmp_path):
    p = tmp_path / "plain.yaml"
    p.write_text(textwrap.dedent("""
        name: plain
        family: hybrid
        spec:
          n_layers: 2
          layer_types: [linear_attention, full_attention]
    """))
    assert load_spec(p).n_full_attention_layers == 1


def test_a_malformed_pattern_is_refused(tmp_path):
    p = tmp_path / "malformed.yaml"
    p.write_text(textwrap.dedent("""
        name: malformed
        family: hybrid
        spec:
          n_layers: 4
          layer_types:
            pattern: []
            repeat: 2
    """))
    with pytest.raises(ValueError, match="non-empty list"):
        load_spec(p)


# ── the config reader refuses to substitute ────────────────────────────────


@pytest.mark.parametrize("missing", [
    "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "linear_num_value_heads", "linear_value_head_dim",
])
def test_a_config_missing_a_required_shape_raises(missing):
    """Substituting another checkpoint's value is how a graph ends up
    confidently describing a model it never read."""
    text = {
        "hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    }
    del text[missing]
    with pytest.raises(ValueError, match=missing):
        hybrid_spec_from_hf_config({"text_config": text})


def test_head_dim_falls_back_to_the_huggingface_convention():
    """Omitting ``head_dim`` is legitimate and means ``hidden / n_heads``. That
    is a function of *this* config, unlike a constant lifted from another
    checkpoint — but it is also the division that is wrong for models which
    widen the query projection, so the distinction is worth keeping visible."""
    hybrid_spec = hybrid_spec_from_hf_config({"text_config": {
        "hidden_size": 2048, "num_hidden_layers": 4, "vocab_size": 1000,
        "num_attention_heads": 16, "num_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 256, "linear_num_value_heads": 8,
        "linear_value_head_dim": 64,
    }})
    assert hybrid_spec.head_dim == 128


# ── prefill ─────────────────────────────────────────────────────────────────
#
# Prefill is not decode with a larger batch. Three things change in kind:
#
#   * attention becomes quadratic within the chunk, so its FLOPs stop tracking
#     `rows x kv_len`;
#   * gated DeltaNet switches algorithm entirely, from a per-token recurrent scan
#     to a chunked matmul form that touches the state once per chunk;
#   * only the last token of a prompt needs logits, so lm_head does not scale
#     with the chunk at all.
#
# Get any of them wrong and the prediction is off by the chunk width, which at
# 8192 tokens is three orders of magnitude.


def _pre(P, ctx=0, reqs=1, **kw):
    return BatchConfig(batch=0, kv_cache_len=0, prefill_tokens=P,
                       prefill_context=ctx, prefill_requests=reqs, **kw)


def _totals(g):
    return (sum(n.prediction.flops for n in g.nodes),
            sum(n.prediction.bytes for n in g.nodes))


def test_a_default_batchconfig_is_still_a_pure_decode_step():
    """Every existing caller predates prefill and must be unaffected."""
    b = BatchConfig(batch=8, kv_cache_len=1024)
    assert not b.is_prefill
    assert b.attention_qk_pairs() == 8 * 1024
    assert b.logits_rows == 8


def test_prefill_attention_is_quadratic_in_the_chunk():
    """P tokens attend to prior context *and* to their own causal prefix. The
    second term is what makes prefill compute-bound; omitting it would price an
    8192-token chunk as if it were 8192 independent decode queries."""
    b = _pre(8192, ctx=0)
    assert b.attention_qk_pairs() == 8192 * 8193 / 2

    with_ctx = _pre(8192, ctx=4096)
    assert with_ctx.attention_qk_pairs() == 8192 * 4096 + 8192 * 8193 / 2


def test_lm_head_charges_one_row_per_prompt_not_one_per_token():
    """The largest overcount available here: a 248k-row vocabulary projection
    against 8192 rows instead of 1."""
    assert _pre(8192, reqs=1).logits_rows == 1
    assert _pre(8192, reqs=4).logits_rows == 4
    # A mixed step bills its decode positions too.
    assert BatchConfig(batch=8, prefill_tokens=512, prefill_requests=2).logits_rows == 10


def test_prefill_flips_the_step_from_memory_bound_to_compute_bound(hybrid_spec):
    """The qualitative result. Decode reads weights to serve a handful of tokens;
    prefill reuses each weight across the whole chunk, so arithmetic intensity
    crosses the hardware ridge and the binding constraint changes."""
    def bound_counts(bc):
        g = predict_hybrid_graph(hybrid_spec, H200, bc)
        return sum(1 for n in g.nodes if n.prediction.bound == "compute"), len(g.nodes)

    dec_compute, _ = bound_counts(BatchConfig(batch=8, kv_cache_len=1024))
    pre_compute, total = bound_counts(_pre(8192))
    assert dec_compute == 0
    assert pre_compute > total / 2


def test_prefill_amortises_weight_traffic_across_the_chunk(hybrid_spec):
    """Bytes are dominated by weights, which are read once however wide the step.
    So 16x the tokens must cost far less than 16x the bytes."""
    _, b512 = _totals(predict_hybrid_graph(hybrid_spec, H200, _pre(512)))
    _, b8192 = _totals(predict_hybrid_graph(hybrid_spec, H200, _pre(8192)))
    assert b8192 / b512 < 2.0


def test_gdn_state_traffic_scales_with_chunks_not_tokens(hybrid_spec):
    """The chunked delta rule touches the recurrent state once per ``GDN_CHUNK``
    tokens, not once per token. Modelling prefill as a wide decode would
    overstate this node's bytes 64x and make the thirty GDN layers look like a
    bottleneck they are not — which is exactly what a first pass produced."""
    from gitm.planner.hybrid_graph import GDN_CHUNK

    def state_bytes(bc):
        g = predict_hybrid_graph(hybrid_spec, H200, bc)
        return sum(n.prediction.bytes for n in g.nodes if n.op == "linattn_recurrent")

    b1, b2 = state_bytes(_pre(GDN_CHUNK * 8)), state_bytes(_pre(GDN_CHUNK * 16))
    assert 1.5 < b2 / b1 < 2.5          # doubles with chunk count

    # And a single chunk costs about what one decode step's state read costs.
    one_chunk = state_bytes(_pre(GDN_CHUNK))
    one_step = state_bytes(BatchConfig(batch=1, kv_cache_len=0))
    assert 0.5 < one_chunk / one_step < 3.0


def test_gdn_arithmetic_prices_against_the_activation_peak_not_the_state_dtype(
    hybrid_spec,
):
    """`ssm_state_dtype` is fp32 and governs how the state is *stored*. The
    arithmetic is tensor-core matmuls in the model dtype. Passing fp32 as the op
    dtype selects the fp32 FLOP peak, which the SKU catalogue does not populate,
    so it falls back to an A100's 19.5 TF/s against an H200's 989 — a 50x penalty
    that made this node read as 57% of a prefill step instead of ~14%."""
    g = predict_hybrid_graph(hybrid_spec, H200, _pre(8192))
    node = next(n for n in g.nodes if n.op == "linattn_recurrent")
    assert node.prediction.peak_flops_per_s == H200.peak_flops_bf16_per_s
    assert not node.prediction.peak_is_fallback


def test_a_mixed_step_costs_both_phases(hybrid_spec):
    """Chunked prefill routinely runs both in one fused forward. Their costs are
    additive within an op, so the graph keeps one node per op per layer and
    residuals stay comparable with a decode-only capture."""
    dec = predict_hybrid_graph(hybrid_spec, H200, BatchConfig(batch=8, kv_cache_len=1024))
    mix = predict_hybrid_graph(
        hybrid_spec, H200,
        BatchConfig(batch=8, kv_cache_len=1024, prefill_tokens=512, prefill_requests=1),
    )
    assert len(mix.nodes) == len(dec.nodes)
    assert _totals(mix)[0] > _totals(dec)[0]


def test_per_token_cost_falls_sharply_with_chunk_width(hybrid_spec):
    """The reason prefill is batched at all."""
    def per_token(P):
        return predict_hybrid_graph(hybrid_spec, H200, _pre(P)).total_pred_s / P

    assert per_token(8192) < per_token(512) / 4


def test_every_gdn_projection_scales_with_prefill_tokens(hybrid_spec):
    """The projections and the convolution process every query token regardless
    of phase — only the recurrent state's *algorithm* differs.

    A first pass converted the recurrent node alone and left the projections on
    `positions`, so a pure prefill step (positions == 0) gave `linattn_in_proj`
    zero FLOPs. It showed as arithmetic intensity 0.0 and 1% of the step, when it
    is in fact 27% and compute-bound.
    """
    g = predict_hybrid_graph(hybrid_spec, H200, _pre(8192))
    for op in ("linattn_in_proj", "linattn_conv", "attn_out_proj"):
        flops = sum(n.prediction.flops for n in g.nodes if n.op == op)
        assert flops > 0, f"{op} has no arithmetic on a pure prefill step"


def test_a_pure_prefill_step_reports_prefill_throughput(capsys):
    """Dividing by a zero decode batch printed `0 tok/s at batch 0`."""
    from gitm.planner.registry import main

    main(["qwen3.6-35b-a3b", "--gpu", "H200", "--prefill-tokens", "8192",
          "--batch", "0", "--kv-len", "0"])
    out = capsys.readouterr().out
    assert "prefilling 8,192 tokens" in out
    assert "0 tok/s at batch 0" not in out


# ── MiMo-V2.5: the windowed-attention hybrid ─────────────────────────────────
#
# A third member of the hybrid family, and the one that breaks the family's
# founding assumption that "not full attention" means "recurrent". MiMo's 39
# sliding-window layers re-read a KV cache exactly as full attention does — they
# just read at most 128 entries of it. Priced as linear their cache read
# vanishes; priced as unbounded it is overstated 64x at 8k context. Each test
# below pins one number that would otherwise be confidently wrong.
#
# Aliased imports, as above: this module hosts three families whose readers share
# function names.

# The shape of XiaomiMiMo/MiMo-V2.5, trimmed to the keys the planner reads —
# the same convention as ``V4_CONFIG`` above, and for the same reason: a test
# that reads a checkpoint off a developer's disk skips everywhere else, so the
# tests that matter most (the family it classifies as, and the inverted
# ``hybrid_layer_pattern`` polarity) would never run in CI.
#
# ``test_the_two_readers_agree`` is what keeps this honest: it compares every
# non-exempt field of the spec built from here against the catalogue YAML, so a
# key dropped in trimming fails loudly rather than quietly weakening the check.
# Verified spec-identical to the full 57-key config.json at the time of writing.
MIMO_CONFIG = {
    'model_type': 'mimo_v2',
    'hidden_size': 4096,
    'num_hidden_layers': 48,
    'vocab_size': 152576,
    'num_attention_heads': 64,
    'num_key_value_heads': 4,
    'head_dim': 192,
    'v_head_dim': 128,
    'partial_rotary_factor': 0.334,
    'sliding_window': 128,
    'sliding_window_size': 128,
    'swa_num_key_value_heads': 8,
    'swa_head_dim': 192,
    'swa_v_head_dim': 128,
    'swa_rope_theta': 10000,
    'add_swa_attention_sink_bias': True,
    'n_routed_experts': 256,
    'num_experts_per_tok': 8,
    'moe_intermediate_size': 2048,
    'intermediate_size': 16384,
    'moe_layer_freq': [
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    'hybrid_layer_pattern': [
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ],
    'dtype': 'bfloat16',
    'quantization_config': {'quant_method': 'fp8'},
}


@pytest.fixture
def mimo():
    """Read from the catalogue, as an offline caller does."""
    return load_spec("mimo-v2.5")


@pytest.fixture
def mimo_cfg():
    """Read from the checkpoint's own config, as the attach path does."""
    return MIMO_CONFIG


# ── G1: it classifies at all ────────────────────────────────────────────────


def test_mimo_classifies_as_hybrid_not_dense(mimo_cfg):
    """Before the key aliases, ``detect_family`` returned ``dense`` and the
    planner raised NotImplementedError: ``is_hybrid_moe_config`` wanted
    ``num_experts`` + ``layer_types`` and MiMo declares neither key, while
    ``is_sparse_moe_config`` wanted ``index_topk``/``compress_ratios``. The most
    interesting model in the brief landed in the one family with no reader."""
    from gitm.planner.registry import detect_family

    assert detect_family(mimo_cfg) == "hybrid"


def test_deepseek_v4_still_classifies_as_sparse_moe():
    """The G1 alias widened ``is_hybrid_moe_config`` to accept
    ``n_routed_experts`` — which **DeepSeek-V4 also declares**, and
    ``detect_family`` consults the hybrid predicate first. V4 declares no
    schedule under either spelling, so it must still fall through. If this ever
    fails, every V4 prediction is being produced by the wrong graph."""
    from gitm.planner.registry import detect_family

    assert detect_family(V4_CONFIG) == "sparse_moe"


def test_hybrid_layer_pattern_polarity_is_inverted(mimo_cfg):
    """``hybrid_layer_pattern[i] == 1`` means WINDOWED, the opposite of how
    ``layer_types`` reads (modeling_mimo_v2.py:400). Reading it the intuitive way
    swaps 39 layers for 9 — the right node count, a plausible total, and every
    per-layer residual compared against the wrong mechanism."""
    spec = hybrid_spec_from_hf_config(mimo_cfg, name="MiMo")
    full = [i for i in range(spec.n_layers) if spec.is_full_attention(i)]
    assert full == [0, 5, 11, 17, 23, 29, 35, 41, 47]
    assert spec.n_sliding_attention_layers == 39
    assert spec.n_linear_attention_layers == 0


def test_no_modulo_rule_reproduces_the_schedule(mimo):
    """The gaps are 5,6,6,6,6,6,6,6 — layer 0 breaks the stride. This is why the
    catalogue writes 48 entries out instead of using {pattern, repeat}: an
    interval of 6 would place full attention at 5,11,...,47 and *miss layer 0*,
    which is also the only dense-MLP layer."""
    full = [i for i in range(mimo.n_layers) if mimo.is_full_attention(i)]
    gaps = [b - a for a, b in zip(full, full[1:], strict=False)]
    assert gaps == [5, 6, 6, 6, 6, 6, 6, 6]
    assert 0 in full


def test_the_two_readers_agree(mimo, mimo_cfg):
    """A hand-written YAML and the checkpoint's own config, compared field by
    field. Drift is what makes a stale catalogue dangerous: it keeps producing
    confident predictions for a model that changed underneath it."""
    from dataclasses import fields

    read = hybrid_spec_from_hf_config(mimo_cfg, name=mimo.name)
    # Fields the config genuinely cannot carry: the MTP head is visible only in
    # the safetensors index, and the collective count is a TP convention.
    exempt = {
        "name",
        # Visible only as tensors in the safetensors index.
        "mtp_num_hidden_layers", "mtp_layer_kind", "mtp_dense",
        # A TP convention and a precision reading, neither declared as a shape.
        "all_reduces_per_layer", "op_dtype_overrides",
        # Gated-DeltaNet geometry: MiMo has no recurrent layers, so the reader
        # leaves these at placeholders rather than demanding fields that govern
        # no kernel.
        "linear_num_value_heads", "linear_value_head_dim",
        "linear_num_key_heads", "linear_key_head_dim", "conv_dim",
    }
    differing = {
        f.name: (getattr(read, f.name), getattr(mimo, f.name))
        for f in fields(HybridMoEModelSpec)
        if f.name not in exempt and getattr(read, f.name) != getattr(mimo, f.name)
    }
    assert differing == {}


def test_mtp_count_is_not_config_derivable(mimo_cfg):
    """MiMo ships three draft layers and declares them nowhere in config.json —
    they exist only as tensors in the safetensors index. A reader that invented a
    count here would predict a head it never saw, so a bare config must yield 0
    and the catalogue must carry the real number."""
    assert "mtp_num_hidden_layers" not in mimo_cfg
    assert hybrid_spec_from_hf_config(mimo_cfg).mtp_num_hidden_layers == 0
    assert load_spec("mimo-v2.5").mtp_num_hidden_layers == 3


# ── G2: the window is a third asymptotic ────────────────────────────────────


def test_sliding_window_layers_are_flat_in_context(mimo):
    """The defining property. A windowed layer reads at most W entries however
    long the context grows; a full-attention layer's read grows without bound.
    If these ever move together, the third layer kind has collapsed back into one
    of the other two."""
    def core_bytes(kv_len, pred):
        by_layer = {}
        for n in pred.nodes:
            if n.op == "attn_score_value":
                by_layer[n.layer] = n.prediction.bytes
        return by_layer

    short = core_bytes(8192, predict_hybrid_graph(
        mimo, H200, BatchConfig(batch=4, kv_cache_len=8192)))
    long = core_bytes(131072, predict_hybrid_graph(
        mimo, H200, BatchConfig(batch=4, kv_cache_len=131072)))

    # 16x the context.
    assert short[1] == pytest.approx(long[1])          # windowed: unchanged
    assert long[0] == pytest.approx(short[0] * 16)     # full: grows with it


def test_priced_as_unbounded_the_window_costs_64x_too_much(mimo):
    """The error the window prevents, stated as a number. At 8k context against a
    128-token window the ratio is exactly kv_len/window."""
    windowed = predict_hybrid_graph(mimo, H200, BatchConfig(batch=1, kv_cache_len=8192))
    unwindowed = predict_hybrid_graph(
        replace(mimo, sliding_window=0), H200, BatchConfig(batch=1, kv_cache_len=8192))

    def swa_core(g):
        return sum(n.prediction.bytes for n in g.nodes
                   if n.op == "attn_score_value" and n.layer == 1)

    assert swa_core(unwindowed) / swa_core(windowed) == pytest.approx(8192 / 128)


def test_a_windowed_layer_priced_as_linear_would_lose_its_cache_read(mimo):
    """The §7.2 failure mode: alias the config keys without adding the third kind
    and MiMo classifies as hybrid, runs, and prices 39 of 48 layers as
    gated-DeltaNet recurrent state — which is *also* flat in context, so even the
    qualitative behaviour looks right while the number is wrong and nothing says
    so. A windowed layer must emit a real cache read."""
    g = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8, kv_cache_len=8192))
    swa = [n for n in g.nodes if n.op == "attn_score_value" and n.layer == 1]
    assert swa and swa[0].prediction.bytes > 0
    # ...and no recurrent nodes anywhere, since there are no linear layers.
    assert not [n for n in g.nodes if n.op == "linattn_recurrent"]


def test_windowed_prefill_is_banded_not_quadratic():
    """Prefill's causal triangle becomes a band once a window caps it. At P=8192
    and W=128 that is 1,040,448 query-key pairs against 33,558,528 — a **32x
    overcount** on the term that decides whether prefill is compute-bound."""
    b = BatchConfig(batch=1, kv_cache_len=0, prefill_tokens=8192, prefill_requests=1)
    assert b.attention_qk_pairs() == 8192 * 8193 / 2          # unwindowed
    assert b.attention_qk_pairs(window=128) == 1_040_448
    assert b.attention_qk_pairs() / b.attention_qk_pairs(128) == pytest.approx(32.3, abs=0.1)


def test_banded_pair_count_matches_a_brute_force_sum():
    """The closed form is an optimisation of a sum over tokens; if they ever
    disagree the closed form is wrong, not the sum."""
    for p, ctx, w in [(100, 0, 128), (100, 50, 128), (100, 500, 128), (37, 7, 13)]:
        b = BatchConfig(batch=0, kv_cache_len=0, prefill_tokens=p,
                        prefill_context=ctx, prefill_requests=1)
        brute = sum(min(ctx + i + 1, w) for i in range(p))
        assert b.attention_qk_pairs(window=w) == pytest.approx(brute)


# ── G3: geometry, and the direction of its error ────────────────────────────


def test_o_proj_input_is_n_heads_times_v_head_dim(mimo):
    """**The current code OVERSTATES this projection, it does not understate
    it.** o_proj maps one v_head_dim-wide result per query head back to hidden:
    64 x 128 = 8192. Using q_dim gives 64 x 192 = 12288, which is 1.5x too
    *large*.

    The sign is the point. Over-priced, a near-zero residual on this node means a
    kernel running 1.5x slower than the hardware allows — it reads as healthy
    when it is not."""
    assert mimo.o_proj_in == 8192
    assert mimo.q_dim == 12288
    assert mimo.q_dim > mimo.o_proj_in          # OVERSTATED, not understated
    assert mimo.q_dim / mimo.o_proj_in == pytest.approx(1.5)


def test_kv_geometry_differs_between_the_two_layer_families(mimo):
    """8 KV heads on windowed layers against 4 on full-attention ones, and K is
    192 wide where V is 128. One head count and one width for the model is wrong
    on 39 of 48 layers — and this is the KV rate, the quantity decode is bound
    by."""
    assert mimo.attn_geometry(0) == (64, 4, 192, 128)     # full
    assert mimo.attn_geometry(1) == (64, 8, 192, 128)     # windowed
    assert mimo.kv_entry_elems(0) == 1280
    assert mimo.kv_entry_elems(1) == 2560


def test_kv_rate_splits_growing_from_fixed(mimo):
    """9 layers linear in S, 39 flat: ``11,520 x S + 12.78M`` elements. Counting
    the windowed layers per-token inflates the rate ~9.7x and caps concurrency
    far below what the hardware allows."""
    kw = weight_bytes(mimo.kv_dtype)
    assert hybrid_kv_bytes_per_token(mimo) / kw == pytest.approx(9 * 1280)
    assert hybrid_kv_bytes_per_token(mimo) / kw == pytest.approx(11_520)
    assert hybrid_kv_fixed_bytes(mimo) / kw == pytest.approx(39 * 2560 * 128)


def test_mimo_predicted_weights_land_near_the_published_checkpoint(mimo):
    """308.85 GB against the index's total_size of 315.03 GB, -2.0%. The residual
    is the two encoders, the MTP head and per-tensor metadata, none of which this
    spec enumerates — the same class of gap as -2.4% on V4 and -3.3% on Qwen."""
    whole = hybrid_model_weight_bytes(mimo, ShardingConfig(tp=1))
    assert whole / 315.031102208e9 == pytest.approx(1.0, abs=0.04)


def test_weight_footprint_would_double_at_the_wrong_dtype(mimo):
    """MiMo stores fp8 weights under a ``bfloat16`` torch dtype. The reader used
    to take the torch dtype for weights, because every previous member of this
    family was bf16 throughout — which prices the whole checkpoint at 2 bytes and
    predicts 617 GB against a 315 GB index."""
    assert mimo.weight_dtype == "fp8"
    assert mimo.act_dtype == "bf16"
    wrong = hybrid_model_weight_bytes(replace(mimo, weight_dtype="bf16"),
                                      ShardingConfig(tp=1))
    right = hybrid_model_weight_bytes(mimo, ShardingConfig(tp=1))
    assert wrong / right == pytest.approx(2.0, abs=0.01)


# ── G5: three precisions in one block ───────────────────────────────────────


def test_three_precisions_inside_one_attention_block(mimo):
    """fp8 qkv_proj, bf16 o_proj (in ``ignored_layers``, no scale tensor
    anywhere), fp32 router (an explicit ``.float()`` cast). One weight_dtype for
    the model prices o_proj at 1 byte when it is 2."""
    g = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8, kv_cache_len=4096))
    seen = {n.op: n.prediction.dtype for n in g.nodes}
    assert seen["qkv_proj"] == "fp8"
    assert seen["attn_out_proj"] == "bf16"
    assert seen["moe_router"] == "fp32"
    assert seen["lm_head"] == "bf16"


def test_an_fp32_router_prices_against_an_fp32_peak(mimo):
    """Unlike a missing fp8 peak, a missing fp32 one is **not flagged** —
    ``resolve_peak`` returns the field directly, so ``peak_is_fallback`` stays
    False and the wrong ceiling looks measured. Without an H200 fp32 entry the
    router priced against A100's 19.5 TF/s: 3.4x slow, enough to move it from
    5th to 3rd in the ranked node table."""
    hw = hardware_spec_for(peak_for_sku("H200"))
    assert hw.peak_flops_fp32_per_s == pytest.approx(67e12)
    g = predict_hybrid_graph(mimo, hw, BatchConfig(batch=32, kv_cache_len=8192))
    router = next(n for n in g.nodes if n.op == "moe_router")
    assert router.prediction.peak_flops_per_s == pytest.approx(67e12)
    assert not router.prediction.peak_is_fallback


# ── G7: the dense layer and the draft head ──────────────────────────────────


def test_layer_zero_is_dense_and_runs_no_router(mimo):
    """MiMo's layer 0 has a 16384-wide dense MLP, not a mixture. Charging it a
    256-expert router and an expert bank it does not have is not a rounding
    error: the expert term dominates the layer."""
    g = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8, kv_cache_len=4096))
    layer0 = {n.op for n in g.nodes if n.layer == 0}
    assert "mlp_gate_up" in layer0 and "mlp_down" in layer0
    assert "moe_router" not in layer0 and "moe_routed" not in layer0
    # ...and exactly 47 of the 48 layers do have a mixture.
    assert len([n for n in g.nodes if n.op == "moe_routed"]) == 47


def test_the_draft_head_is_windowed_and_dense(mimo):
    """Three errors the old MTP path made at once: ``_emit_moe`` unconditionally
    (the expert-free draft charged a 256-expert router), ``layer_kind`` running
    off the end of ``layer_types`` and returning layer 47's FULL attention (an
    O(1)-in-S head priced with a cache growing in S), and one ``lm_head`` for the
    whole step."""
    g = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8, kv_cache_len=4096),
                             with_mtp=True)
    draft = [n for n in g.nodes if n.layer is not None and n.layer >= mimo.n_layers]
    assert draft, "no draft nodes emitted"
    ops = {n.op for n in draft}
    assert "mlp_gate_up" in ops and "moe_routed" not in ops
    assert mimo.layer_kind(mimo.n_layers) == "sliding_attention"


def test_the_draft_head_is_flat_in_context(mimo):
    """The consequence of the kind being right. Priced as full attention the
    draft's cache read grows with S; it is windowed, so it does not."""
    def draft_core(kv):
        g = predict_hybrid_graph(mimo, H200, BatchConfig(batch=4, kv_cache_len=kv),
                                 with_mtp=True)
        return sum(n.prediction.bytes for n in g.nodes
                   if n.op == "attn_score_value" and n.layer >= mimo.n_layers)

    assert draft_core(8192) == pytest.approx(draft_core(131072))


def test_one_lm_head_per_draft_stage(mimo):
    """Each stage samples a token before the next can consume it, so each runs
    its own vocabulary projection. Emitting one for the whole step leaves D of
    them unmodelled — ~312 MB of weight traffic per missing stage at TP4."""
    sh = ShardingConfig(tp=4)
    plain = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8), sh)
    with_mtp = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8), sh, with_mtp=True)
    assert len([n for n in plain.nodes if n.op == "lm_head"]) == 1
    assert len([n for n in with_mtp.nodes if n.op == "lm_head"]) == 1 + 3


# ── G6: the one edge that is genuinely required ─────────────────────────────


def test_the_draft_chain_is_serial_but_the_backbone_is_not(mimo):
    """MiMo puts both kinds of serialisation in one step: the MTP chain is
    genuinely serial (stage i+1 cannot begin before stage i), the 96 all-reduces
    are not — and today both show up as the same positive residual. Edges are
    declared only where the evidence is; with none declared the critical path is
    the sum, which is the honest answer."""
    plain = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8))
    assert not any(n.depends_on for n in plain.nodes)
    assert plain.serial_chain_s == 0.0        # nothing claimed, nothing declared

    mtp = predict_hybrid_graph(mimo, H200, BatchConfig(batch=8), with_mtp=True)
    assert any(n.depends_on for n in mtp.nodes)
    # A floor on the serial part, NOT an estimate of the step: unedged nodes are
    # unconstrained, not known-parallel. Reading this as a step time would treat
    # the whole backbone as infinitely overlappable.
    assert 0 < mtp.serial_chain_s < mtp.total_pred_s


# ── the collective count, without moving the bytes ──────────────────────────


def test_two_all_reduce_nodes_move_the_same_bytes_as_one_did(mimo):
    """A7 says 96 all-reduces per step (2 per layer x 48); the graph emitted 48.
    But the BYTES were already right — the expression carried a leading 2.0 for
    the second reduction. Splitting them must leave the total invariant, or every
    predicted collective doubles. This is the likeliest way to break the thing
    while 'fixing' it."""
    sh = ShardingConfig(tp=4)
    b = BatchConfig(batch=32, kv_cache_len=8192)
    two = predict_hybrid_graph(mimo, H200, b, sh)
    one = predict_hybrid_graph(replace(mimo, all_reduces_per_layer=1), H200, b, sh)

    def coll(g):
        ns = [n for n in g.nodes if n.op == "tp_all_reduce"]
        return len(ns), sum(n.prediction.bytes for n in ns)

    n_two, bytes_two = coll(two)
    n_one, bytes_one = coll(one)
    assert n_two == 2 * n_one == 96
    assert bytes_two == pytest.approx(bytes_one)


def test_qwen_keeps_one_all_reduce_per_layer():
    """The field defaults to the long-standing behaviour. Raising it for every
    model would perturb kernel-to-node alignment on checkpoints nobody has
    counted NCCL kernels for."""
    q = load_spec("qwen3.6-35b-a3b")
    assert q.all_reduces_per_layer == 1
    g = predict_hybrid_graph(q, H200, BatchConfig(batch=8), ShardingConfig(tp=4))
    assert len([n for n in g.nodes if n.op == "tp_all_reduce"]) == q.n_layers


# ── the regression guard ────────────────────────────────────────────────────


def test_new_geometry_fields_are_a_no_op_when_unset():
    """Every G2/G3 expression must reduce algebraically to the one it replaced
    when the new fields are unset. This is what makes the change safe for the
    checkpoints already in the catalogue, and it is checked rather than argued."""
    q = load_spec("qwen3.6-35b-a3b")
    assert q.v_head_dim is None
    assert q.o_proj_in == q.q_dim == 4096
    assert q.kv_entry_elems(0) == 2 * q.kv_dim
    assert q.sliding_window == 0
    assert all(q.attention_window(i) == 0 for i in range(q.n_layers))
    assert q.dtype_for("attn_out_proj", "bf16") == "bf16"
    assert q.n_sliding_attention_layers == 0
    # The reference shape too — it sets none of the new fields either.
    r = HybridMoEModelSpec()
    assert r.o_proj_in == r.q_dim
    assert r.kv_entry_elems(0) == 2 * r.kv_dim


def test_the_spec_stays_hashable(mimo):
    """The dataclass is frozen and therefore hashable, and a Mapping field would
    make ``hash(spec)`` raise — a long way from the YAML that caused it. Hence
    the tuple-of-pairs for the dtype overrides and the frozenset for the dense
    layers."""
    assert hash(mimo)
    assert isinstance(mimo.op_dtype_overrides, tuple)
    assert isinstance(mimo.dense_layers, frozenset)


# ── end to end ──────────────────────────────────────────────────────────────


def test_mimo_decode_is_dominated_by_expert_weight_traffic(mimo):
    """The shape of the answer, not just its parts. At batch 32 the union of
    distinct experts the step must fetch is the overwhelming majority of the
    step, and it is memory-bound."""
    hw = hardware_spec_for(peak_for_sku("H200"))
    g = predict_hybrid_graph(mimo, hw, BatchConfig(batch=32, kv_cache_len=8192),
                             ShardingConfig(tp=4))
    routed = sum(n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed")
    assert routed / g.total_pred_s > 0.85
    assert all(n.prediction.bound == "memory"
               for n in g.nodes if n.op == "moe_routed")


def test_prefill_flips_the_step_compute_bound(mimo):
    """Decode is memory-bound on the expert bank; an 8192-token prefill chunk is
    compute-bound on the same weights, because the arithmetic scales with tokens
    while the weight fetch does not."""
    hw = hardware_spec_for(peak_for_sku("H200"))
    sh = ShardingConfig(tp=4)
    dec = predict_hybrid_graph(mimo, hw, BatchConfig(batch=32, kv_cache_len=8192), sh)
    pre = predict_hybrid_graph(mimo, hw, BatchConfig(
        batch=1, kv_cache_len=0, prefill_tokens=8192, prefill_requests=1), sh)

    def routed_bound(g):
        return {n.prediction.bound for n in g.nodes if n.op == "moe_routed"}

    assert routed_bound(dec) == {"memory"}
    assert routed_bound(pre) == {"compute"}


def test_mimo_step_time_is_strongly_sublinear_in_batch(mimo):
    """The MoE property: 32x the tokens for well under 32x the time, because
    weight traffic follows the *distinct* experts touched and that saturates."""
    hw = hardware_spec_for(peak_for_sku("H200"))
    sh = ShardingConfig(tp=4)
    t1 = predict_hybrid_graph(mimo, hw, BatchConfig(batch=1, kv_cache_len=8192), sh)
    t32 = predict_hybrid_graph(mimo, hw, BatchConfig(batch=32, kv_cache_len=8192), sh)
    assert t32.total_pred_s / t1.total_pred_s < 32 / 2


# ── import hygiene between the two families ──────────────────────────────────


def _colliding_planner_names() -> set[str]:
    """Public top-level names defined in BOTH planner graph modules.

    Derived from the source, not hardcoded, so a collision introduced later is
    caught by the same test rather than needing this list updated.
    """
    import ast
    from pathlib import Path

    planner = Path(__file__).resolve().parent.parent / "gitm" / "planner"

    def public_tops(module: str) -> set[str]:
        tree = ast.parse((planner / f"{module}.py").read_text(encoding="utf-8"))
        return {
            n.name for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.ClassDef)
            and not n.name.startswith("_")
        }

    return public_tops("hybrid_graph") & public_tops("moe_graph")


def test_the_two_families_still_collide_on_names():
    """If this ever comes back empty the guard below is protecting nothing.

    Four functions exist in both families with the same name and different
    meanings — they take different spec types and answer different questions.
    That is not itself a defect: each is the right name inside its own module.
    It only becomes a defect at an import site that pulls both in.
    """
    assert _colliding_planner_names() >= {
        "kv_bytes_per_token",
        "kv_fixed_bytes_per_sequence",
        "model_weight_bytes",
        "spec_from_hf_config",
    }


def test_no_module_imports_a_colliding_planner_name_unaliased():
    """A file that imports from BOTH planner families must alias the collisions.

    ``from gitm.planner.hybrid_graph import kv_bytes_per_token`` and its
    sparse-MoE twin bind the same name. Whichever is imported second silently
    wins for the whole module, and every call afterwards runs against the wrong
    family — producing a number rather than an error.

    **This is not hypothetical.** It happened in this file: a bare
    ``kv_fixed_bytes_per_sequence`` was shadowed by the sparse-MoE version, and a
    MiMo assertion called into ``moe_graph`` and raised
    ``AttributeError: 'HybridMoEModelSpec' object has no attribute
    'compress_ratio'``. That one failed loudly because the two specs have
    different fields; ``kv_bytes_per_token`` would have returned a plausible
    float instead.

    Aliasing was already the documented convention. A convention that has been
    violated once is worth a test — this is the mechanical version.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    colliding = _colliding_planner_names()
    families = {"gitm.planner.hybrid_graph", "gitm.planner.moe_graph"}
    offences = []

    for path in list((root / "gitm").rglob("*.py")) + list((root / "tests").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Every unaliased binding of a colliding name, grouped by name.
        bare: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in families:
                continue
            for alias in node.names:
                if alias.name in colliding and alias.asname is None:
                    bare.setdefault(alias.name, []).append(
                        f"{node.module.rsplit('.', 1)[-1]}:{node.lineno}"
                    )

        # ONE bare binding is a deliberate default - the file has chosen which
        # family that name means. TWO is shadowing: the second silently wins for
        # the whole module, and every call written before it now runs against
        # the wrong family.
        for name, sites in bare.items():
            if len(sites) > 1:
                offences.append(
                    f"{path.relative_to(root).as_posix()}: {name!r} bound "
                    f"unaliased {len(sites)}x ({', '.join(sites)})"
                )

    assert not offences, (
        "a colliding planner name is bound unaliased more than once in the "
        "same file, so the family it resolves to depends on import order:"
        + "; ".join(offences)
    )


def test_no_test_module_defines_the_same_test_name_twice():
    """The same shadowing hazard, one level up: two ``def test_x`` in one file.

    Python keeps the later definition and discards the earlier one, so pytest
    collects one test and reports green while the other stops running. Nothing
    fails and nothing warns — the count just quietly drops.

    **This is not hypothetical either.** It happened in this file: the MiMo
    additions reused ``test_predicted_weights_land_near_the_published_checkpoint``
    and ``test_step_time_is_strongly_sublinear_in_batch``, silently retiring the
    Qwen footprint check and the sparse-MoE batch curve — the Qwen one being the
    exact counterpart of the arithmetic the MiMo work was changing.
    """
    import ast
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offences = []

    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        # Module scope only: a name inside a class or function shadows nothing
        # that pytest would otherwise have collected.
        names = Counter(
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name.startswith("test_")
        )
        offences += [
            f"{path.relative_to(root).as_posix()}: {name!r} defined {count}x"
            for name, count in sorted(names.items())
            if count > 1
        ]

    assert not offences, (
        "a test name is defined more than once at module scope, so only the "
        "last definition runs and the earlier one is silently dead: "
        + "; ".join(offences)
    )
