"""Tests for the kernel bucket breakdown (gitm.tracer.kernel_taxonomy).

The kernel names here are real mangled names off a vLLM/CUPTI trace (the same
fixtures tests/test_deviation_alignment.py uses) plus the MoE and NCCL names a
Qwen3-style MoE decode adds. That matters: the whole module is substring rules over
mangled C++, and rules written against invented names prove nothing.
"""

from __future__ import annotations

import pytest

from gitm.optimizer.deviation import classify_op
from gitm.tracer.kernel_taxonomy import (
    NAME_MAX,
    _active_ns,
    classify_kernel,
    summarize_kernels,
)
from tests.conftest import make_kernel

# --- classification ---------------------------------------------------------


def test_bare_cublas_gemm_is_a_gemm():
    """The kernel the deviation monitor cannot place (it carries no projection tag)
    still has to be visible as GEMM work — it is ~35% of launches."""
    assert classify_kernel("ampere_fp16_s16816gemm_fp16_128x128_ldg8_relu_f2f_stages_32x5_tn") == "gemm"
    assert classify_kernel(
        "_ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8EEE"
    ) == "gemm"


def test_flash_attention_is_attention_not_gemm():
    """FlashAttention's inner loop is a GEMM; bucketing it as one would erase the
    attention cost, so the attention rules must win."""
    name = ("_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traitsILi64E"
            "Li64ELi256ELi4ELb0ELb0EN7cutlass6half_tE19Flash_kernel_traitsILi64E")
    assert classify_kernel(name) == "attention"


def test_moe_kernels_beat_the_gemm_and_elementwise_rules():
    # These names contain "gemm"/"align"/"sort", which later rules would claim.
    assert classify_kernel("fused_moe_kernel") == "moe"
    assert classify_kernel("moe_align_block_size_kernel") == "moe"
    assert classify_kernel("marlin_moe_gemm_kernel") == "moe"
    assert classify_kernel("topk_softmax_kernel") == "moe"


def test_gated_deltanet_kernels_are_linear_attention_not_other():
    """This model is 30 DeltaNet layers to 10 full-attention layers. None of the
    softmax-attention needles match a linear-attention kernel, so without these the
    dominant layer type lands in 'other' and the breakdown describes nothing."""
    for name in ("chunk_gated_delta_rule_fwd_kernel_h",
                 "fused_recurrent_gated_delta_rule_fwd_kernel",
                 "chunk_local_cumsum_scalar_kernel",
                 "solve_tril_kernel",
                 "wy_fast_fwd_prepare_wy_repr_kernel"):
        assert classify_kernel(name) == "linear_attn", name


def test_linear_and_softmax_attention_stay_separable():
    """A hybrid model's whole point is the split — collapsing both into one bucket
    would hide which of the two layer types costs anything."""
    assert classify_kernel("chunk_gated_delta_rule_fwd") == "linear_attn"
    assert classify_kernel("_ZN5flash24flash_fwd_splitkv_kernelI") == "attention"


def test_topp_sampling_cumsum_is_not_mistaken_for_linear_attention():
    """Guard on a needle that was briefly too loose: top-p runs a cumulative sum
    over sorted probs, and 'cumsum' alone would file it as DeltaNet work."""
    assert classify_kernel("top_p_sampling_cumsum_kernel") == "sampling"


def test_nccl_allreduce_is_a_collective():
    """TP=2 puts an all-reduce in the critical path every layer; it must not be
    filed as elementwise 'reduce' work."""
    assert classify_kernel("ncclDevKernel_AllReduce_Sum_f16_RING_LL") == "collective"
    assert classify_kernel("_ZN4vllm18cross_device_reduceILi2EEEvPfS1_") == "collective"


def test_vllm_kv_cache_and_norm_kernels():
    assert classify_kernel(
        "_ZN4vllm30reshape_and_cache_flash_kernelIttLNS_18Fp8KVCacheDataTypeE0EEE"
    ) == "kv_cache"
    assert classify_kernel("_ZN4vllm25fused_add_rms_norm_kernelIN3c104HalfEEEv") == "norm"
    assert classify_kernel("_ZN4vllm22rotary_embedding_kernelIN3c104HalfELb1EEEv") == "rope"


def test_unknown_kernels_are_other_never_none():
    """classify_op returns None for unmodeled work; here everything lands somewhere,
    so 'other' can be measured and acted on instead of silently dropping out."""
    assert classify_kernel("some_kernel_nobody_has_seen") == "other"
    assert classify_kernel("") == "unnamed"
    assert classify_kernel("<anonymous>") == "unnamed"


# --- active time ------------------------------------------------------------


def test_active_time_is_a_union_not_a_sum():
    """Concurrent kernels overlap constantly; summing durations would report more
    busy time than the window contains."""
    assert _active_ns([(0, 100), (50, 150)]) == 150
    assert _active_ns([(0, 100), (200, 300)]) == 200
    assert _active_ns([(0, 100), (10, 20)]) == 100
    assert _active_ns([]) == 0


def test_gpu_active_share_never_exceeds_one_with_overlap():
    kernels = [make_kernel("gemm_a", start_ns=0, end_ns=900),
               make_kernel("gemm_b", start_ns=100, end_ns=950),
               make_kernel("gemm_c", start_ns=200, end_ns=980)]
    bd = summarize_kernels(kernels, window_ns=1000)
    assert bd.gpu_active_share is not None and bd.gpu_active_share <= 1.0


def test_per_device_active_is_tracked_separately_for_tp():
    kernels = [make_kernel("gemm_a", start_ns=0, end_ns=500, device_id=0),
               make_kernel("gemm_b", start_ns=0, end_ns=800, device_id=1)]
    bd = summarize_kernels(kernels, window_ns=1000)
    assert bd.n_devices == 2
    assert bd.per_device_active_ns == {0: 500, 1: 800}
    # the busiest device sets the share — an idle rank must not average away a busy one
    assert bd.gpu_active_share == 0.8


def test_no_window_means_no_invented_share():
    bd = summarize_kernels([make_kernel("gemm_a", start_ns=0, end_ns=500)], window_ns=None)
    assert bd.gpu_active_share is None


# --- coverage warnings ------------------------------------------------------


def test_shares_are_of_kernel_time_and_sum_to_one():
    kernels = [make_kernel("fused_moe_kernel", start_ns=0, end_ns=750),
               make_kernel("ampere_fp16_s16816gemm_tn", start_ns=800, end_ns=1050)]
    bd = summarize_kernels(kernels, window_ns=2000)
    assert bd.n_kernels == 2
    assert abs(sum(b.share for b in bd.buckets) - 1.0) < 1e-9
    assert {b.bucket for b in bd.buckets} == {"moe", "gemm"}
    assert bd.buckets[0].bucket == "moe"  # sorted by time, not name


def test_truncated_names_are_flagged():
    """Two different cutlass instantiations cut at NAME_MAX become one identity —
    the trace looks complete while distinct kernels have merged."""
    long_name = "_ZN7cutlass7Kernel2I" + "x" * NAME_MAX
    kernels = [make_kernel(long_name[:NAME_MAX], start_ns=0, end_ns=10),
               make_kernel(long_name[:NAME_MAX], start_ns=20, end_ns=30)]
    bd = summarize_kernels(kernels, window_ns=100)
    assert bd.n_truncated_names == 2
    assert bd.n_distinct_truncated == 1
    assert any("truncated" in w for w in bd.warnings())


def test_name_max_matches_the_collector_that_produced_the_names():
    """NAME_MAX mirrors GITM_NAME_MAX across a language boundary, bound by nothing
    but a comment. Truncation is detected as `len(name) >= NAME_MAX`, so if the C
    cap is raised alone the detector compares against a bound no name can reach and
    silently reports zero truncation forever; raised on the Python side alone it
    flags every name as clean while they are still being cut. Neither shows up as a
    failure anywhere else — the trace stays well-formed and merged kernels keep
    pooling their time under one identity."""
    import re
    from pathlib import Path

    from gitm.tracer import _cupti

    header = (Path(_cupti.__file__).parent / "cupti_core.h").read_text()
    match = re.search(r"^#define\s+GITM_NAME_MAX\s+(\d+)\s*$", header, re.M)
    assert match, "GITM_NAME_MAX not found in cupti_core.h"
    assert int(match.group(1)) == NAME_MAX


def test_empty_trace_warns_about_graph_replay_and_stops_there():
    bd = summarize_kernels([], window_ns=1000)
    warnings = bd.warnings()
    assert len(warnings) == 1
    assert "CUDA-graph replay" in warnings[0]


def test_nonpositive_kernel_durations_are_excluded_and_named():
    kernels = [
        make_kernel("fused_moe_kernel", start_ns=10, end_ns=10),
        make_kernel("ampere_fp16_s16816gemm_tn", start_ns=20, end_ns=19),
    ]

    bd = summarize_kernels(kernels, window_ns=100)

    assert bd.n_kernels == 0
    assert bd.n_invalid_duration == 2
    assert bd.kernel_time_ns == 0
    assert any("non-positive duration" in warning for warning in bd.warnings())


def test_idle_looking_gpu_is_warned_about():
    bd = summarize_kernels([make_kernel("fused_moe_kernel", start_ns=0, end_ns=50)],
                           window_ns=10_000)
    assert any("GPU active" in w for w in bd.warnings())


def test_missing_gemm_or_moe_is_a_warning_on_a_moe_decode():
    """A MoE decode trace with no MoE kernels is an instrumentation failure, not a
    property of the model — say so rather than reporting a clean breakdown."""
    bd = summarize_kernels(
        [make_kernel("ampere_fp16_s16816gemm_tn", start_ns=0, end_ns=900)],
        window_ns=1000,
    )
    warnings = bd.warnings()
    assert any("no 'moe' kernels" in w for w in warnings)
    assert not any("no 'gemm' kernels" in w for w in warnings)


def test_unreadable_trace_warns_on_other_share():
    bd = summarize_kernels(
        [make_kernel("mystery_kernel_one", start_ns=0, end_ns=900),
         make_kernel("fused_moe_kernel", start_ns=900, end_ns=1000)],
        window_ns=1000,
    )
    assert bd.other_share > 0.25
    assert any("matched no bucket rule" in w for w in bd.warnings())


def test_healthy_moe_trace_produces_no_warnings():
    kernels = [
        make_kernel("fused_moe_kernel", start_ns=0, end_ns=400, device_id=0),
        make_kernel("ampere_fp16_s16816gemm_tn", start_ns=400, end_ns=700, device_id=0),
        make_kernel("ncclDevKernel_AllReduce_Sum_f16_RING_LL", start_ns=700, end_ns=800, device_id=0),
        make_kernel("_ZN5flash24flash_fwd_splitkv_kernelI", start_ns=800, end_ns=950, device_id=0),
    ]
    bd = summarize_kernels(kernels, window_ns=1000)
    assert bd.warnings() == []

# ── Qwen3.6 / hybrid-MoE classification gaps ──────────────────────────────────
#
# Classification gaps found by a real Qwen3.6-35B-A3B capture on an H200.
#
# Every name below was taken from that trace or from the kernel inventory the
# engine's own backend selection implies. The capture recorded 9,473,114 kernels
# and the breakdown put 18.7% of device time in ``other`` and reported the ``gemm``
# bucket at 3.6% — a trace describing a model whose projections had vanished.
#
# Two kinds of defect are pinned here, and they are not equally bad:
#
# * **Unmatched** — the kernel falls to ``other``. Visible, and the breakdown warns
#   about it once it passes a quarter of device time.
# * **Misfiled** — the kernel lands in a bucket that is wrong. Silent, and it
#   inflates a bucket that downstream analysis trusts. This is the worse failure,
#   and the FlashInfer sampler was an instance of it.

# ── gated DeltaNet: 30 of this model's 40 layers ───────────────────────────

GDN_DECODE = [
    "_causal_conv1d_update_kernel",
    "fused_recurrent_gated_delta_rule_packed_decode_kernel",
]

GDN_PREFILL = [
    "_fused_post_conv_kernel",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "solve_tril_16x16_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "_FullyFusedDeltaRuleSm90",
]


@pytest.mark.parametrize("name", GDN_DECODE + GDN_PREFILL)
def test_every_gated_deltanet_kernel_reaches_linear_attention(name):
    """Four separate causes put six of these in ``other``: no convolution needle
    existed at all; ``chunk_o``/``chunk_h`` were written with the suffix in the
    wrong position for the real names; ``wy_fast`` is the module name rather than
    the kernel name; and the CuTeDSL path is CamelCase, which lowercases to a
    single token that no underscored delta needle matches."""
    assert classify_kernel(name) == "linear_attn"


def test_the_cutedsl_fast_path_is_matched_despite_camelcase():
    """``_FullyFusedDeltaRuleSm90`` -> ``_fullyfuseddeltarulesm90``. Every needle
    of the form ``delta_rule`` misses it, so the SM90 fast path — the one Hopper
    actually runs — was the least visible kernel in the model."""
    assert classify_kernel("_FullyFusedDeltaRuleSm90") == "linear_attn"
    assert classify_op("_FullyFusedDeltaRuleSm90") == "linattn_recurrent"


def test_the_convolution_is_its_own_graph_node():
    """A distinct kernel runs it, so folding it into the recurrent node would
    leave that kernel permanently unattributable."""
    assert classify_op("_causal_conv1d_update_kernel") == "linattn_conv"
    assert classify_op("_fused_post_conv_kernel") == "linattn_conv"


def test_recurrent_kernels_do_not_leak_into_softmax_attention():
    """A GDN layer's traffic is constant in context; ``attn_score_value`` is
    predicted to grow with it. Mapping one onto the other would report the
    difference as a deviation on every long-context step."""
    for name in GDN_DECODE + GDN_PREFILL:
        assert classify_op(name) != "attn_score_value"


# ── nvjet: the GEMM family that had no needle ──────────────────────────────

NVJET = [
    "nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT",
    "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT",
    "nvjet_sm90_tst_64x8_64x16_1x1_h_bz_TNT",
    "nvjet_sm90_tst_512x8_64x3_2x1_v_bz_TNT",
    "nvjet_sm90_tst_8x64_64x16_4x1_v_bz_TNN",
]


@pytest.mark.parametrize("name", NVJET)
def test_nvjet_is_recognised_as_a_gemm(name):
    """cuBLAS's JIT-generated Hopper GEMM family. ``TNT``/``TNN`` is BLAS
    transpose notation and ``128x8_64x12`` is a tile shape, but the name carries
    none of ``gemm``/``cublas``/``cutlass``, so 7.98 s of a 42.8 s trace sat
    unclassified."""
    assert classify_kernel(name) == "gemm"


def test_the_nvjet_family_is_no_longer_split_by_an_incidental_needle():
    """The sharper half of the bug: ``..._splitK_...`` variants classified as
    GEMM only because ``splitk`` happened to be a needle, so one kernel family
    landed in two buckets by accident of naming."""
    plain = "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT"
    split = "nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT"
    assert classify_kernel(plain) == classify_kernel(split) == "gemm"


def test_a_bare_gemm_still_maps_to_no_graph_node():
    """``classify_kernel`` and ``classify_op`` deliberately disagree here.
    The bucket answers "what ran"; the op answers "which predicted node is
    this". A GEMM whose name does not carry its projection cannot be assigned
    to one, and guessing would attribute real work to the wrong op."""
    assert classify_kernel(NVJET[0]) == "gemm"
    assert classify_op(NVJET[0]) is None


# ── the misfile ────────────────────────────────────────────────────────────

FLASHINFER_SAMPLER = "void flashinfer::sampling::TopKTopPSamplingFromProbKernel<...>"


def test_the_flashinfer_sampler_is_sampling_not_attention():
    """``flashinfer`` was a *vendor* needle sitting in the attention bucket ahead
    of the sampling rule, and this engine logs "Using FlashInfer for top-p &
    top-k sampling". So sampler cost was reported as attention cost — silent,
    and it inflated a bucket that gets trusted."""
    assert classify_kernel(FLASHINFER_SAMPLER) == "sampling"


def test_removing_the_vendor_needle_did_not_strand_the_sampler():
    """The second-order trap: ``sample`` is not a substring of ``sampling``, and
    ``top_k`` is not a substring of ``TopKTopP``. Deleting the vendor needle
    without adding operation-shaped ones would have moved the kernel from a
    wrong bucket to no bucket."""
    assert classify_kernel("TopPSamplingFromProbsKernel") == "sampling"
    assert classify_kernel("top_k_renorm_probs_kernel") == "sampling"


@pytest.mark.parametrize("name", [
    "flashinfer::BatchPrefillWithPagedKVCacheKernel",
    "flashinfer::BatchDecodeWithPagedKVCacheKernel",
])
def test_flashinfers_actual_attention_kernels_still_classify(name):
    """They are named for what they do, so they are matched by operation rather
    than by vendor."""
    assert classify_kernel(name) == "attention"


def test_moe_routing_is_not_claimed_by_the_loose_sampling_needles():
    """``topk_softmax`` is MoE routing and the ``moe`` rule runs far earlier.
    Adding bare ``topk``/``topp`` to sampling must not reach past it."""
    assert classify_kernel("topk_softmax_kernel") == "moe"


# ── routing versus expert GEMMs ────────────────────────────────────────────


def test_fused_gating_is_routing_not_an_expert_gemm():
    """``topkGating`` was claimed by the generic ``moe`` needle, putting cheap
    routing work inside the node whose weight traffic dominates the step."""
    name = "_ZN4vllm3moe10topkGatingILi8ELi256ELi4ELi16ELi32Ei13__nv_bfloat16E"
    assert classify_kernel(name) == "moe"
    assert classify_op(name) == "moe_router"


def test_expert_gemms_remain_routed():
    assert classify_op("fused_moe_kernel") == "moe_routed"


# ── norm + rotary + insert ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "_triton_mrope_forward",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "rotary_embedding_kernel",
])
def test_rope_kernels_map_to_the_fused_node_every_family_emits(name):
    """Both graph families emit ``attn_qnorm_rope_insert`` as one node because
    vLLM runs it as one fused kernel, but no ``classify_op`` rule existed to
    receive it — so the node was predicted and never matched on either path."""
    assert classify_op(name) == "attn_qnorm_rope_insert"


def test_a_fused_kernel_lands_in_one_coarse_bucket_by_rule_order():
    """The coarse taxonomy has no notion of a kernel doing several things, so a
    norm+rope+quant+insert kernel takes whichever bucket matches first — ``quant``
    here, since it precedes ``norm`` and ``rope``. That is a compromise rather
    than a defect, and it is why ``classify_op`` above (which maps to a *node*,
    not a category) is the mapping the residual path uses."""
    name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    assert classify_kernel(name) == "quant"
    assert classify_op(name) == "attn_qnorm_rope_insert"


# ── the same model on a different GPU picks different kernels ───────────────
#
# vLLM selects backends per SKU. The identical build and checkpoint chose
# FLASH_ATTN + Triton MoE on an H200 and FLASHINFER + TrtLlmBf16Experts on a
# B200. So a rule set validated on one machine can lose whole subsystems on
# another, silently, and the trace still looks complete.
#
# These names are taken from real vLLM startup logs on both machines.


@pytest.mark.parametrize("name", [
    "flashinfer::BatchPrefillWithPagedKVCacheKernel",     # B200: FLASHINFER
    "flashinfer::BatchDecodeWithPagedKVCacheKernel",
    "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_E",  # H200: FLASH_ATTN
    "cutlass___call___vllm_flash_attn_cute_flash_fwd_sm100FlashAttentionForwardSm100",
])
def test_both_attention_backends_classify(name):
    assert classify_kernel(name) == "attention"
    assert classify_op(name) == "attn_score_value"


@pytest.mark.parametrize("name", [
    "vllm::linear_attention",
    "vllm::qwen_gdn_attention_core",
    "vllm::mamba_mixer2",
    "vllm::short_conv",
])
def test_torch_compile_split_ops_are_linear_attention_not_softmax(name):
    """These carry "attention" in their names and would be claimed by the
    softmax rule. That is a misfile, not a gap: a GDN layer's traffic is
    constant in context while `attn_score_value` is predicted to grow with it,
    so every long-context step would report a deviation that isn't there.

    `linear_attn` does not match `linear_attention` — the short needle ends
    "attn", the long name has "atten". Same trap as `sample` vs `sampling`.
    """
    assert classify_kernel(name) == "linear_attn"
    assert classify_op(name) == "linattn_recurrent"


def test_a_short_needle_is_not_assumed_to_prefix_its_longer_form():
    """Guards the class of bug directly, so a future edit that drops the
    explicit long form fails here rather than on a pod."""
    assert "linear_attn" not in "vllm::linear_attention"
    assert "sample" not in "TopKTopPSamplingFromProbKernel".lower()


# ── torch.compile / Inductor kernels ────────────────────────────────────────
#
# A vocabulary that exists only under compilation, so it is absent from every
# eager capture and then arrives all at once. On an H200 graph-mode run it was
# 1.4% of device time sitting in `other`, which had been empty in all four
# preceding eager captures.
#
# `poi` is pointwise, `red` is a reduction. Inductor names a kernel after the ops
# it fused when it can — `mul_sigmoid_view` is legible — and gives it a serial
# number when it cannot.


@pytest.mark.parametrize("name", [
    "triton_poi_fused_5",
    "triton_poi_fused_2",
    "triton_red_fused_3",
    "batch_memcpy_kernel",
    "_ZN2at6native40_GLOBAL__N__f1f992fe_8_Shape_cu30CatArrayBatchedCopy",
])
def test_inductor_and_copy_kernels_are_elementwise(name):
    assert classify_kernel(name) == "elementwise"


def test_a_fused_gate_is_an_activation():
    """`triton_poi_fused_mul_sigmoid_*` is x*sigmoid(x) — SiLU in the MLP, and
    the output gate in a GDN layer. Indistinguishable by name; both are gated
    activations, so the coarse bucket is right either way."""
    assert classify_kernel("triton_poi_fused_mul_sigmoid_view_0") == "activation"


def test_an_anonymous_reduction_is_not_guessed_to_be_a_norm():
    """`triton_red_fused_3` could be an RMSNorm or a plain sum. Filing it as
    `norm` would inflate a bucket that gets trusted; `elementwise` is the honest
    answer, and correlation is the only thing that can do better."""
    assert classify_kernel("triton_red_fused_3") == "elementwise"
    # A norm that still names itself is unaffected.
    assert classify_kernel("layer_norm_fwd_kernel") == "norm"
