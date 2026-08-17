"""Tests for the kernel bucket breakdown (gitm.tracer.kernel_taxonomy).

The kernel names here are real mangled names off a vLLM/CUPTI trace (the same
fixtures tests/test_deviation_alignment.py uses) plus the MoE and NCCL names a
Qwen3-style MoE decode adds. That matters: the whole module is substring rules over
mangled C++, and rules written against invented names prove nothing.
"""

from __future__ import annotations

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
