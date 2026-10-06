"""AMD kernel names must land in the right bucket and op, as NVIDIA's do.

Both rule sets were written against CUDA captures. On MI355X the dominant
GEMMs (Tensile/hipBLASLt ``Cijk_``), every AITER MLA decode and paged-attention
kernel, and AITER's MoE router went to ``other`` or to the wrong op — the
router to sampling, MoE sort/quant kernels to the expert GEMM.

Every name below is a real symbol, from the source that ships it:

* Tensile/hipBLASLt solution names: ``Cijk_<A layout>_<B layout>_<types>_MT...``
  (hipBLASLt library logic YAMLs; rocBLAS uses the same Tensile naming).
* vLLM ``csrc/rocm/skinny_gemms.cu`` (``wvSplitK*``, ``LLGemm1_kernel``) and
  ``csrc/rocm/attention.cu`` (``paged_attention_ll4mi_*``).
* AITER kernel tables ``hsa/gfx950/<family>/*.csv`` (``knl_name`` column) and
  ``csrc/kernels/*.cu`` (``__global__`` symbols).
* Composable Kernel / ck_tile mangled entry points, ROCclr blit kernels, RCCL.

``classify_op`` deliberately returns None for a bare GEMM on either vendor —
its op comes from correlation, never the name.
"""

from __future__ import annotations

import pytest

from gitm.optimizer.deviation import classify_op, observed_op
from gitm.tracer.kernel_taxonomy import classify_kernel
from gitm.tracer.vendor import name_dialect

CORPUS = [
    # name, taxonomy bucket, predicted-graph op
    ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_SN_LDSB1_GRPM1_GSU1_"
     "ISA950_K1_WG32_8_1", "gemm", None),
    ("Cijk_Ailk_Bljk_HHS_BH_MT128x128x32_MI32x32x8x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1",
     "gemm", None),
    ("wvSplitK_hf_sml_", "gemm", None),
    ("wvSplitKQ_hf_", "gemm", None),
    ("LLGemm1_kernel", "gemm", None),
    ("_ZN5aiter36bf16gemm_bf16_tn_256x256_bpreshuffleE", "gemm", None),
    ("_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E", "gemm", None),
    ("_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E", "gemm", None),
    ("_ZN2ck40kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle", "gemm", None),
    ("_ZN5aiter41mla_dec_stage1_bf16_a16w16_subQ128_mqa128E", "attention", "attn_score_value"),
    ("_ZN5aiter42mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_psE", "attention", "attn_score_value"),
    ("_ZN5aiter44PA_A16W8_BLK256_1TG_4W_16mx1_64nx4_MTP_PS_PBE", "attention", "attn_score_value"),
    ("_ZN5aiter19fmha_fwd_hd128_bf16E", "attention", "attn_score_value"),
    ("paged_attention_ll4mi_QKV_mfma16_kernel", "attention", "attn_score_value"),
    ("_ZN7ck_tile6kentryILi1ENS_13FmhaFwdKernelEEEvv", "attention", "attn_score_value"),
    ("kernel_unified_attention_2d", "attention", "attn_score_value"),
    ("_fwd_grouped_kernel_stage1", "attention", "attn_score_value"),
    ("_ZN5aiter19topksoftmax_4x128x4E", "moe", "moe_router"),
    ("moeTopK", "moe", "moe_router"),
    ("moe_align_block_size_kernel", "moe", "moe_router"),
    ("fused_mx_quant_moe_sort_kernel", "moe", "moe_permute"),
    ("mxfp4_moe_sort_kernel", "moe", "moe_permute"),
    ("_ZN7ck_tile6kentryILi1ENS_16MoeSortingKernelINS_16MoeSortingProblemEEEEEvNT0_5KargsE",
     "moe", "moe_permute"),
    ("_ZN5aiter52fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_ps_32x256E", "moe", "moe_routed"),
    ("moe_sum_kernel", "moe", "moe_combine"),
    ("moe_smooth_per_token_scaled_quant_kernel_v1", "moe", "act_quant"),
    ("scaled_quant_kernel", "quant", "act_quant"),
    ("fused_mrope_rms_kv_kernel", "rope", "attn_qnorm_rope_insert"),
    ("concat_and_cache_mla_kernel", "kv_cache", None),
    ("ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)", "collective", "tp_all_reduce"),
    ("cross_device_reduce_1stage", "collective", "tp_all_reduce"),
    ("__amd_rocclr_copyBuffer", "elementwise", None),
    ("__amd_rocclr_fillBufferAligned", "elementwise", None),
]


@pytest.mark.parametrize("name,bucket,op", CORPUS, ids=[c[0][:40] for c in CORPUS])
def test_amd_kernel_classification(name, bucket, op):
    assert classify_kernel(name) == bucket
    assert classify_op(name) == op


def test_no_amd_kernel_in_the_corpus_lands_in_other():
    assert not [n for n, _, _ in CORPUS if classify_kernel(n) == "other"]


@pytest.mark.parametrize("name", ["moe_smooth_per_token_scaled_quant_kernel_v1",
                                  "_ZN5aiter19topksoftmax_4x128x4E",
                                  "fused_mx_quant_moe_sort_kernel"])
def test_moe_side_kernels_inside_an_expert_range_keep_their_own_op(name):
    """Inside ``L3/moe_routed`` these must not be charged to the expert GEMM:
    observed_op keeps the name's own op for quant, router and permute work,
    each of which the MoE graph prices as its own node."""
    assert observed_op(name, "moe_routed") == classify_op(name) != "moe_routed"


def test_a_bare_hipblaslt_gemm_takes_its_op_from_correlation_only():
    name = CORPUS[0][0]
    assert observed_op(name, None) is None
    assert observed_op(name, "mlp_down") == "mlp_down"


@pytest.mark.parametrize("name,dialect", [
    ("Cijk_Alik_Bljk_BBS", "amd"), ("__amd_rocclr_copyBuffer", "amd"),
    ("_ZN5aiter19fmha_fwd_hd128_bf16E", "amd"), ("wvSplitK_hf_", "amd"),
    ("nvjet_sm90_tst_128x8", "nvidia"), ("sm90_xmma_gemm_bf16", "nvidia"),
    ("ampere_bf16_s16816gemm", "nvidia"), ("flash_fwd_splitkv_kernel", None),
    ("triton_poi_fused_mul_0", None),
])
def test_vendor_dialect_of_kernel_names(name, dialect):
    assert name_dialect(name) == dialect
