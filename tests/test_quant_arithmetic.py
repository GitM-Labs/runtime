"""Bytes-per-parameter, execution format and ridge arithmetic.

Each test derives its expected value from the format's definition or the
checkpoint's own tensor shapes, not from a number the planner printed. Two design
notes are reproduced at the bottom (the Kimi K2.6 NVFP4 case and GLM-5.2), and
where a note was wrong the test pins the correction and names the evidence.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.glm_graph import (
    kv_bytes_per_token,
    kv_entry_bytes,
    memory_fit,
    model_weight_bytes,
    predict_glm_graph,
)
from gitm.planner.model_catalogue import load_spec
from gitm.planner.roofline import (
    QUANT_FORMATS,
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    UnsupportedExecution,
    critical_rows,
    distinct_experts,
    expert_pad_factor,
    kv_elem_bytes,
    linear_traffic,
    resolve_execution,
    ridge,
    roofline,
    weight_bytes,
)


def _hw(sku: str) -> HardwareSpec:
    return hardware_spec_for(peak_for_sku(sku))


H200, B200, MI355X = _hw("H200"), _hw("B200"), _hw("MI355X")
BASELINE = BatchConfig(batch=32, kv_cache_len=8192)


# ── storage: bytes per stored weight ─────────────────────────────────────────


def test_nvfp4_bytes_follow_the_k26_shard_headers():
    """gate_proj.weight U8 [2048, 3584] + weight_scale F8_E4M3 [2048, 448] store a
    2048 x 7168 matrix. Two e2m1 per byte, one e4m3 scale per 16 along K."""
    rows, k = 2048, 7168
    payload = rows * (k // 2)
    scales = rows * (k // 16)
    assert (payload, scales) == (2048 * 3584, 2048 * 448)
    assert weight_bytes("nvfp4") == (payload + scales) / (rows * k) == 0.5625


@pytest.mark.parametrize(
    ("name", "bits", "block", "scale_b"),
    [
        ("nvfp4", 4, 16, 1),        # e4m3 per 16
        ("mxfp4", 4, 32, 1),        # e8m0 per 32 (OCP MX)
        ("mxfp8", 8, 32, 1),        # e8m0 per 32 (OCP MX)
        ("int4_g32", 4, 32, 2),     # bf16 per 32 (compressed-tensors)
        ("fp8_block128", 8, 16384, 4),  # fp32 per 128x128
    ],
)
def test_format_bytes_are_payload_plus_block_scale(name, bits, block, scale_b):
    f = QUANT_FORMATS[name]
    assert f.bytes_per_elem == bits / 8 + scale_b / block
    assert f.scale_overhead == pytest.approx((scale_b / block) / (bits / 8 + scale_b / block))


def test_scale_overhead_ranks_the_4bit_formats():
    """NVFP4 and INT4 g32 spend twice MXFP4's scale bytes: 1/9 of the stream vs 1/17."""
    assert QUANT_FORMATS["nvfp4"].scale_overhead == pytest.approx(1 / 9)
    assert QUANT_FORMATS["int4_g32"].scale_overhead == pytest.approx(1 / 9)
    assert QUANT_FORMATS["mxfp4"].scale_overhead == pytest.approx(1 / 17)


def test_nvfp4_and_int4_g32_store_identical_bytes():
    """Why K2.6-NVFP4 and the INT4 K2.5/K2.6 base both weigh 595 GB: a 1-byte scale
    per 16 and a 2-byte scale per 32 cost the same per weight."""
    assert weight_bytes("nvfp4") == weight_bytes("int4") == 0.5625


def test_legacy_labels_keep_their_byte_cost():
    """Every label the catalogue already used prices exactly as before the table."""
    assert weight_bytes("fp8") == 1.0 + 4.0 / (128 * 128)
    assert weight_bytes("int4") == 0.5 + 2.0 / 32
    assert weight_bytes("mxfp4") == weight_bytes("fp4") == 0.5 + 1.0 / 32
    assert weight_bytes("bf16") == weight_bytes("fp16") == 2.0
    assert weight_bytes("fp32") == 4.0
    assert weight_bytes("no-such-dtype") == 2.0


# ── the cache is not a weight ────────────────────────────────────────────────


def test_fp8_kv_is_one_byte_not_the_weight_constant():
    """vLLM kv_cache.py:88-91 accepts only a per-tensor k_scale/v_scale, so an fp8
    cache element is 1 byte. The weight label's 128x128 block scale does not apply."""
    assert kv_elem_bytes("fp8") == 1.0
    assert weight_bytes("fp8") > 1.0
    assert kv_elem_bytes("bf16") == 2.0


def test_fp8_ds_mla_entry_is_656_bytes():
    """DeepSeek's sparse-MLA layout: 512 fp8 latent + one fp32 scale per 128 + 64 bf16 RoPE."""
    spec = replace(load_spec("kimi-k2.6"), kv_dtype="fp8_ds_mla", kv_rope_dtype="bf16")
    assert kv_entry_bytes(spec) == 512 * (1 + 4 / 128) + 64 * 2 == 656


# ── execution: what a stored format becomes on a SKU ─────────────────────────


def test_nvfp4_on_h200_runs_marlin_w4a16():
    """Hopper has no FP4 tensor cores and NVFP4 has no W4A8 Marlin path, so the
    MACs are bf16, the bytes are the checkpoint's, and no activation is quantised."""
    ex = resolve_execution("nvfp4", H200)
    assert (ex.backend, ex.compute_dtype) == ("marlin", "bf16")
    assert ex.streamed_bytes == ex.resident_bytes == 0.5625
    assert ex.temp_bytes == 0.0 and ex.act_format is None
    assert ex.is_upcast and "marlin_utils_fp4.py" in ex.source


def test_nvfp4_on_b200_runs_w4a4_with_activation_quant():
    ex = resolve_execution("nvfp4", B200)
    assert (ex.backend, ex.compute_dtype) == ("native", "fp4")
    assert ex.act_format is QUANT_FORMATS["nvfp4"]
    assert not ex.is_upcast


def test_mxfp4_on_blackwell_defaults_to_bf16_macs():
    """FLASHINFER_TRTLLM_MXFP4_BF16 is first in the v0.19.1 priority list: stored
    fp4, executed bf16, on a part that has fp4 tensor cores."""
    ex = resolve_execution("mxfp4", B200)
    assert ex.compute_dtype == "bf16" and ex.is_upcast
    assert (ex.pad_hidden, ex.pad_inter) == (256, 256)


def test_mxfp4_with_and_without_marlin_upcast_differ_in_padding_not_bytes():
    default = resolve_execution("mxfp4", B200)
    marlin = resolve_execution("mxfp4", B200, backend="marlin")
    assert default.streamed_bytes == marlin.streamed_bytes == 0.53125
    assert (marlin.pad_hidden, marlin.pad_inter) == (256, 128)


def test_backend_padding_is_applied_to_each_ranks_slice():
    """Finding 2 of Jalon's review. vLLM splits intermediate across TP ranks first
    (fused_moe/layer.py:426) and pads the per-rank width afterwards (:537-538).
    Width 2,880 across 8 ranks with a 256 multiple: split-then-pad is 360 -> 512
    per rank; pad-then-split is 3,072 / 8 = 384, which the kernel pads again.
    The old order under-counted by 25% (384 / 512). Hidden is not split."""
    trtllm = resolve_execution("mxfp4", B200)
    per_rank = expert_pad_factor(trtllm, 2880, 2880, shards=8)
    assert per_rank == pytest.approx((3072 * 512) / (2880 * 360))
    wrong_order = (3072 * (3072 / 8)) / (2880 * 360)
    assert wrong_order / per_rank == pytest.approx(384 / 512)
    marlin = resolve_execution("mxfp4", H200, backend="marlin")
    assert expert_pad_factor(marlin, 2880, 2880, shards=8) == pytest.approx((3072 * 384) / (2880 * 360))
    # Unsharded, the old numbers still hold.
    assert expert_pad_factor(trtllm, 2880, 2880) == pytest.approx((3072 / 2880) ** 2)
    # Kimi and GLM are exact multiples per rank at TP8, so nothing moves there.
    assert expert_pad_factor(trtllm, 7168, 2048, shards=8) == 1.0
    assert expert_pad_factor(marlin, 7168, 2048, shards=8) == 1.0
    assert expert_pad_factor(trtllm, 6144, 2048, shards=8) == 1.0


def test_padded_expert_bytes_flow_through_the_graph_per_rank():
    """The graph divides expert bytes by the shard count; the pad factor must
    already be per rank or the two cancel into the wrong order."""
    gpt_oss_like = replace(load_spec("kimi-k2.6"), hidden=2880, moe_intermediate_size=2880,
                           expert_dtype="mxfp4")
    g8 = predict_glm_graph(gpt_oss_like, B200, BASELINE, ShardingConfig(tp=8))
    g1 = predict_glm_graph(gpt_oss_like, B200, BASELINE, ShardingConfig(tp=1))
    r8 = next(n for n in g8.nodes if n.op == "moe_routed").prediction.bytes
    r1 = next(n for n in g1.nodes if n.op == "moe_routed").prediction.bytes
    # Weight bytes per rank at TP8 = (whole-model padded-per-rank) / 8, so the
    # ratio to TP1 is the padding ratio between the two orders, not exactly 1/8.
    assert r8 / r1 > (1 / 8) * 1.2


def test_mxfp8_has_no_hopper_kernel():
    with pytest.raises(UnsupportedExecution, match="sm100"):
        resolve_execution("mxfp8", H200)
    ex = resolve_execution("mxfp8", B200)
    assert ex.compute_dtype == "fp8" and ex.act_format is QUANT_FORMATS["mxfp8"]


def test_block_fp8_quantises_activations_per_128_channels():
    """One fp32 scale per 128 channels per row (GroupShape(1, 128)), not per row."""
    ex = resolve_execution("fp8", H200)
    assert ex.compute_dtype == "fp8"
    assert ex.act_format.bytes_per_elem == 1 + 4 / 128


def test_int4_g32_on_h200_is_marlin_with_unchanged_bytes():
    ex = resolve_execution("int4", H200)
    assert (ex.backend, ex.compute_dtype, ex.bytes_per_use) == ("marlin", "bf16", 0.5625)


def test_emulation_is_temporary_traffic_on_top_of_storage():
    """nvfp4_emulation_utils.py: fp32 unpack, fp32 scale, bf16 cast, matmul read."""
    ex = resolve_execution("nvfp4", H200, backend="emulation")
    assert ex.temp_bytes == 4 + (4 + 4) + (4 + 2) + 2
    assert ex.bytes_per_use == pytest.approx(0.5625 + 20)


def test_unknown_arch_keeps_the_old_ladder():
    """No arch, no rule: the stored bytes stay exact and the rate is a flagged guess."""
    ex = resolve_execution("nvfp4", HardwareSpec())
    assert ex.backend == "unknown" and ex.estimated
    assert roofline("x", 1.0, 1.0, HardwareSpec(), "nvfp4").peak_is_fallback


def test_every_pinned_rule_cites_its_evidence():
    for sku in (H200, B200):
        for dtype in ("fp8", "nvfp4", "mxfp4", "int4"):
            assert resolve_execution(dtype, sku).source, (dtype, sku.arch)


# ── ridges, knees and tiers (H200 first) ─────────────────────────────────────


def test_h200_ridges():
    assert ridge(H200, "bf16") == pytest.approx(989e12 / 4.8e12)
    assert ridge(H200, "fp8") == pytest.approx(1979e12 / 4.8e12)
    assert ridge(H200, "fp32") == pytest.approx(67e12 / 4.8e12)
    assert ridge(H200, "bf16", tier="link") == pytest.approx(989e12 / 900e9)


def test_nvfp4_ridge_follows_execution_not_storage():
    """Marlin NVFP4 on H200 answers to the bf16 ridge (206). The old ladder priced
    it at fp8 (412), doubling its compute ceiling and flagging it as a fallback."""
    assert ridge(H200, "nvfp4") == ridge(H200, "bf16")
    assert ridge(B200, "nvfp4") == pytest.approx(9000e12 / 8e12)
    pred = roofline("moe_routed", 1e12, 1e9, H200, "nvfp4")
    assert pred.compute_dtype == "bf16" and not pred.peak_is_fallback


def test_critical_rows_matches_the_closed_form():
    """AI(r*) equals the ridge exactly, and the large-matrix limit is R w / 2."""
    k, n = 7168, 2048
    r = critical_rows("nvfp4", H200, k, n)
    ai = 2 * r * k * n / (0.5625 * k * n + 2 * r * (k + n))
    assert ai == pytest.approx(ridge(H200, "nvfp4"))
    big = critical_rows("nvfp4", H200, 10**7, 10**7)
    assert big == pytest.approx(ridge(H200, "bf16") * 0.5625 / 2, rel=1e-3)


def test_kimi_decode_is_far_below_the_expert_knee():
    """B=32 wakes 188 of 384 experts: 1.36 rows each, against a knee of ~67 on H200."""
    rows_per_expert = 32 * 8 / distinct_experts(32, 384, 8)
    assert rows_per_expert == pytest.approx(1.36, abs=0.01)
    assert critical_rows("nvfp4", H200, 7168, 2048) > 40 * rows_per_expert


def test_linear_traffic_splits_payload_scales_and_scratch():
    ex = resolve_execution("nvfp4", H200)
    t = linear_traffic(32, 7168, 2048, ex, 2.0)
    w = 7168 * 2048
    assert t.weight_payload == w * 0.5 and t.weight_scales == pytest.approx(w / 16)
    assert t.activations == 2.0 * 32 * (7168 + 2048) and t.temporary == 0.0
    assert t.hbm == pytest.approx(w * 0.5625 + t.activations)


# ── design note 1: the Kimi K2.6 NVFP4 case ──────────────────────────────────


def _case_spec():
    """The submitted YAML's precision: bf16 cache, shared expert bf16, fp32 router."""
    return replace(
        load_spec("kimi-k2.6"), kv_dtype="bf16", kv_rope_dtype="bf16",
        op_dtype_overrides=(("moe_router", "fp32"), ("moe_shared", "bf16")),
    )


def test_case_expert_and_kv_constants():
    assert 3 * 7168 * 2048 * weight_bytes("nvfp4") == 24_772_608  # 24.77 MB/expert
    assert distinct_experts(32, 384, 8) == pytest.approx(188.2, abs=0.05)
    assert kv_bytes_per_token(_case_spec()) == 61 * 576 * 2 == 70_272


def test_case_fp8_kv_figure_was_the_wrong_layout():
    """The case priced fp8 KV at 640 B/layer (fp8 latent + bf16 RoPE) = 39,040 B/token.
    Dense MLA with --kv-cache-dtype fp8 uses vLLM's generic layout, one byte for all
    576 dims (mla_attention.py:316 head_size = 512 + 64; :1130-1137 cache shape
    (blocks, block_size, head_size)), so 35,136 B/token and 9.21 GB at the baseline."""
    assert kv_bytes_per_token(load_spec("kimi-k2.6")) == 61 * 576 == 35_136
    assert 61 * (512 + 64 * 2) == 39_040


def test_case_expert_row_reproduces_on_b200_tp4():
    """H1 in the case: 1,165.7 MB per layer per rank and 8.74 ms over 60 layers."""
    g = predict_glm_graph(_case_spec(), B200, BASELINE, ShardingConfig(tp=4))
    routed = [n for n in g.nodes if n.op == "moe_routed"]
    weight_bytes_per_layer = distinct_experts(32, 384, 8) * 24_772_608 / 4
    assert weight_bytes_per_layer / 1e6 == pytest.approx(1165.7, abs=0.1)
    assert routed[0].prediction.bytes == pytest.approx(weight_bytes_per_layer, rel=0.001)
    assert sum(n.prediction.t_pred_s for n in routed) * 1e3 == pytest.approx(8.74, abs=0.02)


def test_case_router_no_longer_prices_at_a100_fp32():
    """The 13.56 ms planner run carried 0.54 ms of fp32 router compute at A100's
    19.5 TF/s because B200 had no fp32 entry. At B200's 75 TF/s it is 0.14 ms."""
    assert B200.peak_flops_fp32_per_s == 75e12
    g = predict_glm_graph(_case_spec(), B200, BASELINE, ShardingConfig(tp=4))
    flops = 2 * 32 * 7168 * 384
    # Two moe_router nodes per layer share the op name (GEMM, then gating); the
    # GEMM is the one carrying 2 x rows x hidden x experts.
    gemm = [n for n in g.nodes if n.op == "moe_router" and n.prediction.flops == flops]
    assert len(gemm) == 60 and gemm[0].prediction.dtype == "fp32"
    assert sum(n.prediction.t_compute_s for n in gemm) == pytest.approx(60 * flops / 75e12)


def test_case_fit_uses_one_ledger():
    """Workspace was charged in the case's fit but not in its 'available for KV'.
    One ledger: TP4 leaves 162 - 150.5 - 4.5 = 7.0 GB, not 10.9, so 99k tokens
    (12 sequences at 8K rather than 18), and the bf16 baseline does not fit.
    TP8 does. (The planner omits the 0.94 GB vision tower the case counted.)"""
    tp4 = memory_fit(_case_spec(), B200, BASELINE, ShardingConfig(tp=4), workspace_bytes=4.5e9)
    assert tp4.kv_available == pytest.approx(tp4.budget - tp4.weights - 4.5e9)
    assert tp4.kv_tokens == pytest.approx(tp4.kv_available / 70_272)
    assert not tp4.fits
    assert tp4.kv_tokens // 8192 == 12
    tp8 = memory_fit(_case_spec(), B200, BASELINE, ShardingConfig(tp=8), workspace_bytes=4.5e9)
    assert tp8.fits
    # The default NVIDIA's command serves is an fp8 cache (vLLM resolves 'auto'
    # from the checkpoint's kv_cache_scheme); TP4 still does not hold 32 x 8K.
    fp8 = replace(_case_spec(), kv_dtype="fp8", kv_rope_dtype="fp8")
    tp4_fp8 = memory_fit(fp8, B200, BASELINE, ShardingConfig(tp=4), workspace_bytes=4.5e9)
    assert not tp4_fp8.fits and tp4_fp8.kv_tokens // 8192 == 24


# ── design note 2: GLM-5.2 on 8xH200 ─────────────────────────────────────────


def test_glm_ridges_and_kv_per_token():
    """Note §0: 412 / 206 / 14. Note §1 computed 52,618 B/token and flagged it as
    using the weight constant for the cache; the cache format gives its 52,608."""
    assert round(ridge(H200, "fp8")) == 412
    assert round(ridge(H200, "bf16")) == 206
    assert round(ridge(H200, "fp32")) == 14
    glm = load_spec("glm-5.2-fp8")
    assert kv_bytes_per_token(glm) == 78 * (512 + 64 * 2) + 21 * 128 == 52_608


def test_glm_decode_floor_reproduces():
    """Note §4.1: 16.254 ms at B=32, S=8192, TP8/EP8; moe_routed 12.052 ms."""
    g = predict_glm_graph(
        load_spec("glm-5.2-fp8"), H200, BASELINE, ShardingConfig(tp=8, ep=8)
    )
    routed = sum(n.prediction.t_pred_s for n in g.nodes if n.op == "moe_routed")
    assert g.total_pred_s * 1e3 == pytest.approx(16.254, abs=0.001)
    assert routed * 1e3 == pytest.approx(12.052, abs=0.001)
    assert not g.has_fallback_peaks


def test_glm_act_quant_charges_a_scale_per_128_channels():
    """Note A.1 priced 0.590 MB (one scale per row). Per-128 groups at hidden 6144
    are 48 fp32 scales a row, 0.596 MB. Still launch-bound, so the floor holds."""
    g = predict_glm_graph(
        load_spec("glm-5.2-fp8"), H200, BASELINE, ShardingConfig(tp=8, ep=8)
    )
    aq = next(n for n in g.nodes if n.op == "act_quant")
    assert aq.prediction.bytes == pytest.approx(32 * 6144 * (2 + 1 + 4 / 128))
    assert aq.prediction.bound == "launch"


def test_dense_ffn_is_priced_once_per_weight():
    """model_weight_bytes multiplied the dense FFN by bytes-per-weight twice, so a
    bf16 dense layer cost 4 B/weight. GLM-5.2's three dense layers carried 1.36 GB
    of it, the whole of the bf16 note's +0.08% against 1,506,659,919,872 B; on
    Kimi the 0.79 GB it added hid the unmodelled 0.94 GB vision tower."""
    glm = load_spec("glm-5.2")
    dense = 3 * 3 * 6144 * 12288 * 2.0
    assert model_weight_bytes(glm) / 1_506_659_919_872 - 1 == pytest.approx(0, abs=2e-4)
    assert dense == pytest.approx(1.359e9, rel=1e-3)
    k26 = load_spec("kimi-k2.6")
    assert 595_148_192_736 - model_weight_bytes(k26) == pytest.approx(0.94e9, rel=0.05)


def test_resident_weights_equal_storage_when_nothing_pads():
    for name, sku in (("kimi-k2.6", H200), ("glm-5.2-fp8", H200), ("kimi-k2.6", B200)):
        spec = load_spec(name)
        assert model_weight_bytes(spec, hw=sku) == pytest.approx(model_weight_bytes(spec))


def test_mi355x_rules_are_read_from_aiter_and_vllm_rocm_sources():
    """CDNA4 rules come from vLLM b1388b1f plus the AITER tag its ROCm image pins
    (v0.1.10.post2) and CK; none is inferred. INT4 g32 runs the Triton W4A16
    kernel (compressed_tensors_moe.py:177-190; fused_moe.py:288-289 dequantises
    in-kernel), block fp8 runs AITER with per-token group-128 fp32 activation
    scales, MXFP4 runs the CK 2-stage scaled-f8f6f4 MFMA with a separate MXFP4
    activation quant and per-rank padding to 256. NVFP4 has no ROCm backend."""
    assert MI355X.arch == "cdna4" and MI355X.memory_bytes == 288e9
    int4 = resolve_execution("int4", MI355X)
    assert (int4.backend, int4.compute_dtype, int4.act_format, int4.estimated) == (
        "triton_wna16", "bf16", None, False)
    assert int4.bytes_per_use == 0.5625
    fp8 = resolve_execution("fp8", MI355X)
    assert (fp8.backend, fp8.compute_dtype, fp8.estimated) == ("aiter", "fp8", False)
    assert fp8.act_format.name == "fp8_group128"
    mx = resolve_execution("mxfp4", MI355X)
    assert (mx.backend, mx.compute_dtype, mx.estimated) == ("aiter_ck2stages", "fp4", False)
    assert mx.act_format.name == "mxfp4" and (mx.pad_hidden, mx.pad_inter) == (256, 256)
    for dtype in ("fp8", "mxfp4", "int4"):
        assert "aiter/" in resolve_execution(dtype, MI355X).source or "rocm" in resolve_execution(dtype, MI355X).source
    with pytest.raises(UnsupportedExecution, match="no NvFp4 MoE backend"):
        resolve_execution("nvfp4", MI355X)
    with pytest.raises(UnsupportedExecution, match="is_cuda"):
        resolve_execution("int4", MI355X, backend="marlin")


def test_plan_prices_an_explicit_cache_dtype_over_the_resolved_default(capsys):
    """'auto' keeps the catalogue's kv_dtype, which records vLLM's resolution:
    K2.6-NVFP4 declares a static fp8 kv_cache_scheme, so its default is fp8
    (utils/torch_utils.py:262-342). An explicit bf16 prices the wide cache.

    Through the public ``gitm plan`` dispatcher, which rebuilds argv and had
    dropped every flag added here."""
    from gitm.cli import main

    base = ["plan", "kimi-k2.6", "--gpu", "H200", "--batch", "32", "--kv-len", "8192",
            "--tp", "8", "--workspace-gb", "4.5", "--gpu-mem-util", "0.9"]
    assert main(base) == 0
    default = capsys.readouterr().out
    assert main([*base, "--kv-cache-dtype", "bf16"]) == 0
    wide = capsys.readouterr().out
    assert "floor 11.404 ms/step" in default and "35,136 B/token" in default
    assert "floor 13.323 ms/step" in wide and "70,272 B/token" in wide
    assert "4.5 GB workspace" in default and "46.2 GB for KV" in default


def test_plan_kv_cache_dtype_applies_to_every_family(capsys):
    """The hybrid family stores a cache too; the flag must not be a silent no-op."""
    from gitm.cli import main

    base = ["plan", "qwen3.6-35b-a3b", "--gpu", "H200", "--batch", "8", "--json"]
    assert main([*base, "--kv-cache-dtype", "bf16"]) == 0
    bf16 = json.loads(capsys.readouterr().out)["total_pred_s"]
    assert main([*base, "--kv-cache-dtype", "fp8"]) == 0
    fp8 = json.loads(capsys.readouterr().out)["total_pred_s"]
    assert fp8 < bf16


def test_fit_counts_the_prefill_chunk():
    """An empty-cache 8,192-token prefill writes 8,192 x 35,136 B = 288 MB of KV
    on Kimi; a decode-only ledger reported zero need for it."""
    k26 = load_spec("kimi-k2.6")
    prefill = BatchConfig(batch=0, kv_cache_len=0, prefill_tokens=8192, prefill_requests=1)
    fit = memory_fit(k26, H200, prefill, ShardingConfig(tp=8))
    assert fit.kv_needed == 8192 * 35_136
    chunk2 = replace(prefill, prefill_context=8192)
    assert memory_fit(k26, H200, chunk2, ShardingConfig(tp=8)).kv_needed == 2 * 8192 * 35_136


def test_fp32_peak_is_carried_by_every_blackwell_entry():
    """The fp32 fallback is not flagged, so a missing entry is a confident wrong
    number: GB200 priced its router at A100's 19.5 TF/s while B200 used 75."""
    for sku in ("B200", "GB200", "B300", "GB300"):
        assert _hw(sku).peak_flops_fp32_per_s == 75e12, sku


def test_estimated_backend_rules_reach_the_prediction():
    """Finding 3 of Jalon's review. An execution rule inferred rather than read
    from source is estimated=True on the rule; before the fix the graph node it
    priced said estimated=False, so the report presented an inferred kernel as
    a derived floor. MXFP4 on Hopper (Triton, triton_kernels not read) is the
    remaining inferred rule."""
    mx = replace(load_spec("kimi-k2.6"), expert_dtype="mxfp4")
    assert resolve_execution("mxfp4", H200).estimated is True
    g = predict_glm_graph(mx, H200, BASELINE, ShardingConfig(tp=8))
    routed = [n for n in g.nodes if n.op == "moe_routed"]
    assert routed and all(n.prediction.estimated for n in routed)
    # A read rule stays a clean floor.
    g = predict_glm_graph(load_spec("kimi-k2.5"), MI355X, BASELINE, ShardingConfig(tp=8))
    assert not any(n.prediction.estimated for n in g.nodes if n.op == "moe_routed")


def test_plan_json_carries_estimated_per_node(capsys, tmp_path):
    """The flag has to survive to `gitm plan --json`, the report's input."""
    import yaml

    from gitm.planner.registry import main

    entry = yaml.safe_load(open("gitm/planner/models/kimi-k2.6.yaml"))
    entry["spec"]["expert_dtype"] = "mxfp4"
    p = tmp_path / "kimi-mxfp4.yaml"
    p.write_text(yaml.safe_dump(entry))
    assert main([str(p), "--gpu", "H200", "--batch", "32", "--kv-len", "8192",
                 "--tp", "8", "--json"]) == 0
    nodes = json.loads(capsys.readouterr().out)["nodes"]
    assert {n["estimated"] for n in nodes if n["op"] == "moe_routed"} == {True}
