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
