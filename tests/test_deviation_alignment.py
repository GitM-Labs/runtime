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


def _k(name: str, dur_s: float, t0: int = 0) -> KernelEvent:
    return KernelEvent(
        name=name, start_ns=t0, end_ns=t0 + int(dur_s * 1e9), stream_id=0, device_id=0
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


def test_no_predicted_graph_keeps_everything():
    from gitm.planner.graph import Graph
    from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec

    empty = Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(), nodes=[])
    tr = _trace([_k("anything", 1e-6), _k("else", 1e-6)])
    assert deviating_kernel_indices(tr, empty).kept_indices == [0, 1]
