"""Residual pairing across structurally different layers.

A dense transformer repeats one layer, so any predicted node stands for every
other node with the same op. ``residuals()`` relied on that, keeping one
representative per op. The assumption does not survive a heterogeneous stack:
on a DeepSeek-V4-class model layers 0-1 are sliding-window (128 tokens, no
indexer) and 2-42 are compressed (640 tokens), and the compressed layers split a
further 32x apart at the indexer.

These tests pin three behaviours: dense models are untouched, a resolved layer
gives an exact comparison, and an unresolved layer on a heterogeneous op falls
back to an interval rather than to whichever node happened to be emitted first.
"""

from __future__ import annotations

import pytest

from gitm.optimizer.monitor import residuals
from gitm.planner.graph import Graph, PredictedNode
from gitm.planner.moe_graph import predict_moe_graph, spec_from_hf_config
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ModelSpec,
    RooflinePrediction,
)
from gitm.tracer.schema import KernelEvent, Trace


def _node(op: str, layer: int | None, t_pred: float, byts: float = 0.0) -> PredictedNode:
    return PredictedNode(
        op, layer,
        RooflinePrediction(
            op=op, flops=0.0, bytes=byts, t_compute_s=0.0, t_memory_s=t_pred,
            t_pred_s=t_pred, bound="memory",
        ),
    )


def _graph(nodes: list[PredictedNode]) -> Graph:
    return Graph(model=ModelSpec(), hw=HardwareSpec(), batch=BatchConfig(), nodes=nodes)


def _k(name: str, dur_s: float, layer: int | None = None, byts: int | None = None) -> KernelEvent:
    # Round, not truncate. A trace stores whole nanoseconds, and V4's
    # sliding-window attention predicts ~3.66 us — truncating drops it below its
    # own prediction and manufactures a residual out of the helper.
    return KernelEvent(
        name=name, start_ns=0, end_ns=round(dur_s * 1e9), stream_id=0, device_id=0,
        range_layer=layer,
        bytes_read=byts, bytes_written=0 if byts is not None else None,
    )


#: Nanosecond quantisation floor. Real predictions are not whole nanoseconds, so
#: a kernel replayed at exactly its predicted time still lands ~1e-4 off it.
NS_QUANTISATION = 1e-3


def _trace(events: list[KernelEvent]) -> Trace:
    return Trace(
        workload_id="w", fingerprint="f", run_id="r", device_count=1,
        vendor="nvidia", captured_at_ns=0, duration_ns=10**9, events=events,
    )


# ── uniform ops: unchanged ──────────────────────────────────────────────────


def test_uniform_op_keeps_the_exact_point_residual():
    """Every dense model, and 8 of V4's 10 per-layer ops, land here.

    All layers agree, so there is one class and the comparison is a point — the
    behaviour that existed before class-awareness, preserved exactly.
    """
    g = _graph([_node("attn_score_value", i, 1e-3) for i in range(4)])
    res = residuals(_trace([_k("flash_attn_fwd", 2e-3)]), g)

    kr = res.per_kernel[0]
    assert kr.n_classes == 1
    assert not kr.interval_based
    assert kr.r_kt == pytest.approx(1.0)  # 2ms observed against 1ms predicted


# ── heterogeneous ops, layer unknown: interval ──────────────────────────────


def test_observation_inside_the_class_span_scores_zero():
    """The core fix. Layers disagree, the kernel's layer is unknown, and the
    observation is consistent with *some* layer — so there is no evidence of
    deviation and the residual must not manufacture one."""
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 3e-3)]), g)

    kr = res.per_kernel[0]
    assert kr.n_classes == 2
    assert kr.interval_based
    assert kr.r_kt == 0.0


def test_observation_above_the_span_still_deviates():
    """Conservative is not blind — outside every prediction is still a finding."""
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 10e-3)]), g)
    assert res.per_kernel[0].r_kt == pytest.approx(1.0)  # (10-5)/5, against the near edge


def test_observation_below_the_span_still_deviates():
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 0.5e-3)]), g)
    assert res.per_kernel[0].r_kt == pytest.approx(-0.5)  # (0.5-1)/1


def test_memory_traffic_uses_the_same_interval():
    g = _graph([
        _node("attn_score_value", 0, 1e-3, byts=1000.0),
        _node("attn_score_value", 2, 5e-3, byts=9000.0),
    ])
    res = residuals(_trace([_k("flash_attn_fwd", 3e-3, byts=5000)]), g)
    assert res.per_kernel[0].r_mt == 0.0


# ── heterogeneous ops, layer known: exact ───────────────────────────────────


def test_resolved_layer_compares_against_that_layer_only():
    """With NVTX identity there is no ambiguity to be conservative about.

    The same 3ms observation that scored 0.0 as an interval is a real +200%
    against layer 0's own prediction — which is the point of landing NVTX.
    """
    g = _graph([_node("attn_score_value", 0, 1e-3), _node("attn_score_value", 2, 5e-3)])
    res = residuals(_trace([_k("flash_attn_fwd", 3e-3, layer=0)]), g)

    kr = res.per_kernel[0]
    assert kr.layer == 0
    assert not kr.interval_based  # layer resolved, so not interval-based
    assert kr.r_kt == pytest.approx(2.0)


def test_resolved_layer_picks_the_right_one_of_several():
    g = _graph([_node("attn_score_value", i, (i + 1) * 1e-3) for i in range(4)])
    res = residuals(_trace([_k("flash_attn_fwd", 3e-3, layer=2)]), g)
    assert res.per_kernel[0].r_kt == pytest.approx(0.0)  # layer 2 predicts exactly 3ms


# ── the regression this exists for ──────────────────────────────────────────

V4_CONFIG = {
    "model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43,
    "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
    "qk_rope_head_dim": 64, "q_lora_rank": 1024, "o_lora_rank": 1024, "o_groups": 8,
    "vocab_size": 129280, "n_routed_experts": 256, "n_shared_experts": 1,
    "num_experts_per_tok": 6, "moe_intermediate_size": 2048, "index_n_heads": 64,
    "index_head_dim": 128, "index_topk": 512, "sliding_window": 128,
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
    "num_nextn_predict_layers": 1, "expert_dtype": "fp4",
    "quantization_config": {"quant_method": "fp8"}, "torch_dtype": "bfloat16",
}


@pytest.fixture
def v4_graph():
    spec = spec_from_hf_config(V4_CONFIG, name="DeepSeek-V4-Flash")
    return predict_moe_graph(
        spec, HardwareSpec(), BatchConfig(batch=64, kv_cache_len=65536)
    )


def test_healthy_compressed_layer_is_not_flagged(v4_graph):
    """The artefact this change removes.

    Layer 0 is sliding-window and is emitted first, so it used to become the
    representative for `attn_score_value` across all 43 layers. A compressed
    layer running at exactly its own predicted time then scored r_kt = +4.0
    against a +-0.4 band — and `check_invariants` reads a systematic offset as
    confirmation, so multi-basis filtering amplified it rather than rejecting it.
    """
    compressed = next(
        n for n in v4_graph.nodes if n.op == "attn_score_value" and n.layer == 2
    )
    res = residuals(_trace([_k("flash_mla_sparse_fwd", compressed.prediction.t_pred_s)]), v4_graph)

    kr = res.per_kernel[0]
    assert kr.n_classes == 2  # sliding-window vs compressed
    assert abs(kr.r_kt) < NS_QUANTISATION
    assert abs(kr.r_kt) < 0.4  # comfortably inside the kernel-time band


def test_healthy_sliding_window_layer_is_not_flagged(v4_graph):
    """The other direction: layer 0 running at its own prediction is also fine."""
    swa = next(n for n in v4_graph.nodes if n.op == "attn_score_value" and n.layer == 0)
    res = residuals(_trace([_k("flash_mla_sparse_fwd", swa.prediction.t_pred_s)]), v4_graph)
    assert abs(res.per_kernel[0].r_kt) < NS_QUANTISATION


def test_indexer_span_covers_both_compression_ratios(v4_graph):
    """`attn_index_score` splits 32x between ratio-4 and ratio-128 layers.

    Sliding-window layers emit no indexer at all, so this span is entirely
    within what the config calls the compressed layers — a different partition
    from the one `attn_score_value` needs, which is why the fix is per-op rather
    than one global layer taxonomy.
    """
    idx = [n for n in v4_graph.nodes if n.op == "attn_index_score"]
    assert {n.layer for n in idx}.isdisjoint({0, 1})
    ts = sorted(n.prediction.t_pred_s for n in idx)
    assert ts[-1] / ts[0] == pytest.approx(32.0, rel=0.01)

    res = residuals(_trace([_k("lightning_indexer_topk", ts[0]), _k("lightning_indexer_topk", ts[-1])]), v4_graph)
    assert all(abs(kr.r_kt) < NS_QUANTISATION for kr in res.per_kernel)


def test_a_genuinely_slow_kernel_still_surfaces(v4_graph):
    """Not a blanket suppression — 10x the slowest layer is still a violation."""
    slowest = max(
        n.prediction.t_pred_s for n in v4_graph.nodes if n.op == "attn_score_value"
    )
    res = residuals(_trace([_k("flash_mla_sparse_fwd", slowest * 10)]), v4_graph)
    assert res.per_kernel[0].r_kt == pytest.approx(9.0, rel=1e-3)
