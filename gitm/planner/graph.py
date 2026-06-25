"""Predicted execution graph.

A flat list of predicted nodes per decode step. v0 is intentionally simple:
attention QKV projection, attention score (GQA-aware), attention output, MLP
gate+up, MLP down, vocab projection — one decode step worth.

v0 emits one decode step worth of nodes; multi-step and dependency-edge
modeling are on the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ModelSpec,
    RooflinePrediction,
    roofline,
)


@dataclass
class PredictedNode:
    op: str
    layer: int | None
    prediction: RooflinePrediction
    # Streams the planner expects to run on — used by the stream-concurrency
    # invariant.
    expected_stream_id: int = 0


@dataclass
class Graph:
    model: ModelSpec
    hw: HardwareSpec
    batch: BatchConfig
    nodes: list[PredictedNode] = field(default_factory=list)

    @property
    def total_pred_s(self) -> float:
        return sum(n.prediction.t_pred_s for n in self.nodes)


def predict_graph(
    model: ModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
) -> Graph:
    """Emit a predicted execution graph for one decode step.

    GQA-aware: KV-cache reads scale with ``num_kv_heads``, not ``n_heads``.
    """
    model = model or ModelSpec()
    hw = hw or HardwareSpec()
    batch = batch or BatchConfig()

    g = Graph(model=model, hw=hw, batch=batch)
    b = batch.batch
    h = model.hidden
    kv_len = batch.kv_cache_len
    head_dim = model.head_dim
    n_kv = model.num_kv_heads
    n_h = model.n_heads
    dt = model.dtype_bytes

    for layer in range(model.n_layers):
        # QKV projection: matmul (b, h) @ (h, (n_h + 2*n_kv) * head_dim)
        qkv_out = (n_h + 2 * n_kv) * head_dim
        flops = 2 * b * h * qkv_out
        bytes_moved = dt * (b * h + h * qkv_out + b * qkv_out)
        g.nodes.append(
            PredictedNode("qkv_proj", layer, roofline("qkv_proj", flops, bytes_moved, hw))
        )

        # Attention scores + softmax + value: KV-cache traffic dominates at decode.
        # Reads: K, V over kv_len tokens, grouped to n_kv heads.
        kv_bytes = dt * 2 * kv_len * n_kv * head_dim * b
        attn_flops = 2 * b * n_h * head_dim * kv_len * 2  # qk + sv
        g.nodes.append(
            PredictedNode(
                "attn_score_value",
                layer,
                roofline("attn_score_value", attn_flops, kv_bytes, hw),
            )
        )

        # Output projection
        flops = 2 * b * h * h
        bytes_moved = dt * (b * h + h * h + b * h)
        g.nodes.append(
            PredictedNode("attn_out_proj", layer, roofline("attn_out_proj", flops, bytes_moved, hw))
        )

        # MLP gate+up
        flops = 2 * 2 * b * h * model.intermediate
        bytes_moved = dt * (b * h + 2 * h * model.intermediate + 2 * b * model.intermediate)
        g.nodes.append(
            PredictedNode("mlp_gate_up", layer, roofline("mlp_gate_up", flops, bytes_moved, hw))
        )

        # MLP down
        flops = 2 * b * model.intermediate * h
        bytes_moved = dt * (b * model.intermediate + model.intermediate * h + b * h)
        g.nodes.append(
            PredictedNode("mlp_down", layer, roofline("mlp_down", flops, bytes_moved, hw))
        )

    # Final vocab projection
    flops = 2 * b * h * model.vocab
    bytes_moved = dt * (b * h + h * model.vocab + b * model.vocab)
    g.nodes.append(
        PredictedNode("lm_head", None, roofline("lm_head", flops, bytes_moved, hw))
    )

    return g
