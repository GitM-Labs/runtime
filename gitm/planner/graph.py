"""Predicted execution graph.

A flat list of predicted nodes per decode step. v0 is intentionally simple:
attention QKV projection, attention score (GQA-aware), attention output, MLP
gate+up, MLP down, vocab projection — one decode step worth.

v0 emits one decode step worth of nodes; multi-step and dependency-edge
modeling are on the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ModelSpec,
    RooflinePrediction,
    ShardingConfig,
    SparseMoEModelSpec,
    distinct_experts,
    roofline,
)


def _ffn_terms(model: ModelSpec, b: int, *, moe_layer: bool = True, tp: int = 1) -> tuple[
    float, float, float, float
]:
    """``(gate_up_flops, gate_up_bytes, down_flops, down_bytes)`` for one layer.

    ``moe_layer`` selects the mixture arithmetic for *this* layer. MoE
    checkpoints are not uniformly sparse (see :meth:`ModelSpec.is_moe_layer`), so
    a dense block inside an MoE model is priced as a dense FFN.

    Dense (``num_experts == 0``) reproduces the original single-FFN arithmetic
    exactly. For MoE the two costs are driven by *different* counts, which is the
    whole point of the mixture:

    * **flops** scale with the experts each token activates — ``b * top_k``
      routed (plus every shared expert for every token), so compute grows
      linearly with batch;
    * **weight bytes** scale with the *distinct* experts the batch touches
      (:func:`distinct_experts`), because an expert's weights are read once per
      step however many tokens route to it — so traffic grows sublinearly and
      saturates at ``num_experts``.

    Weights use ``model.w_bytes`` and activations ``model.dtype_bytes``, so a
    quantized MoE checkpoint (fp8 weights, bf16 activations) is modeled with the
    right width on the term that dominates.

    The router GEMM (``[b, hidden] @ [hidden, num_experts]``) is folded into
    gate_up rather than given its own node: it is real but ~1% of expert-GEMM
    cost, and adding an op would change the canonical vocabulary that
    ``classify_op`` and ``library.yaml``'s ``applies_to_kernels`` key off.
    """
    h = model.hidden
    dt = model.dtype_bytes  # activations
    wb = model.w_bytes  # weights (may be narrower, e.g. fp8)

    if not (model.is_moe and moe_layer):
        # Dense: one FFN, weights fetched once, every token through all of it.
        ff = model.intermediate / tp
        gate_up_flops = 2 * 2 * b * h * ff
        gate_up_bytes = dt * (b * h + 2 * b * ff) + wb * (2 * h * ff)
        down_flops = 2 * b * ff * h
        down_bytes = dt * (b * ff + b * h) + wb * (ff * h)
        return gate_up_flops, gate_up_bytes, down_flops, down_bytes

    k = model.top_k
    ff = model.expert_intermediate
    n_shared = model.shared_experts
    sff = model.shared_intermediate
    # Expected distinct routed experts whose weights this step must fetch.
    distinct = distinct_experts(b, model.num_experts, k)

    # Compute: every token pays k routed experts plus all shared ones.
    gate_up_flops = 2 * 2 * b * (k * h * ff + n_shared * h * sff) / tp
    down_flops = 2 * b * (k * ff * h + n_shared * sff * h) / tp
    # Router: [b, h] @ [h, num_experts], folded in above.
    gate_up_flops += 2 * b * h * model.num_experts

    # Weight traffic: distinct routed experts once each, plus the shared experts
    # (always resident in the step) and the router matrix.
    gate_up_weight_bytes = wb * (
        (distinct * 2 * h * ff + n_shared * 2 * h * sff) / tp
        + h * model.num_experts
    )
    down_weight_bytes = wb * (distinct * ff * h + n_shared * sff * h) / tp
    # Activations: in [b, h], out [b, k*ff] (+ shared) for gate_up; mirrored for down.
    act_out = b * (k * ff + n_shared * sff) / tp
    gate_up_bytes = dt * (b * h + 2 * act_out) + gate_up_weight_bytes
    down_bytes = dt * (act_out + b * h) + down_weight_bytes
    return gate_up_flops, gate_up_bytes, down_flops, down_bytes


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
    # Dense (:func:`predict_graph`) or sparse-MoE
    # (:func:`gitm.planner.moe_graph.predict_moe_graph`) — the node list is the
    # same shape either way, so everything downstream of the planner (residuals,
    # deviation, attribution) consumes both without branching.
    model: ModelSpec | SparseMoEModelSpec
    hw: HardwareSpec
    batch: BatchConfig
    nodes: list[PredictedNode] = field(default_factory=list)
    # How the model is spread across ranks. The default (1/1/1) means the graph
    # is whole-model, which is what every dense caller wants.
    sharding: ShardingConfig = field(default_factory=ShardingConfig)

    @property
    def total_pred_s(self) -> float:
        return sum(n.prediction.t_pred_s for n in self.nodes)

    @property
    def has_unpriced_nodes(self) -> bool:
        """True when a positive compute or byte term lacks its denominator.

        A priced memory term must not hide missing compute throughput (or vice
        versa), so this cannot be inferred from total predicted time alone.
        """
        return self.has_unpriced_compute or self.has_unpriced_memory

    @property
    def has_unpriced_compute(self) -> bool:
        return any(n.prediction.compute_is_unpriced for n in self.nodes)

    @property
    def has_unpriced_memory(self) -> bool:
        return any(n.prediction.memory_is_unpriced for n in self.nodes)

    @property
    def has_unpriced_collectives(self) -> bool:
        """True if a collective moves bytes but predicts zero time."""
        collective_ops = {"moe_all_to_all", "tp_all_reduce"}
        return any(
            n.op in collective_ops and n.prediction.memory_is_unpriced
            for n in self.nodes
        )

    @property
    def has_fallback_peaks(self) -> bool:
        """True if any node was priced against a dtype it doesn't run in.

        The report must surface this: a graph built on fallback peaks has a
        systematically low ceiling, so its headroom is an overestimate.
        """
        return any(n.prediction.peak_is_fallback for n in self.nodes)

    @property
    def has_fallback_bytes(self) -> bool:
        """True if any node substituted bf16 for an unknown byte-width dtype.

        Kept separate from :attr:`has_fallback_peaks`: compute peak and byte
        width are independent inputs, and decode is commonly memory-bound.
        """
        return any(n.prediction.bytes_are_fallback for n in self.nodes)

    @property
    def hardware_is_fallback(self) -> bool:
        """True when the graph uses substituted rather than detected SKU peaks."""
        return self.hw.is_fallback

    @property
    def resident_weight_bytes_per_rank(self) -> float | None:
        """Sparse-model resident weight footprint for this graph's rank.

        Dense v0 does not yet enumerate a complete resident footprint, so it
        returns ``None`` rather than repurposing per-step traffic as capacity.
        """
        if not isinstance(self.model, SparseMoEModelSpec):
            return None
        # Local import avoids the graph <-> sparse graph construction cycle.
        from gitm.planner.moe_graph import model_weight_bytes

        return model_weight_bytes(self.model, self.sharding)

    @property
    def resident_weight_bytes_is_lower_bound(self) -> bool:
        """True when private DSpark shapes make the footprint a known lower bound."""
        return isinstance(self.model, SparseMoEModelSpec) and bool(self.model.dspark_layer_ids)


def predict_graph(
    model: ModelSpec | None = None,
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
) -> Graph:
    """Emit a predicted execution graph for one decode step.

    GQA-aware: KV-cache reads scale with ``num_kv_heads``, not ``n_heads``.
    """
    model = model or ModelSpec()
    hw = hw or HardwareSpec()
    batch = batch or BatchConfig()
    sharding = sharding or ShardingConfig()
    if sharding.ep != 1:
        raise ValueError("dense graph does not support expert parallelism")
    tp = sharding.tp
    if model.n_heads % tp:
        raise ValueError(f"n_heads={model.n_heads} must be divisible by tp={tp}")
    if model.num_kv_heads >= tp:
        if model.num_kv_heads % tp:
            raise ValueError(
                f"num_kv_heads={model.num_kv_heads} must be divisible by tp={tp}"
            )
        kv_heads_rank = model.num_kv_heads / tp
    else:
        if tp % model.num_kv_heads:
            raise ValueError(
                f"tp={tp} must be divisible by replicated num_kv_heads={model.num_kv_heads}"
            )
        kv_heads_rank = 1

    g = Graph(model=model, hw=hw, batch=batch, sharding=sharding)
    b = batch.positions_per_step
    sequences = batch.batch
    h = model.hidden
    kv_len = batch.kv_cache_len
    head_dim = model.head_dim
    n_h = model.n_heads
    dt = model.dtype_bytes
    wb = model.w_bytes
    q_heads_rank = n_h / tp

    for layer in range(model.n_layers):
        # QKV projection: matmul (b, h) @ (h, (n_h + 2*n_kv) * head_dim)
        qkv_out = (q_heads_rank + 2 * kv_heads_rank) * head_dim
        flops = 2 * b * h * qkv_out
        bytes_moved = dt * (b * h + b * qkv_out) + wb * h * qkv_out
        g.nodes.append(
            PredictedNode(
                "qkv_proj",
                layer,
                roofline("qkv_proj", flops, bytes_moved, hw, dtype=model.compute_dtype),
            )
        )

        # Attention scores + softmax + value. Full-attention layers re-read a KV
        # cache that grows with context; linear/recurrent layers (gated DeltaNet,
        # Mamba) carry a fixed-size state instead, so their traffic is flat in
        # sequence length. Pricing the latter as KV overstates traffic by roughly
        # kv_len / head_dim — over 100x at 16k context.
        if model.is_full_attention_layer(layer):
            # Reads: K, V over kv_len tokens, grouped to n_kv heads.
            kv_bytes = dt * 2 * kv_len * kv_heads_rank * head_dim * sequences
            attn_flops = 2 * b * q_heads_rank * head_dim * kv_len * 2  # qk + sv
        else:
            # Read the recurrent state, update it, write it back: 2x state per
            # sequence. FLOPs are the state-sized matmuls, also context-free.
            state = model.linear_attn_state_elems / tp
            kv_bytes = dt * 2 * state * sequences
            attn_flops = 2 * b * state * 2  # state-vector product + state update
        g.nodes.append(
            PredictedNode(
                "attn_score_value",
                layer,
                roofline(
                    "attn_score_value", attn_flops, kv_bytes, hw, dtype=model.compute_dtype
                ),
            )
        )

        # Output projection
        local_h = h / tp
        flops = 2 * b * local_h * h
        bytes_moved = dt * (b * local_h + b * h) + wb * local_h * h
        g.nodes.append(
            PredictedNode(
                "attn_out_proj",
                layer,
                roofline("attn_out_proj", flops, bytes_moved, hw, dtype=model.compute_dtype),
            )
        )

        # MLP gate+up / down. On an MoE model these two ops carry the expert
        # GEMMs, so their flops/bytes come from the mixture model instead of a
        # single dense FFN (see _ffn_terms).
        gate_up_flops, gate_up_bytes, down_flops, down_bytes = _ffn_terms(
            model, b, moe_layer=model.is_moe_layer(layer), tp=tp
        )
        g.nodes.append(
            PredictedNode(
                "mlp_gate_up",
                layer,
                roofline(
                    "mlp_gate_up",
                    gate_up_flops,
                    gate_up_bytes,
                    hw,
                    dtype=model.compute_dtype,
                ),
            )
        )
        g.nodes.append(
            PredictedNode(
                "mlp_down",
                layer,
                roofline(
                    "mlp_down", down_flops, down_bytes, hw, dtype=model.compute_dtype
                ),
            )
        )

    # Final vocab projection
        if tp > 1:
            link = replace(hw, peak_mem_bw_bytes_per_s=hw.interconnect_bw_bytes_per_s)
            collective_bytes = 4.0 * (tp - 1) / tp * b * h * dt
            g.nodes.append(
                PredictedNode(
                    "tp_all_reduce",
                    layer,
                    roofline(
                        "tp_all_reduce",
                        0.0,
                        collective_bytes,
                        link,
                        dtype=model.compute_dtype,
                        estimated=True,
                    ),
                )
            )

    local_vocab = model.vocab / tp
    flops = 2 * b * h * local_vocab
    bytes_moved = dt * (b * h + b * local_vocab) + wb * h * local_vocab
    g.nodes.append(
        PredictedNode(
            "lm_head",
            None,
            roofline("lm_head", flops, bytes_moved, hw, dtype=model.compute_dtype),
        )
    )

    return g
