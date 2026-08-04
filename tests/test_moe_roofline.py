"""MoE roofline: batch-dependent expert fetch, and dense back-compat.

The property under test is the one that makes a mixture different from a dense
FFN: *compute* scales with the experts each token activates (linear in batch)
while *weight traffic* scales with the distinct experts the batch touches, which
saturates at ``num_experts``. Getting that curve right is what makes an MoE
residual mean anything — a dense ceiling is wrong by ~28x at batch 1, and a
flat "active params" ceiling is wrong by the same factor at large batch.
"""

from __future__ import annotations

import pytest

from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, distinct_experts

# Qwen3.6-35B-A3B-FP8 shaped: narrow hidden, many narrow experts, one shared
# expert, fp8 weights with bf16 activations.
MOE = ModelSpec(
    name="moe-test", hidden=2048, n_layers=40, n_heads=32, num_kv_heads=4,
    head_dim=128, intermediate=0, vocab=151936, num_experts=256,
    experts_per_token=8, moe_intermediate=768, shared_experts=1,
    dtype_bytes=2, weight_dtype_bytes=1,
)
H100 = HardwareSpec(
    name="H100", peak_flops_fp16_per_s=1979e12, peak_mem_bw_bytes_per_s=3.35e12
)


def _node(model, batch, op, hw=H100):
    g = predict_graph(model, hw, BatchConfig(batch=batch))
    return next(n for n in g.nodes if n.op == op).prediction


# --- distinct_experts: the term itself ---------------------------------------


def test_batch_one_touches_exactly_top_k():
    assert distinct_experts(1, 256, 8) == pytest.approx(8.0)


def test_saturates_at_num_experts():
    # Past the knee the union stops growing — the step has fetched every expert,
    # i.e. the whole model, so MoE's bandwidth edge over a dense model is gone.
    assert distinct_experts(10**6, 256, 8) == pytest.approx(256.0)
    assert distinct_experts(128, 256, 8) > 250  # ~98% by batch 128


def test_sublinear_between_the_limits():
    """Strictly below the naive b*k line once collisions start, and monotone."""
    prev = 0.0
    for b in (1, 2, 4, 8, 16, 32, 64, 128):
        d = distinct_experts(b, 256, 8)
        assert d > prev, "distinct count must be monotone in batch"
        assert d <= min(b * 8, 256) + 1e-9, "cannot exceed b*k, nor the expert count"
        if b >= 8:
            assert d < b * 8, f"collisions must make it sublinear by b={b}"
        prev = d


def test_top_k_equal_to_num_experts_touches_all():
    # Every token already routes everywhere; one token suffices.
    assert distinct_experts(1, 8, 8) == pytest.approx(8.0)
    # And top_k is clamped rather than allowed to exceed the expert count.
    assert distinct_experts(4, 8, 99) == pytest.approx(8.0)


def test_degenerate_inputs_are_zero_not_errors():
    assert distinct_experts(0, 256, 8) == 0.0
    assert distinct_experts(4, 0, 8) == 0.0
    assert distinct_experts(4, 256, 0) == 0.0
    assert distinct_experts(-1, 256, 8) == 0.0


# --- ModelSpec surface --------------------------------------------------------


def test_dense_spec_is_not_moe_and_falls_back():
    m = ModelSpec()
    assert not m.is_moe
    assert m.top_k == 0
    assert m.w_bytes == m.dtype_bytes  # no separate weight width configured
    assert m.expert_intermediate == m.intermediate


def test_moe_spec_properties():
    assert MOE.is_moe
    assert MOE.top_k == 8
    assert MOE.w_bytes == 1 and MOE.dtype_bytes == 2  # fp8 weights, bf16 acts
    assert MOE.expert_intermediate == 768
    assert MOE.shared_intermediate == 768  # falls back to the routed width


def test_num_experts_without_top_k_stays_dense():
    """Half-configured is dense, not a mixture — no silent guessing."""
    assert not ModelSpec(num_experts=256).is_moe
    assert not ModelSpec(experts_per_token=8).is_moe


# --- graph: dense back-compat -------------------------------------------------


@pytest.mark.parametrize("b", [1, 8, 64])
def test_dense_ffn_arithmetic_is_unchanged(b):
    """The dense path must reproduce the pre-MoE formulas byte for byte."""
    m = ModelSpec()
    dt, h, ff = m.dtype_bytes, m.hidden, m.intermediate
    gu = _node(m, b, "mlp_gate_up")
    dn = _node(m, b, "mlp_down")
    assert gu.flops == 2 * 2 * b * h * ff
    assert gu.bytes == dt * (b * h + 2 * h * ff + 2 * b * ff)
    assert dn.flops == 2 * b * ff * h
    assert dn.bytes == dt * (b * ff + ff * h + b * h)


def test_moe_does_not_disturb_non_ffn_ops():
    """Only the FFN ops change; attention/lm_head must match the dense model."""
    shape = dict(hidden=2048, n_layers=2, n_heads=32, num_kv_heads=4,
                 head_dim=128, vocab=151936, intermediate=768)
    dense = ModelSpec(**shape)
    moe = ModelSpec(**shape, num_experts=256, experts_per_token=8, moe_intermediate=768)
    for op in ("qkv_proj", "attn_score_value", "attn_out_proj", "lm_head"):
        d, m = _node(dense, 8, op), _node(moe, 8, op)
        assert (d.flops, d.bytes) == (m.flops, m.bytes), f"{op} should be untouched"


def test_graph_node_vocabulary_is_unchanged_for_moe():
    """No new op names — library.yaml/classify_op key off this vocabulary."""
    dense_ops = {n.op for n in predict_graph(ModelSpec(n_layers=2)).nodes}
    moe_ops = {n.op for n in predict_graph(
        ModelSpec(n_layers=2, num_experts=64, experts_per_token=4, moe_intermediate=512)
    ).nodes}
    assert moe_ops == dense_ops


# --- graph: the MoE property that matters -------------------------------------


def test_compute_grows_linearly_while_weight_traffic_saturates():
    """The core asymmetry. Doubling batch past the knee roughly doubles flops but
    barely moves bytes, because the expert union is already nearly complete."""
    big, bigger = _node(MOE, 512, "mlp_gate_up"), _node(MOE, 1024, "mlp_gate_up")
    assert bigger.flops == pytest.approx(2 * big.flops, rel=0.01)
    assert bigger.bytes < 1.30 * big.bytes, "weight traffic must have flattened"


def test_arithmetic_intensity_rises_with_batch():
    """Direct consequence: FLOP/byte climbs, so the op walks toward compute-bound."""
    ai = [
        _node(MOE, b, "mlp_gate_up").flops / _node(MOE, b, "mlp_gate_up").bytes
        for b in (1, 8, 32, 128, 1024)
    ]
    assert ai == sorted(ai), f"AI must be monotone in batch, got {ai}"
    assert ai[0] < 5, "batch-1 decode is a GEMV: a couple of FLOPs per byte"
    assert ai[-1] > 10 * ai[0]


def test_batch_one_decode_is_memory_bound():
    assert _node(MOE, 1, "mlp_gate_up").bound == "memory"


def test_moe_moves_far_less_than_a_dense_model_of_the_same_total_size():
    """At batch 1 only top-k experts are read — the reason MoE is efficient."""
    dense_equiv = ModelSpec(
        hidden=2048, n_layers=40, intermediate=256 * 768, dtype_bytes=2, weight_dtype_bytes=1
    )
    d = _node(dense_equiv, 1, "mlp_gate_up").bytes
    m = _node(MOE, 1, "mlp_gate_up").bytes
    assert d / m > 20, f"expected a large gap at batch 1, got {d / m:.1f}x"


def test_large_batch_fetches_the_entire_expert_set():
    """Past saturation the step reads *every* expert — the whole model's weights —
    so MoE's bandwidth advantage over a dense model of the same total size is gone.

    Asserted as an exact decomposition, which pins the formula rather than a
    ratio. Note total bytes still differ from a same-width dense FFN: that model
    also pushes every token through a 256x wider intermediate, so its
    *activation* traffic is far larger. Only the weight term converges.
    """
    b = 4096
    node = _node(MOE, b, "mlp_gate_up")
    assert distinct_experts(b, MOE.num_experts, MOE.top_k) == pytest.approx(
        MOE.num_experts
    ), "precondition: the expert union must be saturated at this batch"

    weights = MOE.w_bytes * (
        MOE.num_experts * 2 * MOE.hidden * MOE.expert_intermediate  # every routed expert
        + MOE.shared_experts * 2 * MOE.hidden * MOE.shared_intermediate
        + MOE.hidden * MOE.num_experts  # router
    )
    acts = MOE.dtype_bytes * (
        b * MOE.hidden
        + 2 * b * (MOE.top_k * MOE.expert_intermediate
                   + MOE.shared_experts * MOE.shared_intermediate)
    )
    assert node.bytes == pytest.approx(weights + acts, rel=1e-9)

    # And the weight half matches what a dense model of the same total size reads.
    dense_equiv = ModelSpec(
        hidden=2048, n_layers=40, intermediate=256 * 768, dtype_bytes=2, weight_dtype_bytes=1
    )
    dense_weights = dense_equiv.w_bytes * 2 * dense_equiv.hidden * dense_equiv.intermediate
    assert weights == pytest.approx(dense_weights, rel=0.01)


def test_shared_expert_adds_flops_and_bytes():
    base = dict(hidden=2048, n_layers=2, intermediate=768, num_experts=64,
                experts_per_token=4, moe_intermediate=768)
    without = _node(ModelSpec(**base), 8, "mlp_gate_up")
    with_shared = _node(ModelSpec(**base, shared_experts=1), 8, "mlp_gate_up")
    assert with_shared.flops > without.flops
    assert with_shared.bytes > without.bytes


def test_quantized_weights_cut_the_dominant_term():
    """fp8 weights roughly halve batch-1 traffic vs bf16 — it is nearly all weights."""
    base = dict(hidden=2048, n_layers=2, num_experts=256, experts_per_token=8,
                moe_intermediate=768, intermediate=768, dtype_bytes=2)
    bf16 = _node(ModelSpec(**base), 1, "mlp_gate_up").bytes
    fp8 = _node(ModelSpec(**base, weight_dtype_bytes=1), 1, "mlp_gate_up").bytes
    assert 0.45 < fp8 / bf16 < 0.6, f"expected ~half, got {fp8 / bf16:.2f}"


# --- per-layer placement: MoE checkpoints are not uniformly sparse -----------


def test_all_layers_moe_by_default():
    m = ModelSpec(n_layers=4, num_experts=64, experts_per_token=4)
    assert [m.is_moe_layer(i) for i in range(4)] == [True] * 4
    assert m.n_moe_layers == 4


def test_leading_dense_layers_are_dense():
    """DeepSeek-style first_k_dense_replace."""
    m = ModelSpec(n_layers=6, num_experts=64, experts_per_token=4, first_dense_layers=2)
    assert [m.is_moe_layer(i) for i in range(6)] == [False, False, True, True, True, True]
    assert m.n_moe_layers == 4


def test_moe_layer_step_interleaves():
    """Qwen-style decoder_sparse_step: every Nth layer is MoE."""
    m = ModelSpec(n_layers=6, num_experts=64, experts_per_token=4, moe_layer_step=2)
    assert [m.is_moe_layer(i) for i in range(6)] == [True, False, True, False, True, False]
    assert m.n_moe_layers == 3


def test_dense_model_has_no_moe_layers():
    m = ModelSpec(n_layers=4)
    assert not any(m.is_moe_layer(i) for i in range(4))
    assert m.n_moe_layers == 0


def test_dense_layers_are_priced_as_dense_in_the_graph():
    """A dense block inside an MoE model must use dense FFN arithmetic — pricing
    it as MoE would inflate the predicted ceiling for that layer."""
    m = ModelSpec(hidden=2048, n_layers=4, intermediate=768, num_experts=256,
                  experts_per_token=8, moe_intermediate=768, first_dense_layers=2)
    g = predict_graph(m, H100, BatchConfig(batch=16))
    gate_ups = [n for n in g.nodes if n.op == "mlp_gate_up"]
    assert len(gate_ups) == 4
    dense_bytes = gate_ups[0].prediction.bytes   # layer 0 -> dense
    moe_bytes = gate_ups[2].prediction.bytes     # layer 2 -> MoE
    assert dense_bytes != moe_bytes
    # The dense layer must match the plain single-FFN formula exactly.
    dt, wb, h, ff = m.dtype_bytes, m.w_bytes, m.hidden, m.intermediate
    b = 16
    assert dense_bytes == dt * (b * h + 2 * b * ff) + wb * (2 * h * ff)
    # And the MoE layer reads many experts, so it moves strictly more.
    assert moe_bytes > dense_bytes


# --- active vs total params ---------------------------------------------------


def test_dense_model_active_equals_total():
    m = ModelSpec()
    assert m.active_params == m.total_params


def test_moe_active_is_a_small_fraction_of_total():
    """The '35B-A3B' property: only top_k of num_experts participate per token."""
    assert MOE.active_params < MOE.total_params
    assert MOE.active_params / MOE.total_params < 0.2


def test_expert_params_scale_with_k_over_e():
    """Isolate the expert term: doubling top_k roughly doubles the active expert
    params, while total is unchanged."""
    base = dict(hidden=2048, n_layers=8, intermediate=768, num_experts=64,
                moe_intermediate=768, vocab=1000)
    k4 = ModelSpec(**base, experts_per_token=4)
    k8 = ModelSpec(**base, experts_per_token=8)
    assert k4.total_params == k8.total_params
    # Difference is exactly 4 more experts' worth of FFN per MoE layer.
    delta = k8.active_params - k4.active_params
    expected = k4.n_moe_layers * 4 * 3 * 2048 * 768
    assert delta == expected


def test_dense_layers_shift_params_from_experts_to_dense_ffn():
    base = dict(hidden=2048, n_layers=8, intermediate=768, num_experts=64,
                experts_per_token=4, moe_intermediate=768, vocab=1000)
    all_moe = ModelSpec(**base)
    half = ModelSpec(**base, first_dense_layers=4)
    # Fewer MoE layers => far fewer total params (experts dominate the count).
    assert half.total_params < all_moe.total_params
    assert half.n_moe_layers == 4 and all_moe.n_moe_layers == 8


def test_param_accounting_is_internally_consistent():
    """Hand-derive both sides for a small shape so the formula is pinned, not
    just self-consistent."""
    m = ModelSpec(hidden=64, n_layers=2, n_heads=4, num_kv_heads=4, head_dim=16,
                  intermediate=128, vocab=100, num_experts=8, experts_per_token=2,
                  moe_intermediate=32, shared_experts=1)
    attn = 64 * (4 + 2 * 4) * 16 + 4 * 16 * 64          # qkv + out proj
    router = 64 * 8
    experts_total = 8 * 3 * 64 * 32
    experts_active = 2 * 3 * 64 * 32
    shared = 1 * 3 * 64 * 32
    embed = 2 * 100 * 64
    assert m.total_params == 2 * (attn + experts_total + shared + router) + embed
    assert m.active_params == 2 * (attn + experts_active + shared + router) + embed


def test_total_prediction_is_finite_and_positive_across_shapes():
    """Versatility guard: a spread of real MoE shapes all predict sanely."""
    shapes = [
        dict(num_experts=8, experts_per_token=2, moe_intermediate=14336),    # Mixtral-ish
        dict(num_experts=64, experts_per_token=6, moe_intermediate=1408),    # DeepSeek-ish
        dict(num_experts=128, experts_per_token=8, moe_intermediate=768),    # Qwen-ish
        dict(num_experts=256, experts_per_token=1, moe_intermediate=2048),   # extreme top-1
    ]
    for extra in shapes:
        m = ModelSpec(hidden=2048, n_layers=4, intermediate=768, **extra)
        for b in (1, 16, 256):
            g = predict_graph(m, H100, BatchConfig(batch=b))
            assert g.total_pred_s > 0
            assert all(n.prediction.t_pred_s >= 0 for n in g.nodes)
            assert all(n.prediction.bytes > 0 for n in g.nodes)
