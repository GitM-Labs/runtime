"""Catalogue entries for GQA + MoE checkpoints fitted into the ``hybrid`` family.

MiniMax-M2.5, MiniMax-M2.7, MiniMax-M3 and gpt-oss-120b. None of them has a
linear-attention layer; the hybrid family is the only one with GQA attention,
so these exercise its degenerate corners — an all-``full_attention`` schedule,
zero shared experts, a dense prefix, and a checkpoint with two storage dtypes.

Expected values are transcribed from each checkpoint's config.json and its
safetensors shard headers at the revision pinned in the entry's ``checkpoint``
block. They are written out here rather than re-read so the test fails when the
YAML drifts, not when the network is down.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gitm.planner.hybrid_graph import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    HybridMoEModelSpec,
    kv_bytes_per_token,
    kv_fixed_bytes_per_sequence,
    model_weight_bytes,
    predict_hybrid_graph,
)
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.roofline import (
    BatchConfig,
    HardwareSpec,
    ShardingConfig,
    kv_elem_bytes,
    quant_format,
    weight_bytes,
)

IDS = ("minimax-m2.5", "minimax-m2.7", "minimax-m3", "gpt-oss-120b")

H200 = HardwareSpec(
    name="H200",
    peak_flops_bf16_per_s=989e12,
    peak_flops_fp16_per_s=989e12,
    peak_mem_bw_bytes_per_s=4.8e12,
    interconnect_bw_bytes_per_s=900e9,
)

#: (hf id, model_weight_bytes, summed shard-header payload, index total_size)
WEIGHTS = {
    "minimax-m2.5": ("MiniMaxAI/MiniMax-M2.5", 228_744_752_832, 230_121_630_720, 466_083_044_352),
    "minimax-m2.7": ("MiniMaxAI/MiniMax-M2.7", 228_744_752_832, 230_121_630_720, 480_836_588_544),
    "minimax-m3": ("MiniMaxAI/MiniMax-M3", 851_899_318_272, 854_172_958_720, 869_157_697_024),
    "gpt-oss-120b": ("openai/gpt-oss-120b", 62_044_070_400, 65_248_815_744, 65_248_815_744),
}


def _ops(g) -> set[str]:
    return {n.op for n in g.nodes}


# ── loading and identity ────────────────────────────────────────────────────


@pytest.mark.parametrize("model", IDS)
def test_entry_loads_as_hybrid_with_its_hf_name_and_pin(model):
    assert model in available()
    entry = load_entry(model)
    assert entry["family"] == "hybrid"
    assert entry["name"] == WEIGHTS[model][0]
    pin = entry["checkpoint"]
    assert pin["repository"] == WEIGHTS[model][0]
    assert len(pin["revision"]) == 40
    assert pin["payload_bytes"] == WEIGHTS[model][2]
    assert pin["index_total_size"] == WEIGHTS[model][3]
    spec = load_spec(model)
    assert isinstance(spec, HybridMoEModelSpec)
    hash(spec)  # every collection field landed hashable


@pytest.mark.parametrize("model", IDS)
def test_entry_carries_its_own_provenance(model):
    prov = load_entry(model)["provenance"]
    assert prov["verified"] and prov["open"] and prov["unmodelled"]


@pytest.mark.parametrize("model", IDS)
def test_every_dtype_string_resolves(model):
    """An unknown dtype silently prices at 2 B. Every label must be a real format."""
    spec = load_spec(model)
    for dt in (spec.weight_dtype, spec.act_dtype, *(d for _, d in spec.op_dtype_overrides)):
        assert quant_format(dt) is not None, dt
    assert quant_format(spec.kv_dtype) is not None


# ── spec values against config.json / headers ──────────────────────────────


def test_minimax_m25_spec_matches_config():
    s = load_spec("minimax-m2.5")
    assert (s.hidden, s.n_layers, s.vocab) == (3072, 62, 200064)
    assert (s.n_heads, s.num_kv_heads, s.head_dim) == (48, 8, 128)
    assert s.q_dim == 6144                       # wider than hidden
    assert s.rope_dim == 64
    assert (s.num_experts, s.num_experts_per_tok, s.moe_intermediate_size) == (256, 8, 1536)
    assert s.shared_expert_intermediate_size == 0
    assert s.dense_layers == frozenset()
    assert s.mtp_num_hidden_layers == 0          # declared 3, none shipped
    assert s.weight_dtype == "fp8" and weight_bytes("fp8") == pytest.approx(1 + 4 / 16384)
    assert s.dtype_for("moe_router", "x") == "fp32"
    assert s.dtype_for("lm_head", "x") == "bf16"


def test_minimax_m27_extends_m25_with_an_identical_spec():
    a, b = load_spec("minimax-m2.5"), load_spec("minimax-m2.7")
    assert replace(a, name=b.name) == b
    entry = load_entry("minimax-m2.7")
    # Own pin and own provenance — neither is inherited.
    assert entry["checkpoint"]["revision"] != load_entry("minimax-m2.5")["checkpoint"]["revision"]
    assert entry["provenance"] != load_entry("minimax-m2.5")["provenance"]


def test_minimax_m3_spec_matches_text_config():
    s = load_spec("minimax-m3")
    assert (s.hidden, s.n_layers, s.vocab) == (6144, 60, 200064)
    assert (s.n_heads, s.num_kv_heads, s.head_dim) == (64, 4, 128)
    assert s.rope_dim == 64
    assert (s.num_experts, s.num_experts_per_tok, s.moe_intermediate_size) == (128, 4, 3072)
    assert s.shared_expert_intermediate_size == 3072
    assert s.dense_layers == frozenset({0, 1, 2})
    assert s.dense_ffn_width == 12288
    assert s.weight_dtype == "bf16"
    assert s.mtp_num_hidden_layers == 0


def test_gpt_oss_spec_matches_config():
    s = load_spec("gpt-oss-120b")
    assert (s.hidden, s.n_layers, s.vocab) == (2880, 36, 201088)
    assert (s.n_heads, s.num_kv_heads, s.head_dim) == (64, 8, 64)
    assert s.q_dim == 4096
    assert s.rope_dim == 64                      # full-width rotary
    assert (s.num_experts, s.num_experts_per_tok, s.moe_intermediate_size) == (128, 4, 2880)
    assert s.shared_expert_intermediate_size == 0
    assert s.sliding_window == 128
    assert s.weight_dtype == "mxfp4" and weight_bytes("mxfp4") == 0.53125
    for op in ("qkv_proj", "attn_out_proj", "moe_router", "lm_head"):
        assert s.dtype_for(op, s.weight_dtype) == "bf16"


# ── schedules ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model,n", [("minimax-m2.5", 62), ("minimax-m2.7", 62), ("minimax-m3", 60)])
def test_minimax_schedules_are_all_full_attention(model, n):
    s = load_spec(model)
    assert s.layer_types == (FULL_ATTENTION,) * n
    assert s.n_full_attention_layers == n
    assert s.n_linear_attention_layers == 0 and s.n_sliding_attention_layers == 0


def test_gpt_oss_schedule_alternates_starting_sliding():
    s = load_spec("gpt-oss-120b")
    assert [i for i in range(36) if s.is_sliding_attention(i)] == list(range(0, 36, 2))
    assert [i for i in range(36) if s.is_full_attention(i)] == list(range(1, 36, 2))
    assert s.layer_types[0] == SLIDING_ATTENTION


# ── weight bytes ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", IDS)
def test_weight_bytes_are_the_figure_the_entry_reports(model):
    _, predicted, payload, _ = WEIGHTS[model]
    got = model_weight_bytes(load_spec(model))
    assert got == pytest.approx(predicted, abs=1)
    # Within 5% of what is actually on disk. gpt-oss is the loosest (-4.91%)
    # because its bf16 non-expert weights are priced at mxfp4.
    assert abs(got / payload - 1) < 0.05


def test_minimax_m25_residual_closes_on_the_dtype_ledger():
    """Pricing embed/lm_head at bf16 and the router at fp32 leaves only norms."""
    s = load_spec("minimax-m2.5")
    w = weight_bytes("fp8")
    corrected = (
        model_weight_bytes(s)
        - 2 * s.vocab * s.hidden * w + 2 * s.vocab * s.hidden * 2
        - s.n_layers * s.hidden * s.num_experts * w + s.n_layers * s.hidden * s.num_experts * 4
    )
    assert WEIGHTS["minimax-m2.5"][2] - corrected == pytest.approx(1_720_320, abs=1)


def test_gpt_oss_expert_term_is_exact_at_mxfp4():
    s = load_spec("gpt-oss-120b")
    experts = s.n_layers * s.num_experts * 3 * s.hidden * s.moe_intermediate_size * 0.53125
    # gate_up blocks+scales + down blocks+scales, summed from the 15 headers.
    assert experts == 19_110_297_600 + 1_194_393_600 + 38_220_595_200 + 2_388_787_200


# ── prediction: the family's degenerate corners ────────────────────────────


@pytest.mark.parametrize("model", IDS)
@pytest.mark.parametrize("tp", [1, 4, 8])
def test_predicts_a_finite_graph_with_no_linear_attention(model, tp):
    g, family = predict(model, H200, BatchConfig(batch=8, kv_cache_len=4096), ShardingConfig(tp=tp))
    assert family == "hybrid"
    ops = _ops(g)
    assert not any(op.startswith("linattn") for op in ops)
    assert {"qkv_proj", "attn_score_value", "attn_out_proj", "moe_router", "moe_routed", "lm_head"} <= ops
    total = sum(n.prediction.t_pred_s for n in g.nodes)
    assert 0 < total < float("inf")


@pytest.mark.parametrize("model,shared", [
    ("minimax-m2.5", False), ("minimax-m2.7", False), ("minimax-m3", True), ("gpt-oss-120b", False),
])
def test_shared_expert_node_only_where_the_checkpoint_has_one(model, shared):
    g, _ = predict(model, H200, BatchConfig(batch=4, kv_cache_len=512))
    assert ("moe_shared" in _ops(g)) is shared


def test_minimax_m3_dense_prefix_emits_mlp_not_moe():
    g, _ = predict("minimax-m3", H200, BatchConfig(batch=4, kv_cache_len=512))
    mlp_layers = {n.layer for n in g.nodes if n.op == "mlp_gate_up"}
    moe_layers = {n.layer for n in g.nodes if n.op == "moe_routed"}
    assert mlp_layers == {0, 1, 2}
    assert moe_layers == set(range(3, 60))


def test_with_mtp_adds_nothing_when_no_mtp_weights_ship():
    for model in IDS:
        spec = load_spec(model)
        b = BatchConfig(batch=2, kv_cache_len=256)
        assert len(predict_hybrid_graph(spec, H200, b, with_mtp=True).nodes) == len(
            predict_hybrid_graph(spec, H200, b).nodes
        )


def test_gpt_oss_kv_splits_into_growing_and_windowed_parts():
    s = load_spec("gpt-oss-120b")
    kw = kv_elem_bytes(s.kv_dtype)
    assert kv_bytes_per_token(s) == 18 * 8 * (64 + 64) * kw
    assert kv_fixed_bytes_per_sequence(s) == 18 * 128 * 8 * (64 + 64) * kw


def test_gpt_oss_windowed_layers_read_at_most_the_window():
    s = load_spec("gpt-oss-120b")
    g = predict_hybrid_graph(s, H200, BatchConfig(batch=1, kv_cache_len=8192))
    sv = {n.layer: n.prediction.bytes for n in g.nodes if n.op == "attn_score_value"}
    assert sv[1] / sv[0] == pytest.approx(8192 / 128)


@pytest.mark.parametrize("model", ["minimax-m2.5", "minimax-m3"])
def test_minimax_kv_rate_is_every_layer(model):
    s = load_spec(model)
    assert kv_bytes_per_token(s) == s.n_layers * s.num_kv_heads * 2 * s.head_dim * kv_elem_bytes(s.kv_dtype)
    assert kv_fixed_bytes_per_sequence(s) == 0


def test_minimax_m3_block_sparse_attention_is_priced_dense():
    """Documented over-pricing: attention bytes grow linearly in kv_len on the
    sparse layers, where the real kernel reads ~2,176 tokens. If the family
    learns block-sparse attention this must change, along with the entry."""
    s = load_spec("minimax-m3")
    a = predict_hybrid_graph(s, H200, BatchConfig(batch=1, kv_cache_len=4096))
    b = predict_hybrid_graph(s, H200, BatchConfig(batch=1, kv_cache_len=32768))

    def sv(g):
        return next(
            n.prediction.bytes for n in g.nodes if n.op == "attn_score_value" and n.layer == 30
        )

    assert sv(b) / sv(a) == pytest.approx(8.0)


# ── the limitation the entries document ────────────────────────────────────


def test_op_dtype_overrides_set_compute_dtype_but_not_bytes():
    """Why gpt-oss-120b needs a forced weight_dtype choice.

    Pins current behaviour, which the MiniMax and gpt-oss entries document as
    unmodelled: overriding an op's dtype changes the roofline ceiling it is
    priced against but not the bytes it moves. The code change that would
    make this test fail (deliberately) is a per-op *stored* dtype, or an
    ``expert_dtype``, read by ``model_weight_bytes``, ``_emit_moe``,
    ``_emit_full_attention`` and ``emit_lm_head``. When that lands, update
    this test and the four entries together.
    """
    s = load_spec("gpt-oss-120b")
    b = BatchConfig(batch=1, kv_cache_len=128)
    as_is = predict_hybrid_graph(s, H200, b)
    no_override = predict_hybrid_graph(replace(s, op_dtype_overrides=()), H200, b)

    def lm(g):
        return next(n for n in g.nodes if n.op == "lm_head")

    assert lm(as_is).prediction.bytes == lm(no_override).prediction.bytes
    # lm_head moves vocab x hidden at mxfp4 width, not the checkpoint's bf16.
    w = lm(as_is).prediction.bytes
    # Plus one row of bf16 activation in and out: 2 x (hidden + vocab).
    assert w == pytest.approx(s.vocab * s.hidden * 0.53125 + 2 * (s.hidden + s.vocab))
    assert w < s.vocab * s.hidden * 2 / 3.5
