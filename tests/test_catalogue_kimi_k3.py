"""Catalogue entry: kimi-k3 (moonshotai/Kimi-K3 @ f831ab66).

The checkpoint mixes 69 KDA layers with 24 gated MLA layers and runs its routed
experts in a 3584-wide latent space. No planner family carries either, so the
entry is ``glm_moe_dsa`` priced all-MLA with two stand-in MoE fields. These tests
pin three things: the tensor-byte ledger from the shard headers closes on the
published ``total_size``; the stand-ins are exact where the YAML says they are;
and the planner's gap against that ledger is the documented one, to the byte.
A change that moves any figure here should also update the YAML and
docs/kimi-k3/DESIGN-NOTE.md, not only these numbers.
"""

from __future__ import annotations

import pytest

from gitm.planner import glm_graph
from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.roofline import BatchConfig, ShardingConfig

TOTAL_SIZE = 1_560_860_324_864  # model.safetensors.index.json metadata.total_size

H = 7168
N_LAYERS = 93
BF16, F32 = 2, 4
MXFP4 = 0.5 + 1 / 32  # e2m1 packed two per byte + one e8m0 scale per 32

# 1-indexed, verbatim from config.json text_config.linear_attn_config.
CONFIG_FULL_ATTN = [4 * i for i in range(1, 24)] + [93]
CONFIG_KDA = [i for i in range(1, 92) if i % 4 != 0]

# 0-indexed, read from the weight map: layers carrying kv_a_proj_with_mqa (MLA)
# and A_log (KDA).
WEIGHT_MAP_MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
                  67, 71, 75, 79, 83, 87, 91, 92}
WEIGHT_MAP_KDA = set(range(N_LAYERS)) - WEIGHT_MAP_MLA


def _kda_layer_bytes() -> int:
    """One KDA layer's self_attn, from the shard-00001/00002 headers."""
    qkv = 96 * 128
    bf16 = (
        4 * qkv * H      # q_proj, k_proj, v_proj, g_proj (full-rank gate)
        + H * qkv        # o_proj
        + 96 * H         # b_proj
        + 128 * H        # f_a_proj
        + qkv * 128      # f_b_proj
    )
    f32 = 3 * qkv * 4 + 128 + qkv + 128  # q/k/v_conv1d, A_log, dt_bias, o_norm
    return bf16 * BF16 + f32 * F32


def _mla_layer_bytes(with_gate: bool = True) -> int:
    """One gated MLA layer's self_attn, from the shard-00004 header."""
    elems = (
        1536 * H                 # q_a_proj
        + 1536                   # q_a_layernorm
        + 96 * 192 * 1536        # q_b_proj [18432, 1536]
        + 576 * H                # kv_a_proj_with_mqa
        + 512                    # kv_a_layernorm
        + 96 * 256 * 512         # kv_b_proj [24576, 512]
        + H * 96 * 128           # o_proj
    )
    if with_gate:
        elems += 96 * 128 * H    # g_proj
    return elems * BF16


ROUTED_EXPERT_WEIGHTS = 3 * 3584 * 3072  # w1/w3 [3072, 3584], w2 [3584, 3072]

LEDGER = {
    "routed_experts": 92 * 896 * ROUTED_EXPERT_WEIGHTS * MXFP4,
    "kda_attention": 69 * _kda_layer_bytes(),
    "shared_experts": 92 * 3 * H * 6144 * BF16,
    "mla_attention": 24 * _mla_layer_bytes(),
    "latent_proj": 92 * (2 * 3584 * H + 3584) * BF16,
    "lm_head": 163840 * H * BF16,
    "embed_tokens": 163840 * H * BF16,
    "dense_ffn": 3 * H * 33792 * BF16,
    "router": 92 * (896 * H * BF16 + 896 * F32),
    "vision_tower": 802_428_928,   # sum of the shard-00096 header
    "mm_projector": (4096 * 4096 + 7168 * 4096 + 7168) * BF16,
    "attn_res": 93 * 2 * (H + H) * BF16,          # *_res_proj [1, H] + *_res_norm [H]
    "layer_norms": 93 * 2 * H * BF16,             # input / post_attention
    "final_norms": 2 * H * BF16 + H * BF16,       # norm, output_attn_res_{norm,proj}
}


@pytest.fixture(scope="module")
def spec():
    return load_spec("kimi-k3")


def test_entry_loads_with_full_provenance(spec):
    assert "kimi-k3" in available()
    entry = load_entry("moonshotai/Kimi-K3")
    assert entry["family"] == "glm_moe_dsa"
    assert entry.get("execution_status") != "static_only"
    prov = entry["provenance"]
    assert prov["verified"] and prov["estimated"] and prov["open"] and prov["unmodelled"]
    assert spec.n_layers == N_LAYERS and spec.hidden == H and spec.vocab == 163840
    assert spec.expert_dtype == "mxfp4"
    assert spec.n_full_indexer_layers == 0
    assert spec.index_topk == 1_048_576


def test_config_layer_lists_are_one_indexed():
    """configuration_kimi_k3.is_kda_layer tests (i + 1) in kda_layers; the
    weight map is the arbiter and agrees once both lists are shifted by one."""
    assert len(CONFIG_KDA) == 69 and len(CONFIG_FULL_ATTN) == 24
    assert {i - 1 for i in CONFIG_KDA} == WEIGHT_MAP_KDA
    assert {i - 1 for i in CONFIG_FULL_ATTN} == WEIGHT_MAP_MLA
    assert 0 in WEIGHT_MAP_KDA and {91, 92} <= WEIGHT_MAP_MLA


def test_tensor_ledger_closes_on_total_size():
    assert _kda_layer_bytes() == 887_800_832
    assert _mla_layer_bytes() == 464_392_192
    assert LEDGER["routed_experts"] == 1_446_456_066_048
    assert sum(LEDGER.values()) == TOTAL_SIZE


def test_parameter_counts_reproduce_the_readme():
    experts = 92 * 896 * ROUTED_EXPERT_WEIGHTS
    non_expert_text = (
        69 * 443_727_872 + 69 * (128 + 12288 + 128)  # KDA incl. convs; A_log, dt_bias, o_norm
        + 24 * _mla_layer_bytes() // BF16
        + 92 * 3 * H * 6144 + 92 * (2 * 3584 * H + 3584)
        + 2 * 163840 * H + 3 * H * 33792 + 92 * (896 * H + 896)
        + 93 * 4 * H + 93 * 2 * H + 3 * H
    )
    vision = (LEDGER["vision_tower"] + LEDGER["mm_projector"]) // BF16  # all BF16
    total = experts + non_expert_text + vision
    assert total / 1e9 == pytest.approx(2779.93, abs=0.01)  # README "2.8T"
    active_excl_embed = experts * 16 / 896 + non_expert_text - 163840 * H
    assert active_excl_embed == pytest.approx(104.19e9, rel=1e-3)  # README "104B"


def test_latent_moe_stand_ins_are_exact_for_weights_and_flops(spec):
    """moe_intermediate_size 1536 at hidden 7168 is the same weight and MAC
    count as the real 3072 at the 3584 latent width; 4 x 1536 shared = 6144."""
    assert 3 * spec.hidden * spec.moe_intermediate_size == ROUTED_EXPERT_WEIGHTS
    assert spec.n_shared_experts * spec.moe_intermediate_size == 2 * 3072
    n_sparse = spec.n_layers - spec.first_k_dense_replace
    routed = n_sparse * spec.n_routed_experts * 3 * spec.hidden * spec.moe_intermediate_size
    assert routed * MXFP4 == LEDGER["routed_experts"]


def test_weight_bytes_gap_is_the_documented_one(spec):
    predicted = glm_graph.model_weight_bytes(spec)
    assert predicted == 1_504_904_085_504
    assert predicted / TOTAL_SIZE - 1 == pytest.approx(-0.03585, abs=5e-5)
    mla_ungated = _mla_layer_bytes(with_gate=False) - (1536 + 512) * BF16
    gap = (
        -69 * (_kda_layer_bytes() - mla_ungated)                       # KDA as MLA
        - 24 * (_mla_layer_bytes() - mla_ungated)                       # g_proj + norms
        - LEDGER["latent_proj"]
        - LEDGER["vision_tower"] - LEDGER["mm_projector"]
        - LEDGER["attn_res"] - LEDGER["layer_norms"] - LEDGER["final_norms"]
        - 92 * 896 * F32                                                # router bias
    )
    assert gap == -55_956_239_360
    assert predicted - TOTAL_SIZE == gap


def test_kv_rate_is_the_documented_all_mla_overcount(spec):
    """93 layers x 576 x bf16 where the model caches 24 — 3.875x. Pinned so the
    KDA layer kind landing shows up here as an intended change."""
    assert glm_graph.kv_bytes_per_token(spec) == 93 * 576 * BF16 == 107_136
    assert glm_graph.kv_bytes_per_token(spec) / (24 * 576 * BF16) == pytest.approx(3.875)


def test_entry_predicts_a_decode_graph_on_mi355x():
    hw = hardware_spec_for(peak_for_sku("MI355X"))
    g, family = predict("kimi-k3", hw, BatchConfig(batch=32, kv_cache_len=8192),
                        ShardingConfig(tp=8))
    assert family == "glm_moe_dsa"
    assert g.nodes and all(n.prediction.t_pred_s >= 0 for n in g.nodes)
    ops = [n.op for n in g.nodes]
    assert ops.count("attn_score_value") == 93      # all-MLA, documented
    assert ops.count("moe_routed") == 92
    assert ops.count("mlp_gate_up") == 1
    assert not any(op.startswith("attn_index") for op in ops)


def test_whole_model_decode_bytes_match_the_design_note(spec):
    """Design note §3.2: planner node bytes at TP1, B=1 and B=32, S=8k.

    B=32 includes AITER's stage-2 activation quant on the routed experts, 92
    nodes, 0.18 GB (aiter_mxfp4_routed_quants: K3's shape has no tuned fused
    row, so both quants run). B=1 moves 0.006 GB, inside the tolerance."""
    hw = hardware_spec_for(peak_for_sku("MI355X"))
    for batch, expected_gb in ((1, 82.91), (32, 721.62)):
        g = glm_graph.predict_glm_graph(
            spec, hw, BatchConfig(batch=batch, kv_cache_len=8192), ShardingConfig(tp=1)
        )
        total = sum(n.prediction.bytes for n in g.nodes)
        assert total / 1e9 == pytest.approx(expected_gb, abs=0.01)
