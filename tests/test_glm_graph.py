"""The GLM-5.2 (``glm_moe_dsa``) decode graph, pinned against the checkpoint.

Every assertion guards a term that separates GLM-5.2 from the DeepSeek-V4 sparse
family it was forked from, or a wiring seam a plausible-but-wrong graph would slip
through:

* IndexShare — only ``full`` layers emit indexer nodes; ``shared`` layers reuse
  the selection and carry no indexer weights,
* MLA KV traffic scales with the shared latent, never ``n_heads``,
* the dense prefix runs an FFN, not a mixture,
* precision is bf16 (no fp4 experts leaking in from the V4 defaults),
* the predicted footprint matches the published 1.507 TB checkpoint,
* ``detect_family`` routes ``glm_moe_dsa`` before the structural sparse-MoE test.
"""

from __future__ import annotations

import pytest

from gitm.planner.glm_graph import (
    GlmMoeDsaModelSpec,
    is_glm_moe_dsa_config,
    kv_entry_bytes,
    model_weight_bytes,
    predict_glm_graph,
    spec_from_hf_config,
)
from gitm.planner.model_catalogue import available, load_spec, predict
from gitm.planner.registry import detect_family
from gitm.planner.roofline import BatchConfig, ShardingConfig

# GLM-5.2's shape, trimmed to the keys the planner reads. Arrays are the real
# checkpoint's schedules (period-4 IndexShare past a 3-layer dense prefix).
GLM_CONFIG = {
    "model_type": "glm_moe_dsa",
    "architectures": ["GlmMoeDsaForCausalLM"],
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 192,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "indexer_types": (
        ["full", "full", "full"]
        + ["shared", "shared", "shared", "full"] * 18
        + ["shared", "shared", "shared"]
    ),
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "first_k_dense_replace": 3,
    "mlp_layer_types": ["dense", "dense", "dense"] + ["sparse"] * 75,
    "routed_scaling_factor": 2.5,
    "num_nextn_predict_layers": 1,
    "index_share_for_mtp_iteration": True,
    "dtype": "bfloat16",
    "vocab_size": 154880,
}


def _spec() -> GlmMoeDsaModelSpec:
    return spec_from_hf_config(GLM_CONFIG, name="GLM-5.2")


def _ops(g, op: str) -> list:
    return [n for n in g.nodes if n.op == op]


def test_config_reader_reads_schedules_verbatim():
    spec = _spec()
    assert spec.n_layers == 78
    assert spec.n_full_indexer_layers == 21  # 3 dense-prefix + 18 period-4
    assert spec.n_sparse_mlp_layers == 75
    # The dense prefix is dense; a mid-stack layer is sparse.
    assert not spec.is_sparse_mlp(0) and spec.is_sparse_mlp(40)
    # Read verbatim, not derived: layer 2 is 'full' though 2 % 4 != 0.
    assert spec.is_full_indexer(2)
    assert not spec.is_full_indexer(3)


def test_indexshare_shared_layers_emit_no_indexer():
    """The mechanism the fork exists to price: 57 of 78 layers skip the indexer."""
    spec = _spec()
    g = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    n_proj = len(_ops(g, "attn_index_proj"))
    n_score = len(_ops(g, "attn_index_score"))
    # One per full-indexer layer, and the MTP head shares (no indexer node).
    assert n_proj == n_score == spec.n_full_indexer_layers == 21
    # Every layer still runs the attention core over the selected positions.
    assert len(_ops(g, "attn_score_value")) == spec.n_layers + spec.num_nextn_predict_layers


def test_mla_kv_traffic_uses_shared_latent_not_heads():
    """The classic MLA error: charging KV as ``n_heads * head_dim``.

    The cache holds one latent per token, shared across all 64 query heads, so the
    per-entry byte count is ``kv_lora_rank + qk_rope`` — independent of ``n_heads``.
    """
    spec = _spec()
    entry = kv_entry_bytes(spec)
    assert entry == (512 + 64) * 2  # bf16 latent + bf16 rope key
    # A per-head (GQA-style) K+V reading would be tens of times larger.
    gqa_wrong = spec.n_heads * (spec.q_head_dim + spec.v_head_dim) * 2
    assert gqa_wrong > 25 * entry


def test_dense_prefix_runs_ffn_not_mixture():
    spec = _spec()
    g = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    # Exactly the 3 dense layers carry an mlp_gate_up/down; the router runs on the
    # 75 sparse layers plus the sparse MTP head.
    assert len(_ops(g, "mlp_gate_up")) == 3
    assert len(_ops(g, "moe_router")) == spec.n_sparse_mlp_layers + spec.num_nextn_predict_layers


def test_precision_is_bf16_no_fp4_leak():
    """Forked from a fp4-expert default; a leak would deflate the dominant term."""
    spec = _spec()
    assert spec.weight_dtype == "bf16"
    assert spec.expert_dtype == "bf16"
    assert spec.kv_dtype == "bf16"


def test_footprint_matches_published_checkpoint():
    """Predicted weight bytes within a few percent of the 1.507 TB on disk."""
    published = 1_506_659_919_872  # model.safetensors.index.json total_size
    wb = model_weight_bytes(_spec())
    assert abs(wb / published - 1.0) < 0.03


def test_attention_core_flat_indexer_scan_grows_with_context():
    """DSA: the core is bounded by top-k; only the indexer scan grows."""
    spec = _spec()
    short = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=4096))
    long = predict_glm_graph(spec, batch=BatchConfig(batch=1, kv_cache_len=131072))

    def total(g, op):
        return sum(n.prediction.t_pred_s for n in g.nodes if n.op == op)

    # Core read is capped at index_topk (2048) — unchanged from 4K to 128K.
    assert total(long, "attn_score_value") == pytest.approx(
        total(short, "attn_score_value"), rel=1e-9
    )
    # The scan scores the whole history, so it grows.
    assert total(long, "attn_index_score") > 10 * total(short, "attn_index_score")


def test_detect_family_routes_glm_before_sparse_moe():
    """Both families carry index_topk + n_routed_experts; model_type must win."""
    assert detect_family(GLM_CONFIG) == "glm_moe_dsa"
    assert is_glm_moe_dsa_config(GLM_CONFIG)


def test_catalogue_entry_loads_and_predicts():
    assert "glm-5.2" in available()
    spec = load_spec("glm-5.2")
    assert spec.n_layers == 78 and spec.n_full_indexer_layers == 21
    g, family = predict("glm-5.2", batch=BatchConfig(batch=1, kv_cache_len=4096))
    assert family == "glm_moe_dsa"
    assert g.total_pred_s > 0


def test_tp_must_divide_heads():
    spec = _spec()
    with pytest.raises(ValueError, match="does not divide"):
        predict_glm_graph(spec, sharding=ShardingConfig(tp=7))


def test_missing_required_field_refuses_default():
    broken = {k: v for k, v in GLM_CONFIG.items() if k != "kv_lora_rank"}
    with pytest.raises(ValueError, match="kv_lora_rank"):
        spec_from_hf_config(broken)
