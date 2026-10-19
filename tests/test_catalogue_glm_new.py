"""Catalogue entries for GLM-5, GLM-5.1, GLM-5.3 and their FP8 siblings.

Every published number below is read from the checkpoint's
``model.safetensors.index.json`` (``total_size``) or its tensor-to-shard map;
the YAML comments carry the reasoning. Tolerances are tight (0.1%) because the
bf16/fp8 shape arithmetic on this architecture lands within ~0.02% — a dtype or
schedule regression moves the total by far more than that.
"""
import pytest

from gitm.planner.glm_graph import model_weight_bytes
from gitm.planner.model_catalogue import available, load_entry, load_spec, predict
from gitm.planner.roofline import BatchConfig

#: model.safetensors.index.json total_size, per repo.
PUBLISHED = {
    "glm-5": 1_507_728_316_928,  # zai-org/GLM-5, 282 shards
    "glm-5-fp8": 756_162_687_872,  # zai-org/GLM-5-FP8, 142 shards
    "glm-5.1": 1_507_728_316_928,  # zai-org/GLM-5.1 — index byte-identical to GLM-5
    "glm-5.1-fp8": 756_162_687_872,  # zai-org/GLM-5.1-FP8 — identical to GLM-5-FP8
    "glm-5.3": 755_617_140_416,  # zai-org/GLM-5.3, 141 shards — identical map to GLM-5.2-FP8
}

#: Decoder layers carrying ``*.indexer.*`` tensors in the weight map (the MTP
#: layer 78 carries them on every checkpoint and is checked separately).
GLM5_INDEXER_LAYERS = list(range(78))
GLM52_INDEXER_LAYERS = [0, 1, 2] + list(range(6, 78, 4))

NO_INDEXSHARE = ("glm-5", "glm-5-fp8", "glm-5.1", "glm-5.1-fp8")


@pytest.mark.parametrize("entry", sorted(PUBLISHED))
def test_entry_loads_and_predicts(entry):
    assert entry in available()
    spec = load_spec(entry)
    assert hash(spec)
    assert spec.n_layers == 78 and spec.hidden == 6144 and spec.vocab == 154880
    assert spec.n_sparse_mlp_layers == 75  # layers 0-2 dense, per the weight map
    assert (spec.n_routed_experts, spec.num_experts_per_tok) == (256, 8)
    g, family = predict(entry, batch=BatchConfig(batch=32, kv_cache_len=8192))
    assert family == "glm_moe_dsa"
    assert g.total_pred_s > 0


@pytest.mark.parametrize("entry", sorted(PUBLISHED))
def test_weight_bytes_match_published_checkpoint(entry):
    """All five land within 0.03%; pinned at 0.1%."""
    ratio = model_weight_bytes(load_spec(entry)) / PUBLISHED[entry]
    assert ratio == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("entry", NO_INDEXSHARE)
def test_glm5_indexer_on_every_layer(entry):
    """No index_topk_freq / indexer_types in config.json, and the weight map has
    indexer tensors on all 78 decoder layers: no IndexShare."""
    spec = load_spec(entry)
    full = [i for i in range(spec.n_layers) if spec.is_full_indexer(i)]
    assert full == GLM5_INDEXER_LAYERS
    g, _ = predict(entry, batch=BatchConfig(batch=32, kv_cache_len=8192))
    ops = [n.op for n in g.nodes]
    assert ops.count("attn_index_proj") == 78
    assert ops.count("attn_index_score") == 78


def test_glm53_indexer_schedule_matches_weight_map():
    spec = load_spec("glm-5.3")
    full = [i for i in range(spec.n_layers) if spec.is_full_indexer(i)]
    assert full == GLM52_INDEXER_LAYERS
    assert spec.n_full_indexer_layers == 21
    g, _ = predict("glm-5.3", batch=BatchConfig(batch=32, kv_cache_len=8192))
    assert [n.op for n in g.nodes].count("attn_index_proj") == 21


def test_glm5_indexshare_gap_is_exactly_57_indexers():
    """GLM-5 minus GLM-5.2 published bf16 totals is 57 indexers to the byte —
    the checkpoint-size evidence for the schedule difference."""
    one_indexer = (2048 * 4096 + 6144 * 128 + 6144 * 32 + 2 * 128) * 2  # bf16
    assert PUBLISHED["glm-5"] - 1_506_659_919_872 == 57 * one_indexer


def test_fp8_indexer_projections_are_fp8_by_byte_difference():
    """GLM-5-FP8 minus GLM-5.2-FP8 is 57 indexers with wq_b/wk at 1 B/elem plus
    their F32 block scales, weights_proj/k_norm at 2 B. The entries agree."""
    fp8_indexer = (
        (2048 * 4096 + 6144 * 128) * 1  # wq_b, wk: F8_E4M3
        + (6144 * 32 + 2 * 128) * 2  # weights_proj, k_norm: BF16
        + (32 * 16 + 1 * 48) * 4  # weight_scale_inv: F32
    )
    assert PUBLISHED["glm-5-fp8"] - PUBLISHED["glm-5.3"] == 57 * fp8_indexer
    for entry in ("glm-5-fp8", "glm-5.3"):
        spec = load_spec(entry)
        assert spec.dtype_for("attn_index_proj", spec.weight_dtype) == "fp8"
        assert spec.stored_dtype_for("attn_index_weights_proj", "fp8") == "bf16"


def test_glm5_mtp_carries_its_own_indexer():
    """No index_share_for_mtp_iteration key in GLM-5's config and layer 78 ships
    an indexer, so each draft stage runs one; GLM-5.3 shares, as its config says."""
    assert load_spec("glm-5").index_share_for_mtp_iteration is False
    assert load_spec("glm-5.3").index_share_for_mtp_iteration is True
    b = BatchConfig(batch=8, kv_cache_len=8192, speculative_tokens=3)
    g5, _ = predict("glm-5", batch=b)
    g53, _ = predict("glm-5.3", batch=b)
    assert [n.op for n in g5.nodes].count("attn_index_proj") == 78 + 3
    assert [n.op for n in g53.nodes].count("attn_index_proj") == 21


def test_extends_relationships_override_nothing_architectural():
    """GLM-5.1 = GLM-5 shapes (identical index file); GLM-5.3 = GLM-5.2-FP8 shapes."""
    assert load_spec("glm-5.1") == load_spec("glm-5").__class__(
        **{**load_spec("glm-5").__dict__, "name": "zai-org/GLM-5.1"}
    )
    assert load_spec("glm-5.1-fp8") == load_spec("glm-5-fp8").__class__(
        **{**load_spec("glm-5-fp8").__dict__, "name": "zai-org/GLM-5.1-FP8"}
    )
    g53, g52 = load_spec("glm-5.3"), load_spec("glm-5.2-fp8")
    for f in ("indexer_types", "index_topk_freq", "mlp_layer_types", "first_k_dense_replace",
              "q_lora_rank", "kv_lora_rank", "weight_dtype", "expert_dtype", "kv_dtype"):
        assert getattr(g53, f) == getattr(g52, f), f
    assert load_entry("glm-5.3")["name"] == "zai-org/GLM-5.3"


def test_fp8_entries_fit_8xh200_and_bf16_do_not():
    per_gpu_h200 = 141e9
    for bf16, fp8 in (("glm-5", "glm-5-fp8"), ("glm-5.1", "glm-5.1-fp8")):
        assert model_weight_bytes(load_spec(bf16)) / per_gpu_h200 > 8
        assert model_weight_bytes(load_spec(fp8)) / per_gpu_h200 < 8
    assert model_weight_bytes(load_spec("glm-5.3")) / per_gpu_h200 < 8


def test_unmodelled_items_are_declared():
    assert any("index_skip_topk_offset" in u for u in load_entry("glm-5.3")["provenance"]["unmodelled"])
    for entry in ("glm-5", "glm-5.3"):
        assert any("rope_theta" in u for u in load_entry(entry)["provenance"]["unmodelled"])
