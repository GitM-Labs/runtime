"""The loop predicts against the graph the model actually needs.

`gitm run` built the dense graph for every model. A DeepSeek-V4-class checkpoint
needs the sparse-MoE graph (mixed fp4/fp8, compressed/selected attention, MTP)
instead — but a Mixtral, which is also a mixture, does *not*, because its
attention is standard and the MoE graph's compressed-KV nodes would mis-price it.
These tests pin the dispatch on a fake duck-typed engine (no vLLM/GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.planner.moe_graph import is_sparse_moe_config
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, SparseMoEModelSpec
from gitm.scheduler.loop import _execution_graph, _hf_config_dict


def _engine(**hf_fields):
    hf = SimpleNamespace(**hf_fields)
    return SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf)))


_V4 = dict(
    model_type="deepseek_v4", num_hidden_layers=4, hidden_size=4096,
    n_routed_experts=256, num_experts_per_tok=6, moe_intermediate_size=2048,
    index_topk=512, compress_ratios=[0, 0, 4, 128],
    quantization_config={"quant_method": "fp8"}, expert_dtype="fp4", torch_dtype="bfloat16",
)
_MIXTRAL = dict(
    model_type="mixtral", num_hidden_layers=4, hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, intermediate_size=14336, vocab_size=32000,
    num_local_experts=8, num_experts_per_tok=2, sliding_window=4096,
)
_OPT_125M = dict(
    num_hidden_layers=12, hidden_size=768, num_attention_heads=12,
    intermediate_size=3072, vocab_size=50272,
)


def test_predicate_fires_only_for_dsa_moe():
    assert is_sparse_moe_config(_V4)
    assert not is_sparse_moe_config(_MIXTRAL)  # MoE, but standard attention
    assert not is_sparse_moe_config(_OPT_125M)  # dense


def test_v4_engine_routes_to_the_moe_graph():
    graph, is_moe = _execution_graph(_engine(**_V4), HardwareSpec(), BatchConfig(batch=8))
    assert is_moe
    assert isinstance(graph.model, SparseMoEModelSpec)
    assert graph.total_pred_s > 0


def test_mixtral_engine_routes_to_the_dense_graph():
    _, is_moe = _execution_graph(_engine(**_MIXTRAL), HardwareSpec(), BatchConfig())
    assert not is_moe


def test_dense_engine_routes_to_the_dense_graph():
    graph, is_moe = _execution_graph(_engine(**_OPT_125M), HardwareSpec(), BatchConfig())
    assert not is_moe
    assert isinstance(graph.model, ModelSpec)
    assert len(graph.nodes) == 12 * 5 + 1  # the real model, not the Llama default


def test_no_engine_falls_back_to_the_default_dense_graph():
    graph, is_moe = _execution_graph(None, HardwareSpec(), BatchConfig())
    assert not is_moe
    assert isinstance(graph.model, ModelSpec)


def test_object_quantization_config_is_flattened_to_a_dict():
    # A live HF config carries quantization_config as an object, not a dict;
    # spec_from_hf_config expects a dict, so the boundary must normalise it.
    hf = SimpleNamespace(**{**_V4, "quantization_config": SimpleNamespace(quant_method="fp8")})
    assert _hf_config_dict(hf)["quantization_config"] == {"quant_method": "fp8"}


# ── family dispatch shared with the registry ─────────────────────────────────
# The loop used to test ``is_sparse_moe_config`` alone. GLM-5.2 carries
# ``index_topk`` + ``n_routed_experts`` exactly as V4 does, so it matched and the
# live loop priced a GLM run with the V4 graph — the collision
# ``registry.detect_family`` orders its checks to avoid.


def test_glm_engine_routes_to_the_glm_graph_not_the_v4_one():
    from gitm.planner.glm_graph import GlmMoeDsaModelSpec
    from gitm.scheduler.loop import _execution_graph_family
    from tests.test_glm_graph import GLM_CONFIG

    assert is_sparse_moe_config(GLM_CONFIG)  # the structural test *would* claim it
    graph, family = _execution_graph_family(_engine(**GLM_CONFIG), HardwareSpec(),
                                            BatchConfig(batch=8))
    assert family == "glm_moe_dsa"
    assert isinstance(graph.model, GlmMoeDsaModelSpec)
    assert not isinstance(graph.model, SparseMoEModelSpec)
    assert graph.total_pred_s > 0


def test_glm_engine_with_no_scheduler_samples_still_predicts():
    """The loop passes ``batch=None`` when nothing was sampled (CPU box, dry run)."""
    from gitm.scheduler.loop import _execution_graph_family
    from tests.test_glm_graph import GLM_CONFIG

    graph, family = _execution_graph_family(_engine(**GLM_CONFIG), HardwareSpec(), None)
    assert family == "glm_moe_dsa" and graph.total_pred_s > 0


def test_hybrid_engine_routes_to_the_hybrid_graph():
    from gitm.scheduler.loop import _execution_graph_family
    from tests.test_moe_graph import QWEN36

    graph, family = _execution_graph_family(_engine(**QWEN36), HardwareSpec(),
                                            BatchConfig(batch=4))
    assert family == "hybrid"
    assert any(n.op == "linattn_recurrent" for n in graph.nodes)


def test_v4_still_reports_the_sparse_moe_family():
    from gitm.scheduler.loop import _execution_graph_family

    _, family = _execution_graph_family(_engine(**_V4), HardwareSpec(), BatchConfig(batch=8))
    assert family == "sparse_moe"
