"""The production dense-config parser must preserve the live architecture."""

from __future__ import annotations

from gitm.planner.graph import predict_graph
from gitm.scheduler.loop import _dense_spec_from_config

# opt-125m: 12 layers, hidden 768, 12 heads (MHA), intermediate 3072.
_OPT_125M = {
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 50272,
    "torch_dtype": "bf16",
}


def _spec(**overrides):
    spec, error = _dense_spec_from_config({**_OPT_125M, **overrides})
    assert error == ""
    assert spec is not None
    return spec


def test_reads_real_model_arch_from_config():
    spec = _spec()
    assert spec.n_layers == 12
    assert spec.hidden == 768
    assert spec.n_heads == 12
    assert spec.num_kv_heads == 12
    assert spec.head_dim == 64
    assert spec.intermediate == 3072
    assert spec.vocab == 50272


def test_predicted_graph_matches_real_model_not_default():
    assert len(predict_graph(model=_spec()).nodes) == 61
    assert len(predict_graph().nodes) == 161


def test_gqa_and_hybrid_attention_fields_survive_config_parsing():
    spec = _spec(num_key_value_heads=4, full_attention_interval=3)
    assert spec.num_kv_heads == 4
    assert spec.full_attn_layer_step == 3


def test_missing_answer_deciding_field_refuses():
    cfg = dict(_OPT_125M)
    del cfg["vocab_size"]
    spec, error = _dense_spec_from_config(cfg)
    assert spec is None
    assert "vocab_size" in error


def test_unknown_dense_dtype_refuses_instead_of_becoming_bf16():
    spec, error = _dense_spec_from_config({**_OPT_125M, "torch_dtype": "mystery4"})
    assert spec is None
    assert "not priceable" in error
