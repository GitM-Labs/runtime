"""The production dense-config parser must preserve the live architecture."""

from __future__ import annotations

import pytest

from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig, ShardingConfig
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


def test_missing_dense_dtype_refuses_instead_of_becoming_bf16():
    cfg = dict(_OPT_125M)
    del cfg["torch_dtype"]

    spec, error = _dense_spec_from_config(cfg)

    assert spec is None
    assert "torch_dtype" in error


def test_malformed_dense_quantization_config_refuses():
    spec, error = _dense_spec_from_config(
        {**_OPT_125M, "quantization_config": ["fp8"]}
    )

    assert spec is None
    assert "quantization_config" in error


def test_dense_parser_preserves_fp32_compute_dtype():
    spec, error = _dense_spec_from_config({**_OPT_125M, "torch_dtype": "float32"})

    assert error == ""
    assert spec is not None
    assert spec.compute_dtype == "fp32"
    assert spec.dtype_bytes == 4


def test_dense_graph_prices_tensor_parallel_work_per_rank():
    whole = predict_graph(model=_spec(num_key_value_heads=4))
    sharded = predict_graph(
        model=_spec(num_key_value_heads=4), sharding=ShardingConfig(tp=2)
    )

    whole_qkv = next(n.prediction for n in whole.nodes if n.op == "qkv_proj")
    rank_qkv = next(n.prediction for n in sharded.nodes if n.op == "qkv_proj")
    assert sharded.sharding.tp == 2
    assert rank_qkv.flops == whole_qkv.flops / 2
    assert any(n.op == "tp_all_reduce" for n in sharded.nodes)


def test_quantized_dense_width_applies_to_attention_and_lm_head_weights():
    bf16 = predict_graph(model=_spec())
    fp8 = predict_graph(
        model=_spec(quantization_config={"quant_method": "fp8"})
    )

    for op in ("qkv_proj", "attn_out_proj", "lm_head"):
        bf16_bytes = next(n.prediction.bytes for n in bf16.nodes if n.op == op)
        fp8_bytes = next(n.prediction.bytes for n in fp8.nodes if n.op == op)
        assert fp8_bytes < bf16_bytes


def test_dense_graph_honors_speculative_positions():
    plain = predict_graph(model=_spec(), batch=BatchConfig(batch=2, speculative_tokens=0))
    drafted = predict_graph(model=_spec(), batch=BatchConfig(batch=2, speculative_tokens=3))

    plain_qkv = next(n.prediction for n in plain.nodes if n.op == "qkv_proj")
    drafted_qkv = next(n.prediction for n in drafted.nodes if n.op == "qkv_proj")
    assert drafted_qkv.flops == plain_qkv.flops * 4


def test_dense_graph_refuses_incompatible_tensor_parallel_shape():
    with pytest.raises(ValueError, match="n_heads"):
        predict_graph(model=_spec(), sharding=ShardingConfig(tp=5))
