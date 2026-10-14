"""Narrow storage-expression fixes; these fixtures do not claim K3/RDU support."""
from dataclasses import replace

import pytest
import yaml

from gitm.planner.glm_graph import (
    GlmMoeDsaModelSpec,
    kv_bytes_per_token,
    kv_entry_bytes,
    memory_fit,
    predict_glm_graph,
)
from gitm.planner.glm_graph import spec_from_hf_config as glm_reader
from gitm.planner.hybrid_graph import HybridMoEModelSpec, predict_hybrid_graph
from gitm.planner.hybrid_graph import spec_from_hf_config as hybrid_reader
from gitm.planner.model_catalogue import load_spec
from gitm.planner.roofline import BatchConfig, HardwareSpec, ShardingConfig


def node(graph, op):
    return next(n.prediction for n in graph.nodes if n.op == op)


def hybrid():
    return HybridMoEModelSpec(n_layers=1, layer_types=('linear_attention',),
                             conv_dim=128, linear_conv_channels=384,
                             linear_conv_state_dtype='bf16', ssm_state_dtype='fp32')


def test_independent_convolution_storage_and_channels():
    spec = hybrid()
    assert spec.conv_state_elems == 384 * 3
    assert spec.linear_state_bytes_per_layer == 4 * 32 * 32 * 4 + 384 * 3 * 2
    assert spec.linear_state_bytes_per_sequence == spec.linear_state_bytes_per_layer
    g = predict_hybrid_graph(spec, batch=BatchConfig(batch=2), sharding=ShardingConfig(tp=2))
    assert node(g, 'linattn_conv').bytes == 2 * 2 * (384 // 2) * 3 * 2 + 2 * 2 * (384 // 2) * 2
    legacy_dtype = replace(spec, linear_conv_state_dtype=None)
    old = predict_hybrid_graph(legacy_dtype, batch=BatchConfig(batch=2), sharding=ShardingConfig(tp=2))
    assert node(old, 'linattn_conv').bytes - node(g, 'linattn_conv').bytes == 2 * 2 * 192 * 3 * 2
    assert node(old, 'linattn_recurrent') == node(g, 'linattn_recurrent')
    # Explicit geometry does not turn estimated physical lowering into a fact.
    assert node(g, 'linattn_conv').estimated


@pytest.mark.parametrize('dtype', ['bf16', 'bfloat16', 'torch.bfloat16'])
def test_convolution_dtype_aliases(dtype):
    assert replace(hybrid(), linear_conv_state_dtype=dtype).conv_state_element_bytes == 2


@pytest.mark.parametrize('value', [True, 0, -1, 2.5, '384'])
def test_invalid_explicit_channels(value):
    with pytest.raises(ValueError):
        replace(hybrid(), linear_conv_channels=value)


@pytest.mark.parametrize('value', ['mxfp4', 'typo', 2, True])
def test_invalid_explicit_state_dtype(value):
    with pytest.raises(ValueError):
        replace(hybrid(), linear_conv_state_dtype=value)


def test_explicit_convolution_channels_cannot_truncate_under_tp():
    with pytest.raises(ValueError, match='convolution channels'):
        predict_hybrid_graph(replace(hybrid(), linear_conv_channels=383), sharding=ShardingConfig(tp=2))


def test_hybrid_reader_keeps_explicit_channels_and_state_dtype():
    cfg = dict(hidden_size=512, num_hidden_layers=1, vocab_size=1024,
               num_attention_heads=4, num_key_value_heads=2,
               linear_num_value_heads=4, linear_value_head_dim=32,
               num_experts=8, num_experts_per_tok=2, moe_intermediate_size=256,
               layer_types=['linear_attention'], linear_conv_channels=384,
               linear_conv_state_dtype='bf16')
    spec = hybrid_reader({'text_config': cfg})
    assert spec.conv_channels == 384 and spec.conv_dim == 128
    assert spec.conv_state_element_bytes == 2
    cfg.pop('linear_conv_channels')
    cfg.pop('linear_conv_state_dtype')
    legacy = hybrid_reader({'text_config': cfg})
    assert legacy.conv_channels == 128 and legacy.conv_state_element_bytes == 4


def test_catalogue_loader_accepts_new_storage_fields(tmp_path):
    path = tmp_path / 'explicit.yaml'
    path.write_text(yaml.safe_dump({'name': 'synthetic', 'family': 'hybrid', 'spec': {
        'n_layers': 1, 'layer_types': ['linear_attention'],
        'linear_conv_channels': 384, 'linear_conv_state_dtype': 'bf16'}}))
    spec = load_spec(path)
    assert spec.conv_channels == 384 and spec.conv_state_element_bytes == 2
    path.write_text(yaml.safe_dump({'name': 'synthetic', 'family': 'glm_moe_dsa', 'spec': {
        'index_kv_dtype': 'fp8', 'index_scale_bytes_per_token': 4}}))
    assert load_spec(path).index_entry_bytes == 64 + 4


def glm():
    return GlmMoeDsaModelSpec(n_layers=2, indexer_types=('full', 'shared'),
                             index_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                             index_kv_dtype='fp8', index_scale_bytes_per_token=4)


@pytest.mark.parametrize('tp', [1, 2])
def test_index_key_storage_reaches_fit_read_and_write(tp):
    spec = glm()
    batch = BatchConfig(batch=2, kv_cache_len=128)
    sh = ShardingConfig(tp=tp)
    assert spec.index_entry_bytes == 132  # 128 FP8 keys plus one FP32 scale
    assert kv_entry_bytes(spec) == 1152  # BF16 latent and RoPE stay unchanged
    assert kv_bytes_per_token(spec) == 2 * 1152 + 132
    fit = memory_fit(spec, HardwareSpec(memory_bytes=1e12), batch, sh)
    assert fit.kv_needed == 2 * 128 * (2 * 1152 + 132)
    g = predict_glm_graph(spec, batch=batch, sharding=sh)
    legacy = predict_glm_graph(replace(spec, index_kv_dtype=None, index_scale_bytes_per_token=0),
                               batch=batch, sharding=sh)
    assert node(g, 'attn_index_score').bytes == 2 * 128 * 132
    assert node(legacy, 'attn_index_score').bytes - node(g, 'attn_index_score').bytes == 2 * 128 * (256 - 132)
    assert node(legacy, 'attn_index_proj').bytes - node(g, 'attn_index_proj').bytes == 2 * (256 - 132)
    assert node(g, 'attn_index_score').flops == node(legacy, 'attn_index_score').flops
    assert node(g, 'attn_score_value') == node(legacy, 'attn_score_value')


def test_index_prefill_read_and_write_include_scales():
    batch = BatchConfig(batch=0, prefill_tokens=32, prefill_context=128)
    spec = glm()
    g = predict_glm_graph(spec, batch=batch)
    assert node(g, 'attn_index_score').bytes == (128 + 32) * 132
    legacy = predict_glm_graph(replace(spec, index_kv_dtype=None, index_scale_bytes_per_token=0), batch=batch)
    assert node(legacy, 'attn_index_proj').bytes - node(g, 'attn_index_proj').bytes == 32 * (256 - 132)


@pytest.mark.parametrize('value', [True, -1, 4.5, '4', None])
def test_invalid_index_scale_bytes(value):
    with pytest.raises(ValueError):
        replace(glm(), index_scale_bytes_per_token=value)


@pytest.mark.parametrize('value', ['mxfp4', 'typo', True, []])
def test_invalid_index_dtype(value):
    with pytest.raises(ValueError):
        replace(glm(), index_kv_dtype=value)


def test_glm_reader_requires_explicit_cache_layout():
    cfg = dict(hidden_size=512, num_hidden_layers=2, vocab_size=1024,
               num_attention_heads=4, q_lora_rank=128, kv_lora_rank=512,
               n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=256,
               indexer_types=['full', 'shared'], dtype='bfloat16',
               index_kv_dtype='fp8', index_scale_bytes_per_token=4)
    spec = glm_reader(cfg)
    assert spec.index_kv_dtype == 'fp8' and spec.index_scale_bytes_per_token == 4
    cfg.pop('index_kv_dtype')
    cfg.pop('index_scale_bytes_per_token')
    legacy = glm_reader(cfg)
    assert legacy.index_kv_dtype is None and legacy.index_scale_bytes_per_token == 0
    assert legacy.kv_dtype == 'bf16'
