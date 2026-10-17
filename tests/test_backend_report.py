"""Backend scope, source-derived payloads and preserved mathematical graph."""
from dataclasses import asdict, replace

import pytest

from gitm.planner.backend_report import QwenStaticCacheBackend, qwen_static_cache_report
from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec, ShardingConfig


def qwen_graph(batch=1, context=2048):
    spec = ModelSpec(name='Qwen/Qwen2.5-3B-Instruct', hidden=2048, n_layers=36,
                     n_heads=16, num_kv_heads=2, head_dim=128, intermediate=11008,
                     vocab=151936, dtype_bytes=2, weight_dtype_bytes=2)
    return predict_graph(spec, HardwareSpec(), BatchConfig(batch=batch, kv_cache_len=context))


@pytest.mark.parametrize('batch', [1, 8, 16])
@pytest.mark.parametrize('context', [128, 2048])
@pytest.mark.parametrize('mode', ['eager', 'cuda_graph'])
def test_report_maps_source_payload_without_repricing(batch, context, mode):
    graph = qwen_graph(batch, context)
    before = asdict(graph)
    result = qwen_static_cache_report(graph, QwenStaticCacheBackend(execution_mode=mode))
    payload = 2 * batch * 16 * (context+1) * 128 * 2
    assert result['expanded_kv_payload_per_layer_bytes'] == payload
    assert result['total_logical_expanded_writes_bytes'] == 36*payload
    assert result['shape']['attention_cache_slots'] == context+1
    assert len(result['components']) == 36
    for component in result['components']:
        index, = component['canonical_node_indices']
        assert graph.nodes[index].op == 'attn_score_value'
        assert component['physical_hbm_bytes'] is None
        assert component['kernel_launch_count'] is None
        assert component['latency_estimate_s'] is None
    assert result['coverage'] == 'partial'
    assert result['unpriced_components']
    assert result['physical_hbm_bytes'] is result['latency_estimate_s'] is None
    assert asdict(graph) == before


@pytest.mark.parametrize('field,value', [
    ('checkpoint_revision', 'other'), ('engine', 'vllm'), ('engine_version', '4.47.0'),
    ('torch_version', '2.5.0'), ('attention', 'flash_attention_2'), ('cache', 'DynamicCache'),
    ('precision', 'fp32'), ('execution_mode', 'compiled'), ('context_contract', 'includes_current_token'),
])
def test_unknown_backend_rejected(field, value):
    with pytest.raises(ValueError):
        qwen_static_cache_report(qwen_graph(), replace(QwenStaticCacheBackend(), **{field: value}))


@pytest.mark.parametrize('mutation', [
    lambda g: setattr(g, 'model', replace(g.model, name='other')),
    lambda g: setattr(g, 'model', replace(g.model, num_kv_heads=1)),
    lambda g: setattr(g, 'hw', replace(g.hw, name='H200')),
    lambda g: setattr(g, 'sharding', ShardingConfig(tp=2)),
    lambda g: setattr(g, 'batch', replace(g.batch, batch=0)),
    lambda g: setattr(g, 'batch', replace(g.batch, kv_cache_len=32768)),
    lambda g: setattr(g, 'batch', replace(g.batch, prefill_tokens=128)),
    lambda g: setattr(g, 'batch', replace(g.batch, speculative_tokens=3)),
    lambda g: g.nodes.pop(0),
    lambda g: g.nodes.append(g.nodes[0]),
])
def test_unsupported_graph_rejected(mutation):
    graph = qwen_graph()
    mutation(graph)
    with pytest.raises(ValueError):
        qwen_static_cache_report(graph, QwenStaticCacheBackend())
