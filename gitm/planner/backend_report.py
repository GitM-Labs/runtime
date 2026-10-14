"""Optional, source-derived lowering report for one pinned engine boundary.

This report supplements a mathematical graph. It never reprices that graph or
turns logical payload into measured HBM traffic or an empirical latency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gitm.planner.graph import Graph


@dataclass(frozen=True)
class QwenStaticCacheBackend:
    checkpoint_revision: str = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    engine: str = 'transformers'
    engine_version: str = '4.46.3'
    torch_version: str = '2.4.1'
    attention: str = 'sdpa'
    cache: str = 'StaticCache'
    precision: str = 'bf16'
    execution_mode: str = 'eager'
    # The graph contract is prefix length; the backend includes the next token.
    context_contract: str = 'prefix_excluding_current_token'


def qwen_static_cache_report(graph: Graph, backend: QwenStaticCacheBackend) -> dict[str, Any]:
    """Describe the pinned Qwen2.5-3B decode path; reject unsupported variants.

    Exact model/engine scope is intentional: a different backend can consume
    grouped KV directly and need no expanded buffers. Unknown costs stay null.
    """
    expected = QwenStaticCacheBackend(execution_mode=backend.execution_mode)
    if backend != expected or backend.execution_mode not in {'eager', 'cuda_graph'}:
        raise ValueError('unsupported engine, version, cache, precision or execution contract')
    m, b = graph.model, graph.batch.batch
    if graph.batch.prefill_tokens != 0 or graph.batch.speculative_tokens != 0:
        raise ValueError('report requires a single-token decode without prefill or speculation')
    geometry = (m.hidden, m.n_layers, m.n_heads, m.num_kv_heads,
                m.head_dim, m.intermediate, m.vocab, m.dtype_bytes, m.w_bytes)
    if m.name != 'Qwen/Qwen2.5-3B-Instruct' or geometry != (2048, 36, 16, 2, 128, 11008, 151936, 2, 2):
        raise ValueError('report requires the pinned Qwen2.5-3B BF16 model geometry')
    if m.is_moe or any(not m.is_full_attention_layer(i) for i in range(m.n_layers)):
        raise ValueError('report requires dense full attention')
    if graph.sharding.tp != 1 or graph.sharding.ep != 1 or graph.sharding.dp != 1:
        raise ValueError('report has no evidence for sharded execution')
    if graph.hw.name != 'A100-SXM4-80GB':
        raise ValueError('report has no recorded platform evidence for this device')
    prefix = graph.batch.kv_cache_len
    if type(b) is not int or b <= 0 or type(prefix) is not int or not 1 <= prefix < 32768:
        raise ValueError('positive integer batch and a prefix within the pinned context limit are required')
    expected_ops = {'qkv_proj', 'attn_score_value', 'attn_out_proj', 'mlp_gate_up', 'mlp_down'}
    by_layer: dict[int, dict[str, int]] = {}
    head_nodes = []
    for index, node in enumerate(graph.nodes):
        if node.layer is None:
            if node.op != 'lm_head':
                raise ValueError('unsupported global node')
            head_nodes.append(index)
            continue
        if node.layer not in range(m.n_layers) or node.op not in expected_ops:
            raise ValueError('unsupported graph node')
        layer = by_layer.setdefault(node.layer, {})
        if node.op in layer:
            raise ValueError('duplicate canonical node')
        layer[node.op] = index
    if len(head_nodes) != 1 or len(by_layer) != m.n_layers or any(set(v) != expected_ops for v in by_layer.values()):
        raise ValueError('incomplete canonical dense graph')

    length = prefix + 1
    unique = 2 * b * m.num_kv_heads * length * m.head_dim * m.dtype_bytes
    expanded = 2 * b * m.n_heads * length * m.head_dim * m.dtype_bytes
    source = 'https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py'
    components = []
    for layer, indices in sorted(by_layer.items()):
        components.append({
            'component_id': f'layer.{layer}.expanded_kv',
            'canonical_node_indices': [indices['attn_score_value']],
            'producer': 'repeat_kv.expand_then_reshape',
            'consumer': 'scaled_dot_product_attention',
            'unique_input_payload_bytes': unique,
            'materialized_intermediate_payload_bytes': expanded,
            'logical_write_payload_bytes': expanded,
            'logical_consumer_input_payload_bytes': expanded,
            'physical_hbm_bytes': None,
            'kernel_launch_count': None,
            'latency_estimate_s': None,
            'evidence': {'kind': 'source-derived', 'artifact': source,
                         'symbols': ['repeat_kv', 'Qwen2SdpaAttention.forward'],
                         'qualification': 'buffer payload, not physical bus traffic'},
        })
    return {
        'schema_version': 1,
        'kind': 'backend_lowering_report',
        'coverage': 'partial',
        'backend': asdict(backend),
        'execution_region': {'kind': 'cached_single_token_forward',
                             'includes': ['decoder', 'lm_head'],
                             'excludes': ['tokenization', 'prefill', 'sampling', 'network']},
        'shape': {'batch': b, 'prefix_context': prefix, 'attention_cache_slots': length},
        'canonical_graph': {'node_count': len(graph.nodes), 'total_pred_s': graph.total_pred_s,
                            'numeric_predictions_modified': False},
        'components': components,
        'total_logical_expanded_writes_bytes': expanded * m.n_layers,
        'expanded_kv_payload_per_layer_bytes': expanded,
        'physical_hbm_bytes': None,
        'latency_estimate_s': None,
        'unpriced_components': ['projection splitting and bias', 'rotary position operations',
                               'cache updates', 'norms and residuals', 'MLP pointwise operations',
                               'embedding and final norm', 'physical attention access pattern',
                               'kernel utilization', 'launch and boundary scheduling'],
        'warnings': ['Logical payload cannot be added directly to legacy node bytes.',
                     'Per-layer payload is not total live memory or an allocator peak.',
                     'The saved trace supports a materialization inference but lacks input shapes and Python stacks.',
                     'No point-latency or complete physical-traffic claim is available.'],
    }
