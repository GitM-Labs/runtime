"""Result semantics are explicit for every existing catalogue entry."""
import json
from pathlib import Path

import pytest

from gitm.planner.registry import main

MODELS = sorted(path.stem for path in (Path(__file__).resolve().parents[1]/'gitm/planner/models').glob('*.yaml'))


@pytest.mark.parametrize('model', MODELS)
def test_catalogue_json_has_conditional_semantics(model, capsys):
    rc = main([model, '--gpu', 'H200', '--batch', '1', '--kv-len', '128', '--json'])
    out = capsys.readouterr().out
    if rc == 2 and 'checkpoint evidence only' in out:
        pytest.skip(f'{model}: checkpoint evidence only; planner integration pending')
    assert rc == 0, out
    result = json.loads(out)
    semantics = result['prediction_semantics']
    assert semantics['kind'] == 'serial_resource_proxy'
    assert semantics['aggregation'] == 'sum_of_node_t_pred_s'
    assert semantics['latency_estimate_s'] is None
    assert semantics['whole_step_bound_status'] == 'not_validated'
    assert len(semantics['assumptions']) == 3
    assert result['total_pred_s'] == sum(node['t_pred_s'] for node in result['nodes'])
