"""Recheck supplied K3 scalar flags against the actual hybrid schema.

Run from the repository root. This checks field coverage and loader rejection;
it does not establish numerical correctness or make this submission loadable.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path

import yaml

from gitm.planner.hybrid_graph import HybridMoEModelSpec
from gitm.planner.model_catalogue import load_spec

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/cross-architecture/sources/k3/kimi-k3.yaml'
REGISTER = ROOT / 'docs/cross-architecture/schema-gap-register.md'
ROWS = {
    'K3-01': ('q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim',
              'qk_rope_head_dim', 'mla_use_nope', 'mla_cache_elems_per_token'),
    'K3-02': ('linear_conv_channels', 'linear_conv_state_dtype'),
    'K3-03': ('routed_expert_hidden_size',),
    'K3-04': ('latent_moe_use_norm',),
    'K3-05': ('moe_router_activation_func', 'topk_method'),
    'K3-06': ('expert_dtype', 'expert_group_size', 'expert_scale_dtype'),
}


def reconcile() -> dict:
    text = SOURCE.read_text()
    spec = yaml.safe_load(text)['spec']
    flagged = set(re.findall(r'^  (\w+):[^\n]*#.*schema gap', text, re.MULTILINE))
    mapped = {key for keys in ROWS.values() for key in keys}
    if flagged != mapped:
        raise AssertionError(f'flag/register mapping drift: {flagged ^ mapped}')
    known = {field.name for field in fields(HybridMoEModelSpec)}
    register = REGISTER.read_text()
    records = []
    for row, keys in ROWS.items():
        for key in keys:
            if row not in register or f'`{key}=' not in register:
                raise AssertionError(f'missing register evidence: {row}/{key}')
            records.append({'register_id': row, 'field': key, 'value': spec[key],
                            'present_in_hybrid_schema': key in known})
    unknown = sorted(set(spec) - known)
    try:
        load_spec(SOURCE)
    except ValueError as error:
        rejection = str(error).replace(str(ROOT) + '/', '')
        if not all(key in rejection for key in unknown):
            raise AssertionError('loader rejection does not identify all unknown fields') from error
    else:
        raise AssertionError('submission unexpectedly loaded; reconcile its consumers before closure')
    return {
        'scope': 'field coverage and real loader rejection; semantic closure still requires consumer review',
        'submission_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'flagged_scalar_count': len(flagged), 'reconciled_scalars': records,
        'unknown_scalar_spec_keys': unknown,
        'unflagged_unknown_scalar_keys': sorted(set(unknown) - flagged),
        'precision_overrides': spec['op_dtype_overrides'],
        'precision_override_review': 'K3-07a and K3-07b; not inferred from field presence',
        'loader_rejection': rejection,
        'model_loadable': False,
    }


if __name__ == '__main__':
    print(json.dumps(reconcile(), indent=2))
