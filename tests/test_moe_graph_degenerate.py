"""Degenerate inputs must fail loudly or stay sane — never collapse quietly.

The graph branches on layer kind, compression level, sharding degree and a dozen
config fields. Every branch is an opportunity to produce a *plausible* total with
a whole mechanism costed at zero, which is worse than a crash: a crash gets
fixed, a cheap confident number gets believed and then billed against.

Three real collapses this file pins, all found by sweeping rather than reasoning:

* ``tp > n_heads`` floor-divided heads to zero and priced the entire attention
  path at 0 FLOPs;
* a config without ``sliding_window`` made every uncompressed layer read
  ``min(kv_len, 0)`` tokens, so attention was free;
* ``qk_rope_head_dim > head_dim`` drove the KV-entry byte split negative.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from gitm.planner.moe_graph import (
    effective_kv_tokens,
    index_candidates,
    kv_bytes_per_token,
    kv_entry_bytes,
    model_weight_bytes,
    predict_moe_graph,
    spec_from_hf_config,
)
from gitm.planner.roofline import BatchConfig, HardwareSpec, ShardingConfig
from tests.test_moe_graph import V4_BASE_CONFIG


@pytest.fixture
def spec():
    return spec_from_hf_config(V4_BASE_CONFIG)


@pytest.fixture
def hw():
    return HardwareSpec()


# ── refused rather than mispriced ───────────────────────────────────────────


@pytest.mark.parametrize("tp", [7, 128, 96])
def test_sharding_the_model_cannot_take_is_refused(spec, hw, tp):
    """Head counts floor-divide, so an indivisible TP silently zeroes attention."""
    with pytest.raises(ValueError, match="does not divide"):
        predict_moe_graph(spec, hw, BatchConfig(batch=8), ShardingConfig(tp=tp))


def test_rope_slice_larger_than_the_head_is_refused(spec, hw):
    """It would make ``kv_entry_bytes`` negative on the FP8 portion."""
    bad = replace(spec, qk_rope_head_dim=spec.head_dim + 1)
    with pytest.raises(ValueError, match="exceeds head_dim"):
        predict_moe_graph(bad, hw, BatchConfig(batch=8))


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32, 64])
def test_every_divisor_of_the_head_count_is_accepted(spec, hw, tp):
    g = predict_moe_graph(spec, hw, BatchConfig(batch=8), ShardingConfig(tp=tp))
    attn = sum(n.prediction.flops for n in g.nodes if n.op == "attn_score_value")
    assert attn > 0


# ── uncompressed without a window is global, not free ───────────────────────


def test_no_sliding_window_means_global_not_zero(spec, hw):
    """``min(kv_len, 0)`` is the trap. An uncompressed layer with no window
    attends to everything; it does not attend to nothing."""
    no_win = replace(spec, sliding_window=0)
    assert effective_kv_tokens(no_win, 0, 4096) == 4096

    g = predict_moe_graph(no_win, hw, BatchConfig(batch=8, kv_cache_len=4096))
    attn = sum(n.prediction.flops for n in g.nodes if n.op == "attn_score_value")
    assert attn > 0


def test_a_config_stripped_to_nothing_is_refused_instead_of_defaulted(hw):
    """A missing live config cannot inherit a plausible sparse architecture."""
    with pytest.raises(ValueError, match="hidden_size"):
        spec_from_hf_config({})


# ── degenerate but legal shapes stay finite and non-negative ────────────────


def _all_finite(g) -> bool:
    return all(
        math.isfinite(n.prediction.t_pred_s)
        and n.prediction.t_pred_s >= 0
        and n.prediction.bytes >= 0
        and n.prediction.flops >= 0
        for n in g.nodes
    )


@pytest.mark.parametrize(
    "label,mutation",
    [
        ("no compression at all", {"compress_ratios": ()}),
        ("one compression level", {"compress_ratios": tuple([4] * 43)}),
        ("all sliding-window", {"compress_ratios": tuple([0] * 43)}),
        ("mHC disabled", {"hc_width": 1}),
        ("every layer hash-routed", {"num_hash_layers": 99}),
        ("no top-k selection", {"index_topk": 0}),
        ("single layer", {"n_layers": 1}),
        ("no experts", {"n_routed_experts": 0, "num_experts_per_tok": 0}),
        ("no shared expert", {"n_shared_experts": 0}),
        ("no MTP head", {"num_nextn_predict_layers": 0}),
        ("no o-grouping", {"o_groups": 1}),
    ],
)
def test_degenerate_shapes_stay_finite(spec, hw, label, mutation):
    g = predict_moe_graph(replace(spec, **mutation), hw, BatchConfig(batch=8, kv_cache_len=4096))
    assert _all_finite(g), label
    assert g.total_pred_s > 0, label


@pytest.mark.parametrize("batch,ctx", [(1, 0), (1, 1), (1, 1 << 24), (4096, 1)])
def test_extreme_batch_and_context_stay_finite(spec, hw, batch, ctx):
    g = predict_moe_graph(spec, hw, BatchConfig(batch=batch, kv_cache_len=ctx))
    assert _all_finite(g)
    assert math.isfinite(g.total_pred_s)


def test_zero_batch_is_refused_instead_of_becoming_a_zero_work_graph(spec, hw):
    with pytest.raises(ValueError, match="batch must be a positive integer"):
        BatchConfig(batch=0)


def test_expert_parallelism_beyond_the_expert_count_stays_finite(spec, hw):
    g = predict_moe_graph(
        spec, hw, BatchConfig(batch=8, kv_cache_len=4096), ShardingConfig(tp=8, ep=512)
    )
    assert _all_finite(g)


# ── the scalar helpers ──────────────────────────────────────────────────────


def test_kv_helpers_are_non_negative_on_every_layer_kind(spec):
    assert kv_entry_bytes(spec) > 0
    assert kv_bytes_per_token(spec) > 0
    assert model_weight_bytes(spec) > 0
    for layer in range(spec.n_layers):
        assert effective_kv_tokens(spec, layer, 65536) > 0
        assert index_candidates(spec, layer, 65536) >= 0


def test_zero_context_reads_nothing(spec):
    """The one case where zero is the right answer."""
    assert effective_kv_tokens(spec, 0, 0) == 0
    assert index_candidates(spec, 2, 0) == 0


def test_layers_past_the_ratio_list_reuse_the_last_entry(spec):
    """The MTP head sits at ``layer == n_layers`` and must classify as something."""
    assert spec.attention_kind(spec.n_layers) in {"swa", "csa", "hca"}
    assert effective_kv_tokens(spec, spec.n_layers, 4096) > 0
