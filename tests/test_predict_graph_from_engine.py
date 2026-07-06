"""The predicted graph must match the model that actually ran.

`predict_graph()` defaults to Llama-2-7B (32 layers). When the loop has a live
engine, `_model_spec_from_engine` reads the real architecture off its HF config
so residuals/deviation score against the right model. These tests use a fake
duck-typed engine (no vLLM/GPU) to pin the config-reading + fallback behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.planner.graph import predict_graph
from gitm.scheduler.loop import _model_spec_from_engine


def _fake_engine(**hf_fields):
    """A stand-in vLLM engine exposing ``llm_engine.model_config.hf_config``."""
    hf = SimpleNamespace(**hf_fields)
    return SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf)))


# opt-125m: 12 layers, hidden 768, 12 heads (MHA, no GQA), intermediate 3072.
_OPT_125M = dict(
    num_hidden_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    vocab_size=50272,
)


def test_reads_real_model_arch_from_engine():
    spec = _model_spec_from_engine(_fake_engine(**_OPT_125M))
    assert spec is not None
    assert spec.n_layers == 12
    assert spec.hidden == 768
    assert spec.n_heads == 12
    assert spec.num_kv_heads == 12  # no num_key_value_heads -> falls back to n_heads
    assert spec.head_dim == 64  # 768 / 12
    assert spec.intermediate == 3072
    assert spec.vocab == 50272


def test_predicted_graph_matches_the_real_model_not_llama():
    """opt-125m -> 12*5+1 = 61 nodes, not the 32*5+1 = 161 Llama default."""
    spec = _model_spec_from_engine(_fake_engine(**_OPT_125M))
    assert len(predict_graph(model=spec).nodes) == 61
    assert len(predict_graph().nodes) == 161  # default is still Llama-7B


def test_gqa_uses_num_key_value_heads_when_present():
    spec = _model_spec_from_engine(
        _fake_engine(**{**_OPT_125M, "num_key_value_heads": 4})
    )
    assert spec is not None
    assert spec.num_kv_heads == 4


def test_falls_back_to_none_when_no_engine_or_config():
    # No engine at all -> None (loop then uses the default graph).
    assert _model_spec_from_engine(None) is None
    # An engine with no readable HF config -> None, never a crash.
    assert _model_spec_from_engine(SimpleNamespace(nope=1)) is None
    # A config missing required fields -> None (the int() raises, caught).
    assert _model_spec_from_engine(_fake_engine(hidden_size=768)) is None
