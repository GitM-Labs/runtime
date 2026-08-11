"""The loop must dispatch the live model to a priceable execution graph or refuse."""

from __future__ import annotations

from types import SimpleNamespace

from gitm.planner.context import peak_for_sku
from gitm.planner.roofline import SparseMoEModelSpec
from gitm.scheduler.loop import _execution_graph


def _pctx(sku: str | None = "NVIDIA B200", *, kv_cache_len: int | None = None):
    return SimpleNamespace(
        peak=peak_for_sku(sku),
        sku=sku,
        gate=SimpleNamespace(kv_cache_len=kv_cache_len),
    )


def _engine(cfg: dict):
    hf = SimpleNamespace(**cfg)
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf))


def _moe_cfg(**over) -> dict:
    cfg = {
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 32,
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
        "torch_dtype": "bfloat16",
    }
    cfg.update(over)
    return cfg


def test_sparse_engine_dispatches_to_sparse_graph():
    resolved = _execution_graph(_engine(_moe_cfg()), _pctx(), sched=None)

    assert resolved.ok
    assert isinstance(resolved.graph.model, SparseMoEModelSpec)
    assert any(node.op == "moe_routed" for node in resolved.graph.nodes)
    assert not resolved.graph.has_fallback_bytes
    assert resolved.model_source == "live_hf_config"


def test_sparse_dispatch_refuses_unpriceable_final_dtype():
    resolved = _execution_graph(
        _engine(_moe_cfg(expert_dtype="future_fp3")), _pctx(), sched=None
    )

    assert not resolved.ok and resolved.graph is None
    assert "expert_dtype='future_fp3'" in resolved.refusal_reason


def test_partial_sparse_config_is_refused_not_sent_to_dense_graph():
    resolved = _execution_graph(
        _engine({"hidden_size": 64, "n_routed_experts": 4}), _pctx(), sched=None
    )

    assert not resolved.ok and resolved.graph is None
    assert "experts per token" in resolved.refusal_reason


def test_missing_live_model_refuses_instead_of_defaulting_to_llama():
    resolved = _execution_graph(None, _pctx(), sched=None)

    assert not resolved.ok and resolved.graph is None
    assert "no live engine" in resolved.refusal_reason


def test_unknown_hardware_refuses_instead_of_recording_a100_as_observed():
    resolved = _execution_graph(_engine(_moe_cfg()), _pctx("Unknown GPU"), sched=None)

    assert not resolved.ok and resolved.graph is None
    assert "Unknown GPU" in resolved.refusal_reason
    assert "hardware catalogue" in resolved.refusal_reason


def test_config_conversion_bug_is_a_named_refusal():
    class BrokenConfig:
        def to_dict(self):
            raise RuntimeError("parser exploded")

    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=BrokenConfig()))
    resolved = _execution_graph(engine, _pctx(), sched=None)

    assert not resolved.ok and resolved.graph is None
    assert "RuntimeError" in resolved.refusal_reason
    assert "parser exploded" in resolved.refusal_reason


def test_accepted_batch_and_kv_defaults_are_diagnostics():
    resolved = _execution_graph(_engine(_moe_cfg()), _pctx(), sched=None)

    assert resolved.ok
    assert resolved.graph.batch.batch == 1
    assert resolved.graph.batch.kv_cache_len == 4096
    assert any("batch=1" in note for note in resolved.diagnostics)
    assert any("kv_cache_len=4096" in note for note in resolved.diagnostics)
