"""A wrapped config must reach the family that models it.

Some checkpoints ship a multimodal wrapper: the top level carries
``architectures``, the vision tower and the token ids, and every shape the
decode graph needs sits under ``text_config``. ``hybrid_graph`` learned to
descend into it; the sparse-MoE and GLM predicates and the dense reader did not.

The consequence is not a worse graph, it is a graph of a different model.
``detect_family`` returns ``"dense"``, ``_model_spec_from_hf`` raises on the
wrapper's missing ``hidden_size`` and is swallowed by its own ``except``, and
``predict_graph(model=None)`` returns the Llama-2-7B default. Observed on an
8x MI355X run against Kimi K2.5: ``"family": "dense"``, 161 nodes, every
residual measured against a model that was not running.
"""

from __future__ import annotations

from gitm.planner.registry import detect_family

#: Kimi K2.5-shaped: DeepSeek-V4-class sparse MoE (routed experts + sparse
#: attention machinery) behind a multimodal wrapper. Trimmed to the fields the
#: predicates read.
WRAPPED_MOE = {
    "architectures": ["KimiForConditionalGeneration"],
    "model_type": "kimi",
    "vision_config": {"hidden_size": 1152},
    "text_config": {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 18432,
        "vocab_size": 163840,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "index_topk": 2048,
    },
}

#: The same model with nothing wrapped — the shape the predicates were written
#: against. Pins that the descent did not change behaviour for configs that
#: never needed it.
FLAT_MOE = dict(WRAPPED_MOE["text_config"])


def test_a_wrapped_sparse_moe_is_not_read_as_dense():
    """The bug, directly. Without the descent this returns "dense" and the run
    is priced against Llama-2-7B."""
    assert detect_family(WRAPPED_MOE) == "sparse_moe"


def test_an_unwrapped_config_is_unaffected():
    assert detect_family(FLAT_MOE) == "sparse_moe"


def test_a_wrapped_config_builds_a_spec_describing_the_inner_model():
    """Detection alone is not enough — the reader has to see the same dict, or
    the family is right and every shape in it is a default."""
    from gitm.planner.registry import spec_from_hf_config

    spec = spec_from_hf_config(WRAPPED_MOE, name="kimi-k2.5")

    assert spec.hidden == 7168
    assert spec.n_layers == 61
    assert spec.vocab == 163840


def test_the_predicted_graph_is_not_the_llama_default():
    """What the MI355X run actually produced: 161 nodes of dense Llama-2-7B for
    a 61-layer sparse-MoE checkpoint."""
    from gitm.planner.registry import predict_for_config

    graph, family = predict_for_config(WRAPPED_MOE, name="kimi-k2.5")

    assert family == "sparse_moe"
    assert len(graph.nodes) != 161, "this is the Llama-2-7B default graph"


# --------------------------------------------------------------------------- #
# the loop's own path, which is where the silent fallback fires                #
# --------------------------------------------------------------------------- #
class _Cfg:
    """A HF config object, as a live engine exposes it."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return {k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in self.__dict__.items()}


def test_the_dense_reader_descends_through_the_wrapper():
    """``_model_spec_from_hf`` reads the config as an *object*. On a wrapper it
    raised on the missing ``hidden_size``, its own ``except`` swallowed that,
    and the caller took ``predict_graph(model=None)`` — Llama-2-7B, 32 layers,
    for whatever was actually running."""
    from gitm.scheduler.loop import _model_spec_from_hf

    inner = _Cfg(hidden_size=7168, num_hidden_layers=61, num_attention_heads=128,
                 num_key_value_heads=128, intermediate_size=18432, vocab_size=163840)
    wrapped = _Cfg(architectures=["KimiForConditionalGeneration"], text_config=inner)

    spec = _model_spec_from_hf(wrapped)

    assert spec is not None, "fell back to the Llama-2-7B default"
    assert spec.hidden == 7168
    assert spec.n_layers == 61


def test_an_unwrapped_config_object_still_reads():
    from gitm.scheduler.loop import _model_spec_from_hf

    spec = _model_spec_from_hf(_Cfg(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=32, intermediate_size=11008, vocab_size=32000))

    assert spec is not None
    assert spec.n_layers == 32
