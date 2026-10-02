"""A checkpoint has to be findable by the names it actually arrives under.

Catalogue entries are keyed by YAML file stem — ``kimi-k2.5``. Nothing a real
run holds looks like that. vLLM reports the model id, ``moonshotai/Kimi-K2.5``,
which is the entry's own ``name:`` field; and under ``HF_HUB_OFFLINE=1`` it
reports the local snapshot directory instead, because there is no hub call to
resolve the id against.

So the entry that knows a checkpoint's family and its hand-fitted fields was
reachable only by a name no caller has.
"""

from __future__ import annotations

import json

import pytest

from gitm.planner.model_catalogue import load_entry


def test_the_file_stem_still_resolves():
    assert load_entry("kimi-k2.5")["family"] == "glm_moe_dsa"


def test_the_model_id_resolves():
    """``moonshotai/Kimi-K2.5`` is what vLLM reports and what the entry calls
    itself. It did not resolve: `_resolve` took the last path segment and looked
    for `Kimi-K2.5.yaml`, which is not the file's name."""
    assert load_entry("moonshotai/Kimi-K2.5")["family"] == "glm_moe_dsa"


def test_resolution_is_case_insensitive():
    assert load_entry("Kimi-K2.5")["family"] == "glm_moe_dsa"


def test_a_huggingface_cache_snapshot_path_resolves(tmp_path):
    """Under HF_HUB_OFFLINE the engine reports a snapshot directory. The id is
    still in it — the cache encodes `org/name` as `models--org--name` — so the
    entry is recoverable without a hub call."""
    snap = tmp_path / "models--moonshotai--Kimi-K2.5" / "snapshots" / "abc123"
    snap.mkdir(parents=True)

    assert load_entry(str(snap))["family"] == "glm_moe_dsa"


def test_an_unknown_name_still_says_what_is_available():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_entry("not-a-model")


def test_a_local_model_directory_is_read_as_a_config(tmp_path):
    """A plain checkpoint directory is not in the catalogue and should not need
    to be: `config.json` is sitting in it."""
    from gitm.planner.registry import _load

    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 64,
        "vocab_size": 163840, "intermediate_size": 18432,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "n_routed_experts": 384, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
    }))

    spec, family, note = _load(str(tmp_path))

    assert family == "glm_moe_dsa"
    assert spec.n_layers == 61


# --------------------------------------------------------------------------- #
# the `gitm plan` path, which has its own gate                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("probe", ["kimi-k2.5", "moonshotai/Kimi-K2.5", "Kimi-K2.5"])
def test_plan_reaches_the_catalogue_by_any_name_the_run_holds(probe):
    """`_load` gated the catalogue on `model in available()`, which holds only
    file stems — so resolving the id was not enough on its own."""
    from gitm.planner.registry import _load

    _spec, family, note = _load(probe)

    assert family == "glm_moe_dsa"
    assert "catalogue" in note


def test_a_snapshot_path_prefers_the_catalogue_over_its_own_config(tmp_path):
    """Both exist in a real cache directory. The entry carries a corrected
    family and hand-fitted fields the raw config does not, so it wins."""
    from gitm.planner.registry import _load

    snap = tmp_path / "models--moonshotai--Kimi-K2.5" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"hidden_size": 1, "model_type": "x"}))

    _spec, family, note = _load(str(snap))

    assert family == "glm_moe_dsa"
    assert "catalogue" in note, "read the raw config instead of the entry"
