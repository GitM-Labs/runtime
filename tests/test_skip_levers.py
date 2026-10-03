"""Excluding a lever from a run.

A lever can hang the model rather than merely regress it. On Kimi at TP=8 on
ROCm, n-gram speculative decoding stalls in an RCCL all-gather, and it ranks
first in the catalogue because it carries the largest hand-authored estimate —
so every run met it before anything else, and the only way past was to edit the
library. These pin the two halves: what a pattern matches, and that a skipped
lever stays visible as skipped.
"""

from __future__ import annotations

import pytest

from gitm.kernels.library import load_library, parse_skips, skipped_by
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate


def _spec(name, knob="k", knobs=None):
    return InterventionSpec(
        name=name, summary="s", knob=knob, value=1, knobs=knobs or {},
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )


# --- what a pattern means ---------------------------------------------------


def test_an_exact_name_excludes_that_lever_and_nothing_else():
    specs = [_spec("speculative_decode_ngram_5"), _spec("cuda_graphs_enable")]
    hit = [s.name for s in specs if skipped_by(s, ("speculative_decode_ngram_5",))]
    assert hit == ["speculative_decode_ngram_5"]


def test_a_glob_excludes_a_family():
    specs = [_spec("speculative_decode_ngram_5"), _spec("speculative_decode_eagle"),
             _spec("cuda_graphs_enable")]
    hit = [s.name for s in specs if skipped_by(s, ("speculative*",))]
    assert hit == ["speculative_decode_ngram_5", "speculative_decode_eagle"]


def test_a_knob_excludes_every_lever_that_sets_it():
    """The knob is what hangs the model, and one knob can be reached by several
    levers under different names."""
    specs = [_spec("via_name_a", knob="num_speculative_tokens"),
             _spec("via_name_b", knob="num_speculative_tokens"),
             _spec("unrelated", knob="block_size")]
    hit = [s.name for s in specs if skipped_by(s, ("num_speculative_tokens",))]
    assert hit == ["via_name_a", "via_name_b"]


def test_a_multi_knob_lever_is_matched_on_any_of_its_knobs():
    spec = _spec("atomic_pair", knob="", knobs={"speculative_config": 1, "other": 2})
    assert skipped_by(spec, ("speculative_config",)) == "speculative_config"
    assert skipped_by(spec, ("other",)) == "other"
    assert skipped_by(spec, ("absent",)) is None


def test_matching_ignores_case_and_reports_the_pattern_that_hit():
    """The pattern comes back rather than a bool so a run can say *why* a lever
    never ran."""
    spec = _spec("Speculative_Decode_Ngram_5")
    assert skipped_by(spec, ("other*", "SPECULATIVE*")) == "SPECULATIVE*"


def test_no_patterns_excludes_nothing():
    assert skipped_by(_spec("anything"), ()) is None


# --- how patterns arrive ----------------------------------------------------


def test_patterns_parse_from_a_list_or_a_delimited_string():
    assert parse_skips(["a", "b"]) == ("a", "b")
    assert parse_skips("a,b") == ("a", "b")          # env var or k8s manifest
    assert parse_skips("a b") == ("a", "b")
    assert parse_skips(["a,b", "c"]) == ("a", "b", "c")  # repeated flag, mixed


def test_empty_fragments_and_duplicates_drop_out():
    """A trailing comma is a typo, not a pattern that matches nothing."""
    assert parse_skips("a,,b,") == ("a", "b")
    assert parse_skips("a, a ,b") == ("a", "b")
    assert parse_skips(None) == ()
    assert parse_skips("") == ()


# --- against the real catalogue ---------------------------------------------


def test_the_ngram_lever_this_exists_for_is_actually_reachable():
    """If the catalogue renames it, this fails rather than silently skipping
    nothing on the next Kimi run."""
    lib = load_library(workload="vllm-decode")
    by_name = [s for s in lib if skipped_by(s, ("speculative*",))]
    assert [s.name for s in by_name] == ["speculative_decode_ngram_5"]

    # And by the knob, which is the handle an operator reads off a hang.
    by_knob = [s for s in lib if skipped_by(s, ("num_speculative_tokens",))]
    assert by_knob == by_name

    # It is top-ranked by the library's own estimate, which is why it is met
    # first and why a hang on it blocks everything behind it.
    assert max(lib, key=lambda s: s.expected_delta_mean).name == by_name[0].name


def test_excluding_one_lever_leaves_the_rest_of_the_catalogue_intact():
    lib = load_library(workload="vllm-decode")
    kept = [s for s in lib if not skipped_by(s, ("speculative*",))]
    assert len(kept) == len(lib) - 1
    assert "speculative_decode_ngram_5" not in {s.name for s in kept}


# --- plumbing ---------------------------------------------------------------


def test_the_flag_reaches_the_loop_config():
    import gitm.api as api

    seen = {}
    real = api.run_loop
    api.run_loop = lambda cfg: seen.setdefault("cfg", cfg) or {}
    try:
        api.optimize(workload="vllm-decode", skip_levers=["speculative*", "a,b"])
    finally:
        api.run_loop = real
    assert seen["cfg"].skip_levers == ("speculative*", "a", "b")


def test_skip_levers_defaults_to_nothing_excluded():
    from gitm.scheduler import LoopConfig

    assert LoopConfig().skip_levers == ()


def test_flag_and_env_patterns_merge_without_duplicating():
    """Both sources exist because a flag suits a person and the env suits a
    Kubernetes manifest; naming the same pattern in both is not two patterns."""
    merged = parse_skips(["speculative*", "other", "speculative*"])
    assert merged == ("speculative*", "other")


# --- the paths that return before the catalogue -----------------------------


def test_the_curated_workloads_each_have_a_lever_a_pattern_can_reach():
    """hft, openfold and edge each return from the loop before the catalogue is
    read, so their one curated lever has to be checked on its own path. If a
    pattern cannot name it, --skip-lever is silently ignored on those
    workloads — asked for and not done, which is worse than absent."""
    pytest.importorskip("torch")
    specs = []
    for mod, fn in (("gitm.benchmarks.hft.optimize", "hft_intervention_spec"),
                    ("gitm.benchmarks.edge.optimize", "edge_intervention_spec")):
        m = pytest.importorskip(mod)
        specs.append(getattr(m, fn)())

    for spec in specs:
        assert skipped_by(spec, (spec.name,)) == spec.name
        assert skipped_by(spec, ("*",)) == "*"
        assert skipped_by(spec, ("something-else",)) is None
