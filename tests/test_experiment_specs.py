"""Emitting the arms a harness runs — and getting the same levers back.

S-1: the epic says the loop *"emits experiment specs for the runtime experiment
harness"*, and nothing in ``gitm/`` wrote one. Every manifest was hand-written,
and the nine intervention slots in ``scripts/kimi_loop/gen_pods.py`` are a
person doing this from the ranked list by eye.

The central test here is the round trip. #125 settled that resolution matches on
knob *and* value, so a catalogue-generated manifest round-trips by construction;
this turns that claim into something that fails when it stops being true.
"""

from __future__ import annotations

import json

import pytest

from gitm.kernels.library import load_library
from gitm.optimizer.experiment_specs import (
    DEFAULT_LOAD,
    SCHEMA,
    flag_for,
    is_env_knob,
    plan_arms,
    write_experiments,
)
from gitm.optimizer.harness_results import knob_difference, resolve_lever

LIB = load_library(workload="vllm-decode")
BASE = ["--tensor-parallel-size", "8", "--enforce-eager", "--gpu-memory-utilization", "0.9"]


class _Cap:
    """The only part of a Capture that knob_difference reads."""

    def __init__(self, argv):
        self.serve_argv = tuple(argv)


def _resolve(base, arm_argv):
    """What the return path makes of this arm: the lever it resolves to."""
    knobs = knob_difference(_Cap(base), _Cap(arm_argv))
    assert len(knobs) == 1, f"arm changed {len(knobs)} flags: {knobs}"
    knob, value = next(iter(knobs.items()))
    return resolve_lever(knob, value, LIB)


# --------------------------------------------------------------------------- #
# the round trip                                                              #
# --------------------------------------------------------------------------- #
def test_every_emitted_arm_resolves_back_to_the_lever_it_was_emitted_for():
    """The whole contract, over the real catalogue. If this breaks, the loop
    emits experiments whose results it cannot attribute."""
    arms, unreachable = plan_arms(BASE, LIB)
    assert arms, "the catalogue produced no arms at all"

    for arm in arms:
        if not arm.ingestable:
            continue  # an env lever is invisible to a serve-argv diff, by design
        got = _resolve(BASE, arm.serve_argv)
        assert got is not None, f"{arm.lever}: emitted an arm nothing resolves"
        assert got.name == arm.lever, f"{arm.lever} resolved back as {got.name}"

    # And the ones that could not be emitted each say why, rather than vanishing.
    assert all(u.reason for u in unreachable)


def test_flag_for_inverts_the_normalisation_resolve_lever_applies():
    """The two sides of the round trip, stated as one property."""
    for spec in LIB:
        if is_env_knob(spec.knob):
            continue
        assert flag_for(spec.knob).lstrip("-").replace("-", "_") == spec.knob


# --------------------------------------------------------------------------- #
# how a value becomes a flag                                                  #
# --------------------------------------------------------------------------- #
def test_a_true_lever_becomes_a_bare_flag():
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == "enable_expert_parallel"])
    assert list(arms[0].serve_argv) == [*BASE, "--enable-expert-parallel"]


def test_a_valued_lever_carries_its_value():
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == "max_num_seqs_dynamic"])
    assert list(arms[0].serve_argv) == [*BASE, "--max-num-seqs", "256"]


def test_a_valued_lever_replaces_rather_than_repeats_a_flag():
    """Two settings of one flag is a server-dependent precedence question, and
    the reader's parser reports only one of them."""
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == "gpu_memory_utilization_dynamic"])
    argv = list(arms[0].serve_argv)
    assert argv.count("--gpu-memory-utilization") == 1
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.92"
    assert _resolve(BASE, argv).name == "gpu_memory_utilization_dynamic"


def test_a_false_lever_is_realised_by_removing_the_flag():
    """``cuda_graphs_enable`` is ``enforce_eager: false`` and exists only as the
    absence of the flag."""
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == "cuda_graphs_enable"])
    assert "--enforce-eager" not in arms[0].serve_argv
    assert _resolve(BASE, arms[0].serve_argv).name == "cuda_graphs_enable"


def test_removing_a_flag_also_removes_the_value_that_belonged_to_it():
    base = ["--max-num-seqs", "512", "--enforce-eager"]
    arms, _ = plan_arms(base, [s for s in LIB if s.name == "cuda_graphs_enable"])
    assert list(arms[0].serve_argv) == ["--max-num-seqs", "512"]


# --------------------------------------------------------------------------- #
# what cannot become an arm, and why it is reported                           #
# --------------------------------------------------------------------------- #
def test_a_false_lever_against_a_baseline_that_does_not_set_the_flag():
    """The arm would be the baseline, and compare rightly refuses two arms that
    ran the same flags. Reachability depends on the baseline, not the lever."""
    _, out = plan_arms(["--tensor-parallel-size", "8"],
                       [s for s in LIB if s.name == "cuda_graphs_enable"])
    assert len(out) == 1
    assert "does not set" in out[0].reason and out[0].lever == "cuda_graphs_enable"


def test_a_lever_the_baseline_already_runs_measures_nothing():
    base = ["--tensor-parallel-size", "8", "--enable-expert-parallel"]
    _, out = plan_arms(base, [s for s in LIB if s.name == "enable_expert_parallel"])
    assert len(out) == 1 and "already runs" in out[0].reason


def test_a_lever_the_baseline_already_runs_under_a_different_spelling():
    """The value comparison is the catalogue's own, so '256' and 256 are the
    same setting and the arm is still a no-op."""
    base = ["--max-num-seqs", "256"]
    _, out = plan_arms(base, [s for s in LIB if s.name == "max_num_seqs_dynamic"])
    assert len(out) == 1 and "already runs" in out[0].reason


def test_an_env_lever_is_emitted_but_marked_unattributable():
    """Worth running; its result cannot be read back, because knob_difference
    sees only serve_argv."""
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == "attention_backend_flashinfer"])
    arm = arms[0]
    assert arm.env == {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
    assert list(arm.serve_argv) == BASE   # the server differs only in environment
    assert arm.ingestable is False


def test_a_candidate_the_ranking_rejected_is_not_emitted():
    """The gate's answer is categorical; re-asking it on the cluster would spend
    GPU time on a lever the loop already declined."""
    class _Ranked:
        def __init__(self, spec, reason=None):
            self.spec, self.rejected_reason, self.predicted_delta = spec, reason, 0.04

    ep = next(s for s in LIB if s.name == "enable_expert_parallel")
    eplb = next(s for s in LIB if s.name == "enable_eplb")
    arms, out = plan_arms(BASE, [_Ranked(ep), _Ranked(eplb, "safety.requires_qualification_commit")])
    assert [a.lever for a in arms] == ["enable_expert_parallel"]
    assert len(out) == 1 and "not ranked" in out[0].reason


def test_a_multi_knob_lever_cannot_be_credited_to_one_lever():
    from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate

    spec = InterventionSpec(
        name="pair", summary="s", knob="", value=None,
        knobs={"enable_expert_parallel": True, "enable_eplb": True},
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )
    _, out = plan_arms(BASE, [spec])
    assert len(out) == 1 and "2 knobs at once" in out[0].reason


def test_max_arms_caps_the_sweep():
    arms, _ = plan_arms(BASE, LIB, max_arms=3)
    assert len(arms) == 3


# --------------------------------------------------------------------------- #
# the file                                                                    #
# --------------------------------------------------------------------------- #
def test_the_file_names_the_baseline_explicitly(tmp_path):
    """Which arm is the baseline is not recoverable from a set of directories
    afterwards, and a wrong guess inverts the sign of every delta."""
    arms, out = plan_arms(BASE, LIB, max_arms=2)
    path = write_experiments(tmp_path / "experiments.json", baseline_argv=BASE,
                             arms=arms, unreachable=out, served_model="Kimi-K2.5",
                             load=DEFAULT_LOAD, run_id="sweep1")
    doc = json.loads(open(path).read())

    assert doc["schema"] == SCHEMA
    assert doc["baseline"]["serve_argv"] == BASE
    assert doc["served_model"] == "Kimi-K2.5" and doc["run_id"] == "sweep1"
    assert len(doc["arms"]) == 2
    assert doc["load"] == DEFAULT_LOAD


def test_the_file_carries_the_command_that_reads_the_results_back(tmp_path):
    arms, out = plan_arms(BASE, LIB, max_arms=3)
    path = write_experiments(tmp_path / "e.json", baseline_argv=BASE, arms=arms,
                             unreachable=out, served_model="Kimi-K2.5",
                             load=DEFAULT_LOAD)
    doc = json.loads(open(path).read())
    assert doc["ingest"]["command"].startswith("gitm ingest --baseline")
    for arm in arms:
        if arm.ingestable:
            assert f"'<{arm.lever} dir>'" in doc["ingest"]["command"]
        else:
            assert arm.lever in doc["ingest"]["not_attributable"]


def test_the_command_survives_being_pasted_into_a_shell(tmp_path):
    """It is a line someone copies. A SKU has spaces, a fingerprint is whatever
    the operator passed, and the directory placeholders are angle brackets — a
    shell reads those as redirection rather than as blanks to fill in."""
    import shlex

    arms, out = plan_arms(BASE, LIB, max_arms=2)
    path = write_experiments(
        tmp_path / "e.json", baseline_argv=BASE, arms=arms, unreachable=out,
        served_model="Kimi-K2.5", load=DEFAULT_LOAD,
        gpu_sku="Bob's Instinct MI355X", fingerprint="Kimi MI355X",
    )
    cmd = json.loads(open(path).read())["ingest"]["command"]

    # Tokenises the way a shell would, with each value arriving as one argument.
    tokens = shlex.split(cmd)
    assert tokens[tokens.index("--gpu-sku") + 1] == "Bob's Instinct MI355X"
    assert tokens[tokens.index("--fingerprint") + 1] == "Kimi MI355X"
    assert tokens[tokens.index("--baseline") + 1] == "<baseline dir>"
    # No stray redirection left in the line.
    assert "<" not in cmd.replace("'<", "").replace(" dir>'", "")


def test_unreachable_levers_are_always_in_the_file(tmp_path):
    """A sweep that quietly dropped three candidates looks like a sweep of the
    ones that remained."""
    _, out = plan_arms(["--tensor-parallel-size", "8"],
                       [s for s in LIB if s.name == "cuda_graphs_enable"])
    path = write_experiments(tmp_path / "e.json", baseline_argv=["--tensor-parallel-size", "8"],
                             arms=[], unreachable=out, served_model="m",
                             load=DEFAULT_LOAD)
    doc = json.loads(open(path).read())
    assert doc["arms"] == []
    assert len(doc["unreachable"]) == 1 and doc["unreachable"][0]["reason"]


@pytest.mark.parametrize("name", ["enable_expert_parallel", "kv_cache_dtype_fp8",
                                  "preemption_mode_swap", "quantization_awq"])
def test_named_levers_each_round_trip(name):
    """A spread of value kinds: bool, string, enum-ish string, string."""
    arms, _ = plan_arms(BASE, [s for s in LIB if s.name == name])
    assert _resolve(BASE, arms[0].serve_argv).name == name


def test_speculative_decoding_is_emitted_as_the_flag_vllm_accepts():
    """L-3. `--num-speculative-tokens 5` is rejected by the server, so the
    top-ranked arm of every sweep was a cluster job that could not start."""
    import json

    lever = next(s for s in LIB if s.knob == "num_speculative_tokens")
    arms, _ = plan_arms(BASE, [lever])
    argv = list(arms[0].serve_argv)
    assert "--num-speculative-tokens" not in argv
    cfg = json.loads(argv[argv.index("--speculative-config") + 1])
    assert cfg == {"method": "ngram", "num_speculative_tokens": lever.value}
    # And it reads back as the same lever.
    assert _resolve(BASE, arms[0].serve_argv).name == lever.name


def test_a_baseline_already_speculating_is_not_proposed_again():
    import json

    lever = next(s for s in LIB if s.knob == "num_speculative_tokens")
    base = [*BASE, "--speculative-config",
            json.dumps({"method": "ngram", "num_speculative_tokens": lever.value})]
    arms, unreachable = plan_arms(base, [lever])
    assert not arms and "already runs" in unreachable[0].reason


def test_an_arm_changes_only_the_token_count_of_a_baseline_speculative_config():
    """The baseline's own method and lookup window stay, so the arm differs from
    it in the lever and nothing else."""
    import json

    lever = next(s for s in LIB if s.knob == "num_speculative_tokens")
    base = [*BASE, "--speculative-config",
            json.dumps({"method": "ngram", "prompt_lookup_max": 4,
                        "num_speculative_tokens": lever.value + 1})]
    arms, _ = plan_arms(base, [lever])
    argv = list(arms[0].serve_argv)
    cfg = json.loads(argv[argv.index("--speculative-config") + 1])
    assert cfg == {"method": "ngram", "prompt_lookup_max": 4,
                   "num_speculative_tokens": lever.value}
    assert argv.count("--speculative-config") == 1
