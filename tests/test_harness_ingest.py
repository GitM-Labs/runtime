"""`gitm ingest` — the command that closes the loop's return edge.

The converter in :mod:`gitm.optimizer.harness_results` has existed since #125
and had no caller anywhere: no CLI command, no call from the loop. To use it you
would open a REPL. So every result measured on the cluster was invisible to the
ranking that reads history, which is the failure the history reader exists to
prevent, one layer out.

There is nothing to add to the loop itself — it drives a local engine and never
sees a harness directory — so the operator is the caller, after a sweep lands.
These tests go through the CLI entry point rather than the converter, because
the converter was already tested and the missing piece was the caller.
"""

from __future__ import annotations

import json

import pytest

from gitm.cli import main as cli_main
from gitm.optimizer.history import load_history, record_for

from .test_harness_results import BASE_ARGV, _arm

SKU = "AMD Instinct MI355X"
FP = "kimi-k2.5-mi355x"


def _ingest(tmp_path, *args, fingerprint=FP):
    argv = ["ingest", "--scratch", str(tmp_path / "scratch"), *args]
    if fingerprint is not None and "--fingerprint" not in args:
        argv += ["--fingerprint", fingerprint]
    return cli_main(argv)


def _sweep(tmp_path):
    """One baseline and two arms, as a harness sweep lands on disk.

    The arms deliberately resolve to real catalogue levers: ``kv_cache_block_size_16``
    is reached by ``--block-size 16``, and a flag that is the *opposite* of a
    lever (``--enforce-eager``, where the lever is ``enforce_eager: false``) has
    no entry and is refused, which is what the mixed-sweep test uses.
    """
    base = _arm(tmp_path, "tp2", rps=40.0)
    ep = _arm(tmp_path, "tp2-ep", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)
    blocks = _arm(tmp_path, "tp2-b16", argv=[*BASE_ARGV, "--block-size", "16"], rps=36.0)
    return base, ep, blocks


# --------------------------------------------------------------------------- #
# the edge is closed                                                           #
# --------------------------------------------------------------------------- #
def test_ingested_cluster_results_are_read_by_the_ranking(tmp_path):
    """The whole point. After ingest, the levers the cluster measured are in the
    history the loop ranks from, keyed by GPU and workload like a local run."""
    base, ep, eager = _sweep(tmp_path)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                 "--candidate", str(eager), "--gpu-sku", SKU, "--run-id", "sweep1")
    assert rc == 0

    hist = load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU)
    assert hist.runs_read == 1
    names = {r.intervention_name for r in hist.records.values()}
    assert "enable_expert_parallel" in names

    won = record_for(hist, "enable_expert_parallel", gpu_sku=SKU, fingerprint=FP)
    assert won is not None and won.mean_delta > 0.4   # 40.0 -> 59.6 rps
    assert won.wins == 1
    # A loss is evidence too, and has to survive the trip as a loss.
    lost = record_for(hist, "kv_cache_block_size_16", gpu_sku=SKU, fingerprint=FP)
    assert lost is not None and lost.mean_delta < 0   # 40.0 -> 36.0 rps
    assert lost.losses == 1


def test_a_sweep_lands_as_one_run_not_one_per_arm(tmp_path):
    """Same baseline, same workload, same box: it is one run. Splitting it would
    also mean inventing a run id per arm."""
    base, ep, eager = _sweep(tmp_path)
    _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
            "--candidate", str(eager), "--gpu-sku", SKU, "--run-id", "sweep1")

    runs = sorted(p.name for p in (tmp_path / "scratch" / "runs").iterdir())
    assert runs == ["sweep1"]
    doc = json.loads((tmp_path / "scratch" / "runs" / "sweep1" / "verification.json").read_text())
    assert len(doc["results"]) == 2


# --------------------------------------------------------------------------- #
# ingesting twice must not count twice                                         #
# --------------------------------------------------------------------------- #
def test_re_ingesting_the_same_sweep_is_refused(tmp_path):
    """The default run id is a digest of the arms, so the second attempt hits
    the existing export instead of filing the same measurements again. Double
    counting would show up as a lever with twice the attempts and the ranking
    treating one sweep as corroboration of itself."""
    base, ep, _ = _sweep(tmp_path)
    first = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                    "--gpu-sku", SKU)
    second = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                     "--gpu-sku", SKU)
    assert first == 0 and second == 1

    hist = load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU)
    rec = record_for(hist, "enable_expert_parallel", gpu_sku=SKU, fingerprint=FP)
    assert rec is not None and rec.attempts == 1


def test_the_default_run_id_does_not_depend_on_argument_order(tmp_path):
    base, ep, eager = _sweep(tmp_path)
    a = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                "--candidate", str(eager), "--gpu-sku", SKU)
    b = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(eager),
                "--candidate", str(ep), "--gpu-sku", SKU)
    assert a == 0 and b == 1  # the second collided, so it is the same sweep


# --------------------------------------------------------------------------- #
# partial sweeps and bad input                                                 #
# --------------------------------------------------------------------------- #
def test_one_unusable_arm_does_not_discard_the_others(tmp_path):
    """An arm measured on a different model is not an A/B of this baseline. That
    refusal belongs to that arm, and the arms that were comparable are still
    evidence."""
    base, ep, _ = _sweep(tmp_path)
    other = _arm(tmp_path, "other-model", model="Llama-3-70B",
                 argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=50.0)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                 "--candidate", str(other), "--gpu-sku", SKU, "--run-id", "mixed")
    assert rc == 0

    doc = json.loads((tmp_path / "scratch" / "runs" / "mixed" / "verification.json").read_text())
    assert len(doc["results"]) == 1
    # Visible, or a sweep of three that ingested one looks like a sweep of one.
    refused = json.loads(
        (tmp_path / "scratch" / "runs" / "mixed" / "ingest_refused.json").read_text())
    assert len(refused["refused"]) == 1 and "other-model" in refused["refused"][0]


def test_a_sweep_where_nothing_is_comparable_writes_no_export(tmp_path):
    """An export with no records is a file asserting that a sweep produced no
    evidence, which the ranking would then read as a run it had already seen."""
    base, _, _ = _sweep(tmp_path)
    other = _arm(tmp_path, "other-model", model="Llama-3-70B",
                 argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=50.0)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(other),
                 "--gpu-sku", SKU, "--run-id", "nothing")
    assert rc == 1
    assert not (tmp_path / "scratch" / "runs" / "nothing" / "verification.json").exists()


def test_an_unreadable_capture_fails_before_writing_anything(tmp_path):
    base, ep, _ = _sweep(tmp_path)
    rc = _ingest(tmp_path, "--baseline", str(base),
                 "--candidate", str(tmp_path / "does-not-exist"),
                 "--gpu-sku", SKU, "--run-id", "bad")
    assert rc == 2
    assert not (tmp_path / "scratch" / "runs").exists()


# --------------------------------------------------------------------------- #
# the SKU is what makes a record readable                                      #
# --------------------------------------------------------------------------- #
def test_without_a_sku_the_records_are_written_but_never_read(tmp_path, capsys):
    """History filters on the GPU, because a result from another box is not
    evidence about this one. So a SKU-less ingest succeeds and is then invisible,
    which is the quieter of the two failures and has to be said out loud."""
    base, ep, _ = _sweep(tmp_path)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                 "--run-id", "nosku")
    assert rc == 0
    assert "--gpu-sku" in capsys.readouterr().err

    assert load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU).records == {}


def test_dry_run_writes_nothing_and_names_the_destination(tmp_path, capsys):
    base, ep, _ = _sweep(tmp_path)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                 "--gpu-sku", SKU, "--dry-run")
    assert rc == 0
    out = capsys.readouterr().out
    assert "verification.json" in out and "harness-" in out
    # scratch_root creates its subdirectories on sight, so the export itself is
    # what must be absent.
    assert not list((tmp_path / "scratch" / "runs").glob("*/verification.json"))


@pytest.mark.parametrize("missing", ["--baseline", "--candidate"])
def test_both_arms_are_required(tmp_path, missing):
    """Which arm is the baseline is not recoverable from two directories, and a
    wrong guess inverts the sign of every delta."""
    base, ep, _ = _sweep(tmp_path)
    args = {"--baseline": str(base), "--candidate": str(ep)}
    del args[missing]
    with pytest.raises(SystemExit):
        _ingest(tmp_path, *[x for kv in args.items() for x in kv])


# --------------------------------------------------------------------------- #
# the run id identifies a sweep, not a copy of one                            #
# --------------------------------------------------------------------------- #
def test_the_same_sweep_copied_elsewhere_is_still_the_same_sweep(tmp_path):
    """The id is derived from what the arms measured, not where they sit. Hashing
    paths meant an operator who re-downloaded results into a second directory
    bypassed the refusal, and history counted one sweep twice — reading it as
    corroboration of itself."""
    import shutil

    base, ep, _ = _sweep(tmp_path)
    first = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                    "--gpu-sku", SKU)
    assert first == 0

    copy = tmp_path / "redownloaded"
    copy.mkdir()
    shutil.copytree(base, copy / "tp2")
    shutil.copytree(ep, copy / "tp2-ep")

    second = _ingest(tmp_path, "--baseline", str(copy / "tp2"),
                     "--candidate", str(copy / "tp2-ep"), "--gpu-sku", SKU)
    assert second == 1, "a copy of the same measurements was ingested again"

    rec = record_for(load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU),
                     "enable_expert_parallel", gpu_sku=SKU, fingerprint=FP)
    assert rec is not None and rec.attempts == 1


def test_a_rerun_that_measured_something_different_is_a_different_sweep(tmp_path):
    """Same arms, new numbers, so it is new evidence and must not collide."""
    base, ep, _ = _sweep(tmp_path)
    assert _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                   "--gpu-sku", SKU) == 0

    rerun = _arm(tmp_path, "tp2-ep-again",
                 argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=55.0)
    assert _ingest(tmp_path, "--baseline", str(base), "--candidate", str(rerun),
                   "--gpu-sku", SKU) == 0

    rec = record_for(load_history(tmp_path / "scratch" / "runs", gpu_sku=SKU),
                     "enable_expert_parallel", gpu_sku=SKU, fingerprint=FP)
    assert rec is not None and rec.attempts == 2


@pytest.mark.parametrize("bad", ["../escape", "a/b", "/abs", "..", "."])
def test_a_run_id_cannot_escape_the_history_directory(tmp_path, bad):
    """It is appended to runs/, so anything that is not a single directory name
    writes the export where the ranking does not look and where nobody asked."""
    base, ep, _ = _sweep(tmp_path)
    assert _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                   "--gpu-sku", SKU, "--run-id", bad) == 2
    assert not list((tmp_path / "scratch").rglob("verification.json"))


# --------------------------------------------------------------------------- #
# the fingerprint comes from an arm that was accepted                         #
# --------------------------------------------------------------------------- #
def test_the_fingerprint_is_not_taken_from_a_refused_arm(tmp_path, capsys):
    """Filing accepted results under a refused arm's workload digest hides them
    behind the filter history applies."""
    base, ep, _ = _sweep(tmp_path)
    other = _arm(tmp_path, "other-model", model="Llama-3-70B",
                 argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=50.0)

    # The refused arm is named first, so it would have supplied the fingerprint.
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(other),
                 "--candidate", str(ep), "--gpu-sku", SKU, "--run-id", "mixed",
                 fingerprint=None)
    assert rc == 0

    doc = json.loads(
        (tmp_path / "scratch" / "runs" / "mixed" / "verification.json").read_text())
    assert len(doc["results"]) == 1
    # The accepted arm's own digest, so the record is findable.
    from gitm.optimizer.harness_results import fingerprint_of, read_capture
    assert doc["provenance"]["fingerprint"] == fingerprint_of(read_capture(ep))


def test_an_untraced_first_arm_does_not_discard_the_sweep(tmp_path):
    """It used to: fingerprinting the first candidate raised, and the arms that
    were fine went with it."""
    base, ep, _ = _sweep(tmp_path)
    untraced = _arm(tmp_path, "no-trace", argv=[*BASE_ARGV, "--block-size", "16"],
                    rps=38.0, trace=False)

    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(untraced),
                 "--candidate", str(ep), "--gpu-sku", SKU, "--run-id", "partial",
                 fingerprint=None)
    assert rc == 0
    doc = json.loads(
        (tmp_path / "scratch" / "runs" / "partial" / "verification.json").read_text())
    assert {r["intervention_name"] for r in doc["results"]} >= {"enable_expert_parallel"}


# --------------------------------------------------------------------------- #
# dry run predicts what the real command does                                 #
# --------------------------------------------------------------------------- #
def test_dry_run_refuses_what_the_real_command_would_refuse(tmp_path, capsys):
    """It is consulted exactly when the operator is unsure, so a prediction that
    contradicts the command is worse than no prediction."""
    base, ep, _ = _sweep(tmp_path)
    assert _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                   "--gpu-sku", SKU, "--run-id", "once") == 0

    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(ep),
                 "--gpu-sku", SKU, "--run-id", "once", "--dry-run")
    assert rc == 1
    assert "already exists" in capsys.readouterr().err


def test_dry_run_refuses_a_sweep_where_nothing_is_comparable(tmp_path, capsys):
    base, _, _ = _sweep(tmp_path)
    other = _arm(tmp_path, "other-model", model="Llama-3-70B",
                 argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=50.0)
    rc = _ingest(tmp_path, "--baseline", str(base), "--candidate", str(other),
                 "--gpu-sku", SKU, "--run-id", "nope", "--dry-run")
    assert rc == 1
    assert "would refuse" in capsys.readouterr().err
    assert not list((tmp_path / "scratch").rglob("verification.json"))
