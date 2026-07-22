"""Customer-verification export: the A/B as structured, re-measurable data."""

from __future__ import annotations

import json

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import ApplyResult, EngineABResult
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import (
    MIN_NOISE_BAND,
    SCHEMA,
    build_export,
    build_record,
    write_verification,
)


def _spec(name="raise-batch", knob="max_num_seqs", value=512) -> InterventionSpec:
    return InterventionSpec(
        name=name,
        summary="Raise max concurrent sequences",
        knob=knob,
        value=value,
        expected_delta_mean=0.08,
        expected_delta_lo=0.02,
        expected_delta_hi=0.14,
        source="https://docs.vllm.ai/example",
    )


def _ab(**over) -> EngineABResult:
    base = dict(
        knob="max_num_seqs", value=512, baseline_tps=100.0, candidate_tps=112.0,
        speedup=1.12, kept=True, via="hot-swap", baseline_std=1.0,
        candidate_std=1.0, reps=3, significant=True,
    )
    base.update(over)
    return EngineABResult(**base)


def _prov() -> Provenance:
    return Provenance(
        workload_id="vllm-decode", fingerprint="fp", run_id="run-1",
        git_sha="abc1234", gitm_version="0.1.11",
        started_at_ns=0, ended_at_ns=1,
    )


# --------------------------------------------------------------------------- #
# the measured numbers survive intact                                         #
# --------------------------------------------------------------------------- #
def test_record_carries_both_sides_not_just_the_delta():
    rec = build_record(_spec(), _ab(), ApplyResult(True, False, 0.12))
    # The markdown report states only the percentage; re-measuring needs the
    # absolute numbers on both sides.
    assert rec.baseline_tps == 100.0
    assert rec.candidate_tps == 112.0
    assert abs(rec.delta - 0.12) < 1e-9
    assert rec.reps == 3
    assert rec.source.startswith("https://")


def test_full_config_travels_on_both_sides():
    rec = build_record(
        _spec(), _ab(), ApplyResult(True, False, 0.12),
        baseline_config={"max_num_seqs": 256, "gpu_memory_utilization": 0.9},
        candidate_config={"max_num_seqs": 512, "gpu_memory_utilization": 0.9},
    )
    # Not just the knob that moved: reproducing at a different
    # gpu_memory_utilization measures a different system.
    assert rec.baseline_config["gpu_memory_utilization"] == 0.9
    assert rec.baseline_config["max_num_seqs"] == 256
    assert rec.candidate_config["max_num_seqs"] == 512


# --------------------------------------------------------------------------- #
# `kept` follows the gate, not the measurement                                #
# --------------------------------------------------------------------------- #
def test_kept_follows_the_rollback_gate_not_the_ab_indicator():
    # EngineABResult.kept is a measure-time delta>=0 indicator. The gate can
    # still roll the candidate back (min_keep_delta, a failed correctness
    # check). The export must report what actually happened to the workload.
    ab = _ab(kept=True)
    rec = build_record(_spec(), ab, ApplyResult(True, rolled_back=True, measured_delta=0.001))
    assert ab.kept is True
    assert rec.kept is False


def test_kept_true_when_gate_kept_it():
    rec = build_record(_spec(), _ab(kept=True), ApplyResult(True, False, 0.12))
    assert rec.kept is True


# --------------------------------------------------------------------------- #
# agreement band                                                              #
# --------------------------------------------------------------------------- #
def test_single_rep_gets_a_floored_agreement_band():
    # reps=1 → zero measured scatter. Publishing a 0% band would imply a
    # precision no GPU benchmark has, and any customer jitter would read as
    # a failed reproduction.
    rec = build_record(
        _spec(),
        _ab(reps=1, baseline_std=0.0, candidate_std=0.0),
        ApplyResult(True, False, 0.12),
    )
    assert rec.agreement_band == MIN_NOISE_BAND


def test_real_scatter_widens_the_band_beyond_the_floor():
    # baseline 100 tok/s with std 5+5 → 10% scatter, well above the floor.
    rec = build_record(
        _spec(),
        _ab(reps=5, baseline_std=5.0, candidate_std=5.0),
        ApplyResult(True, False, 0.12),
    )
    assert abs(rec.agreement_band - 0.10) < 1e-9


# --------------------------------------------------------------------------- #
# document shape                                                              #
# --------------------------------------------------------------------------- #
def test_export_is_json_serializable_with_provenance():
    rec = build_record(_spec(), _ab(), ApplyResult(True, False, 0.12))
    doc = build_export([rec], _prov(), gpu_sku="NVIDIA A100-SXM4-80GB")
    round_tripped = json.loads(json.dumps(doc, default=str))
    assert round_tripped["schema"] == SCHEMA
    assert round_tripped["provenance"]["git_sha"] == "abc1234"
    assert round_tripped["environment"]["gpu_sku"] == "NVIDIA A100-SXM4-80GB"
    assert len(round_tripped["results"]) == 1


def test_unknown_environment_reports_none_not_a_guess():
    doc = build_export([], _prov())
    # No SKU passed and (on a CPU box) no driver — these must read as unknown.
    assert doc["environment"]["gpu_sku"] is None


def test_write_verification_writes_readable_json(tmp_path):
    rec = build_record(_spec(), _ab(), ApplyResult(True, False, 0.12))
    out = write_verification([rec], _prov(), tmp_path / "verification.json")
    doc = json.loads(open(out).read())
    assert doc["results"][0]["knob"] == "max_num_seqs"
    assert doc["protocol"]["metric"].startswith("decode throughput")
