"""Every fallback the loop takes is recorded, and has the consequence it claims.

Each test forces one fallback and checks two things: that the degradation is on
the record, and that what it is supposed to change actually changes — a probe
with nothing to time fails the A/B instead of timing noise, a run that flagged
its own A/B is not read back into history, a unit that was not tokens is not
reported as tok/s.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gitm.optimizer.degradation import (
    AB_PROBE,
    AB_UNIT,
    AFFECTS_AB,
    APPROXIMATE,
    AR_CATALOG,
    AR_PROPOSER,
    AR_TARGET,
    FILE_NAME,
    GRAPH_BATCH,
    GRAPH_HARDWARE,
    GRAPH_MODEL,
    UNRELIABLE,
    WORKLOAD_RUNNER,
    Degradation,
    DegradationLog,
    unreliable_ab,
)

from .conftest import make_kernel, make_trace


@contextmanager
def _quiet():
    """The fallbacks these tests force warn by design; the tests assert the record."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# ── the log itself ────────────────────────────────────────────────────────────


def test_record_warns_dedupes_and_writes_even_when_clean(tmp_path: Path):
    log = DegradationLog()
    log.write(tmp_path)
    clean = json.loads((tmp_path / FILE_NAME).read_text())
    assert clean["clean"] is True and clean["items"] == []

    with pytest.warns(RuntimeWarning, match="graph.batch"):
        log.record(GRAPH_BATCH, used="batch=1", reason="no samples")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a repeat must not warn again
        log.record(GRAPH_BATCH, used="batch=1", reason="no samples")
    assert len(log) == 1

    log.write(tmp_path)
    doc = json.loads((tmp_path / FILE_NAME).read_text())
    assert doc["clean"] is False and doc["approximate"] == [GRAPH_BATCH]


def test_severity_is_checked():
    with pytest.raises(ValueError, match="severity"):
        Degradation("x", used="y", reason="z", severity="fine")


def test_unreliable_ab_reads_serialised_records_only_for_the_ab():
    assert unreliable_ab([
        {"stage": AB_PROBE, "severity": UNRELIABLE, "affects": [AFFECTS_AB]},
        {"stage": GRAPH_MODEL, "severity": UNRELIABLE, "affects": ["residuals"]},
        {"stage": AB_UNIT, "severity": APPROXIMATE, "affects": [AFFECTS_AB]},
        "not a dict",
    ]) == [AB_PROBE]


# ── the A/B probe ─────────────────────────────────────────────────────────────


def test_no_runner_probe_refuses_instead_of_timing_nothing():
    from gitm.scheduler.loop import _engine_throughput_fn

    engine = SimpleNamespace()
    log = DegradationLog()
    with _quiet():
        probe = _engine_throughput_fn(engine, None, log)
    with pytest.raises(RuntimeError, match="nothing to time"):
        probe(engine)
    assert [d.stage for d in log.unreliable] == [AB_PROBE]


def test_no_runner_probe_fails_the_ab_rather_than_deciding_on_noise():
    """Through the real applicator: the measure raises, the gate restores, and
    nothing is measured — so no verification record can be written."""
    from gitm.kernels.spec import InterventionSpec
    from gitm.optimizer.apply import LiveEngineApplicator, apply_intervention
    from gitm.scheduler.loop import _engine_throughput_fn

    engine = SimpleNamespace(gitm_llm_kwargs={})
    spec = InterventionSpec.model_validate(dict(
        name="probe_test", summary="s", knob="max_num_seqs", value=64,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="test",
    ))
    with _quiet():
        applicator = LiveEngineApplicator(
            engine, throughput_fn=_engine_throughput_fn(engine, None, DegradationLog()))
        result = apply_intervention(spec, applicator, min_keep_delta=0.0)
    assert result.measured_delta is None
    assert result.error and "nothing to time" in result.error
    assert not result.applied and not result.rolled_back  # never touched the engine


def test_default_probe_refuses_a_restarted_engine():
    from gitm.scheduler.loop import _engine_throughput_fn

    original, restarted = SimpleNamespace(), SimpleNamespace()
    log = DegradationLog()
    probe = _engine_throughput_fn(original, lambda: {"generated_tokens": 10}, log)
    assert probe(original) > 0 and not log
    with _quiet(), pytest.raises(RuntimeError, match="restarted"):
        probe(restarted)
    assert [d.stage for d in log.unreliable] == [AB_PROBE]


def test_restart_ab_under_the_default_probe_is_an_error_not_a_result():
    """Through the real applicator's restart path: the new engine cannot be timed
    by a runner bound to the old one, so the candidate is restored with an error
    and nothing is measured."""
    from gitm.kernels.spec import InterventionSpec
    from gitm.optimizer.apply import LiveEngineApplicator, apply_intervention
    from gitm.scheduler.loop import _engine_throughput_fn

    original = SimpleNamespace(gitm_llm_kwargs={})
    rebuilt = SimpleNamespace(gitm_llm_kwargs={})
    log = DegradationLog()
    spec = InterventionSpec.model_validate(dict(
        name="restart_test", summary="s", knob="max_num_seqs", value=64,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="test",
    ))
    with _quiet():
        applicator = LiveEngineApplicator(
            original,
            throughput_fn=_engine_throughput_fn(original, lambda: {"generated_tokens": 50}, log),
            restart_fn=lambda _old, _values: rebuilt,
            force_restart=True,
        )
        result = apply_intervention(spec, applicator, min_keep_delta=0.0)
    assert result.measured_delta is None
    assert result.error and "restarted" in result.error
    assert [d.stage for d in log.unreliable] == [AB_PROBE]


def test_probe_without_a_token_count_says_runs_per_second():
    from gitm.scheduler.loop import _ab_evidence, _engine_throughput_fn

    engine = SimpleNamespace()
    log = DegradationLog()
    probe = _engine_throughput_fn(engine, lambda: {"something_else": 3}, log)
    with _quiet():
        probe(engine)
        probe(engine)
    assert [d.stage for d in log] == [AB_UNIT]  # once, not once per rep
    ab = SimpleNamespace(speedup=1.1, via="hot-swap", baseline_tps=2.0, candidate_tps=2.2)
    text = _ab_evidence(ab, rolled_back=False, degradations=log)
    assert "runs/s" in text and "tok/s" not in text and "workload throughput" in text
    assert "tok/s" in _ab_evidence(ab, rolled_back=False, degradations=DegradationLog())


# ── the predicted graph's basis ──────────────────────────────────────────────


def _hw():
    from gitm.planner.roofline import HardwareSpec

    return HardwareSpec()


def test_no_engine_graph_says_it_is_the_default():
    from gitm.scheduler.loop import _execution_graph_basis

    _graph, family, why = _execution_graph_basis(None, _hw(), None)
    assert family == "dense" and why == "no engine attached"


def test_unreadable_config_names_the_missing_field():
    from gitm.scheduler.loop import _execution_graph_basis

    hf = SimpleNamespace(hidden_size=512, num_attention_heads=8, num_hidden_layers=4,
                         to_dict=lambda: {"hidden_size": 512})
    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf))
    _g, _f, why = _execution_graph_basis(engine, _hw(), None)
    assert why is not None and "vocab_size" in why


def test_readable_config_is_not_a_degradation():
    from gitm.scheduler.loop import _execution_graph_basis

    hf = SimpleNamespace(hidden_size=512, num_attention_heads=8, num_hidden_layers=4,
                         vocab_size=1000, intermediate_size=2048,
                         to_dict=lambda: {"hidden_size": 512, "vocab_size": 1000})
    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf))
    graph, family, why = _execution_graph_basis(engine, _hw(), None)
    assert why is None and family == "dense"
    assert graph.model.n_layers == 4  # the engine's model, not Llama-2-7B's 32


def test_graph_basis_records_model_hardware_and_batch():
    from gitm.scheduler.loop import _record_graph_basis

    log = DegradationLog()
    with _quiet():
        _record_graph_basis(log, pctx=SimpleNamespace(peak=None, sku="Mystery GPU"),
                            batch=None, sched=None, graph_default_why="no engine attached")
    by = {(d.stage, d.severity) for d in log}
    assert (GRAPH_MODEL, UNRELIABLE) in by
    assert (GRAPH_HARDWARE, APPROXIMATE) in by
    assert (GRAPH_BATCH, APPROXIMATE) in by
    assert any("Mystery GPU" in d.reason for d in log)


# ── history: an unreliable A/B is not evidence ───────────────────────────────


def _export(tmp_path: Path, run_id: str, degradations: list[dict]) -> None:
    d = tmp_path / run_id
    d.mkdir(parents=True)
    (d / "verification.json").write_text(json.dumps({
        "schema": 1,
        "provenance": {"run_id": run_id, "workload_id": "vllm-decode",
                       "fingerprint": "fp", "degradations": degradations},
        "environment": {"gpu_sku": "NVIDIA H100 80GB"},
        "protocol": {},
        "results": [{"intervention_name": "lever", "delta": 0.3, "kept": True,
                     "significant": True, "speedup": 1.3}],
    }))


def test_history_skips_a_run_that_flagged_its_own_ab(tmp_path: Path):
    from gitm.optimizer.history import load_history

    _export(tmp_path, "good", [])
    _export(tmp_path, "bad", [{"stage": AB_PROBE, "severity": UNRELIABLE,
                               "affects": [AFFECTS_AB], "used": "x", "reason": "y"}])
    h = load_history(tmp_path)
    assert "bad" in h.skipped and AB_PROBE in h.skipped["bad"]
    assert "good" not in h.skipped


def test_an_approximate_ab_is_still_read(tmp_path: Path):
    from gitm.optimizer.history import load_history

    _export(tmp_path, "runs-per-s", [{"stage": AB_UNIT, "severity": APPROXIMATE,
                                      "affects": [AFFECTS_AB], "used": "x", "reason": "y"}])
    assert "runs-per-s" not in load_history(tmp_path).skipped


# ── report and verification export ───────────────────────────────────────────


def _provenance(degradations):
    from gitm.optimizer.report import build_provenance

    return build_provenance("vllm-decode", "fp", "run", 0, degradations=degradations)


def test_report_lists_degradations_and_does_not_count_an_unreliable_ab():
    from gitm.optimizer.report import Claim, write_report

    log = DegradationLog([Degradation(AB_PROBE, used="u", reason="r", severity=UNRELIABLE,
                                      affects=(AFFECTS_AB,))])
    claim = Claim(summary="s", residual_invariant="kernel_time", residual_value=0.1,
                  causal_evidence="e", intervention_name="lever", predicted_delta=0.05,
                  measured_delta=0.2)
    md = write_report([claim], _provenance(log))
    assert "## Degradations" in md and AB_PROBE in md
    assert "none counted as verified" in md


def test_a_callers_own_summary_still_carries_the_ab_caveat():
    """The loop passes its own headline on every live run with scheduler samples."""
    from gitm.optimizer.report import write_report

    log = DegradationLog([Degradation(AB_PROBE, used="u", reason="r", severity=UNRELIABLE,
                                      affects=(AFFECTS_AB,))])
    md = write_report([], _provenance(log), summary="vLLM decode on H100: 3 candidates.")
    assert "vLLM decode on H100: 3 candidates." in md
    assert "not counted as verified" in md and AB_PROBE in md
    clean = write_report([], _provenance(DegradationLog()), summary="plain headline.")
    assert "not counted as verified" not in clean


def test_a_clean_report_has_no_degradations_section():
    from gitm.optimizer.report import write_report

    assert "## Degradations" not in write_report([], _provenance(DegradationLog()))


def test_verification_export_names_the_unit_it_measured():
    from gitm.optimizer.verification_export import build_export

    runs = DegradationLog([Degradation(AB_UNIT, used="runs/s", reason="r",
                                       affects=(AFFECTS_AB,))])
    doc = build_export([], _provenance(runs))
    assert "runs/sec" in doc["protocol"]["metric"]
    assert doc["provenance"]["degradations"][0]["stage"] == AB_UNIT
    assert "tokens/sec" in build_export([], _provenance(DegradationLog()))["protocol"]["metric"]


# ── autoresearch contingencies ───────────────────────────────────────────────


def test_version_drift_cuts_the_frozen_catalog_to_the_installed_fields(monkeypatch):
    from gitm.agents import autoresearch as ar

    frozen = [k.name for k in ar._FALLBACK_KNOBS]
    keep = frozen[0]

    # An EngineArgs from a vLLM that kept one of the frozen knobs and dropped the rest.
    FakeEngineArgs = dataclasses.make_dataclass(  # noqa: N806
        "FakeEngineArgs", [("model", str, "m"), (keep, int, 1)])

    def boom(*_a, **_k):
        raise TypeError("annotation drift")

    monkeypatch.setattr(ar, "_knobs_from_engine_args", boom)
    surface = ar.resolve_knobs_from(FakeEngineArgs)
    assert [k.name for k in surface.knobs] == [keep]
    assert surface.source == "frozen:introspection-failed"
    d = surface.degradation
    assert d is not None and d.stage == AR_CATALOG and "annotation drift" in d.reason
    for dropped in frozen[1:]:
        assert dropped in d.used


def test_empty_introspection_is_its_own_reason():
    from gitm.agents import autoresearch as ar

    Only = dataclasses.make_dataclass("Only", [("model", str, "m"), ("seed", int, 0)])  # noqa: N806
    surface = ar.resolve_knobs_from(Only)
    assert surface.degradation is not None
    assert "no tunable knobs" in surface.degradation.reason


def test_live_engineargs_is_not_a_degradation():
    from gitm.agents import autoresearch as ar

    Live = dataclasses.make_dataclass("Live", [("max_num_seqs", int, 256)])  # noqa: N806
    surface = ar.resolve_knobs_from(Live, gpu_count=1)
    assert surface.source == "engineargs" and surface.degradation is None


def test_fallback_proposer_says_which_source_it_used():
    from gitm.agents.autoresearch import FallbackProposer, TableProposer

    class Empty:
        def propose(self, bottleneck_class, *, target_op=None):
            return []

    p = FallbackProposer(Empty(), TableProposer())
    p.propose("memory_bound")
    assert [d.stage for d in p.degradations()] == [AR_PROPOSER]


def test_unscoped_search_is_recorded_but_target_is_kept():
    from gitm.agents.autoresearch import autoresearch
    from gitm.optimizer.apply import DictApplicator
    from gitm.optimizer.monitor import KernelResidual, Residuals

    events = [make_kernel("gemm", start_ns=i * 100, end_ns=i * 100 + 90) for i in range(4)]
    res = Residuals()
    res.per_kernel = [KernelResidual(op="paged_attention", layer=None, r_kt=0.9, r_mt=None)]
    run = autoresearch(make_trace(events=events), applicator=DictApplicator({}), residuals=res)
    assert run.target is not None and run.target.op == "paged_attention"
    assert AR_TARGET in [d.stage for d in run.degradations]


# ── end to end ───────────────────────────────────────────────────────────────


def test_every_run_writes_degradations_and_summarises_them(tmp_path: Path):
    from gitm import optimize

    with _quiet():
        result = optimize(workload="vllm-decode", budget="1s", target=0.15,
                          scratch=str(tmp_path))
    summary = result["summary"]
    doc = json.loads((Path(result["run_dir"]) / FILE_NAME).read_text())
    assert summary["degradations"]["n"] == len(doc["items"])
    # This box has no runner for vllm-decode, and the run says so.
    assert WORKLOAD_RUNNER in summary["degradations"]["unreliable"]
    assert summary["degraded"] is True


def test_cli_echoes_a_degraded_run(capsys):
    from gitm.cli import _warn_degraded

    _warn_degraded({"degradations": {"n": 1, "unreliable": [GRAPH_MODEL],
                                     "approximate": []}}, "/runs/x")
    err = capsys.readouterr().err
    assert "unreliable: graph.model" in err and "/runs/x/degradations.json" in err
    _warn_degraded({"degradations": {"n": 0}}, "/runs/x")
    assert capsys.readouterr().err == ""
