"""The 24-hour autonomous loop.

This is the orchestration glue — it composes tracer, planner, optimizer,
kernels, and agents in the 5 phases below. Each phase writes its artifact
to local scratch under ``<scratch>/runs/<run_id>/`` (see ``gitm._paths``) so a
partial run is still useful; the durable copy is synced to S3 afterwards.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitm._paths import runs_dir, traces_dir
from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library
from gitm.optimizer.apply import DryRunApplicator, apply_intervention
from gitm.optimizer.attribution import attribute
from gitm.optimizer.dr import attribute_dr
from gitm.optimizer.measure import measure_trace, measurement_claims, measurement_summary
from gitm.optimizer.monitor import check_invariants, residuals
from gitm.optimizer.qualification import qualify
from gitm.optimizer.report import Claim, build_provenance, write_report
from gitm.planner.graph import predict_graph
from gitm.tracer.capture import capture
from gitm.workloads import WorkloadRunner, get_factory, sync_device

_BUDGET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$")

# Workloads the predicted graph + intervention library actually model. Anything
# else gets a measurement-only report (see _measurement_result) rather than
# vLLM-specific intervention claims that wouldn't apply.
_LIBRARY_WORKLOADS = {"vllm-decode"}

# Workloads with a real, output-verified intervention applied through the
# rollback gate (not the vLLM library). Their runner carries an ``.applicator``
# (see gitm.workloads) so the loop can observe → attribute → select → apply →
# prove with a *measured* delta instead of a measurement-only report.
_HFT_INTERVENTION_WORKLOADS = {"hft", "hft-lob"}

# OpenFold/AF2 has a real, plDDT-gated intervention (bf16 inference) applied
# through the same rollback gate. Its runner carries an ``.applicator``.
_OPENFOLD_INTERVENTION_WORKLOADS = {"openfold", "alphafold", "af2"}

# Edge (3D LiDAR detection) has a real, detection-equivalence-gated intervention
# (fp16 autocast inference) applied through the same rollback gate. Its runner
# carries an ``.applicator``.
_EDGE_INTERVENTION_WORKLOADS = {"edge", "kitti", "nuscenes"}


def _parse_budget_s(budget: str) -> float:
    m = _BUDGET_RE.match(budget.lower())
    if not m:
        raise ValueError(f"unparseable budget: {budget!r} (use 24h, 90m, 3600s, 1d)")
    value, unit = float(m.group(1)), m.group(2)
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]


@dataclass
class LoopConfig:
    engine: Any | None = None
    workload: str | None = None
    budget: str = "24h"
    target: float = 0.15
    scratch: str | None = None
    top_n_interventions: int = 5
    # Optional explicit driver for the embedded/engine path. When unset, the
    # loop looks up ``workload`` in the workload registry (gitm.workloads).
    workload_runner: WorkloadRunner | None = None


def run_loop(cfg: LoopConfig) -> dict[str, Any]:
    """Execute the 24-hour loop and return ``{summary, report_md, ...}``."""
    workload = cfg.workload or (getattr(cfg.engine, "workload_id", None) or "vllm-decode")
    run_id = uuid.uuid4().hex
    budget_s = _parse_budget_s(cfg.budget)
    started_ns = time.time_ns()

    run_dir = runs_dir(cfg.scratch) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir(cfg.scratch) / f"{run_id}.jsonl"

    # Phase 1 — capture, fingerprint, predict graph
    # Resolve a workload runner: an explicit one wins, else the registry. The
    # runner launches GPU work *inside* the capture window so the trace reflects
    # the real workload instead of an empty no-op. Resolution happens outside
    # capture (data loading / warmup shouldn't be traced).
    runner = cfg.workload_runner
    runner_error: str | None = None
    if runner is None:
        factory = get_factory(workload)
        if factory is not None:
            try:
                runner = factory(cfg)
            except Exception as exc:  # missing deps/data on this box — degrade, don't crash
                runner_error = f"workload runner unavailable for {workload!r}: {exc}"
        else:
            runner_error = f"no workload runner registered for {workload!r}"

    with capture(trace_path, workload_id=workload, run_id=run_id) as trace:
        if runner is not None:
            try:
                runner()
                sync_device()  # ensure all kernels land in the trace before stop
            except Exception as exc:
                runner_error = f"workload run failed: {exc}"

    qual = qualify(trace, target_floor=cfg.target)
    (run_dir / "qualification.json").write_text(
        json.dumps(
            {
                "commit": qual.commit,
                "floor": qual.floor,
                "fingerprint": qual.fingerprint,
                "diagnostic": qual.diagnostic,
            },
            indent=2,
        )
    )

    # HFT carries a real, output-verified intervention on its runner. Apply+prove
    # it through the rollback gate — the A/B runs on the active backend, so the
    # delta is measured even on a box without CUPTI. (Runs before the empty-trace
    # guard for that reason; attribution below is included only if kernels exist.)
    if workload in _HFT_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _hft_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # OpenFold/AF2 carries the bf16 intervention on its runner. Same pattern as
    # HFT: apply+prove through the rollback gate (measure() runs the fp32-vs-bf16
    # A/B, gated on plDDT-equivalence). Before the empty-trace guard so the A/B
    # still runs on a box without CUPTI; attribution is included if kernels exist.
    if workload in _OPENFOLD_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _openfold_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # Edge (kitti/nuscenes) carries the fp16 intervention on its runner. Same
    # pattern as HFT/AF2: apply+prove through the rollback gate (measure() runs
    # the fp32-vs-fp16 A/B, gated on detection-equivalence). Before the empty-
    # trace guard so the A/B still runs on a box without CUPTI.
    if workload in _EDGE_INTERVENTION_WORKLOADS:
        applicator = getattr(runner, "applicator", None)
        if applicator is not None:
            return _edge_intervention_result(
                run_dir=run_dir,
                run_id=run_id,
                workload=workload,
                trace=trace,
                qual=qual,
                applicator=applicator,
                started_ns=started_ns,
                trace_path=trace_path,
            )

    # Guard: if the tracer captured nothing (no GPU/shim, or the workload never
    # ran), do NOT proceed to attribution + emit claims — that fabricates a
    # result from an empty trace. Report no-data honestly instead.
    if trace.vendor == "none" or not trace.kernels():
        diagnostic = runner_error or qual.diagnostic or (
            "Tracer captured no GPU kernels. Either no GPU/CUPTI shim is present, "
            "or the workload did not run under the runtime."
        )
        return _no_data_result(
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
            diagnostic=diagnostic,
        )

    # The predicted graph + intervention library model vLLM decode specifically.
    # For any other workload, pairing the real trace with that transformer graph
    # produces vLLM serving-knob "claims" that don't apply. Instead, emit an
    # honest measurement report computed from the actual captured kernels.
    if workload not in _LIBRARY_WORKLOADS:
        return _measurement_result(
            run_dir=run_dir,
            run_id=run_id,
            workload=workload,
            trace=trace,
            qual=qual,
            started_ns=started_ns,
            trace_path=trace_path,
        )

    graph = predict_graph()
    (run_dir / "predicted_graph.json").write_text(
        json.dumps({"nodes": len(graph.nodes), "total_pred_s": graph.total_pred_s}, indent=2)
    )

    # Phase 2 — residuals + attribution
    res = residuals(trace, graph)
    violations = check_invariants(res)  # multi-basis confirmed (GITM-008)
    hypotheses = attribute(res, graph)  # Granger
    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating (GITM-008)

    (run_dir / "violations.json").write_text(
        json.dumps(
            [
                {
                    "invariant": v.invariant,
                    "node_op": v.node_op,
                    "layer": v.layer,
                    "residual": v.residual,
                    "severity": v.severity,
                }
                for v in violations
            ],
            indent=2,
        )
    )
    (run_dir / "residuals.json").write_text(
        json.dumps(
            {
                "n_kernel_residuals": len(res.per_kernel),
                "n_violations": len(violations),
                "serialized_concurrency_fraction": res.serialized_concurrency_fraction,
                "top_hypotheses_granger": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value}
                    for h in hypotheses.top(5)
                ],
                "top_hypotheses_doubly_robust": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value,
                     "notes": h.notes}
                    for h in dr_hypotheses.top(5)
                ],
            },
            indent=2,
        )
    )

    # Phase 3 — library + counterfactual replay ranking
    library = load_library()
    policy = Policy(require_qualification_commit=qual.commit, skip_high_risk=not qual.commit)
    ranked = select_interventions(trace, library, policy, top_n=cfg.top_n_interventions)
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [
                {
                    "name": c.spec.name,
                    "predicted_delta": c.predicted_delta,
                    "rejected_reason": c.rejected_reason,
                }
                for c in ranked
            ],
            indent=2,
        )
    )

    # Phase 4 — apply with rollback gates
    claims: list[Claim] = []
    rolled_back: list[str] = []
    rejected: list[str] = []
    for c in ranked:
        if c.rejected_reason is not None:
            rejected.append(f"{c.spec.name} ({c.rejected_reason})")
            continue
        # W1 skeleton: no live engine attached -> predict-only, unverified claims.
        # A live run passes an engine applicator here (GITM-020).
        result = apply_intervention(c.spec, DryRunApplicator())
        if result.rolled_back:
            rolled_back.append(c.spec.name)
        claims.append(
            Claim(
                summary=c.spec.summary,
                residual_invariant="kernel_time",
                residual_value=0.0,
                causal_evidence=", ".join(
                    f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
                )
                or "no strong causal signal",
                intervention_name=c.spec.name,
                predicted_delta=c.predicted_delta,
                measured_delta=result.measured_delta,
                rolled_back=result.rolled_back,
            )
        )
        if time.time_ns() - started_ns >= int(budget_s * 1e9):
            break

    # Phase 5 — stabilize + write report
    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rejected_candidates = rejected
    provenance.rolled_back = rolled_back

    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": len(rejected),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _measurement_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Honest measurement report for a workload with no intervention library.

    Computes residuals/attribution from the *actual* captured kernels and emits
    observations (not optimization claims) — so an HFT or edge run describes its
    real cuDF/CUB kernels instead of fabricating vLLM serving-knob claims.
    """
    result = measure_trace(trace)
    claims = measurement_claims(result)

    (run_dir / "measurement.json").write_text(
        json.dumps(
            {
                "n_kernels": result.n_kernels,
                "n_memcpy": result.n_memcpy,
                "serialized_concurrency_fraction": result.serialized_fraction,
                "n_violations": len(result.violations),
                "families": result.families,
                "top_hypotheses": [
                    {"cause": h.cause_op, "effect": h.effect_op, "p_value": h.p_value}
                    for h in result.top_hypotheses
                ],
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=(
            "Measurement-only run: the runtime observed the workload and reports "
            "its real kernels. No intervention library applies to this workload."
        ),
        summary=measurement_summary(workload, result),
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "measurement",
        "fingerprint": qual.fingerprint,
        "commit": False,
        "floor": qual.floor,
        "n_observations": len(claims),
        "n_claims": 0,
        "n_rolled_back": 0,
        "n_rejected": 0,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _hft_intervention_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for HFT.

    Reuses the real pieces: attribution from the captured kernels
    (:func:`measure_trace`), the curated lever (:func:`hft_intervention_spec`)
    ranked by counterfactual replay (:func:`predict_delta`), and the rollback
    gate (:func:`apply_intervention`) whose measure runs the output-verified A/B.
    The claim's ``measured_delta`` is the A/B speedup — a real number, gated on
    byte-identical output, so a wrong or slower candidate is rolled back.
    """
    from gitm.benchmarks.hft.optimize import hft_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    # Attribute: residuals → invariants → Granger over the actual kernels. Empty
    # when no CUPTI trace was captured (CPU box) — the apply+prove still runs.
    mres = measure_trace(trace)

    # Select: the one curated HFT lever, ranked by predicted delta on this trace.
    spec = hft_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    # Apply behind the rollback gate — measure() runs the verified baseline-vs-
    # candidate A/B and returns the signed speedup (raises → rollback if output
    # diverges; negative delta → rollback if slower).
    apply_res = apply_intervention(spec, applicator, min_keep_delta=0.0)
    ab = applicator.last_result

    # Prove: one claim carrying the measured delta, gated on identical output.
    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend baseline-vs-candidate A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="stream_concurrency",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                predicted_delta=predicted,
                measured_delta=(ab.speedup - 1.0) if ab.identical else None,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "identical_output": getattr(ab, "identical", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_events_per_second": getattr(ab, "baseline_eps", None),
                "candidate_events_per_second": getattr(ab, "candidate_eps", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"HFT intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _openfold_intervention_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for AF2 (OpenFold).

    Mirrors :func:`_hft_intervention_result` but the gate is plDDT-equivalence,
    not byte-identical output: the applicator's measure() runs the fp32-vs-bf16
    A/B and keeps bf16 only if median plDDT stays within tolerance AND it is
    faster, else rolls back to fp32. The claim's ``measured_delta`` is the
    measured speedup, so a quality regression is never reported as a win.
    """
    from benchmarks.biotech.optimize import openfold_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    mres = measure_trace(trace)

    spec = openfold_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    apply_res = apply_intervention(spec, applicator, min_keep_delta=0.0)
    ab = applicator.last_result  # AF2ABResult

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-bf16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="kernel_time",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                # plDDT-equivalence is the AF2 correctness gate (vs byte-identical).
                measured_delta=(ab.speedup - 1.0) if ab.equivalent else None,
                predicted_delta=predicted,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "plddt_equivalent": getattr(ab, "equivalent", None),
                "plddt_delta": getattr(ab, "plddt_delta", None),
                "plddt_tol": getattr(ab, "plddt_tol", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_structures_per_hour": getattr(ab, "baseline_sph", None),
                "candidate_structures_per_hour": getattr(ab, "candidate_sph", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"AF2 intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _edge_intervention_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    trace: Any,
    qual: Any,
    applicator: Any,
    started_ns: int,
    trace_path: Path,
) -> dict[str, Any]:
    """Full observe → attribute → select → apply → prove for edge (kitti/nuscenes).

    Mirrors :func:`_hft_intervention_result`/:func:`_openfold_intervention_result`
    but the gate is detection-equivalence (count + sorted scores within
    tolerance), not byte-identical output: the applicator's measure() runs the
    fp32-vs-fp16 A/B and keeps fp16 only if detections stay equivalent AND it is
    faster, else rolls back to fp32. The claim's ``measured_delta`` is the
    measured speedup, so a detection regression is never reported as a win.
    """
    from gitm.benchmarks.edge.optimize import edge_intervention_spec
    from gitm.optimizer.apply import apply_intervention
    from gitm.optimizer.replay import predict_delta

    mres = measure_trace(trace)

    # The applicator carries its own spec; fall back to the module factory.
    spec = getattr(applicator, "spec", None) or edge_intervention_spec()
    predicted = predict_delta(trace, spec) if trace.kernels() else spec.expected_delta_mean
    (run_dir / "ranked_candidates.json").write_text(
        json.dumps(
            [{"name": spec.name, "predicted_delta": predicted, "rejected_reason": None}],
            indent=2,
        )
    )

    apply_res = apply_intervention(spec, applicator, min_keep_delta=0.0)
    ab = applicator.last_result  # EdgeABResult

    top = mres.top_hypotheses
    if top:
        evidence = (
            f"top hypothesis: {top[0].cause_op[:30]} → {top[0].effect_op[:30]} "
            f"(p={top[0].p_value:.3g}); serialized-concurrency={mres.serialized_fraction:.3f}"
        )
    elif mres.n_kernels:
        evidence = (
            f"serialized-concurrency={mres.serialized_fraction:.3f} over "
            f"{mres.n_kernels} kernels"
        )
    else:
        evidence = (
            "no CUPTI trace captured on this box; intervention proven by the "
            "on-backend fp32-vs-fp16 A/B"
        )

    claims: list[Claim] = []
    rolled_back: list[str] = []
    if ab is not None:
        claims.append(
            Claim(
                summary=spec.summary,
                residual_invariant="kernel_time",
                residual_value=float(mres.serialized_fraction),
                causal_evidence=evidence,
                intervention_name=spec.name,
                # detection-equivalence is the edge correctness gate.
                measured_delta=(ab.speedup - 1.0) if ab.identical else None,
                predicted_delta=predicted,
                rolled_back=apply_res.rolled_back,
            )
        )
        if apply_res.rolled_back:
            rolled_back.append(spec.name)

    (run_dir / "apply_result.json").write_text(
        json.dumps(
            {
                "intervention": spec.name,
                "applied": apply_res.applied,
                "rolled_back": apply_res.rolled_back,
                "measured_delta": apply_res.measured_delta,
                "error": apply_res.error,
                "detections_equivalent": getattr(ab, "identical", None),
                "kept": getattr(ab, "kept", None),
                "verdict": getattr(ab, "verdict", None),
                "baseline_frames_per_second": getattr(ab, "baseline_eps", None),
                "candidate_frames_per_second": getattr(ab, "candidate_eps", None),
                "speedup": getattr(ab, "speedup", None),
                "serialized_concurrency_fraction": mres.serialized_fraction,
                "families": mres.families,
            },
            indent=2,
        )
    )

    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    provenance.rolled_back = rolled_back
    verdict = getattr(ab, "verdict", "no A/B result")
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"edge intervention {spec.name!r}: {verdict}. "
            f"{mres.n_kernels:,} kernels observed, serialized-concurrency="
            f"{mres.serialized_fraction:.3f}."
        ),
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "ok",
        "mode": "intervention",
        "fingerprint": qual.fingerprint,
        "commit": qual.commit,
        "floor": qual.floor,
        "n_claims": len(claims),
        "n_rolled_back": len(rolled_back),
        "n_rejected": 0,
        "speedup": getattr(ab, "speedup", None),
        "kept": getattr(ab, "kept", None),
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}


def _no_data_result(
    *,
    run_dir: Path,
    run_id: str,
    workload: str,
    qual: Any,
    started_ns: int,
    trace_path: Path,
    diagnostic: str,
) -> dict[str, Any]:
    """Write an honest no-data report and return its summary (status=no_data).

    Used when the trace has no kernels — a misconfigured box or a workload that
    never ran. We emit zero claims rather than fabricating results from nothing.
    """
    provenance = build_provenance(
        workload_id=workload,
        fingerprint=qual.fingerprint,
        run_id=run_id,
        started_at_ns=started_ns,
        trace_path=str(trace_path),
    )
    report_md = write_report(
        claims=[],
        provenance=provenance,
        qualification_diagnostic=diagnostic,
        summary="NO DATA — tracer captured no GPU kernels; nothing was measured.",
    )
    (run_dir / "report.md").write_text(report_md)

    summary = {
        "run_id": run_id,
        "workload": workload,
        "status": "no_data",
        "fingerprint": qual.fingerprint,
        "commit": False,
        "floor": qual.floor,
        "n_claims": 0,
        "n_rolled_back": 0,
        "n_rejected": 0,
        "diagnostic": diagnostic,
        "report_path": str(run_dir / "report.md"),
    }
    return {"summary": summary, "report_md": report_md, "run_dir": str(run_dir)}
