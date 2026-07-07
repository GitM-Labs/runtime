"""The 24-hour autonomous loop.

This is the orchestration glue — it composes tracer, planner, optimizer,
kernels, and agents in the 5 phases below. Each phase writes its artifact
to local scratch under ``<scratch>/runs/<run_id>/`` (see ``gitm._paths``) so a
partial run is still useful; the durable copy is synced to S3 afterwards.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gitm._paths import runs_dir, traces_dir
from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.library import load_library
from gitm.optimizer.apply import (
    Applicator,
    DryRunApplicator,
    LiveEngineApplicator,
    apply_intervention,
)
from gitm.optimizer.attribution import attribute
from gitm.optimizer.deviation import deviation_summary, deviation_trace, write_deviation_jsonl
from gitm.optimizer.dr import attribute_dr
from gitm.optimizer.measure import measure_trace, measurement_claims, measurement_summary
from gitm.optimizer.monitor import check_invariants, residuals
from gitm.optimizer.qualification import qualify
from gitm.optimizer.report import Claim, build_provenance, write_report
from gitm.optimizer.scheduler_attribution import scheduler_causes
from gitm.optimizer.vllm_knobs import knob_kind
from gitm.planner.context import build_planner_context
from gitm.planner.graph import predict_graph
from gitm.safety.audit import AuditLog, _write_report
from gitm.tracer.capture import capture
from gitm.tracer.vllm_stats import sample_scheduler_stats
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

def _engine_throughput_fn(engine: Any, runner: Any) -> Any:
    """Resolve a decode-throughput probe for the live A/B.

    Prefers an explicit ``engine.gitm_throughput_fn`` (the engine owns what "a
    decode" means); otherwise times the workload ``runner`` and divides generated
    tokens by elapsed seconds. Re-running the runner re-runs the decode under the
    engine's current config, so an in-place hot-swap is reflected in the measurement.

    Contract: this default probe re-runs the (potentially expensive) full workload
    and is bound to the *original* engine, so it is only valid for in-place
    hot-swap A/Bs. A deployment that supplies ``gitm_restart_fn`` (structural-knob
    restart-apply, which swaps in a *new* engine) MUST also supply an engine-aware
    ``gitm_throughput_fn`` — the default cannot measure the restarted engine.
    """
    explicit = getattr(engine, "gitm_throughput_fn", None)
    if callable(explicit):
        return explicit

    def _tps(_engine: Any) -> float:
        t0 = time.perf_counter()
        out = runner() if runner is not None else {}
        dt = max(time.perf_counter() - t0, 1e-9)
        # First key that is actually present wins — `or` would treat a legitimate
        # 0 (a window that produced no tokens) as missing and fabricate a count.
        toks: float = 1.0
        if isinstance(out, dict):
            for key in ("generated_tokens", "decode_steps", "events"):
                if out.get(key) is not None:
                    toks = float(out[key])
                    break
        return toks / dt

    return _tps


def _scheduler_note(s: Any) -> str | None:
    """One-line scheduler-stats sentence for the report, or None if no samples.

    ``s`` is a :class:`gitm.tracer.vllm_stats.SchedulerStatsSummary`; read
    duck-typed so an empty/absent summary degrades to no note rather than a crash.
    """
    if s is None or getattr(s, "n_samples", 0) == 0:
        return None
    parts: list[str] = []
    if s.peak_queue_depth is not None:
        parts.append(f"peak queue depth {s.peak_queue_depth}")
    if s.mean_batch_occupancy is not None:
        parts.append(f"mean batch occupancy {s.mean_batch_occupancy:.0%}")
    if s.total_preemptions is not None:
        parts.append(f"{s.total_preemptions} preemption(s)")
    if s.peak_gpu_cache_usage is not None:
        parts.append(f"peak KV-cache {s.peak_gpu_cache_usage:.0%}")
    if not parts:
        return None
    return "Engine scheduler: " + ", ".join(parts) + f" (over {s.n_samples} samples)."


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


def _model_spec_from_engine(engine: Any):
    """Build a ``ModelSpec`` from a live vLLM engine's HF config, or ``None``.

    ``predict_graph()`` with no model defaults to Llama-2-7B (32 layers). A run
    of a *different* model (e.g. opt-125m, 12 layers) is then scored against the
    wrong predicted graph, which makes residuals and deviation meaningless. When
    the loop has the live engine, read the real architecture off its HF config so
    the predicted graph matches the model that actually ran. Duck-typed across
    vLLM version drift; any failure returns ``None`` and the caller falls back to
    the default graph rather than crashing.
    """
    if engine is None:
        return None
    hf: Any = None
    for path in (
        "llm_engine.model_config.hf_config",
        "llm_engine.vllm_config.model_config.hf_config",
        "model_config.hf_config",
    ):
        obj: Any = engine
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            hf = obj
            break
    if hf is None:
        return None
    try:
        from gitm.planner.roofline import ModelSpec

        hidden = int(hf.hidden_size)
        n_heads = int(hf.num_attention_heads)
        n_kv = int(getattr(hf, "num_key_value_heads", n_heads) or n_heads)
        head_dim = int(getattr(hf, "head_dim", 0) or (hidden // n_heads))
        return ModelSpec(
            hidden=hidden,
            n_layers=int(hf.num_hidden_layers),
            n_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate=int(getattr(hf, "intermediate_size", 4 * hidden)),
            vocab=int(hf.vocab_size),
        )
    except Exception:
        return None


def _agg_kt_residual(res: Any) -> float:
    """Aggregate kernel-time residual for the report — the median of per-kernel
    ``r_kt`` (observed-vs-predicted kernel time). Median (not mean) so a few
    badly-aligned kernels don't dominate; ``0.0`` when there are no residuals.
    """
    kts = sorted(kr.r_kt for kr in res.per_kernel)
    if not kts:
        return 0.0
    mid = len(kts) // 2
    return kts[mid] if len(kts) % 2 else (kts[mid - 1] + kts[mid]) / 2.0


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

    # If the workload built a live engine (e.g. vLLM), expose it so the
    # scheduler-stats sampler AND the Phase-4 live A/B can drive it. The runner
    # carries it as ``.engine`` (see the vllm-decode factory). Without this the
    # loop stays predict-only (DryRunApplicator, live=False) — the engine is
    # built but never handed to the applicator.
    if cfg.engine is None and runner is not None:
        cfg.engine = getattr(runner, "engine", None)

    # Sample the engine scheduler (queue depth, batch occupancy, preemptions)
    # over the same window as the CUPTI capture — engine-level telemetry the GPU
    # trace can't see. A no-op when no engine is attached (empty series).
    with (
        capture(trace_path, workload_id=workload, run_id=run_id) as trace,
        sample_scheduler_stats(cfg.engine) as sched_stats,
    ):
        if runner is not None:
            try:
                runner()
                sync_device()  # ensure all kernels land in the trace before stop
            except Exception as exc:
                runner_error = f"workload run failed: {exc}"

    # Persist the scheduler series + summary when an engine actually produced one.
    # Turn the summary into ranked causal hypotheses (feeds attribution / claim
    # evidence below) — empty when no engine produced samples.
    sched_summary = sched_stats.summary()
    sched_causes = scheduler_causes(sched_summary)
    if sched_stats.samples:
        (run_dir / "scheduler_stats.json").write_text(
            json.dumps(
                {"summary": asdict(sched_summary), "samples": sched_stats.to_records()},
                indent=2,
            )
        )

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

    # Predict against the model that ACTUALLY ran (read from the live engine),
    # not the Llama-2-7B default — otherwise residuals/deviation score the real
    # kernels against the wrong graph. Falls back to the default graph when there
    # is no engine or its config can't be read (CPU boxes, tests, dry-run).
    _spec = _model_spec_from_engine(cfg.engine)
    graph = predict_graph(model=_spec) if _spec is not None else predict_graph()
    (run_dir / "predicted_graph.json").write_text(
        json.dumps({"nodes": len(graph.nodes), "total_pred_s": graph.total_pred_s}, indent=2)
    )

    # Phase 2 — residuals + attribution
    res = residuals(trace, graph)
    violations = check_invariants(res)  # multi-basis confirmed
    hypotheses = attribute(res, graph)  # Granger
    dr_hypotheses = attribute_dr(res, graph)  # doubly-robust, corroborating

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
                # Engine-scheduler causes (from the vLLM stats adapter) ranked
                # alongside the kernel-level hypotheses (the engine-signal causal link).
                "scheduler_causes": [
                    {"signal": c.signal, "effect": c.effect, "severity": c.severity,
                     "note": c.note, "motivates_knobs": c.motivates_knobs}
                    for c in sched_causes
                ],
            },
            indent=2,
        )
    )

    # Deviation-only tracing: record only the kernels that *departed* from the
    # predicted graph — trace storage scales with deviation, not duration. We
    # always write the compact summary (n_kept, reduction, which ops departed);
    # the full reduced JSONL is written only under GITM_DEVIATION_ONLY=1 (it is
    # the storage-saving artifact, off by default while capture-time integration
    # is still on the roadmap).
    (run_dir / "deviations.json").write_text(
        json.dumps(deviation_summary(trace, graph), indent=2)
    )
    if os.environ.get("GITM_DEVIATION_ONLY") == "1":
        write_deviation_jsonl(deviation_trace(trace, graph), run_dir / "deviation_trace.jsonl")

    # Phase 3 — library + counterfactual replay ranking
    pctx = build_planner_context(cfg.engine, workload = workload)
    library = load_library(workload = workload)
    policy = Policy(require_qualification_commit=qual.commit, skip_high_risk=not qual.commit)
    ranked = select_interventions(trace, library, policy, top_n=cfg.top_n_interventions, ctx=pctx.gate)
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

    # Phase 4 — apply with rollback gates.
    # With a live engine attached, each candidate runs the rollback-gated decode-
    # throughput A/B (LiveEngineApplicator): snapshot baseline tps → hot-swap the
    # knob → measure candidate tps → keep only on a non-negative delta, else
    # restore. Scheduling knobs are hot-swapped; structural knobs are routed
    # through ``engine.gitm_restart_fn`` (if the deployment provides one) or
    # rolled back rather than silently no-op'd. With no engine it's predict-only
    # (DryRunApplicator): candidates land in the report as unverified
    # (measured_delta=None), never claimed as won.
    live_restart_fn = getattr(cfg.engine, "gitm_restart_fn", None) if cfg.engine else None
    if cfg.engine is not None:
        applicator: Applicator = LiveEngineApplicator(
            cfg.engine,
            throughput_fn=_engine_throughput_fn(cfg.engine, runner),
            restart_fn=live_restart_fn,
            reps=int(os.environ.get("GITM_AB_REPS", "1")),
            # GITM_KNOBS_VIA_RESTART=1 applies scheduling knobs via engine rebuild
            # too — for V1, which ignores a live scheduler-config mutation.
            force_restart=os.environ.get("GITM_KNOBS_VIA_RESTART") == "1",
        )
    else:
        applicator = DryRunApplicator()

    claims: list[Claim] = []
    rolled_back: list[str] = []
    rejected: list[str] = []
    # Aggregate kernel-time residual for the report (was hardcoded 0.0). Same for
    # every claim in a run — it describes the run's gap vs the predicted graph.
    kt_residual = _agg_kt_residual(res)
    for c in ranked:
        if c.rejected_reason is not None:
            rejected.append(f"{c.spec.name} ({c.rejected_reason})")
            continue
        # Live + structural knob + no restart hook → it *cannot* be enacted on the
        # running engine, so it's "not evaluable here", not a regression. Mark it
        # rejected (honest) instead of attempting an apply that would roll back and
        # read as "tried and lost" — and skip the wasted baseline benchmark.
        if cfg.engine is not None and live_restart_fn is None and knob_kind(c.spec.knob) == "structural":
            rejected.append(f"{c.spec.name} (structural knob: needs engine restart, no restart_fn)")
            continue
        result = apply_intervention(c.spec, applicator, min_keep_delta=0.0)
        ab = getattr(applicator, "last_result", None)
        if result.rolled_back:
            rolled_back.append(c.spec.name)
        # Causal evidence: the measured A/B verdict when live, else the Granger
        # signal that motivated the candidate. The kept/rolled-back wording comes
        # from the authoritative ApplyResult (the real gate decision), not from
        # EngineABResult.kept (a measure-time delta>=0 indicator).
        if ab is not None:
            outcome = "rolled back" if result.rolled_back else "kept"
            causal_evidence = (
                f"live A/B: {outcome} ({ab.speedup - 1.0:+.1%} decode throughput, via {ab.via}); "
                f"baseline {ab.baseline_tps:.1f} → candidate {ab.candidate_tps:.1f} tok/s"
            )
        else:
            causal_evidence = ", ".join(
                f"{h.cause_op}→{h.effect_op} (p={h.p_value:.2g})" for h in hypotheses.top(2)
            ) or "no strong causal signal"
        # Attach the top scheduler cause that argues for *this* knob — the
        # engine-level signal (C) tied to the specific lever it motivates (B).
        motivating = next((sc for sc in sched_causes if c.spec.knob in sc.motivates_knobs), None)
        if motivating is not None:
            causal_evidence += f"; scheduler[{motivating.signal}]: {motivating.note}"
        claims.append(
            Claim(
                summary=c.spec.summary,
                residual_invariant="kernel_time",
                residual_value=kt_residual,
                causal_evidence=causal_evidence,
                intervention_name=c.spec.name,
                predicted_delta=c.predicted_delta,
                # Display the TRUE measured delta (speedup-1); the gate uses the noise-adjusted
                # return, so a within-noise gain reads as rolled back
                # with its real (small) number, not a distorted one.
                measured_delta=((ab.speedup - 1.0) if ab is not None else result.measured_delta),
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

    sched_note = _scheduler_note(sched_summary)
    report_md = write_report(
        claims=claims,
        provenance=provenance,
        qualification_diagnostic=qual.diagnostic,
        summary=(
            f"vLLM decode on {pctx.sku or 'unknown SKU'}: {len(claims)} candidate(s) "
            f"evaluated, {len(rolled_back)} rolled back. {sched_note}"
            if sched_note
            else None
        ),
    )
    _write_report(run_dir, report_md)

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
        "scheduler_stats": asdict(sched_summary) if sched_stats.samples else None,
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
    _write_report(run_dir, report_md)

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
    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
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
    _write_report(run_dir, report_md)

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

    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
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
    _write_report(run_dir, report_md)

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

    apply_res = apply_intervention(
        spec, applicator, min_keep_delta=0.0, audit=AuditLog(run_dir / "audit.jsonl")
    )
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
    _write_report(run_dir, report_md)

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
    _write_report(run_dir, report_md)

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
