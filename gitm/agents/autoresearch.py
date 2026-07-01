"""Autoresearch — propose non-catalog levers within the attributed bottleneck class.

The curated library (``library.yaml``) is finite. Autoresearch is the agentic
half of the README's "select from a library of known optimizations *and* run
agentic search for novel ones": it proposes real vLLM config knobs *outside*
that catalog, constrained to the bottleneck class the attribution layer
identified (idle / memory / compute).

Every proposal is then routed through the exact same path as a catalog lever:

1. the selection gate — :func:`gitm.agents.policy.select_interventions` — which
   pre-filters on the safety tier and qualification commit, then ranks the
   survivors by counterfactual replay (:func:`gitm.optimizer.replay.predict_delta`);
2. the rollback-gated live apply — :func:`gitm.optimizer.apply.apply_intervention` —
   which snapshots, applies, measures, and keeps only on a measured win.

A proposal the gate rejects is recorded and dropped; one that applies but does
not measurably help is rolled back. Autoresearch is a *candidate source*, not a
new trust path — nothing it proposes can bypass the gate or be kept without a
measured win.

The proposed knobs are real, current vLLM arguments (verified against
docs.vllm.ai); their expected deltas, however, are unproven estimates. The
``source`` field says so, and only the measured A/B keeps or discards them.

v0 classifies the bottleneck from coarse trace telemetry (:func:`classify_bottleneck`)
and emits a small, fixed candidate table per class. Later versions repoint at
the largest *measured* residual and learn an effect model instead of a static
table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.apply import Applicator, apply_intervention
from gitm.optimizer.monitor import _serialized_fraction
from gitm.tracer.schema import Trace

if TYPE_CHECKING:
    from gitm.safety.audit import AuditLog

# --- bottleneck classification ----------------------------------------------
#
# The attribution vocabulary autoresearch searches within. Nothing upstream
# emits these labels yet, so v0 derives them from coarse trace telemetry. These
# are deliberately simple heuristics, not a tuned model — the thresholds only
# have to route the search into the right candidate table; the rollback gate is
# what actually protects a wrong route (a bad proposal is measured and reverted).

#: Serialized-concurrency fraction above this ⇒ kernels ran back-to-back on one
#: stream instead of overlapping: scheduling gaps / launch-bound idle time.
_SC_THRESHOLD = 0.5
#: memcpy share of GPU operations above this ⇒ data movement dominates.
_MEMCPY_THRESHOLD = 0.25


def classify_bottleneck(trace: Trace) -> str:
    """Map a captured trace to one of ``idle_stall`` / ``memory_bound`` / ``compute_bound``.

    v0 heuristic on two signals read straight from the trace: the
    serialized-concurrency fraction (poor kernel overlap ⇒ idle/scheduling gaps)
    and the memcpy fraction (data movement dominating ⇒ memory bound). Each is
    scored against its threshold; the stronger signal wins, and if neither
    crosses its threshold the workload is treated as compute bound. An empty
    trace has no stall/movement signal, so it defaults to ``compute_bound``.
    """
    kernels = trace.kernels()
    if not kernels:
        return "compute_bound"

    memcpys = [e for e in trace.events if e.kind == "memcpy"]
    sc = _serialized_fraction(kernels)
    memcpy_frac = len(memcpys) / (len(memcpys) + len(kernels))

    sc_score = sc / _SC_THRESHOLD
    mem_score = memcpy_frac / _MEMCPY_THRESHOLD
    if max(sc_score, mem_score) < 1.0:
        return "compute_bound"
    return "idle_stall" if sc_score >= mem_score else "memory_bound"


# --- candidate table --------------------------------------------------------
#
# Per-bottleneck candidate perturbations: (knob, value, one-line rationale).
# Every knob is a real, current vLLM argument (docs.vllm.ai) that is NOT in the
# curated library.yaml — autoresearch proposes *outside* the catalog. The
# rationales are plausibility arguments, not measured claims.
_RULES: dict[str, list[tuple[str, object, str]]] = {
    "idle_stall": [
        ("max_num_partial_prefills", 4,
         "raise partial-prefill concurrency so prefill overlaps decode instead of stalling it"),
        ("long_prefill_token_threshold", 2048,
         "lower the long-prefill threshold so big prompts chunk and interleave, closing decode gaps"),
    ],
    "memory_bound": [
        ("cpu_offload_gb", 4,
         "offload cold weights to host RAM to free HBM for a larger KV cache"),
        ("preemption_mode", "swap",
         "swap preempted KV blocks to host instead of recomputing them under memory pressure"),
    ],
    "compute_bound": [
        ("compilation_config", 3,
         "raise torch.compile to level 3 for kernel fusion + piecewise CUDA graphs"),
    ],
}


@dataclass
class AutoresearchResult:
    spec: InterventionSpec
    bottleneck_class: str
    predicted_delta: float
    applicable: bool
    rejected_reason: str | None
    measured_delta: float | None
    rolled_back: bool


@dataclass
class AutoresearchRun:
    """One end-to-end autoresearch pass: the classified bottleneck + its results."""

    bottleneck_class: str
    results: list[AutoresearchResult] = field(default_factory=list)


def propose(bottleneck_class: str) -> list[InterventionSpec]:
    """Emit candidate specs for a bottleneck class (empty if the class is unknown)."""
    out: list[InterventionSpec] = []
    for knob, value, why in _RULES.get(bottleneck_class, []):
        out.append(
            InterventionSpec(
                name=f"autoresearch:{bottleneck_class}:{knob}",
                summary=why,
                knob=knob,
                value=value,
                # Proposed, not measured: an honest, modest range. The A/B is
                # what turns this into a real number.
                expected_delta_mean=0.05,
                expected_delta_lo=0.0,
                expected_delta_hi=0.15,
                source="autoresearch-v0 (proposed knob, not catalog; verified real vLLM arg)",
                applicability=Applicability(
                    workloads=["vllm-decode"], other=f"targets {bottleneck_class}"
                ),
                # Unproven ⇒ never high-risk (topology/weights changes stay in
                # the reviewed catalog). Moderate + the rollback gate is the
                # whole safety story for a proposal.
                safety=SafetyGate(
                    tier="moderate",
                    notes="autoresearch proposal — kept only on a measured, rollback-gated win.",
                ),
            )
        )
    return out


def autoresearch_v0(
    trace: Trace,
    bottleneck_class: str,
    *,
    applicator: Applicator,
    policy: Policy | None = None,
    min_keep_delta: float = 0.0,
    audit: AuditLog | None = None,
) -> list[AutoresearchResult]:
    """Propose → gate → (apply + measure + rollback) for one bottleneck class.

    Proposals are ranked and pre-filtered by :func:`select_interventions` (the
    same gate the catalog goes through), then each survivor is applied behind the
    rollback gate so a proposal that doesn't clear ``min_keep_delta`` is reverted.
    An ``audit`` log, if given, is forwarded to the apply gate so a live
    proposal's apply/rollback lands on the durable safety trail.
    """
    proposals = propose(bottleneck_class)
    if not proposals:
        return []

    ranked = select_interventions(trace, proposals, policy or Policy(), top_n=len(proposals))

    results: list[AutoresearchResult] = []
    for c in ranked:
        if c.rejected_reason is not None:
            results.append(
                AutoresearchResult(
                    spec=c.spec,
                    bottleneck_class=bottleneck_class,
                    predicted_delta=c.predicted_delta,
                    applicable=False,
                    rejected_reason=c.rejected_reason,
                    measured_delta=None,
                    rolled_back=False,
                )
            )
            continue
        res = apply_intervention(c.spec, applicator, min_keep_delta=min_keep_delta, audit=audit)
        results.append(
            AutoresearchResult(
                spec=c.spec,
                bottleneck_class=bottleneck_class,
                predicted_delta=c.predicted_delta,
                applicable=True,
                rejected_reason=None,
                measured_delta=res.measured_delta,
                rolled_back=res.rolled_back,
            )
        )
    return results


def autoresearch(
    trace: Trace,
    *,
    applicator: Applicator,
    policy: Policy | None = None,
    min_keep_delta: float = 0.0,
    audit: AuditLog | None = None,
) -> AutoresearchRun:
    """Classify the trace's bottleneck, then run the full propose→gate→apply pass.

    This is the end-to-end entry point: hand it a captured trace and a live
    applicator and it decides which class to search, proposes non-catalog levers
    for that class, and routes each through the selection + rollback gates. An
    ``audit`` log, if given, records every live apply/rollback.
    """
    bottleneck_class = classify_bottleneck(trace)
    return AutoresearchRun(
        bottleneck_class=bottleneck_class,
        results=autoresearch_v0(
            trace,
            bottleneck_class,
            applicator=applicator,
            policy=policy,
            min_keep_delta=min_keep_delta,
            audit=audit,
        ),
    )
