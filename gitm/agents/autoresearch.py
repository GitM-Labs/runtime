"""Autoresearch — propose non-catalog levers within the attributed bottleneck.

The catalog (library.yaml) is curated and finite. Autoresearch proposes config
perturbations outside it, constrained to the bottleneck class attribution
already identified (idle/memory/compute), then routes each proposal through the
exact same path as a catalog lever: the applicability gate, then apply + measure
+ rollback behind the live applicator. Nothing it proposes can bypass the gate
or be kept without a measured win — it is a candidate source, not a new trust
path.

v0 is a small rule table per bottleneck class; later versions repoint at the
largest residual and learn an effect model.
"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec
from gitm.optimizer.apply import Applicator, apply_intervention
from gitm.optimizer.preconditions import GateContext
from gitm.tracer.schema import Trace


# Per-bottleneck candidate perturbations: (knob, value, one-line rationale).
# vLLM runtime knobs that plausibly relieve each class — proposed, never assumed.
_RULES: dict[str, list[tuple[str, object, str]]] = {
    "idle_stall": [
        ("scheduler_config.max_num_batched_tokens", 8192, "raise batch to fill idle gaps"),
        ("scheduler_config.max_num_seqs", 256, "admit more sequences to reduce starvation"),
    ],
    "memory_bound": [
        ("cache_config.gpu_memory_utilization", 0.95, "grow KV cache, fewer preemptions"),
        ("cache_config.kv_cache_dtype", "fp8", "halve KV traffic with fp8 cache"),
    ],
    "compute_bound": [
        ("model_config.enforce_eager", False, "enable CUDA graphs to cut launch overhead"),
    ],
}


@dataclass
class AutoresearchResult:
    spec: InterventionSpec
    bottleneck_class: str
    applicable: bool
    rejected_reason: str | None
    measured_delta: float | None
    rolled_back: bool


def propose(bottleneck_class: str) -> list[InterventionSpec]:
    """Emit candidate specs for a bottleneck class (empty if none known)."""
    out: list[InterventionSpec] = []
    for knob, value, why in _RULES.get(bottleneck_class, []):
        out.append(
            InterventionSpec(
                name=f"autoresearch:{bottleneck_class}:{knob.split('.')[-1]}",
                summary=why,
                knob=knob,
                value=value,
                expected_delta_mean=0.05,
                expected_delta_lo=0.0,
                expected_delta_hi=0.15,
                source="autoresearch-v0 (proposed, not catalog)",
                applicability=Applicability(workloads=["vllm-decode"], other=bottleneck_class),
            )
        )
    return out


def autoresearch_v0(
    trace: Trace,
    bottleneck_class: str,
    ctx: GateContext,
    *,
    applicator: Applicator,
    policy: Policy | None = None,
    min_keep_delta: float = 0.0,
) -> list[AutoresearchResult]:
    """Propose → gate → (apply + measure + rollback) for one bottleneck class.

    Proposals are ranked/gated by :func:select_interventions (same gate as the
    catalog), then each applicable one is applied behind the rollback gate so a
    proposal that doesn't measurably help is reverted.
    """
    proposals = propose(bottleneck_class)
    ranked = select_interventions(
        trace, proposals, policy or Policy(), top_n=len(proposals) or 1, ctx=ctx
    )

    results: list[AutoresearchResult] = []
    for c in ranked:
        if c.rejected_reason is not None:
            results.append(
                AutoresearchResult(
                    spec=c.spec, bottleneck_class=bottleneck_class, applicable=False,
                    rejected_reason=c.rejected_reason, measured_delta=None, rolled_back=False,
                )
            )
            continue
        res = apply_intervention(c.spec, applicator, min_keep_delta=min_keep_delta)
        results.append(
            AutoresearchResult(
                spec=c.spec, bottleneck_class=bottleneck_class, applicable=True,
                rejected_reason=None, measured_delta=res.measured_delta,
                rolled_back=res.rolled_back,
            )
        )
    return results