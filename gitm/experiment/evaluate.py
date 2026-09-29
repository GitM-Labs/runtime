"""Run a contract through an :class:`~gitm.optimizer.apply.Applicator` and return a verdict.

    evaluate(contract, applicator, sampler) -> Verdict

The applicator is the same seam :func:`~gitm.optimizer.apply.apply_intervention`
drives (snapshot / apply / restore), so any existing applicator works here. The
evaluator does **not** call ``apply_intervention`` itself: that gate reduces an
experiment to one scalar against ``min_keep_delta``, which cannot express an
interval, a correctness gate or a latency gate. Keep-or-rollback is still the
same rule: the candidate stays applied only when the verdict is ``improved``,
and otherwise the snapshot is restored.

Measurement is behind a second seam, :class:`Sampler`: one call runs the
contract's workload once against whatever configuration is currently applied
and returns the raw metrics. On the MI355X that is a timed replay through
``vllm bench serve``; in tests it is a deterministic fake.

The terminal state is decided by fixed rules, applied in this order:

1. an apply, sample or restore raised → ``failed_to_execute``
2. a known-effect control failed for this operating point → ``invalid``
3. the budget ran out before the protocol finished → ``inconclusive``
4. too few samples, or the primary metric or a correctness score is missing → ``invalid``
5. the correctness gate or any latency gate failed → ``regressed``
6. the interval lies entirely above 0 → ``improved``; entirely below 0 → ``regressed``
7. otherwise → ``inconclusive``
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from scipy import stats

from gitm.experiment.contract import (
    Cost,
    ExperimentContract,
    GateResult,
    RawSample,
    Verdict,
)
from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import Applicator

Arm = Literal["baseline", "candidate"]


@dataclass
class Sample:
    """One run of the workload against the currently applied configuration."""

    metrics: dict[str, float]
    wall_s: float
    accelerator_s: float
    #: Correctness score in [0, 1] against the contract's reference, or ``None``
    #: when it was not measured.
    correctness: float | None = None


class Sampler(Protocol):
    def sample(self, arm: Arm, index: int) -> Sample: ...


def intervention_spec(contract: ExperimentContract) -> InterventionSpec:
    """The contract's intervention as the spec every applicator already takes."""
    h = contract.hypothesis
    knobs = dict(contract.intervention)
    return InterventionSpec(
        name=contract.candidate_id,
        summary=h.claim,
        knob=",".join(f"{k}={v}" for k, v in knobs.items()),
        knobs=knobs,
        expected_delta_mean=h.expected_effect.mean,
        expected_delta_lo=h.expected_effect.lo,
        expected_delta_hi=h.expected_effect.hi,
        source=f"hypothesis {h.id} ({h.submitter})",
    )


def run_order(contract: ExperimentContract) -> list[Arm]:
    """The arm of every kept sample, in run order. Warmup is added at switches."""
    p = contract.protocol
    n = p.repetitions
    if p.comparison_order == "ABAB":
        return ["baseline", "candidate"] * n
    if p.comparison_order == "ABBA":
        pattern: list[Arm] = ["baseline", "candidate", "candidate", "baseline"]
        return (pattern * math.ceil(n / 2))[: 2 * n]
    order = cast("list[Arm]", ["baseline"] * n + ["candidate"] * n)
    random.Random(p.order_seed).shuffle(order)  # noqa: S311 - reproducible order, not security
    return order


def effect_interval(
    baseline: list[float], candidate: list[float], *, higher_is_better: bool,
    paired: bool, alpha: float,
) -> tuple[float, float, float, str]:
    """(effect, lo, hi, method) for the relative change, positive = better."""
    sign = 1.0 if higher_is_better else -1.0
    if paired:
        diffs = [sign * (c / b - 1.0) for b, c in zip(baseline, candidate, strict=True)]
        n = len(diffs)
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        half = stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
        method = (f"paired t-interval on per-pair relative change c/b-1, n={n} pairs, "
                  f"{1 - alpha:.0%} two-sided")
        return mean, mean - half, mean + half, method

    nb, nc = len(baseline), len(candidate)
    mb, mc = sum(baseline) / nb, sum(candidate) / nc
    vb = sum((x - mb) ** 2 for x in baseline) / (nb - 1)
    vc = sum((x - mc) ** 2 for x in candidate) / (nc - 1)
    ratio = mc / mb
    # Delta method for the ratio of means; Welch-Satterthwaite degrees of freedom.
    se = ratio * math.sqrt(vc / (nc * mc**2) + vb / (nb * mb**2))
    sb, sc = vb / nb, vc / nc
    df = (sb + sc) ** 2 / (sb**2 / (nb - 1) + sc**2 / (nc - 1)) if sb + sc > 0 else nb + nc - 2
    half = stats.t.ppf(1 - alpha / 2, df) * se
    mean = sign * (ratio - 1.0)
    method = (f"ratio of means mc/mb-1, delta-method SE, Welch df={df:.1f}, "
              f"nb={nb}, nc={nc}, {1 - alpha:.0%} two-sided")
    return mean, mean - half, mean + half, method


def evaluate(
    contract: ExperimentContract,
    applicator: Applicator,
    sampler: Sampler,
    *,
    control_status: Literal["passed", "failed", "not_run"] = "not_run",
    provenance: dict[str, Any] | None = None,
    usd_per_accelerator_s: float | None = None,
) -> Verdict:
    """Run ``contract`` and return its verdict, leaving the candidate applied only if improved."""
    spec = intervention_spec(contract)
    raw: list[RawSample] = []
    wall = accel = 0.0
    error: str | None = None
    budget_hit = False

    def verdict(state: str, reason: str, rollback: str, **extra: Any) -> Verdict:
        cost = Cost(
            wall_clock_s=wall, accelerator_s=accel,
            estimated_usd=accel * usd_per_accelerator_s if usd_per_accelerator_s else None,
        )
        fields: dict[str, Any] = dict(
            candidate_id=contract.candidate_id,
            hypothesis_id=contract.hypothesis.id,
            contract_sha256=contract.sha256(),
            state=state, reason=reason, method="none",
            n_baseline=sum(1 for s in raw if s.arm == "baseline" and not s.warmup),
            n_candidate=sum(1 for s in raw if s.arm == "candidate" and not s.warmup),
            correctness=GateResult(name="correctness", status="not_run"),
            latency=[GateResult(name=g.metric, status="not_run") for g in contract.latency_gates],
            control_status=control_status,
            detectability=contract.noise_floor_ref or "not_established",
            cost=cost, rollback=rollback, provenance=dict(provenance or {}), raw=raw,
        )
        fields.update(extra)
        return Verdict(**fields)

    if control_status == "failed":
        return verdict("invalid", "a known-effect control failed for this operating point; "
                       "no performance verdict until the evaluator passes again", "not_applied")

    snapshot = applicator.snapshot()
    applied = False
    try:
        current: Arm | None = None
        for arm in run_order(contract):
            if arm != current:
                if arm == "candidate":
                    # Marked before the call: a failed apply may leave a partial
                    # change, so it is restored like apply_intervention does.
                    applied = True
                    applicator.apply(spec)
                elif applied:
                    applicator.restore(snapshot)
                    applied = False
                current = arm
                warm = contract.protocol.warmup_reps
            else:
                warm = 0
            for w in range(warm + 1):
                if wall >= contract.budget.max_wall_clock_s or accel >= contract.budget.max_accelerator_s:
                    budget_hit = True
                    break
                s = sampler.sample(arm, len(raw))
                wall += s.wall_s
                accel += s.accelerator_s
                raw.append(RawSample(
                    arm=arm, order_index=len(raw), warmup=w < warm, metrics=s.metrics,
                    correctness=s.correctness, wall_s=s.wall_s, accelerator_s=s.accelerator_s,
                ))
            if budget_hit:
                break
    except Exception as exc:  # noqa: BLE001 - any failure is a terminal state, not a crash
        error = f"{type(exc).__name__}: {exc}"

    state, reason, extra = _decide(contract, raw, error=error, budget_hit=budget_hit)

    # Keep only on improved; otherwise put the baseline back.
    rollback = "kept"
    if state != "improved":
        rollback = "rolled_back" if applied else "not_applied"
        if applied:
            try:
                applicator.restore(snapshot)
            except Exception as exc:  # noqa: BLE001
                rollback = "restore_failed"
                state, reason = "failed_to_execute", f"{reason}; restore failed: {exc}"
    return verdict(state, reason, rollback, **extra)


def _decide(
    contract: ExperimentContract, raw: list[RawSample], *, error: str | None, budget_hit: bool,
) -> tuple[str, str, dict[str, Any]]:
    if error is not None:
        return "failed_to_execute", error, {}
    if budget_hit:
        return "inconclusive", "budget exhausted before the protocol completed", {}

    metric = contract.primary_metric.name
    kept = [s for s in raw if not s.warmup]
    base = [s for s in kept if s.arm == "baseline"]
    cand = [s for s in kept if s.arm == "candidate"]
    if len(base) < 2 or len(cand) < 2:
        return "invalid", f"too few samples (baseline={len(base)}, candidate={len(cand)})", {}
    missing = [s.order_index for s in kept if metric not in s.metrics]
    if missing:
        return "invalid", f"primary metric {metric!r} missing from samples {missing}", {}

    extra: dict[str, Any] = {}
    gate = contract.correctness_gate
    scores = [s.correctness for s in cand if s.correctness is not None]
    if len(scores) < len(cand):
        return "invalid", "correctness was not measured on every candidate sample", {}
    worst = min(scores)
    correctness = GateResult(
        name="correctness", status="pass" if worst >= gate.min_score else "fail",
        detail=f"{gate.method} vs {gate.reference}: min score {worst:.4f}, need >= {gate.min_score}",
    )
    extra["correctness"] = correctness

    latency: list[GateResult] = []
    for g in contract.latency_gates:
        b = [s.metrics.get(g.metric) for s in base]
        c = [s.metrics.get(g.metric) for s in cand]
        if any(x is None for x in b + c):
            latency.append(GateResult(name=g.metric, status="not_run", detail="metric not reported"))
            continue
        mb, mc = sum(b) / len(b), sum(c) / len(c)  # type: ignore[arg-type]
        worse = (mc / mb - 1.0) if g.direction == "lower_is_better" else (1.0 - mc / mb)
        latency.append(GateResult(
            name=g.metric, status="pass" if worse <= g.max_regression else "fail",
            detail=f"worsened {worse:+.2%}, limit {g.max_regression:.2%}",
        ))
    extra["latency"] = latency
    if any(r.status == "not_run" for r in latency):
        return "invalid", "a latency gate's metric was not reported", extra

    p = contract.protocol
    effect, lo, hi, method = effect_interval(
        [s.metrics[metric] for s in base], [s.metrics[metric] for s in cand],
        higher_is_better=contract.primary_metric.direction == "higher_is_better",
        paired=p.pairing == "paired", alpha=p.alpha,
    )
    extra.update(effect=effect, interval=(lo, hi), method=method,
                 claim_rejected=hi < contract.hypothesis.reject_if_effect_below)

    if correctness.status == "fail":
        return "regressed", "correctness gate failed", extra
    failed = [r.name for r in latency if r.status == "fail"]
    if failed:
        return "regressed", f"latency gate failed: {', '.join(failed)}", extra
    if lo > 0:
        return "improved", f"interval [{lo:+.4f}, {hi:+.4f}] lies above 0", extra
    if hi < 0:
        return "regressed", f"interval [{lo:+.4f}, {hi:+.4f}] lies below 0", extra
    return "inconclusive", f"interval [{lo:+.4f}, {hi:+.4f}] spans 0", extra
