"""Seam 3 — join what ``bench serve`` writes back to the workload that produced it.

Deliverable 1's stated purpose includes *"the harness writes the regime label into
every result row"*. That was true of the tagger and false of the pipeline: the
result JSON comes from vLLM and nothing read it. This module reads it.

**The result JSON is not merely incomplete — two of its fields are wrong.**
Measured, not assumed, against a real 0.28.0 run (all 33 keys are enumerated in
the standup's `verification.md`):

```
result.json    request_rate  "inf"    burstiness  1.0     <- the CLI's defaults
the regime     rate_rps      2.837    D           6.74    <- what actually happened
```

Under ``--self-timed`` the request schedule comes from the trace, so vLLM never
uses those two flags — but it records them anyway, untouched, beside real
metrics. They are exactly the two axes the playbook keys on. So they are
**dropped with a reason** rather than merged, renamed, or silently overwritten:
:attr:`BenchRun.dropped` shows a reader that they were considered and why they
did not survive. A field quietly removed looks like a field nobody thought about.

**Reconciliation is the other half.** A result that cannot be tied back to its
trace is not evidence, and the tie has to be checked rather than assumed — the
run may have been fired at the wrong file, the wrong block size, or a server that
dropped half of it. :func:`join_result` runs the checks and records them; a
:class:`BenchRun` that does not reconcile still exists, and says so, because a
failed run is a finding.

The one this exists for: **``total_input_tokens`` against the trace's own total.**
At vLLM's default ``--timed-trace-chunk-hash-size 16`` against Mooncake's
512-token blocks, every prompt is 32x short while completed, duration, throughput
and every percentile still read perfectly. The emitter refuses to write such a
file (:func:`gitm.traffic.replay.write_timed_trace`); this catches the case where
one was fired anyway, from a plan built elsewhere.

**R1 is not solved here.** Knob and environment fields stay ``pending-adit``.
This module makes the *trace* half of a playbook row's provenance real; the
config-capture half arrives when Adit's types do, and the joiner is where they
land.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gitm.traffic.regime import Regime
from gitm.traffic.replay import ReplayPlan
from gitm.traffic.schema import TraceMeta

#: Schema identity, in the style of ``gitm.bench.manifest.SCHEMA``.
SCHEMA = "gitm.traffic.benchrun/v1"

#: Waiting on the shared config-capture schema (risk R1). Grep-able.
PENDING_ADIT = "pending-adit"

#: Result fields that are **wrong** under ``--self-timed``, and why. Dropped, not
#: merged: the trace decided the schedule, so these are the CLI defaults vLLM
#: never consulted, sitting next to real metrics on the two axes a playbook row
#: is keyed on.
MISLEADING_UNDER_SELF_TIMED = {
    "request_rate": "CLI default, not the trace's; the real value is regime.rate_rps",
    "burstiness": "CLI default, not the trace's; the real value is regime.burstiness",
}

#: Metrics kept from the result JSON. Deliberately a list rather than "everything
#: except the dropped ones": a future vLLM adding a field should not silently
#: enlarge what a playbook row claims to have measured.
KEPT_METRICS = (
    "duration", "completed", "failed", "num_prompts",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "request_goodput", "max_output_tokens_per_s", "max_concurrent_requests",
    "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms",
    "model_id", "tokenizer_id", "backend", "endpoint_type", "date", "label",
    "max_concurrency",
    # Real-time factor. 0.0 on text serving — it is an ASR metric — but it is a
    # field vLLM measured, and `unjoined_keys` exists precisely so a measured
    # field cannot fall out of the record without someone deciding it should.
    "rtfx",
)

#: How far a paced run may overrun the trace's span before the schedule is
#: considered to have drifted. Startup and teardown cost a little; a saturating
#: server costs a lot. 5 % of the span plus one second, so short traces are not
#: judged by a percentage of nothing.
DRIFT_TOLERANCE = 0.05
DRIFT_FLOOR_S = 1.0


class Check(BaseModel):
    """One reconciliation between the result and the trace it should describe."""

    model_config = ConfigDict(extra="forbid")

    name: str
    expected: float | int | str | None
    actual: float | int | str | None
    ok: bool
    detail: str = ""


class BenchRun(BaseModel):
    """One measured run, joined to the workload that produced it.

    This is what deliverable 2 promotes from and what deliverable 4's provenance
    is built out of. It is deliberately *not* a playbook row: a row is a **delta
    between two arms**, and this is one arm.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = SCHEMA

    # --- what workload this was, which the result JSON does not say ------------
    source: TraceMeta
    regime: Regime
    regime_label: str

    # --- the conditions it was fired under -------------------------------------
    chunk_hash_size: int
    self_timed: bool
    prefix_synthesized: bool

    # --- what came back ---------------------------------------------------------
    metrics: dict[str, Any] = Field(default_factory=dict)
    #: Fields removed from ``metrics``, mapped to why. Kept visible on purpose.
    dropped: dict[str, str] = Field(default_factory=dict)
    #: The values those fields held, so the record shows what was rejected.
    dropped_values: dict[str, Any] = Field(default_factory=dict)

    # --- does it describe the trace it claims to? -------------------------------
    checks: list[Check] = Field(default_factory=list)

    # --- R1 ---------------------------------------------------------------------
    knobs: dict[str, bool | int | float | str] = Field(default_factory=dict)
    config_capture: str = PENDING_ADIT

    #: The result JSON exactly as vLLM wrote it. Nothing is lost by the join —
    #: a dropped field is dropped from ``metrics``, not from the record.
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def reconciled(self) -> bool:
        """Whether this result actually describes the trace it is joined to."""
        return all(c.ok for c in self.checks)

    @property
    def promotable(self) -> bool:
        """Whether deliverable 2 may even look at this run.

        Necessary, never sufficient: D2 owns the promotion rule. What this says
        is that the run reconciles with its trace and completed without failures
        — below that bar there is nothing for a rule to judge.
        """
        return self.reconciled and self.metrics.get("failed", 1) == 0

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def summary(self) -> str:
        head = "reconciled" if self.reconciled else f"NOT RECONCILED ({len(self.failures())})"
        return (
            f"{self.source.source} / {self.regime_label} — {head}: "
            f"{self.metrics.get('completed')} completed, "
            f"{self.metrics.get('duration', 0.0):.2f}s, "
            f"p99 TTFT {self.metrics.get('p99_ttft_ms', float('nan')):.1f}ms"
        )

    def render(self) -> str:
        lines = [self.summary(), ""]
        w = max((len(c.name) for c in self.checks), default=4)
        for c in self.checks:
            mark = "ok  " if c.ok else "FAIL"
            lines.append(f"  {mark} {c.name.ljust(w)}  expected {c.expected}  actual {c.actual}"
                         + (f"  — {c.detail}" if c.detail else ""))
        if self.dropped:
            lines.append("")
            lines.append("  dropped from metrics (recorded, not merged):")
            for k, why in sorted(self.dropped.items()):
                lines.append(f"    {k} = {self.dropped_values.get(k)!r}: {why}")
        return "\n".join(lines)


def _num(v: Any) -> float | None:
    """Coerce a result value to a float, or ``None``.

    ``request_rate`` arrives as the **string** ``"inf"`` — ``json.dumps`` cannot
    write a bare ``Infinity``, so vLLM stringifies it. A joiner that assumed
    float would raise on the one field it is trying to throw away.
    """
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def join_result(
    result: dict[str, Any],
    plan: ReplayPlan,
    regime: Regime,
    *,
    knobs: dict[str, bool | int | float | str] | None = None,
    drift_tolerance: float = DRIFT_TOLERANCE,
) -> BenchRun:
    """Join a ``bench serve`` result to the plan and regime that produced it.

    Keeps the metrics deliverable 2 needs, drops the two fields that are wrong
    under ``--self-timed``, attaches the workload identity the result JSON has no
    room for, and checks that the numbers actually describe this trace.
    """
    metrics = {k: result[k] for k in KEPT_METRICS if k in result}
    dropped, dropped_values = {}, {}
    for k, why in MISLEADING_UNDER_SELF_TIMED.items():
        if k in result:
            dropped_values[k] = result[k]
            dropped[k] = why if plan.self_timed else f"{why} (run was NOT self-timed)"

    checks: list[Check] = []

    def add(name, expected, actual, ok, detail=""):
        checks.append(Check(name=name, expected=expected, actual=actual, ok=ok, detail=detail))

    # 1. Every request the plan wrote should have been sent.
    completed = result.get("completed")
    add("requests_completed", plan.requests, completed, completed == plan.requests,
        "" if completed == plan.requests else "the run did not replay the whole trace")

    # 2. Failures are disqualifying, not a footnote.
    failed = result.get("failed")
    add("no_failed_requests", 0, failed, failed == 0,
        "" if failed == 0 else "a run with failures measures the failures too")

    # 3. THE one. Input tokens must match the trace, or the prompts were not the
    #    trace's prompts — the 32x truncation that leaves every other count right.
    ti = result.get("total_input_tokens")
    expected_in = plan.input_tokens_total
    ok_tokens = ti == expected_in
    detail = ""
    if not ok_tokens and ti and expected_in:
        ratio = expected_in / ti
        detail = f"{ratio:.1f}x short — check --timed-trace-chunk-hash-size " \
                 f"(the plan wrote {plan.chunk_hash_size}-token blocks)" if ratio > 1.5 else \
                 "prompt lengths do not match the trace"
    add("input_tokens_match_trace", expected_in, ti, ok_tokens, detail)

    # 4. Pacing, and the two directions mean different things. Only meaningful
    #    when the run was self-timed; a rate-driven run has no schedule to hold.
    duration = _num(result.get("duration"))
    if plan.self_timed and duration is not None and plan.span_s > 0:
        budget = plan.span_s * (1 + drift_tolerance) + DRIFT_FLOOR_S
        if duration < plan.span_s * (1 - drift_tolerance):
            ok, why = False, ("finished FASTER than the trace's own span — the timestamps "
                              "were not honoured, so this is not a replay of this trace")
        elif duration > budget:
            ok, why = False, ("overran the trace's span — the schedule drifted, which usually "
                              "means the server saturated; the arrival pattern was not delivered")
        else:
            ok, why = True, ""
        add("paced_to_trace_span", round(plan.span_s, 3), round(duration, 3), ok, why)

    # 5. A synthesized-prefix run cannot speak about cache reuse. Not a failure —
    #    a label, because the number is a floor and D4 has to know.
    if plan.prefix_synthesized:
        add("prefix_identity", "from source", "synthesized", True,
            "source had no prefix identity; blocks were synthesized unique per request, so any "
            "cache-reuse reading from this run is a FLOOR, never an estimate")

    return BenchRun(
        source=plan.source,
        regime=regime,
        regime_label=regime.label(),
        chunk_hash_size=plan.chunk_hash_size,
        self_timed=plan.self_timed,
        prefix_synthesized=plan.prefix_synthesized,
        metrics=metrics,
        dropped=dropped,
        dropped_values=dropped_values,
        checks=checks,
        knobs=dict(knobs or {}),
        raw=dict(result),
    )


def unjoined_keys(result: dict[str, Any]) -> list[str]:
    """Result keys that are neither kept nor deliberately dropped.

    A new vLLM release adding a field should be a decision, not a silent
    omission. `check_join_accounts_for_every_result_key` fails when this is
    non-empty for the recorded real-run keys.
    """
    known = set(KEPT_METRICS) | set(MISLEADING_UNDER_SELF_TIMED)
    return sorted(k for k in result if k not in known)


def is_infinite(v: Any) -> bool:
    """``request_rate`` arrives as ``"inf"``. Named so the string is not a surprise."""
    n = _num(v)
    return n is not None and math.isinf(n)
