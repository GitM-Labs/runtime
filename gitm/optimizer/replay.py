"""Counterfactual replay sandbox.

    predict_delta(trace, intervention_spec) -> float

Given a captured trace and an intervention spec (one entry from
``gitm.kernels.library``), simulate the predicted delta without applying live.
Used to rank candidate interventions before any rollback-gated live attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.deviation import observed_op
from gitm.tracer.schema import Trace


def predict_delta(
    trace: Trace, spec: InterventionSpec, *, delta_mean: float | None = None
) -> float:
    """Predicted fractional delta in wall-clock time on this trace.

    v0 model: apply the spec's ``expected_delta_mean`` weighted by the
    fraction of trace time spent in ops the spec is applicable to. The
    trace-driven replay engine that replaces this v0 is on the roadmap.

    ``delta_mean`` replaces the spec's estimate of the effect *on the time it
    covers*, leaving coverage — a property of this trace — untouched. It is not
    the place for a measured A/B delta: that is an end-to-end ``speedup - 1``,
    already the quantity this function returns, and scaling it by coverage again
    discounts it by the lever's scope. :func:`gitm.agents.policy.select_interventions`
    uses a measured delta directly for that reason.
    """
    # Device time, not wall time. Summed kernel durations over the wall window is
    # not a fraction: two GPUs busy for the same second sum to two seconds of work
    # in one second of wall clock, and coverage came out at 2.0 — a lever worth
    # 10% predicted 20% on an eight-GPU box. Idle gaps bent it the other way. Both
    # denominators are already distinguished in the deviation table, where
    # share_of_device divides by observed device time for exactly this reason.
    kernels = list(trace.kernels())
    device_ns = sum(max(0, k.end_ns - k.start_ns) for k in kernels)
    if device_ns <= 0:
        return 0.0
    applicable_ns = sum(
        max(0, k.end_ns - k.start_ns) for k in kernels if _applies(spec, k.name, k.range_op)
    )
    coverage = applicable_ns / device_ns
    mean = spec.expected_delta_mean if delta_mean is None else delta_mean
    return coverage * mean


def _applies(spec: InterventionSpec, kernel_name: str, range_op: str | None = None) -> bool:
    """Does ``kernel_name`` fall within ``spec``'s declared scope?

    A ``whole_step`` lever covers every kernel in the step, including those the
    predicted graph does not model. Batch shape, admission order and graph
    capture do not care which ops a checkpoint happens to have, and enumerating
    one architecture's op names is what made those levers score zero coverage on
    every other architecture.

    Otherwise, prefers op-identity via :func:`gitm.optimizer.deviation.observed_op`
    — the NVTX range identity when the capture has one, else the name guess —
    which is exactly what ``residuals()`` pairs on. Ranking by the name alone
    meant a bare cuBLAS GEMM that its NVTX range identified as ``mlp_gate_up``
    counted toward that op's residual but toward no lever's coverage. Falls back
    to substring matching for tags it doesn't cover (other workloads' own
    vocabularies, e.g. HFT's ``cudf_groupby_scan``). An empty
    ``applies_to_kernels`` still means 0 coverage, not 100% — a blank scope does
    not win ranking by default.
    """
    if spec.whole_step:
        return True
    if not spec.applies_to_kernels:
        return False
    op = observed_op(kernel_name, range_op)
    if op is not None and op in spec.applies_to_kernels:
        return True
    return any(pat in kernel_name for pat in spec.applies_to_kernels)


def predict_delta_from_files(trace_path: Path, intervention_path: Path) -> float:
    """CLI helper: load trace JSONL + intervention YAML, return predicted delta."""
    trace = _load_trace_jsonl(trace_path)
    with open(intervention_path) as fh:
        data = yaml.safe_load(fh)
    spec = InterventionSpec.model_validate(data)
    return predict_delta(trace, spec)


def _load_trace_jsonl(path: Path) -> Trace:
    """Read a trace file into a :class:`Trace`.

    Streamed line by line rather than read whole. A real capture runs to
    hundreds of thousands of kernels, and ``read_text().splitlines()`` holds the
    entire file *and* the list of its lines before a single event is parsed — two
    full copies on top of the events themselves. The events still have to fit in
    memory; this just stops the file contents doubling that.
    """
    header: dict | None = None
    events_raw = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if header is None:
                header = json.loads(line).get("_header", {})
                continue
            events_raw.append(json.loads(line))
    if header is None:
        raise ValueError(f"empty trace file: {path}")
    # Pydantic discriminates the union by ``kind``
    return Trace.model_validate({**header, "events": events_raw})
