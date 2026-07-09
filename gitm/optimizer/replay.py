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
from gitm.optimizer.deviation import classify_op
from gitm.tracer.schema import Trace


def predict_delta(trace: Trace, spec: InterventionSpec) -> float:
    """Predicted fractional delta in wall-clock time on this trace.

    v0 model: apply the spec's ``expected_delta_mean`` weighted by the
    fraction of trace time spent in ops the spec is applicable to. The
    trace-driven replay engine that replaces this v0 is on the roadmap.
    """
    total_ns = max(trace.duration_ns, 1)
    applicable_ns = 0
    for k in trace.kernels():
        if _applies(spec, k.name):
            applicable_ns += k.end_ns - k.start_ns
    coverage = applicable_ns / total_ns
    return coverage * spec.expected_delta_mean


def _applies(spec: InterventionSpec, kernel_name: str) -> bool:
    """Does ``kernel_name`` fall within ``spec``'s declared scope?

    Prefers op-identity via :func:`gitm.optimizer.deviation.classify_op` (same
    vocabulary ``residuals()`` uses), falling back to substring matching for
    tags it doesn't cover (other workloads' own vocabularies, e.g. HFT's
    ``cudf_groupby_scan``). An empty ``applies_to_kernels`` means 0 coverage,
    not 100% — a blank scope no longer wins ranking by default.
    """
    if not spec.applies_to_kernels:
        return False
    op = classify_op(kernel_name)
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
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty trace file: {path}")
    header = json.loads(lines[0]).get("_header", {})
    events_raw = [json.loads(line) for line in lines[1:] if line.strip()]
    # Pydantic discriminates the union by ``kind``
    return Trace.model_validate({**header, "events": events_raw})
