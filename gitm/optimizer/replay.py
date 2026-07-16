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

    v0 model: the spec's ``expected_delta_mean``, weighted by the headroom it can
    actually act on. The trace-driven replay engine that replaces this v0 is on the
    roadmap.

    Which headroom depends on the spec's mechanism, and getting this wrong once cost
    us a whole report:

    * **Kernel-attributable** specs (fp8 KV, quantization, attention backend) make
      specific kernels cheaper. Their headroom is the share of trace time spent in
      the kernels they name.
    * **Non-kernel-attributable** specs — an empty ``applies_to_kernels``, i.e. the
      scheduler knobs (batch size, concurrency, policy) — do not make any kernel
      faster. They fill the GAPS: better batching claws back time the GPU spent idle.
      Their headroom is the *non-kernel* time.

    Previously an empty list meant "matches every kernel", so a scheduler knob scored
    against the whole trace while a targeted lever scored only its own kernels — a
    spec was rewarded for declining to say what it touches, and broad levers swept
    the top-N by construction. The five claims in the first vllm-decode report were
    exactly the five specs with an empty list; no structural lever could ever have
    been selected.

    Note the two numbers are still not commensurable — they are computed against
    different denominators. ``select_interventions`` therefore balances the slate
    across the two classes rather than trusting a single ranking.
    """
    total_ns = max(trace.duration_ns, 1)
    kernel_ns = sum(k.end_ns - k.start_ns for k in trace.kernels())

    if not spec.applies_to_kernels:
        headroom = max(total_ns - kernel_ns, 0) / total_ns   # idle/stall time
    else:
        matched_ns = sum(
            k.end_ns - k.start_ns for k in trace.kernels() if _applies(spec, k.name)
        )
        headroom = matched_ns / total_ns

    return headroom * spec.expected_delta_mean


def is_kernel_attributable(spec: InterventionSpec) -> bool:
    """Does this spec claim to act on specific kernels?

    The split that decides which headroom ``predict_delta`` scores it against, and
    which half of the slate it competes for.
    """
    return bool(spec.applies_to_kernels)


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
