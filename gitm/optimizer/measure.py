"""Workload-agnostic measurement from a captured trace.

Turns a raw kernel trace into honest, workload-specific observations — kernel
families, per-kernel timing residuals vs each kernel's median, the
serialized-concurrency fraction, and Granger hypotheses over the *actual*
kernels. No model-specific predicted graph, no intervention library.

This is what we report for any workload the optimizer has no tuned intervention
set for (everything except ``vllm-decode``): the report then describes what
really ran on the GPU instead of pairing the trace with a transformer graph and
emitting vLLM serving-knob "claims" that don't apply. Shared by the autonomous
loop (:mod:`gitm.scheduler.loop`) and the driver (:mod:`gitm.runtime_driver`).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gitm.optimizer.attribution import (
    AnalysisStatus,
    RankedHypotheses,
    attribute,
    granger_evidence,
)
from gitm.optimizer.monitor import (
    LIMITATIONS,
    KernelResidual,
    Residuals,
    _serialized_fraction,
    check_invariants,
    measured_serialized_fraction,
    serialized_text,
)
from gitm.optimizer.report import Claim
from gitm.planner.graph import predict_graph
from gitm.tracer.schema import Trace

# Mangled CUDA kernel names → a small set of stable "families" so the residual
# series feeding Granger are well-populated (a variable = a kernel *type*, not
# every template instantiation). Noise tokens are dropped; the first distinctive
# identifier after the library prefix names the family.
_NOISE = {
    "detail", "kernel", "void", "const", "unsigned", "int", "long", "float",
    "double", "global", "device", "functor", "impl", "internal", "type",
    "types", "common", "native", "operator", "policy", "dispatch", "agent",
}


def kernel_family(name: str) -> str:
    """Collapse a mangled kernel name to a ``{lib}.{fn}`` family label."""
    lib = ("cub" if "cub" in name else "cudf" if "cudf" in name
           else "thrust" if "thrust" in name else "k")
    toks = [t for t in re.findall(r"[a-z][a-z_]{3,}", name) if t not in _NOISE]
    fn = toks[0] if toks else "anon"
    return f"{lib}.{fn}"


#: What a measurement run's numbers cannot say. Its "kernel_time" residual is
#: each kernel against its own median, so it measures jitter, not distance from
#: the roofline.
MEASURE_LIMITATIONS = (
    *LIMITATIONS,
    "kernel_time here is relative to each kernel's median, not the roofline",
)


@dataclass
class MeasureResult:
    n_kernels: int
    n_memcpy: int
    serialized_fraction: float
    violations: list = field(default_factory=list)
    top_hypotheses: list = field(default_factory=list)  # granger.top(5), kept for readers of the list
    families: list[str] = field(default_factory=list)
    granger: RankedHypotheses = field(default_factory=lambda: RankedHypotheses(hypotheses=[]))

    @property
    def serialized_measured(self) -> float | None:
        """``serialized_fraction`` for artifacts: None when under 2 kernels."""
        return measured_serialized_fraction(self.serialized_fraction, self.n_kernels)

    @property
    def serialized_text(self) -> str:
        return serialized_text(self.serialized_fraction, self.n_kernels)


def attribute_families(
    res_attr: Residuals, fam_counts: Counter, min_attr: int
) -> RankedHypotheses:
    """Granger over kernel families, with the families too small to test on record.

    Families under ``min_attr`` samples are filtered out before ``attribute``
    sees them, so its own ``excluded_ops`` would not list them; they are added
    here, and ``min_obs`` reports the threshold that actually applied.
    """
    small = {fam: n for fam, n in fam_counts.items() if n < min_attr}
    if res_attr.per_kernel:
        granger = attribute(res_attr, predict_graph())
    else:
        granger = RankedHypotheses(
            hypotheses=[],
            status=AnalysisStatus.INSUFFICIENT_DATA,
            reason=f"no kernel family had {min_attr} samples",
        )
    granger.series_lengths.update(small)
    granger.excluded_ops.update(small)
    granger.min_obs = max(granger.min_obs or 0, min_attr)
    if granger.status is AnalysisStatus.INSUFFICIENT_DATA and res_attr.per_kernel:
        # attribute() wrote its own, lower threshold; state the one that applied.
        granger.reason = f"fewer than 2 kernel families with at least {granger.min_obs} samples"
    return granger


def measure_trace(trace: Trace, *, min_attr: int = 16) -> MeasureResult:
    """Compute residuals → invariants → attribution from the real kernels.

    Residuals are each kernel's duration vs its own name's median (so the op
    labels are the *actual* kernels). Attribution groups kernels into families
    with at least ``min_attr`` samples so Granger's per-op series are well-formed.
    """
    kernels = trace.kernels()
    memcpys = [e for e in trace.events if e.kind == "memcpy"]
    if not kernels:
        return MeasureResult(0, len(memcpys), 0.0)

    sc = _serialized_fraction(kernels)
    by_name: dict[str, list[int]] = {}
    for k in kernels:
        by_name.setdefault(k.name, []).append(k.end_ns - k.start_ns)
    med = {nm: float(np.median(v)) for nm, v in by_name.items()}

    res = Residuals()
    res.serialized_concurrency_fraction = sc
    for k in kernels:
        m = med[k.name] or 1.0
        res.per_kernel.append(
            KernelResidual(op=k.name[:40], layer=None, r_kt=((k.end_ns - k.start_ns) - m) / m, r_mt=None)
        )
    violations = check_invariants(res, multi_basis=True)

    fam_of = {nm: kernel_family(nm) for nm in by_name}
    fam_counts = Counter(fam_of[k.name] for k in kernels)
    res_attr = Residuals()
    res_attr.serialized_concurrency_fraction = sc
    for k in kernels:
        fam = fam_of[k.name]
        if fam_counts[fam] < min_attr:
            continue
        m = med[k.name] or 1.0
        res_attr.per_kernel.append(
            KernelResidual(op=fam, layer=None, r_kt=((k.end_ns - k.start_ns) - m) / m, r_mt=None)
        )
    families = sorted({kr.op for kr in res_attr.per_kernel})
    granger = attribute_families(res_attr, fam_counts, min_attr)

    return MeasureResult(
        n_kernels=len(kernels),
        n_memcpy=len(memcpys),
        serialized_fraction=sc,
        violations=violations,
        top_hypotheses=granger.top(5),
        families=families,
        granger=granger,
    )


def measurement_claims(
    result: MeasureResult, *, limit: int = 5, pairs_in: str = "measurement.json"
) -> list[Claim]:
    """Build measurement observations (not optimization claims) from deviations.

    Each carries the real invariant deviation and the Granger run's status; the
    intervention column is explicitly ``(none — measurement run)``.
    """
    evidence = granger_evidence(result.granger, pairs_in=pairs_in)
    claims: list[Claim] = []
    for v in result.violations[:limit]:
        claims.append(
            Claim(
                summary=f"{v.invariant} deviation on {v.node_op}",
                residual_invariant=v.invariant,
                residual_value=float(v.residual),
                causal_evidence=evidence,
                intervention_name="(none — measurement run)",
                predicted_delta=0.0,
                measured_delta=None,
            )
        )
    return claims


def measurement_summary(workload: str, result: MeasureResult) -> str:
    fams = ", ".join(result.families[:6]) or "none with enough samples"
    return (
        f"Measurement run for {workload!r}: {result.n_kernels:,} kernels "
        f"({result.n_memcpy:,} memcpy) captured, {len(result.violations)} invariant "
        f"deviation(s), serialized-concurrency={result.serialized_text}. "
        f"Kernel families: {fams}. No interventions applied — this workload has no "
        f"tuned intervention library, so the runtime reports what it measured."
    )
