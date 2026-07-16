"""Workload qualification gate.

Stock vLLM on commodity infra commits to the floor. Aggressively tuned
deployments get a diagnostic, no commitment broken. The refund clause hangs
off this gate — it must be honest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from gitm.tracer.schema import Trace


@dataclass
class QualificationResult:
    commit: bool
    floor: float  # fraction, e.g. 0.15
    fingerprint: str
    diagnostic: str = ""


def fingerprint(trace: Trace) -> str:
    """Stable workload fingerprint from kernel-mix + shapes.

    Two equivalent workloads (same model, same batch shape, same backend
    config) produce the same fingerprint regardless of trace timestamps.
    """
    kernels = trace.kernels()
    summary = sorted(
        {
            (k.name, k.grid_x * k.grid_y * k.grid_z, k.block_x * k.block_y * k.block_z)
            for k in kernels
        }
    )
    digest = hashlib.sha256(repr(summary).encode("utf-8")).hexdigest()[:16]
    return f"{trace.vendor}:{digest}"


_IMPORT_SOURCES = frozenset({"nsys-import", "torch-import"})

_IMPORT_DIAGNOSTIC = (
    "Imported trace: diagnostic headroom read only. Floor commitment requires "
    "a gitm-captured run."
)


def qualify(trace: Trace, target_floor: float = 0.15) -> QualificationResult:
    """Decide whether to commit to ``target_floor`` for this workload.

    Heuristic v0: refuse to commit when the trace looks already-tuned, where
    "already-tuned" is signaled by low residual headroom in the kernel mix.
    A richer fingerprint check is on the roadmap; this version routes the
    plumbing end-to-end.

    Imported profiler traces (``nsys-import`` / ``torch-import``) never commit:
    the 15% floor and refund clause require a gitm-captured run.
    """
    fp = fingerprint(trace)
    source = getattr(trace, "source", "cupti") or "cupti"
    if source in _IMPORT_SOURCES:
        return QualificationResult(
            commit=False,
            floor=target_floor,
            fingerprint=fp,
            diagnostic=_IMPORT_DIAGNOSTIC,
        )

    kernels = trace.kernels()
    if not kernels:
        return QualificationResult(
            commit=False,
            floor=target_floor,
            fingerprint=fp,
            diagnostic="No kernels in trace — capture failed or workload did not run.",
        )

    # Heuristic: if the top-10 kernels by time account for >95% of duration,
    # the workload is likely already well-shaped. Real gate uses the residual
    # distribution after the deviation monitor.
    durs = sorted(((k.end_ns - k.start_ns) for k in kernels), reverse=True)
    top_n = durs[:10]
    head_share = sum(top_n) / max(sum(durs), 1)
    if head_share > 0.95 and len(durs) > 50:
        return QualificationResult(
            commit=False,
            floor=target_floor,
            fingerprint=fp,
            diagnostic=(
                "Workload appears aggressively tuned: top-10 kernels account for "
                f"{head_share:.0%} of GPU time. Refusing to commit to floor; "
                "diagnostic report only."
            ),
        )

    return QualificationResult(commit=True, floor=target_floor, fingerprint=fp)
