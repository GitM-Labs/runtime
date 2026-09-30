"""Run the runtime (monitor + attribution) on a real captured A100 trace.

Unit tests (tests/test_runtime_on_trace.py) prove the algorithms are correct on
synthetic inputs. This proves they *run on real GPU data*: it captures a live
CUDA workload via the CUPTI tracer, derives a residual per kernel as its
deviation from that kernel's own median duration, and feeds the real residual
series through:

  * stream-concurrency (real stream IDs + timestamps),
  * the multi-basis filter (vs the raw band check),
  * Granger + doubly-robust attribution.

Exit 0 on success, 2 if no CUDA / shim, 1 on an unexpected failure. Run on the pod.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path


def main() -> int:
    try:
        import numpy as np
        import torch
    except Exception:
        print("SKIP: torch/numpy not available")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 2

    from gitm.optimizer.attribution import (
        AnalysisStatus,
        RankedHypotheses,
        attribute,
        granger_evidence,
    )
    from gitm.optimizer.dr import attribute_dr
    from gitm.optimizer.monitor import (
        KernelResidual,
        Residuals,
        _serialized_fraction,
        check_invariants,
        serialized_text,
    )
    from gitm.planner.graph import predict_graph
    from gitm.tracer import capture

    out = Path("/tmp/w2_real_trace.jsonl")
    # The tracer warns when it loses records (unreadable shards, collector
    # drops); they are listed below, not hidden. Granger and DR record their
    # own warnings in their run records.
    with warnings.catch_warnings(record=True) as capture_warnings, \
            capture(out, workload_id="real-trace") as tr:
        warnings.simplefilter("always")
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = a
        for _ in range(40):       # repeated sgemm -> a real residual series
            c = (c @ b) * 1.0001
        for _ in range(20):       # a second op family
            c = torch.relu(c)
        _ = c.sum().item()
        torch.cuda.synchronize()

    for w in capture_warnings:
        print(f"capture warning: {w.category.__name__}: {w.message}")
    kernels = [e for e in tr.events if e.kind == "kernel"]
    if not kernels:
        print("FAIL: no kernels captured (is the CUPTI shim built? run gpu_setup.sh)")
        return 1
    print(f"captured {len(kernels)} real kernels on {torch.cuda.get_device_name(0)}")

    # Real stream-concurrency computed from the trace stream IDs.
    sc = _serialized_fraction(kernels)
    print(f"serialized_concurrency_fraction (REAL): {serialized_text(sc, len(kernels))}")

    # Residual per kernel = deviation from that kernel name's median duration.
    by_name: dict[str, list[int]] = {}
    for k in kernels:
        by_name.setdefault(k.name, []).append(k.end_ns - k.start_ns)
    med = {n: float(np.median(v)) for n, v in by_name.items()}

    res = Residuals()
    res.serialized_concurrency_fraction = sc
    for k in kernels:
        m = med[k.name] or 1.0
        res.per_kernel.append(
            KernelResidual(op=k.name[:30], layer=None, r_kt=((k.end_ns - k.start_ns) - m) / m, r_mt=None)
        )

    v_mb = check_invariants(res, multi_basis=True)
    v_raw = check_invariants(res, multi_basis=False)
    print(f"violations: multi-basis={len(v_mb)}  raw={len(v_raw)}  "
          f"(filter dropped {len(v_raw) - len(v_mb)} single-basis blips)")

    graph = predict_graph()
    g = attribute(res, graph)
    try:  # as in the loop: a DR failure is recorded, not raised
        d = attribute_dr(res, graph)
    except Exception as exc:
        d = RankedHypotheses(hypotheses=[], status=(
            AnalysisStatus.UNAVAILABLE if isinstance(exc, ImportError) else AnalysisStatus.FAILED))
        d.record_failure(type(exc).__name__, exc)

    print(granger_evidence(g, pairs_in="the line below"))
    print("top Granger pairs (exploratory):",
          [(h.cause_op[:18], h.effect_op[:18], round(h.p_value, 3)) for h in g.top(3)] or "none")
    print(f"doubly-robust: {d.status.value}, {d.pairs_completed}/{d.pairs_attempted} op pairs"
          + (f", failures {d.failures}" if d.failures else "")
          + (f", warnings {d.warnings}" if d.warnings else "")
          + (f", not tried as cause {d.skipped_causes}" if d.skipped_causes else "")
          + (f" ({d.reason})" if d.reason else ""))
    print("top doubly-robust:",
          [(h.cause_op[:18], h.effect_op[:18], h.notes) for h in d.top(2)] or "none")

    # Insufficient data is a correct answer (e.g. DR on a trace with no
    # anomalies), not a failure to run; it passes with a note.
    broken = (AnalysisStatus.NOT_RUN, AnalysisStatus.UNAVAILABLE, AnalysisStatus.FAILED)
    if g.status in broken or d.status in broken:
        print(f"FAIL: attribution did not run (Granger {g.status.value}, DR {d.status.value})")
        return 1
    notes = [f"{name} {h.status.value}" for name, h in (("Granger", g), ("DR", d))
             if h.status is not AnalysisStatus.OK]
    if capture_warnings:
        notes.append(f"{len(capture_warnings)} capture warning(s)")
    print("PASS: the runtime ran end-to-end on real A100 kernel data"
          + (f" (note: {'; '.join(notes)})" if notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
