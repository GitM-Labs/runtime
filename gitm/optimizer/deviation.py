"""Deviation-only tracing — keep only the kernels that depart from prediction.

The behavioral compiler predicts a per-op execution graph; most kernels land
inside their roofline band and are *uninteresting* — they behaved exactly as
predicted, so storing them buys nothing. The optimization signal lives in the
*departures*: kernels slower (or heavier) than predicted, and kernels with no
predicted counterpart at all (unmodeled work). This module reduces a captured
trace to just those, so trace storage scales with deviation, not duration (the
monitor's design principle, applied to the trace itself).

    dev = deviating_kernel_indices(trace, graph)   # which observed kernels departed
    reduced = deviation_trace(trace, graph)         # a Trace of only those kernels

The band check mirrors :func:`gitm.optimizer.monitor.check_invariants` (same
``INVARIANTS`` band widths) but is applied per *observed kernel index* so each
departure maps back to its original event — the residual pass loses that link
because it keys by predicted op name (many kernels share one op).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gitm.optimizer.invariants import INVARIANTS, Invariant
from gitm.planner.graph import Graph
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace


@dataclass
class DeviationResult:
    """Which observed kernels departed from the predicted graph, and how much it compresses."""

    kept_indices: list[int]  # indices into trace.kernels() that departed
    n_observed: int
    n_predicted: int

    @property
    def n_kept(self) -> int:
        return len(self.kept_indices)

    @property
    def reduction(self) -> float:
        """Fraction of observed kernels dropped as in-band (0.0 if none observed)."""
        return 1.0 - (self.n_kept / self.n_observed) if self.n_observed else 0.0


def _departs(
    ok: KernelEvent,
    node_pred_s: float,
    node_pred_bytes: float,
    inv_kt: Invariant | None,
    inv_mt: Invariant | None,
) -> bool:
    """True if observed kernel ``ok`` is out-of-band vs its predicted node."""
    t_obs = max((ok.end_ns - ok.start_ns) / 1e9, 1e-12)
    t_pred = max(node_pred_s, 1e-12)
    r_kt = (t_obs - t_pred) / t_pred
    if inv_kt is not None and abs(r_kt) > inv_kt.band_width:
        return True
    if (
        inv_mt is not None
        and ok.bytes_read is not None
        and ok.bytes_written is not None
        and node_pred_bytes > 0
    ):
        r_mt = ((ok.bytes_read + ok.bytes_written) - node_pred_bytes) / node_pred_bytes
        if abs(r_mt) > inv_mt.band_width:
            return True
    return False


def deviating_kernel_indices(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> DeviationResult:
    """Indices of observed kernels that depart from the predicted graph.

    Decode is a *repeated* step: the predicted graph models one decode step's
    nodes (see :func:`gitm.planner.graph.predict_graph`), but a real trace spans
    many steps. So we pair the observed kernels to the predicted step
    **cyclically** — kernel ``i`` is compared against predicted node
    ``i % len(pred)`` — instead of truncating at one step's worth (which would
    label every kernel after the first step as "unmodeled" and keep ~everything,
    defeating the point of deviation-only storage). A kernel is a *departure* when
    its kernel-time or memory-traffic residual is out-of-band. With no predicted
    graph at all, every kernel is unmodeled work and is kept.
    """
    obs = trace.kernels()
    pred = graph.nodes
    inv_kt = next((i for i in invariants if i.id == "kernel_time"), None)
    inv_mt = next((i for i in invariants if i.id == "memory_traffic"), None)

    if not pred:
        # Nothing predicted → all observed kernels are unmodeled departures.
        return DeviationResult(kept_indices=list(range(len(obs))), n_observed=len(obs),
                               n_predicted=0)

    kept: list[int] = []
    for i, ok in enumerate(obs):
        pn = pred[i % len(pred)]  # cycle the predicted step across repeated decode steps
        if _departs(ok, pn.prediction.t_pred_s, pn.prediction.bytes, inv_kt, inv_mt):
            kept.append(i)

    return DeviationResult(kept_indices=kept, n_observed=len(obs), n_predicted=len(pred))


def deviation_trace(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> Trace:
    """Return a copy of ``trace`` keeping only kernels that depart from prediction.

    Non-kernel events (memcpy/sync) are dropped — the predicted graph models
    kernels, so deviation is only defined over them. The header (workload id,
    fingerprint, run id, duration) is preserved so the reduced trace is still a
    well-formed, self-describing :class:`Trace`.
    """
    obs = trace.kernels()
    dev = deviating_kernel_indices(trace, graph, invariants)
    kept_events = [obs[i] for i in dev.kept_indices]
    return trace.model_copy(update={"events": kept_events})


def deviation_summary(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> dict:
    """Compact summary of the deviation filter — for the run dir / report.

    ``kept_ops`` counts departures per predicted op so the report can say *which*
    ops are the ones that didn't behave as predicted.
    """
    pred = graph.nodes
    dev = deviating_kernel_indices(trace, graph, invariants)
    kept_ops: dict[str, int] = {}
    for i in dev.kept_indices:
        op = pred[i % len(pred)].op if pred else "<unpredicted>"
        kept_ops[op] = kept_ops.get(op, 0) + 1
    return {
        "n_observed": dev.n_observed,
        "n_predicted": dev.n_predicted,
        "n_kept": dev.n_kept,
        "reduction": dev.reduction,
        "kept_ops": kept_ops,
    }


def write_deviation_jsonl(reduced: Trace, path: str | Path) -> None:
    """Write a deviation-only trace as JSONL via the canonical trace writer.

    Delegates to :func:`gitm.tracer.capture.write_trace_jsonl` so the reduced
    trace uses the exact same on-disk format as a full capture (one definition,
    no drift) and round-trips through the same loaders.
    """
    write_trace_jsonl(path, reduced)
