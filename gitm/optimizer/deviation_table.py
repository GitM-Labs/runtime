"""The deviation table — one typed row per (region, phase).

    table = from_trace("kimi.jsonl", graph, steps=100)
    rank_by_recoverable(table.rows, top=3)

The runtime already measures observed-against-predicted per op, and
:func:`gitm.optimizer.deviation.render_deviation` already prints it. What has not
existed is a *typed row* carrying the four things a lever has to be chosen
against: which region, in which phase, what it is bound by, and how much time is
recoverable there. Every consumer has therefore re-derived its own view —
``largest_residual`` ranks by mean fractional overshoot, ``kernel_roi`` by a
p10-of-self floor, the renderer by a ratio — and none of them agree.

Ranking here is by **recoverable milliseconds**, not by ratio. An op 10x over
prediction that runs twice is worth less than a dominant op 20% over, and a
ranking that says otherwise sends the next experiment at the wrong target.
(:func:`gitm.agents.autoresearch.largest_residual` deliberately still ranks the
other way; the divergence is pinned by a test rather than left to be discovered.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitm.optimizer.bound_classes import normalize_bound
from gitm.optimizer.deviation import predicted_per_op, stream_observed_by_phase
from gitm.optimizer.invariants import INVARIANTS
from gitm.planner.graph import Graph

__all__ = [
    "DeviationRow",
    "DeviationTable",
    "from_trace",
    "from_deviate_json",
    "rank_by_recoverable",
    "render_table",
]

UNMODELED = "<unmodeled>"

#: Verdicts. ``below_floor`` is not headroom — it means the observation came in
#: under what the model said was possible, which is a coverage or attribution
#: defect to investigate, never time to go and claim.
OVER_FLOOR = "over_floor"
WITHIN_BAND = "within_band"
BELOW_FLOOR = "below_floor"
UNMODELED_VERDICT = "unmodeled"


def _default_band() -> float:
    return next((i.band_width for i in INVARIANTS if i.id == "kernel_time"), 0.4)


@dataclass(frozen=True)
class DeviationRow:
    """One region, in one phase, measured against its predicted floor."""

    region: str
    op: str
    layer: int | None
    phase: str  # prefill | decode | unknown
    bound: str | None  # normalized BOUND_CLASSES member
    roofline_bound: str | None  # raw compute | memory | launch
    bound_mixed: bool  # this op's layers disagree about what binds it
    kernels: int
    observed_ms: float
    predicted_ms: float | None
    gap_ms: float | None  # signed: observed - predicted
    recoverable_ms: float  # max(0, gap); 0 when unmodeled
    share_of_device: float
    gap_share: float  # recoverable as a share of observed device time
    modeled: bool
    #: Fraction of this row's time whose phase the kernel named itself. The rest
    #: was inherited from the nearest anchor in time — an inference, wrong
    #: wherever a step mixes phases. A row at 0.0 is a phase *guess*.
    phase_confidence: float
    floor_attribution: str  # graph | prorata | none
    verdict: str


@dataclass(frozen=True)
class DeviationTable:
    rows: list[DeviationRow] = field(default_factory=list)
    observed_ms: float = 0.0
    window_ms: float = 0.0
    kernels: int = 0
    steps: int | None = None
    band: float = 0.4
    phase_stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)


def _bound_per_op(graph: Graph) -> dict[str, tuple[str | None, bool]]:
    """``{op: (bound, layers_disagree)}``, the bound weighted by predicted time.

    An op whose layers disagree still gets a single bound — the one holding the
    most predicted time — but is flagged, because a lever chosen against a
    majority that only just won is a weaker match than one chosen against a
    unanimous op.
    """
    weights: dict[str, dict[str, float]] = {}
    for node in graph.nodes:
        b = getattr(node.prediction, "bound", None)
        if b is None:
            continue
        weights.setdefault(node.op, {})
        weights[node.op][b] = weights[node.op].get(b, 0.0) + node.prediction.t_pred_s
    out: dict[str, tuple[str | None, bool]] = {}
    for op, by_bound in weights.items():
        if not by_bound:
            continue
        top = max(by_bound.items(), key=lambda kv: kv[1])[0]
        out[op] = (top, len(by_bound) > 1)
    return out


def _verdict(observed_s: float, predicted_s: float | None, band: float) -> str:
    if predicted_s is None:
        return UNMODELED_VERDICT
    if predicted_s <= 0:
        return OVER_FLOOR if observed_s > 0 else WITHIN_BAND
    ratio = observed_s / predicted_s
    if ratio > 1.0 + band:
        return OVER_FLOOR
    if ratio < 1.0 - band:
        return BELOW_FLOOR
    return WITHIN_BAND


def _rows_from_parts(
    per_key, *, predicted: dict[str, float] | None,
    bounds: dict[str, tuple[str | None, bool]],
    total_ns: int, span_ns: int, steps: int | None, band: float,
    floor_attribution: str, phase_floors: dict[str, dict[str, float]] | None = None,
) -> list[DeviationRow]:
    scale = steps if steps else 1
    # Observed time per op across all phases — the pro-rata weights.
    op_total: dict[str, int] = {}
    for (op, _layer, _phase), slot in per_key.items():
        op_total[op] = op_total.get(op, 0) + slot[1]

    rows: list[DeviationRow] = []
    for (op, layer, phase), slot in per_key.items():
        count, ns, direct_ns = slot[0], slot[1], slot[2]
        observed_s = ns / 1e9
        modeled = op != UNMODELED

        pred_s: float | None = None
        if modeled:
            if phase_floors is not None:
                per_step = (phase_floors.get(phase) or {}).get(op)
                pred_s = per_step * scale if per_step is not None else None
            elif predicted is not None and op in predicted:
                # One graph for both phases: split its floor by this phase's
                # share of the op's observed time. An assumption, flagged.
                share = ns / op_total[op] if op_total.get(op) else 1.0
                pred_s = predicted[op] * scale * share

        gap_s = (observed_s - pred_s) if pred_s is not None else None
        recoverable_s = max(0.0, gap_s) if gap_s is not None else 0.0
        bound_raw, mixed = bounds.get(op, (None, False))

        rows.append(DeviationRow(
            region=op if layer is None else f"{op}@L{layer}",
            op=op,
            layer=layer,
            phase=phase,
            bound=normalize_bound(bound_raw),
            roofline_bound=bound_raw,
            bound_mixed=mixed,
            kernels=count,
            observed_ms=observed_s * 1e3,
            predicted_ms=pred_s * 1e3 if pred_s is not None else None,
            gap_ms=gap_s * 1e3 if gap_s is not None else None,
            recoverable_ms=recoverable_s * 1e3,
            share_of_device=(ns / total_ns) if total_ns else 0.0,
            gap_share=(recoverable_s * 1e9 / total_ns) if total_ns else 0.0,
            modeled=modeled,
            phase_confidence=(direct_ns / ns) if ns else 0.0,
            floor_attribution=floor_attribution if pred_s is not None else "none",
            verdict=_verdict(observed_s, pred_s, band),
        ))
    return rows


def from_trace(
    path: str | Path,
    graphs: Graph | dict[str, Graph] | None = None,
    *,
    steps: int | None,
    pid: int | None = None,
    device: int | None = None,
    propagate: bool = True,
    band: float | None = None,
) -> DeviationTable:
    """Build the table from a captured trace and the predicted graph(s).

    ``graphs`` may be one :class:`Graph` — whose floor is split across phases by
    observed time share (``floor_attribution="prorata"``) — or a mapping of phase
    to graph, where each phase is measured against its own prediction
    (``"graph"``). Predicting costs no hardware, so two graphs is the better
    input where both shapes are known.

    ``steps`` scales a per-step floor to the captured window and is **required**:
    passing ``None`` with a graph raises rather than silently comparing a
    one-step prediction against a whole-window observation. That mistake is
    already live in ``gitm deviate --as-json``, which computes
    ``pred * (steps or 1)`` and prints no warning, while the renderer beside it
    does warn. Pass ``steps=1`` explicitly if the trace really is one step.
    """
    if graphs is not None and steps is None:
        raise ValueError(
            "steps is required when a graph is given: the graph predicts ONE step, "
            "and comparing that against a whole-window observation is not a ratio. "
            "Pass steps=1 explicitly if the capture really is a single step."
        )
    band = _default_band() if band is None else band
    per_key, stats, n, total_ns, span_ns = stream_observed_by_phase(
        path, propagate=propagate, pid=pid, device=device)

    predicted: dict[str, float] | None = None
    phase_floors: dict[str, dict[str, float]] | None = None
    bounds: dict[str, tuple[str | None, bool]] = {}
    attribution = "none"

    if isinstance(graphs, dict):
        phase_floors = {ph: predicted_per_op(g) for ph, g in graphs.items()}
        for g in graphs.values():
            bounds.update(_bound_per_op(g))
        attribution = "graph"
    elif graphs is not None:
        predicted = predicted_per_op(graphs)
        bounds = _bound_per_op(graphs)
        attribution = "prorata"

    rows = _rows_from_parts(
        per_key, predicted=predicted, bounds=bounds, total_ns=total_ns,
        span_ns=span_ns, steps=steps, band=band,
        floor_attribution=attribution, phase_floors=phase_floors)

    return DeviationTable(
        rows=rows, observed_ms=total_ns / 1e6, window_ms=span_ns / 1e6,
        kernels=n, steps=steps, band=band, phase_stats=stats)


def from_deviate_json(src: str | Path | dict, *, steps: int | None = None) -> DeviationTable:
    """Rehydrate the artifact ``gitm deviate --as-json`` already emits.

    That payload has no per-op phase or bound, so every row comes back
    ``phase="unknown"`` with ``bound=None``. This reads what is there rather than
    guessing at what is not — use :func:`from_trace` when the trace is available.
    """
    doc = src if isinstance(src, dict) else json.loads(Path(src).read_text(encoding="utf-8"))
    ops = doc.get("ops") or {}
    total_ns = int(float(doc.get("device_time_s") or 0.0) * 1e9)
    band = float(doc.get("band_width") or _default_band())
    steps = steps if steps is not None else doc.get("steps")

    per_key: dict[tuple[str, int | None, str], list] = {}
    predicted: dict[str, float] = {}
    for op, rec in ops.items():
        ns = int(float(rec.get("observed_s") or 0.0) * 1e9)
        per_key[(op, None, "unknown")] = [int(rec.get("kernels") or 0), ns, 0]
        floor = rec.get("floor_s")
        if floor is not None and op != UNMODELED:
            predicted[op] = float(floor)

    # floor_s in the artifact is ALREADY scaled by steps, so do not scale again.
    rows = _rows_from_parts(
        per_key, predicted=predicted or None, bounds={}, total_ns=total_ns,
        span_ns=int(float(doc.get("window_s") or 0.0) * 1e9), steps=1, band=band,
        floor_attribution="graph")

    return DeviationTable(
        rows=rows, observed_ms=total_ns / 1e6,
        window_ms=float(doc.get("window_s") or 0.0) * 1e3,
        kernels=int(doc.get("n_kernels") or 0), steps=steps, band=band,
        phase_stats=doc.get("phase_stats") or {})


def rank_by_recoverable(
    rows: list[DeviationRow], *, top: int | None = None,
    phase: str | None = None, bound: str | None = None, min_share: float = 0.0,
) -> list[DeviationRow]:
    """Rows worth an experiment, most recoverable time first.

    Unmodeled work never ranks: it is the graph's coverage gap, not headroom, and
    reading it as time to recover is the error the modeled/unmodeled split exists
    to prevent. ``phase`` and ``bound`` filter to one cell — taking the top row
    from each is how a batch gets spread across the lever space instead of
    stacking three specs on the same op.
    """
    out = [r for r in rows if r.modeled and r.recoverable_ms > 0]
    if phase is not None:
        out = [r for r in out if r.phase == phase]
    if bound is not None:
        out = [r for r in out if r.bound == bound]
    if min_share > 0:
        out = [r for r in out if r.gap_share >= min_share]
    out.sort(key=lambda r: (-r.recoverable_ms, -r.observed_ms, r.region))
    return out[:top] if top else out


def render_table(table: DeviationTable, *, top: int = 20) -> str:
    """The table as text, most recoverable first."""
    head = (f"observed  {table.kernels:,} kernels, {table.observed_ms / 1e3:.3f} s device time"
            f" over a {table.window_ms / 1e3:.1f} s window")
    out = [head]
    if table.steps:
        out.append(f"window    {table.steps:,} steps")
    else:
        out.append("window    steps unknown — floors are UNSCALED")

    ranked = rank_by_recoverable(table.rows, top=top)
    if not ranked:
        out.append("\nno recoverable time found")
        return "\n".join(out)

    inferred = [r for r in ranked if r.phase != "unknown" and r.phase_confidence < 0.5]
    if inferred:
        out.append(f"  NOTE: {len(inferred)} of {len(ranked)} rows have a phase inferred from "
                   "neighbouring kernels rather than observed; see phase_conf.")
    out.append("")
    out.append(f"  {'region':24s} {'phase':8s} {'bound':14s} {'obs_ms':>9s} {'floor_ms':>9s} "
               f"{'gap_ms':>9s} {'%dev':>6s} {'conf':>5s}")
    for r in ranked:
        floor = f"{r.predicted_ms:9.2f}" if r.predicted_ms is not None else "        -"
        gap = f"{r.gap_ms:9.2f}" if r.gap_ms is not None else "        -"
        flag = " *" if r.bound_mixed else ""
        out.append(
            f"  {r.region[:24]:24s} {r.phase:8s} {(r.bound or '-')[:14]:14s} "
            f"{r.observed_ms:9.2f} {floor} {gap} {r.gap_share:5.1%} "
            f"{r.phase_confidence:4.0%}{flag}")
    if any(r.bound_mixed for r in ranked):
        out.append("  * this op's layers disagree about what binds it")
    return "\n".join(out)
