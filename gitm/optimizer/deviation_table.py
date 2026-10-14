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


def _layer_floors(graph: Graph) -> dict[tuple[str, int], float]:
    """Predicted seconds per ``(op, layer)`` for a graph that models layers apart.

    :func:`predicted_per_op` sums over layers, which is the right shape for a row
    keyed by op alone and the wrong one for a row keyed by ``(op, layer)`` — that
    row would be measured against every layer's prediction at once, so a 32-layer
    op reads ~32x under its floor and drops out of the ranking entirely. A graph
    whose nodes carry no layer yields ``{}``, and the caller splits the op floor
    by observed share instead.
    """
    out: dict[tuple[str, int], float] = {}
    for node in graph.nodes:
        if node.layer is None:
            continue
        key = (node.op, node.layer)
        out[key] = out.get(key, 0.0) + node.prediction.t_pred_s
    return out


def _merge_bounds(
    per_phase: dict[str, dict[str, tuple[str | None, bool]]],
) -> dict[str, tuple[str | None, bool]]:
    """One bound per op, for rows whose phase has no graph of its own.

    Unanimous across phases is safe to carry; disagreement is not, so it reads
    ``None`` and flags mixed rather than taking whichever graph came last in the
    mapping. Attention is compute bound in prefill and memory bound in decode,
    and handing a row the other phase's answer aims the search at the wrong
    lever — silently, and differently if the mapping is reordered.
    """
    seen: dict[str, set[str | None]] = {}
    mixed_any: dict[str, bool] = {}
    for by_op in per_phase.values():
        for op, (b, mixed) in by_op.items():
            seen.setdefault(op, set()).add(b)
            mixed_any[op] = mixed_any.get(op, False) or mixed
    out: dict[str, tuple[str | None, bool]] = {}
    for op, values in seen.items():
        out[op] = (next(iter(values)), mixed_any[op]) if len(values) == 1 else (None, True)
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
    layer_floors: dict[tuple[str, int], float] | None = None,
    phase_floors: dict[str, dict[str, float]] | None = None,
    phase_layer_floors: dict[str, dict[tuple[str, int], float]] | None = None,
    phase_bounds: dict[str, dict[str, tuple[str | None, bool]]] | None = None,
) -> list[DeviationRow]:
    scale = steps if steps else 1
    # A per-phase graph prices only its own phase; a single graph prices them all,
    # so a share is never taken against time the floor being divided does not cover.
    scoped = phase_floors is not None

    def _floor_sets(phase: str):
        floors = (phase_floors.get(phase) or {}) if scoped else (predicted or {})
        lfloors = (((phase_layer_floors or {}).get(phase) or {}) if scoped
                   else (layer_floors or {}))
        return floors, lfloors

    # Which rows the graph prices by layer on their own, and how much observed
    # time is left for the rest of the op. A graph may hold both shapes at once —
    # GLM prices a final ``rms_norm`` apart from its twelve per-layer norms — and
    # ``predicted_per_op`` sums all thirteen, so an unscoped row that took that
    # sum would be handed the layer predictions a second time.
    claimed: dict[tuple[str, str | None], set[int]] = {}
    claimed_obs: dict[tuple[str, int, str | None], int] = {}
    unclaimed_obs: dict[tuple[str, str | None], int] = {}
    for (op, layer, phase), slot in per_key.items():
        sc = phase if scoped else None
        _, lfloors = _floor_sets(phase)
        if op != UNMODELED and layer is not None and (op, layer) in lfloors:
            claimed.setdefault((op, sc), set()).add(layer)
            claimed_obs[(op, layer, sc)] = claimed_obs.get((op, layer, sc), 0) + slot[1]
        else:
            unclaimed_obs[(op, sc)] = unclaimed_obs.get((op, sc), 0) + slot[1]

    rows: list[DeviationRow] = []
    for (op, layer, phase), slot in per_key.items():
        count, ns, direct_ns = slot[0], slot[1], slot[2]
        observed_s = ns / 1e9
        sc = phase if scoped else None

        # The floor, and the observed time that same floor prices. A row keyed by
        # layer takes the layer's own prediction where the graph has one; every
        # other row of that op shares what is left of the op floor once the layer
        # rows have taken theirs, divided by observed time, and reads ``prorata``
        # rather than claiming a graph number.
        floors, lfloors = _floor_sets(phase)
        base: float | None = None
        denom = 0
        if op != UNMODELED:
            if layer is not None and (op, layer) in lfloors:
                # Within the floor's own scope, not this row alone: a single
                # graph prices one step across both phases, so one layer seen in
                # prefill and again in decode is two rows sharing one prediction.
                # Handing each the whole floor would price that layer twice.
                base, denom = lfloors[(op, layer)], claimed_obs.get((op, layer, sc), ns)
            elif op in floors:
                taken = sum(lfloors.get((op, ly), 0.0) for ly in claimed.get((op, sc), ()))
                residual = floors[op] - taken
                # Nothing left means every prediction this op has is already on a
                # layer row. Re-using one here would count it twice, and a floor
                # of zero would read as a measured no-op, so the row has none.
                if residual > 0:
                    base, denom = residual, unclaimed_obs.get((op, sc), 0)

        pred_s: float | None = None
        attribution = "none"
        if base is not None:
            share = (ns / denom) if denom else 1.0
            pred_s = base * scale * share
            # "graph" only when a per-phase graph priced this row whole. A single
            # graph cannot speak per phase at all, and a divided floor is an
            # attribution either way, so both read "prorata".
            attribution = "graph" if (scoped and share >= 1.0) else "prorata"

        # A row has a floor or it does not. An op the trace recognizes but the
        # graph never predicts is the graph's coverage gap exactly as an
        # unclassified kernel is, and calling it modeled would let a coverage
        # consumer count unpriced work as priced.
        modeled = pred_s is not None
        gap_s = (observed_s - pred_s) if pred_s is not None else None
        recoverable_s = max(0.0, gap_s) if gap_s is not None else 0.0

        by_op = (phase_bounds or {}).get(phase) if scoped else None
        bound_raw, mixed = (by_op if by_op is not None else bounds).get(op, (None, False))

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
            floor_attribution=attribution,
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
    one-step prediction against a whole-window observation. ``gitm deviate
    --json`` takes the softer form of the same position — it states no floor at
    all rather than an unscaled one — and the renderer beside it prints UNSCALED
    instead of a ratio. Pass ``steps=1`` explicitly if the trace really is one
    step.
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
    layer_floors: dict[tuple[str, int], float] | None = None
    phase_floors: dict[str, dict[str, float]] | None = None
    phase_layer_floors: dict[str, dict[tuple[str, int], float]] | None = None
    phase_bounds: dict[str, dict[str, tuple[str | None, bool]]] | None = None
    bounds: dict[str, tuple[str | None, bool]] = {}

    if isinstance(graphs, dict):
        phase_floors = {ph: predicted_per_op(g) for ph, g in graphs.items()}
        phase_layer_floors = {ph: _layer_floors(g) for ph, g in graphs.items()}
        phase_bounds = {ph: _bound_per_op(g) for ph, g in graphs.items()}
        # Only for a phase with no graph of its own, e.g. an unanchored capture.
        bounds = _merge_bounds(phase_bounds)
    elif graphs is not None:
        predicted = predicted_per_op(graphs)
        layer_floors = _layer_floors(graphs)
        bounds = _bound_per_op(graphs)

    rows = _rows_from_parts(
        per_key, predicted=predicted, bounds=bounds, total_ns=total_ns,
        span_ns=span_ns, steps=steps, band=band, layer_floors=layer_floors,
        phase_floors=phase_floors, phase_layer_floors=phase_layer_floors,
        phase_bounds=phase_bounds)

    return DeviationTable(
        rows=rows, observed_ms=total_ns / 1e6, window_ms=span_ns / 1e6,
        kernels=n, steps=steps, band=band, phase_stats=stats)


def from_deviate_json(src: str | Path | dict, *, steps: int | None = None) -> DeviationTable:
    """Rehydrate the artifact ``gitm deviate --json`` already emits.

    That payload has no per-op phase or bound, so every row comes back
    ``phase="unknown"`` with ``bound=None``. This reads what is there rather than
    guessing at what is not — use :func:`from_trace` when the trace is available.

    A payload emitted without ``--steps`` carries no floors at all (its
    ``floors_scaled`` is false), so every row rehydrates unmodeled. That is the
    honest reading: there was never a floor to state.
    """
    doc = src if isinstance(src, dict) else json.loads(Path(src).read_text(encoding="utf-8"))
    ops = doc.get("ops") or {}
    total_ns = int(float(doc.get("device_time_s") or 0.0) * 1e9)
    # ``or`` would read an exact-zero tolerance as missing and silently widen it
    # to the default, changing every verdict in the rehydrated table.
    band_value = doc.get("band_width")
    band = _default_band() if band_value is None else float(band_value)
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
        span_ns=int(float(doc.get("window_s") or 0.0) * 1e9), steps=1, band=band)

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
