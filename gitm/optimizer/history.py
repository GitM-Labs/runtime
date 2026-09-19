"""Read back what previous runs measured.

    load_history(runs_dir) -> History
    record_for(history, "kv_cache_dtype_fp8", gpu_sku=sku) -> LeverRecord | None

Every run writes ``runs/<run_id>/verification.json`` — each lever it tried, the
A/B behind it, and whether the rollback gate kept it. Nothing has ever opened one
again, so the loop ranks candidates from the same hand-authored constants on every
run no matter what it measured last time. This module is the missing direction.

No new writer is needed: the export already carries the knob, the speedup, the rep
count, whether the gain cleared the noise band, the gate's keep/rollback verdict,
and the GPU it ran on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LeverRecord",
    "History",
    "load_history",
    "record_for",
    "render_history",
]

EXPORT_NAME = "verification.json"


@dataclass(frozen=True)
class LeverRecord:
    """What every past run, taken together, says about one lever on one GPU.

    ``conflicted`` rather than a single verdict: a lever that won twice and lost
    twice has not "averaged out to neutral", it has behaved differently under
    conditions this record does not capture. Collapsing that into a mean would
    present a disagreement as a measurement.
    """

    intervention_name: str
    gpu_sku: str | None
    #: Distinct run folders this lever appears in. Kept apart from ``attempts``
    #: because five A/Bs inside one run is far weaker evidence than five across
    #: five runs, and a single count cannot tell those apart.
    runs: int
    #: Individual A/Bs. ``wins + losses + inconclusive`` sums to this, not to
    #: ``runs`` — one run can measure the same lever more than once.
    attempts: int
    wins: int
    losses: int
    inconclusive: int
    #: ``None`` when no attempt recorded a usable delta, for the same reason
    #: :func:`record_for` returns ``None`` for a lever never tried: "we have no
    #: number" and "the number was zero" point opposite ways, and 0.0 for both
    #: would read as a measured no-op.
    mean_delta: float | None
    best_delta: float | None
    worst_delta: float | None
    last_run_id: str | None

    @property
    def conflicted(self) -> bool:
        return self.wins > 0 and self.losses > 0


@dataclass(frozen=True)
class History:
    """Every lever seen across the runs that were readable.

    ``skipped`` is part of the result, not a log line. A history assembled from
    three of twenty runs looks exactly like a history of three runs, and a caller
    that cannot tell those apart will read a thin record as a weak lever.

    ``filtered`` is counted separately and deliberately: a run excluded because it
    ran on a different GPU is the filter working, not history going missing. Adding
    the two together would make a clean read of one box look like a damaged one.
    """

    records: dict[tuple[str, str | None], LeverRecord] = field(default_factory=dict)
    runs_read: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    filtered: int = 0

    def __len__(self) -> int:
        return len(self.records)


def _verdict(result: dict[str, Any]) -> str:
    """win / loss / inconclusive for one A/B.

    ``kept`` is the rollback gate's decision and is authoritative — the export is
    explicit that the gate decides, not the raw delta. ``significant`` only says
    the gain cleared the measured noise band, so a kept-but-insignificant result
    is a lever that was allowed to stay without having proven anything.
    """
    if not result.get("kept"):
        return "loss"
    return "win" if result.get("significant") else "inconclusive"


def load_history(runs_dir: str | Path, *, gpu_sku: str | None = None) -> History:
    """Aggregate every readable ``verification.json`` under ``runs_dir``.

    ``gpu_sku`` filters to one GPU: a result measured on an H100 says nothing
    about an MI355X, so a caller ranking for one box should not see the other's
    record. Runs are ordered by the export's mtime — the export carries a
    ``run_id`` but no timestamp, and the run directories are UUIDs, so the file
    is the only ordering available for ``last_run_id``.
    """
    runs_dir = Path(runs_dir)
    skipped: dict[str, str] = {}
    filtered = 0
    if not runs_dir.is_dir():
        return History(skipped={str(runs_dir): "runs dir does not exist"})

    exports: list[tuple[float, str, str | None, list[dict[str, Any]]]] = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        path = d / EXPORT_NAME
        if not path.exists():
            skipped[d.name] = f"no {EXPORT_NAME}"
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            skipped[d.name] = f"unreadable: {exc}"
            continue
        if not isinstance(doc, dict):
            skipped[d.name] = "not a JSON object"
            continue
        results = doc.get("results")
        if not isinstance(results, list):
            skipped[d.name] = "no results array"
            continue
        sku = (doc.get("environment") or {}).get("gpu_sku")
        if gpu_sku is not None and sku != gpu_sku:
            filtered += 1
            continue
        run_id = (doc.get("provenance") or {}).get("run_id") or d.name
        exports.append((path.stat().st_mtime, run_id, sku, results))

    exports.sort(key=lambda e: e[0])

    acc: dict[tuple[str, str | None], dict[str, Any]] = {}
    for _mtime, run_id, sku, results in exports:
        for r in results:
            name = r.get("intervention_name")
            if not name:
                continue
            key = (name, sku)
            a = acc.setdefault(
                key,
                {"runs": set(), "attempts": 0, "win": 0, "loss": 0,
                 "inconclusive": 0, "deltas": [], "last_run_id": None},
            )
            a["runs"].add(run_id)
            a["attempts"] += 1
            a[_verdict(r)] += 1
            delta = r.get("delta")
            if delta is None and r.get("speedup") is not None:
                delta = r["speedup"] - 1.0
            if isinstance(delta, (int | float)):
                a["deltas"].append(float(delta))
            a["last_run_id"] = run_id

    records = {}
    for (name, sku), a in acc.items():
        deltas = a["deltas"]
        records[(name, sku)] = LeverRecord(
            intervention_name=name,
            gpu_sku=sku,
            runs=len(a["runs"]),
            attempts=a["attempts"],
            wins=a["win"],
            losses=a["loss"],
            inconclusive=a["inconclusive"],
            mean_delta=(sum(deltas) / len(deltas)) if deltas else None,
            best_delta=max(deltas) if deltas else None,
            worst_delta=min(deltas) if deltas else None,
            last_run_id=a["last_run_id"],
        )

    return History(records=records, runs_read=len(exports), skipped=skipped,
                   filtered=filtered)


def record_for(
    history: History, intervention_name: str, *, gpu_sku: str | None = None
) -> LeverRecord | None:
    """The record for one lever, or ``None`` if no run ever tried it.

    ``None`` rather than a zeroed record on purpose: "never measured" and
    "measured, and it did nothing" must not look alike to a caller deciding
    whether this lever is worth an experiment. They point opposite ways.
    """
    return history.records.get((intervention_name, gpu_sku))


def _fit(value: str, width: int) -> str:
    """``value`` inside ``width``, marked when it did not fit.

    A silent cut reads as the whole value, which is how a truncated SKU came to
    name a board it was not measured on.
    """
    return value if len(value) <= width else value[: width - 1] + "\u2026"


def _column(rows: list[LeverRecord], of, header: str, *, cap: int) -> int:
    """Width of a column: the longest value shown, bounded by ``cap``."""
    return min(max([len(header)] + [len(of(r)) for r in rows]), cap)


def render_history(history: History, *, top: int = 20) -> str:
    """The record as a table, most-tried first."""
    head = f"read {history.runs_read} runs"
    if history.filtered:
        head += f", {history.filtered} filtered out by gpu"
    if history.skipped:
        head += f", skipped {len(history.skipped)}"
    out = [head]
    if history.skipped:
        reasons: dict[str, int] = {}
        for reason in history.skipped.values():
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        out.append("  skipped: " + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())))
    if not history.records:
        out.append("\nno levers recorded")
        return "\n".join(out)

    rows = sorted(history.records.values(), key=lambda r: (-r.runs, r.intervention_name))
    unknown = sum(1 for r in rows if r.gpu_sku is None)
    if unknown:
        out.append(
            f"  WARNING: {unknown} record(s) have no GPU. Runs that did not report a SKU "
            "all key together, so results measured on DIFFERENT boxes are merged here. "
            "Set GITM_GPU_SKU on boxes where NVML cannot name the device (any ROCm/AMD "
            "part) so these separate correctly."
        )
    shown = rows[:top]
    # Sized to what is actually on screen, because a fixed width silently ate the
    # part of a SKU that identifies the board: "AMD Instinct MI355X" rendered as
    # "MI355", and an H100 HBM3 and HBM3e came out byte-identical. Two boxes that
    # look like one box is the exact confusion keying on gpu_sku exists to stop.
    name_w = _column(shown, lambda r: r.intervention_name, "lever", cap=40)
    gpu_w = _column(shown, lambda r: r.gpu_sku or "-", "gpu", cap=30)

    out.append("")
    out.append(f"  {'lever':{name_w}s} {'gpu':{gpu_w}s} {'runs':>5s} {'a/b':>4s} "
               f"{'won':>4s} {'lost':>5s} {'incon':>6s} {'mean':>8s}  last")
    for r in shown:
        flag = "  CONFLICTED" if r.conflicted else ""
        mean = f"{r.mean_delta:+.1%}" if r.mean_delta is not None else "n/a"
        out.append(
            f"  {_fit(r.intervention_name, name_w):{name_w}s} "
            f"{_fit(r.gpu_sku or '-', gpu_w):{gpu_w}s} "
            f"{r.runs:5d} {r.attempts:4d} {r.wins:4d} {r.losses:5d} {r.inconclusive:6d} "
            f"{mean:>7s}  {(r.last_run_id or '-')[:8]}{flag}"
        )
    if len(rows) > top:
        out.append(f"  ... {len(rows) - top} more")
    return "\n".join(out)
