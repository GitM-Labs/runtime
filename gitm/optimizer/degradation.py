"""Degradations — every place a run fell back instead of measuring, on the record.

The loop has many honest reasons to fall back: no engine attached, a config it
cannot read, a GPU it does not recognise, a runner that reports no token count,
vLLM absent so autoresearch searches a frozen catalog. Each fallback is fine *as
long as it is said*. What is not fine is the artifact that results: residuals
scored against Llama-2-7B, an A/B timed against a runner that does nothing, a
ranking built from a catalog the installed engine no longer has — written to disk
beside real measurements and indistinguishable from them.

One :class:`DegradationLog` is threaded through a run. Every fallback site records
what it used instead, why, and which artifacts rest on it. The log is then:

* written to ``degradations.json`` on **every** run, so a missing file never
  reads as "clean";
* carried on :class:`~gitm.optimizer.report.Provenance`, so it lands in the
  report and in ``verification.json``;
* summarised in the run summary (``degraded`` / ``degradations``), which
  ``gitm run`` echoes to stderr.

Severity is defined by what it *changes*, not as a label:

* ``unreliable`` — the affected numbers do not describe this run. A run
  carrying an unreliable entry that affects the A/B is excluded from history
  (:func:`gitm.optimizer.history.load_history`), so a fabricated measurement
  cannot rank a lever on every run that follows.
* ``approximate`` — the numbers describe this run under a stated default (a
  batch of 1, A100 peaks, runs/s standing in for tokens/s). Reported, otherwise
  unchanged.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

UNRELIABLE = "unreliable"
APPROXIMATE = "approximate"
SEVERITIES = (UNRELIABLE, APPROXIMATE)

# Stages. Named once so a producer and the consumers that key on them (history,
# the verification export's metric) cannot drift apart.
GRAPH_MODEL = "graph.model"
GRAPH_HARDWARE = "graph.hardware"
GRAPH_BATCH = "graph.batch"
WORKLOAD_RUNNER = "workload.runner"
AB_PROBE = "ab.throughput_probe"
AB_UNIT = "ab.throughput_unit"
AR_CATALOG = "autoresearch.catalog"
AR_PROPOSER = "autoresearch.proposer"
AR_TARGET = "autoresearch.target"
AR_EMPTY = "autoresearch.no_proposals"
AR_SKIPPED = "autoresearch.skipped"

# Artifacts a degradation can rest under. ``ab`` is the one history keys on.
AFFECTS_RESIDUALS = "residuals"
AFFECTS_RANKING = "ranking"
AFFECTS_AB = "ab"
AFFECTS_CLAIMS = "claims"

#: Where the degradations of a run are written.
FILE_NAME = "degradations.json"


@dataclass(frozen=True)
class Degradation:
    """One fallback: what ran instead of the real thing, and why."""

    stage: str
    used: str
    reason: str
    severity: str = APPROXIMATE
    affects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affects"] = list(self.affects)
        return d

    def line(self) -> str:
        """One human-readable sentence, for the report and stderr."""
        on = f" (affects: {', '.join(self.affects)})" if self.affects else ""
        return f"[{self.severity}] {self.stage}: used {self.used} — {self.reason}{on}"


class DegradationLog:
    """The run's fallbacks, in the order they happened.

    ``record`` also raises a :class:`RuntimeWarning`, matching how the telemetry
    and fail-open layers surface a degraded path, so an embedded caller sees it
    without opening a file. Recording the same fallback twice (a probe that falls
    back on every rep) keeps one entry.
    """

    def __init__(self, items: Iterable[Degradation] = ()) -> None:
        self._items: list[Degradation] = []
        for d in items:
            self._add(d, warn=False)

    def record(
        self,
        stage: str,
        *,
        used: str,
        reason: str,
        severity: str = APPROXIMATE,
        affects: Iterable[str] = (),
    ) -> Degradation:
        d = Degradation(stage, used, reason, severity, tuple(affects))
        self._add(d, warn=True)
        return d

    def extend(self, items: Iterable[Degradation]) -> None:
        for d in items:
            self._add(d, warn=True)

    def _add(self, d: Degradation, *, warn: bool) -> None:
        if d in self._items:
            return
        self._items.append(d)
        if warn:
            warnings.warn(f"gitm degraded: {d.line()}", RuntimeWarning, stacklevel=3)

    def __iter__(self) -> Iterator[Degradation]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def unreliable(self) -> list[Degradation]:
        return [d for d in self._items if d.severity == UNRELIABLE]

    def has(self, stage: str) -> bool:
        return any(d.stage == stage for d in self._items)

    def affecting(self, artifact: str) -> list[Degradation]:
        return [d for d in self._items if artifact in d.affects]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._items]

    def summary(self) -> dict[str, Any]:
        """The compact form the run summary carries."""
        return {
            "n": len(self._items),
            "unreliable": [d.stage for d in self.unreliable],
            "approximate": [d.stage for d in self._items if d.severity == APPROXIMATE],
        }

    def write(self, run_dir: str | Path) -> Path:
        """Write ``degradations.json``. Always written: an empty list is the
        statement that nothing fell back, which an absent file cannot make."""
        path = Path(run_dir) / FILE_NAME
        path.write_text(json.dumps({
            "clean": not self._items,
            **self.summary(),
            "items": self.to_dicts(),
        }, indent=2))
        return path


def unreliable_ab(degradations: Iterable[dict[str, Any]]) -> list[str]:
    """Stages of serialised degradations that make a run's A/B untrustworthy.

    Read from exported JSON (``verification.json`` provenance), so it takes dicts
    and tolerates missing keys rather than requiring :class:`Degradation`.
    """
    out: list[str] = []
    for d in degradations:
        if not isinstance(d, dict):
            continue
        affects = d.get("affects") or ()
        if d.get("severity") == UNRELIABLE and AFFECTS_AB in affects:
            out.append(str(d.get("stage", "?")))
    return out
