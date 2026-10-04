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

* ``unreliable`` — the affected numbers do not describe this run. An A/B
  measured under an unreliable entry is excluded from history
  (:func:`gitm.optimizer.history.load_history`), so a fabricated measurement
  cannot rank a lever on every run that follows.
* ``approximate`` — the numbers describe this run under a stated default (a
  batch of 1, A100 peaks, runs/s standing in for tokens/s). Reported, otherwise
  unchanged.

**Scope.** A degradation is either run-wide (``scope=None``: the runner failed,
the graph is a default) or belongs to the one candidate whose A/B it happened
during (``scope=<candidate name>``: the probe refused a restarted engine). The
loop and autoresearch open :meth:`DegradationLog.scope` around each apply, and
each verification record carries :meth:`DegradationLog.measured_under` for its
candidate. So one bad A/B late in a run costs that record, not the valid A/Bs
measured before it.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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

#: What an A/B probe can count, keyed by the runner-output field it read (or
#: ``"runs"`` when it read none): ``(what, short unit, export unit)``. Only
#: ``generated_tokens`` is decode throughput in tokens; every other key records
#: an :data:`AB_UNIT` degradation whose ``used`` is its export unit.
AB_UNITS: dict[str, tuple[str, str, str]] = {
    "generated_tokens": ("decode throughput", "tok/s", "tokens/sec"),
    "decode_steps": ("decode-step throughput", "steps/s", "decode_steps/sec"),
    "events": ("event throughput", "events/s", "events/sec"),
    "runs": ("workload throughput", "runs/s", "runs/sec"),
}
TOKENS = "generated_tokens"


@dataclass(frozen=True)
class Degradation:
    """One fallback: what ran instead of the real thing, and why."""

    stage: str
    used: str
    reason: str
    severity: str = APPROXIMATE
    affects: tuple[str, ...] = ()
    #: ``None`` for the whole run; otherwise the candidate whose A/B it belongs to.
    scope: str | None = None

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
        during = f" [during {self.scope}]" if self.scope else ""
        return f"[{self.severity}] {self.stage}{during}: used {self.used} — {self.reason}{on}"


class DegradationLog:
    """The run's fallbacks, in the order they happened.

    ``record`` also raises a :class:`RuntimeWarning`, matching how the telemetry
    and fail-open layers surface a degraded path, so an embedded caller sees it
    without opening a file. Recording the same fallback twice in one scope (a
    probe that falls back on every rep) keeps one entry. The same fallback in
    another candidate's scope is its own entry, since it taints that A/B too,
    but it warns only once.
    """

    def __init__(self, items: Iterable[Degradation] = ()) -> None:
        self._items: list[Degradation] = []
        self._scope: str | None = None
        self._warned: set[tuple[str, str, str, str]] = set()
        for d in items:
            self._add(d, warn=False)

    @property
    def current_scope(self) -> str | None:
        """The candidate whose A/B is being measured right now, if any."""
        return self._scope

    @contextmanager
    def scope(self, candidate: str) -> Iterator[None]:
        """Attribute what is recorded inside to ``candidate``'s A/B."""
        outer, self._scope = self._scope, candidate
        try:
            yield
        finally:
            self._scope = outer

    def measured_under(self, candidate: str) -> list[Degradation]:
        """The A/B-affecting degradations ``candidate`` was measured under: the
        run-wide ones and those recorded during its own A/B, never another
        candidate's."""
        return [d for d in self._items
                if AFFECTS_AB in d.affects and d.scope in (None, candidate)]

    def record(
        self,
        stage: str,
        *,
        used: str,
        reason: str,
        severity: str = APPROXIMATE,
        affects: Iterable[str] = (),
    ) -> Degradation:
        d = Degradation(stage, used, reason, severity, tuple(affects), self._scope)
        self._add(d, warn=True)
        return d

    def extend(self, items: Iterable[Degradation]) -> None:
        for d in items:
            self._add(d, warn=True)

    def _add(self, d: Degradation, *, warn: bool) -> None:
        if d in self._items:
            return
        self._items.append(d)
        key = (d.stage, d.used, d.reason, d.severity)
        if warn and key not in self._warned:
            self._warned.add(key)
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
        def stages(severity: str) -> list[str]:
            return list(dict.fromkeys(d.stage for d in self._items if d.severity == severity))

        return {
            "n": len(self._items),
            "unreliable": stages(UNRELIABLE),
            "approximate": stages(APPROXIMATE),
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


def ab_unit(degradations: Iterable[Any]) -> str:
    """The :data:`AB_UNITS` key an A/B was measured in, from its own
    degradations (objects or dicts). Tokens unless an :data:`AB_UNIT` entry
    says otherwise."""
    by_export = {v[2]: k for k, v in AB_UNITS.items()}
    for d in degradations:
        stage = d.get("stage") if isinstance(d, dict) else getattr(d, "stage", None)
        used = d.get("used") if isinstance(d, dict) else getattr(d, "used", None)
        if stage == AB_UNIT and used in by_export:
            return by_export[used]
    return TOKENS


def unreliable_ab(degradations: Iterable[Any]) -> list[str]:
    """Stages that make an A/B untrustworthy, from :class:`Degradation` objects
    or their serialised dicts (a ``verification.json`` record's
    ``degradations``). Tolerates missing keys and junk entries."""
    out: list[str] = []
    for d in degradations:
        if isinstance(d, Degradation):
            d = d.to_dict()
        if not isinstance(d, dict):
            continue
        affects = d.get("affects") or ()
        if d.get("severity") == UNRELIABLE and AFFECTS_AB in affects:
            out.append(str(d.get("stage", "?")))
    return out
