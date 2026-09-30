"""Granger tests on the residual subgraph, with a record of what actually ran.

Granger-causality ranks op pairs by Granger F p-value. The series are each op's
residuals in launch order, not per engine step, so the lag axis is not time
and a ranked pair is an exploratory association, never a cause. Claims therefore carry only the
analysis status (``granger_evidence``); the pairs go to the run's JSON artifact.

Every outcome is recorded on ``RankedHypotheses``: whether statsmodels imported,
how many pair tests ran and completed, and the first error and warning of each
kind. An empty hypothesis list alone never means "no signal".

A doubly-robust estimator runs alongside Granger (see gitm/optimizer/dr.py).
"""

from __future__ import annotations

import functools
import inspect
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from gitm.optimizer.monitor import Residuals
from gitm.planner.graph import Graph

_MAX_MESSAGE = 200
# The directory part of an absolute POSIX path; the lookbehind leaves relative
# text such as "a/b/c" alone, what follows the "~" / "<python>" placeholders,
# and URLs ("https://host/a/b": each slash follows a word character or a slash).
_ABS_PATH = re.compile(r"(?<![\w.~>/])(?:/[^\s/:'\"]+)+/")
# System roots a Python install can live under (a distro python3 has
# sys.prefix "/usr"). Naming them "<python>" would mislabel e.g. /usr/local/cuda.
_SYSTEM_ROOTS = frozenset({"", "/", "/usr", "/usr/local", "/opt"})
# Failures that point at the statsmodels API rather than the data.
_API_FAILURES = ("TypeError", "AttributeError")


class AnalysisStatus(str, Enum):
    """Outcome of a Granger or doubly-robust run (see RankedHypotheses)."""

    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"  # no pair produced an estimate
    PARTIAL = "partial"  # some pairs produced no estimate
    DEGRADED = "degraded"  # every pair produced one, some only with a warning or fallback
    OK = "ok"


# Warnings about the library's API, not about the data or the fit: recorded,
# but they do not make an estimate degraded (a dependency bump that adds one
# must not flip every pair).
_API_WARNINGS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


def degrading(warning_category: type[Warning]) -> bool:
    """Whether a warning raised during a fit makes that estimate degraded."""
    return not issubclass(warning_category, _API_WARNINGS)


@dataclass
class Hypothesis:
    cause_op: str
    effect_op: str
    p_value: float
    direction: str  # "+ slower", "- faster"
    notes: str = ""
    #: The estimate rests on a fit fallback or a warning (see the run record);
    #: its p-value is not to be read at face value.
    degraded: bool = False


@dataclass
class RankedHypotheses:
    hypotheses: list[Hypothesis]
    # Defaults describe "nobody ran Granger", so other constructors (dr.py)
    # never read as a completed analysis.
    status: AnalysisStatus = AnalysisStatus.NOT_RUN
    pairs_attempted: int = 0
    pairs_completed: int = 0
    pairs_degraded: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    first_errors: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    first_warnings: dict[str, str] = field(default_factory=dict)
    max_lag: int | None = None
    min_obs: int | None = None
    series_lengths: dict[str, int] = field(default_factory=dict)
    excluded_ops: dict[str, int] = field(default_factory=dict)
    n_obs_used: int | None = None
    #: Why an INSUFFICIENT_DATA result had nothing to test, when set.
    reason: str | None = None
    #: Ops never tried as a cause, with the count that ruled them out (DR: the
    #: number of out-of-band samples, too few or too many to estimate from).
    skipped_causes: dict[str, int] = field(default_factory=dict)

    def top(self, n: int = 5) -> list[Hypothesis]:
        return self.hypotheses[:n]

    def record_failure(self, kind: str, message: object) -> None:
        """Count one failure of ``kind``; keep the first message of each kind."""
        _record(self.failures, self.first_errors, kind, message)

    def record_warning(self, kind: str, message: object) -> None:
        _record(self.warnings, self.first_warnings, kind, message)

    def finalize_status(self) -> None:
        """Status from the pair counts, once pairs were attempted.

        ``pairs_completed`` counts pairs that produced an estimate;
        ``pairs_degraded`` is the subset that needed a fallback or raised a
        warning. A run whose estimates all exist is never FAILED.
        """
        if self.pairs_completed == 0:
            self.status = AnalysisStatus.FAILED
        elif self.pairs_completed < self.pairs_attempted:
            self.status = AnalysisStatus.PARTIAL
        elif self.pairs_degraded:
            self.status = AnalysisStatus.DEGRADED
        else:
            self.status = AnalysisStatus.OK

    def rank(self, key) -> None:
        """Sort hypotheses by ``key``, clean estimates before degraded ones.

        A degraded estimate (e.g. a separated propensity) can carry p = 0; it
        must not push clean results out of a top-N list read by p-value alone.
        """
        self.hypotheses.sort(key=lambda h: (h.degraded, key(h)))

    def summary(self, top_n: int = 5) -> dict:
        """JSON-ready record of the run, for the artifact that holds the pairs."""
        return {
            "status": self.status.value,
            "pairs_attempted": self.pairs_attempted,
            "pairs_completed": self.pairs_completed,
            "pairs_degraded": self.pairs_degraded,
            "failures": dict(self.failures),
            "first_errors": dict(self.first_errors),
            "warnings": dict(self.warnings),
            "first_warnings": dict(self.first_warnings),
            "max_lag": self.max_lag,
            "min_obs": self.min_obs,
            "series_lengths": dict(self.series_lengths),
            "excluded_ops": dict(self.excluded_ops),
            "n_obs_used": self.n_obs_used,
            "reason": self.reason,
            "skipped_causes": dict(self.skipped_causes),
            "top_pairs": [
                {
                    "cause": h.cause_op,
                    "effect": h.effect_op,
                    "p_value": h.p_value,
                    "exploratory": True,
                    "degraded": h.degraded,
                }
                for h in self.top(top_n)
            ],
        }


@functools.cache
def _private_prefixes() -> tuple[tuple[re.Pattern[str], str], ...]:
    """Home and install prefixes, longest first, each matched only as a whole path.

    Replaced before the regex: they may contain spaces it cannot see past, and
    they carry the user name. Computed on first use, not at import: Path.home()
    raises when HOME is unset and the uid has no passwd entry (arbitrary-uid
    containers), and that must not stop the CLI from importing.
    """
    named: dict[str, str] = {}
    for prefix in (sys.prefix, sys.base_prefix):
        if prefix.rstrip("/") not in _SYSTEM_ROOTS:
            named[prefix.rstrip("/")] = "<python>"
    try:
        home = str(Path.home()).rstrip("/")
    except (RuntimeError, KeyError, OSError):
        home = ""
    if home.startswith("/") and home not in _SYSTEM_ROOTS:  # a relative HOME names no path
        named[home] = "~"  # wins over an install prefix that equals it
    return tuple(
        (re.compile(r"(?<![\w/.])" + re.escape(prefix) + r"(?=/|\s|$|['\":,;)\]])"), label)
        for prefix, label in sorted(named.items(), key=lambda kv: len(kv[0]), reverse=True)
    )


def _clean_message(exc_or_msg: object) -> str:
    """One-line message without home/install paths, capped for the artifact.

    Artifacts and reports get shared, so the user's home directory and the
    Python install prefix are replaced first, then the directory part of any
    other absolute path is dropped (the file name stays).
    """
    text = " ".join(str(exc_or_msg).split())[: _MAX_MESSAGE * 4]
    text = text.replace("file:///", "/")  # a file URI is a path; the rules below apply
    for pattern, label in _private_prefixes():
        text = pattern.sub(label, text)
    return _ABS_PATH.sub("", text)[:_MAX_MESSAGE]


def _record(counts: dict[str, int], firsts: dict[str, str], kind: str, message: object) -> None:
    counts[kind] = counts.get(kind, 0) + 1
    if kind not in firsts:  # only the first of each kind is kept, so only it is cleaned
        firsts[kind] = _clean_message(message)


def attribute(
    residuals: Residuals,
    graph: Graph,
    max_lag: int = 2,
) -> RankedHypotheses:
    """Granger tests over every ordered op pair, ranked by p-value, with a run record.

    For each ordered pair (cause, effect) of distinct ops, fit a VAR-style
    Granger F-test on the residual series. The ranking is exploratory (see the
    module docstring): it is written to the run's JSON, not used as a cause in
    claims or to choose interventions. The returned record says what ran.
    """
    # The test's regression with a constant needs more than 3*max_lag + 1
    # observations; shorter series always raise "Insufficient observations".
    min_obs = 3 * max_lag + 2
    out = RankedHypotheses(hypotheses=[], max_lag=max_lag, min_obs=min_obs)
    try:
        from statsmodels.tsa.stattools import (
            grangercausalitytests,  # type: ignore[import-not-found]
        )
    except Exception as exc:  # ImportError, or e.g. a ValueError from a binary mismatch
        out.status = AnalysisStatus.UNAVAILABLE
        out.record_failure(type(exc).__name__, exc)
        return out
    # statsmodels < 0.15 prints every test table unless verbose=False, an
    # argument 0.15 removed. Passing it only where it exists avoids
    # redirecting stdout, which is process-global and unsafe across threads.
    kwargs = {"verbose": False} if "verbose" in inspect.signature(grangercausalitytests).parameters else {}

    # Group residuals by op into series in launch order (not per engine step).
    series: dict[str, list[float]] = {}
    for kr in residuals.per_kernel:
        series.setdefault(kr.op, []).append(kr.r_kt)
    out.series_lengths = {op: len(vals) for op, vals in series.items()}
    out.excluded_ops = {op: n for op, n in out.series_lengths.items() if n < min_obs}

    ops = [op for op in series if op not in out.excluded_ops]
    if len(ops) < 2:
        out.status = AnalysisStatus.INSUFFICIENT_DATA
        out.reason = f"fewer than 2 ops with at least {min_obs} samples"
        return out

    n = min(len(series[op]) for op in ops)
    out.n_obs_used = n
    for cause in ops:
        for effect in ops:
            if cause == effect:
                continue
            out.pairs_attempted += 1
            arr = np.column_stack([np.asarray(series[effect][:n]), np.asarray(series[cause][:n])])
            # Scoped to one pair so the record never grows with the number of pairs.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Our own verbose=False, not a problem with the data.
                warnings.filterwarnings("ignore", message="verbose is deprecated",
                                        category=FutureWarning)
                try:
                    result = grangercausalitytests(arr, maxlag=max_lag, **kwargs)
                    pvals = [float(result[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)]
                except Exception as exc:  # statsmodels raises many types; each is counted by name
                    out.record_failure(type(exc).__name__, exc)
                    pvals = None
            degraded = False
            for w in caught:
                out.record_warning(w.category.__name__, w.message)
                degraded = degraded or degrading(w.category)
            if pvals is None:
                continue
            # min() would skip a NaN in any position but the first; check every lag.
            if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in pvals):
                out.record_failure("nan_pvalue", f"p-values {pvals}")
                continue
            out.pairs_completed += 1
            out.pairs_degraded += degraded
            direction = "+ slower" if np.mean(series[cause]) > 0 else "- faster"
            out.hypotheses.append(
                Hypothesis(cause_op=cause, effect_op=effect, p_value=min(pvals),
                           direction=direction, degraded=degraded)
            )

    out.rank(lambda h: h.p_value)
    out.finalize_status()
    return out


def _failure_types(hyps: RankedHypotheses) -> str:
    return ", ".join(f"{kind}: {count}" for kind, count in sorted(hyps.failures.items()))


def _dominant_failure(hyps: RankedHypotheses) -> str:
    return Counter(hyps.failures).most_common(1)[0][0] if hyps.failures else ""


def _failed_hint(hyps: RankedHypotheses) -> str:
    if _dominant_failure(hyps) in _API_FAILURES:
        return (
            "Likely cause: statsmodels API change; check "
            "`python -c 'import statsmodels; print(statsmodels.__version__)'`."
        )
    return "Likely cause: degenerate, constant or too-short series."


def granger_evidence(hyps: RankedHypotheses, pairs_in: str = "residuals.json") -> str:
    """Claim text for a Granger run: what ran, never which pair "caused" what.

    Pairs are withheld from claims until residuals are aligned per engine
    step; ``pairs_in`` names the artifact where ``summary()`` put them.
    """
    status = hyps.status
    if status is AnalysisStatus.NOT_RUN:
        return "Granger not run"
    if status is AnalysisStatus.UNAVAILABLE:
        err = next(iter(hyps.first_errors.values()), "statsmodels not importable")
        # statsmodels is a core dependency; production pins it in constraints.txt.
        return (f"Granger unavailable: {err}. Fix: reinstall statsmodels "
                f"(pip install -c constraints.txt statsmodels)")
    if status is AnalysisStatus.INSUFFICIENT_DATA:
        n_ok = len(hyps.series_lengths) - len(hyps.excluded_ops)
        return f"Granger not run: {n_ok} ops had ≥ {hyps.min_obs} samples, need 2"
    if status is AnalysisStatus.FAILED:
        # The most common failure, the same one the hint is chosen from.
        kind = _dominant_failure(hyps) or "unknown"
        message = hyps.first_errors.get(kind, "")
        return (
            f"Granger failed: 0/{hyps.pairs_attempted} op pairs completed "
            f"({_failure_types(hyps)}). Most common error: {kind}: {message}. "
            f"{_failed_hint(hyps)}"
        )
    if status in (AnalysisStatus.OK, AnalysisStatus.PARTIAL, AnalysisStatus.DEGRADED):
        text = (
            f"Granger ran ({hyps.pairs_completed}/{hyps.pairs_attempted} op pairs) but is "
            f"not used as evidence: series are ordered by launch, not step; "
            f"pairs in {pairs_in}"
        )
        if status is AnalysisStatus.PARTIAL:
            failed = hyps.pairs_attempted - hyps.pairs_completed
            text += f"; {failed} of {hyps.pairs_attempted} op pairs failed ({_failure_types(hyps)})"
        if hyps.pairs_degraded:
            kinds = ", ".join(f"{k}: {v}" for k, v in sorted(hyps.warnings.items()))
            text += f"; {hyps.pairs_degraded} raised fit warnings ({kinds})"
        return text
    raise ValueError(f"no evidence text for Granger status {status!r}")
