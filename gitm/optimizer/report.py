"""Provenance report writer.

Every claim carries the full chain: residual → causal evidence → intervention
→ measured delta. Incomplete chain = no claim. Rejected candidates and
rolled-back interventions stay visible. The report is the moat.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class Claim:
    summary: str
    residual_invariant: str  # "kernel_time" | "memory_traffic" | "stream_concurrency"
    residual_value: float
    causal_evidence: str  # human-readable from RankedHypotheses
    intervention_name: str
    predicted_delta: float
    measured_delta: float | None
    rolled_back: bool = False
    #: The gate rejected this candidate and the baseline could not be put back.
    #: Not rolled back, and not kept either, so it is counted as neither.
    restore_failed: bool = False
    #: Unreliable degradations this claim's own A/B was measured under. Its delta
    #: is still shown, but it is not a verified claim.
    unreliable_ab: list[str] = field(default_factory=list)


@dataclass
class Provenance:
    workload_id: str
    fingerprint: str
    run_id: str
    git_sha: str
    gitm_version: str
    started_at_ns: int
    ended_at_ns: int
    trace_path: str | None = None
    rejected_candidates: list[str] = field(default_factory=list)
    rolled_back: list[str] = field(default_factory=list)
    #: Every fallback the run took (:mod:`gitm.optimizer.degradation`), as dicts
    #: so the verification export and the history reader see the same record.
    degradations: list[dict[str, Any]] = field(default_factory=list)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _template_dir() -> Path:
    return Path(__file__).parent / "templates"


def write_report(
    claims: list[Claim],
    provenance: Provenance,
    *,
    qualification_diagnostic: str = "",
    summary: str | None = None,
) -> str:
    """Render the provenance report as markdown."""
    env = Environment(
        loader=FileSystemLoader(_template_dir()),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )
    tpl = env.get_template("report.md.j2")
    ctx: dict[str, Any] = {
        "claims": claims,
        "provenance": provenance,
        "qualification_diagnostic": qualification_diagnostic,
        "summary": _with_ab_caveat(summary, claims) or _default_summary(claims),
        "degradations": provenance.degradations,
        "now_ns": time.time_ns(),
    }
    return tpl.render(**ctx)


def _unreliable(claims: list[Claim]) -> tuple[list[Claim], list[str]]:
    """The claims whose own A/B is unreliable, and the stages behind them."""
    bad = [c for c in claims if c.unreliable_ab]
    stages = list(dict.fromkeys(s for c in bad for s in c.unreliable_ab))
    return bad, stages


def _with_ab_caveat(summary: str | None, claims: list[Claim]) -> str | None:
    """A caller's own summary, with the unreliable-A/B caveat appended.

    The loop writes its own headline whenever the engine produced scheduler
    samples — which is every live run, and so exactly the runs where an A/B can
    be unreliable. The caveat cannot depend on the default headline being used.
    """
    if not summary:
        return summary
    bad, stages = _unreliable(claims)
    if not bad:
        return summary
    return (f"{summary} {len(bad)} claim(s) not counted as verified: their A/B is "
            f"unreliable ({', '.join(stages)}). See Degradations below.")


def _default_summary(claims: list[Claim]) -> str:
    # A claim whose own A/B was flagged unreliable keeps its row, but the
    # headline does not add it up as verified. Judged per claim: one bad A/B does
    # not discount the others.
    measured = [c for c in claims if c.measured_delta is not None
                and not c.rolled_back and not c.restore_failed]
    verified = [c for c in measured if not c.unreliable_ab]
    bad, stages = _unreliable(measured)
    note = (f" {len(bad)} more not counted: their A/B is unreliable ({', '.join(stages)})."
            if bad else "")
    if not verified:
        return "No claims verified within budget. See diagnostic below." + note
    total = sum(c.measured_delta or 0.0 for c in verified)
    return f"{len(verified)} verified claims, aggregate measured delta {total:+.1%}." + note


def build_provenance(
    workload_id: str,
    fingerprint: str,
    run_id: str,
    started_at_ns: int,
    trace_path: str | None = None,
    degradations: Iterable[Any] | None = None,
) -> Provenance:
    """``degradations`` takes a :class:`~gitm.optimizer.degradation.DegradationLog`
    (or any iterable of ``Degradation`` / dicts) and is stored as dicts."""
    from gitm import __version__

    return Provenance(
        degradations=[
            d if isinstance(d, dict) else d.to_dict() for d in (degradations or ())
        ],
        workload_id=workload_id,
        fingerprint=fingerprint,
        run_id=run_id,
        git_sha=_git_sha(),
        gitm_version=__version__,
        started_at_ns=started_at_ns,
        ended_at_ns=time.time_ns(),
        trace_path=trace_path,
    )
