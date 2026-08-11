"""Provenance report writer.

Every claim carries the full chain: residual → causal evidence → intervention
→ measured delta. Incomplete chain = no claim. Rejected candidates and
rolled-back interventions stay visible. The report is the moat.
"""

from __future__ import annotations

import math
import subprocess
import time
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
    # Whether the residual is specific to this claim, aggregated over the run,
    # or tied to an autoresearch target op. The report hides the default label.
    residual_scope: str = "claim"
    rolled_back: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.residual_value):
            raise ValueError("residual_value must be finite")

    @property
    def residual_display_value(self) -> float:
        """Value used in the compact table cell; raw truth remains available."""
        return max(-1.0, min(1.0, self.residual_value))

    @property
    def residual_saturated(self) -> bool:
        return abs(self.residual_value) > 1.0


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
    runtime_diagnostics: list[str] | None = None,
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
        "runtime_diagnostics": runtime_diagnostics or [],
        "summary": summary or _default_summary(claims),
        "now_ns": time.time_ns(),
    }
    return tpl.render(**ctx)


def _default_summary(claims: list[Claim]) -> str:
    verified = [c for c in claims if c.measured_delta is not None and not c.rolled_back]
    if not verified:
        return "No claims verified within budget. See diagnostic below."
    total = sum(c.measured_delta or 0.0 for c in verified)
    return f"{len(verified)} verified claims, aggregate measured delta {total:+.1%}."


def build_provenance(
    workload_id: str,
    fingerprint: str,
    run_id: str,
    started_at_ns: int,
    trace_path: str | None = None,
) -> Provenance:
    from gitm import __version__

    return Provenance(
        workload_id=workload_id,
        fingerprint=fingerprint,
        run_id=run_id,
        git_sha=_git_sha(),
        gitm_version=__version__,
        started_at_ns=started_at_ns,
        ended_at_ns=time.time_ns(),
        trace_path=trace_path,
    )
