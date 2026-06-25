"""Render ``benchmarks/<name>/results.md`` — deliverable #4.

Folds the aggregated baseline summary and a representative stall breakdown into
the markdown artifact the benchmark suite emits: the canonical table plus the
GPU-active confirmation and the pass/fail of every sign-off gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from gitm.bench.baseline import BaselineSummary
from gitm.bench.schema import BaselineRun, StallPhase


def _template_dir() -> Path:
    return Path(__file__).parent / "templates"


def load_runs_for_breakdown(paths: list) -> list[BaselineRun]:
    """Load BaselineRun JSONs for stall-breakdown selection (no aggregation)."""
    return [BaselineRun.model_validate_json(Path(p).read_text()) for p in paths]


def representative_breakdown(runs: list[BaselineRun]) -> list[StallPhase]:
    """Pick the stall breakdown to publish.

    Uses the run whose metric is nearest the mean — the most representative of
    the three rather than the best or worst.
    """
    runs = [r for r in runs if r.stall_breakdown]
    if not runs:
        return []
    mean = sum(r.metric_value for r in runs) / len(runs)
    nearest = min(runs, key=lambda r: abs(r.metric_value - mean))
    return nearest.stall_breakdown


def render_results(
    summary: BaselineSummary,
    breakdown: list[StallPhase],
    *,
    gpu_active_ceiling: float,
    manifest_sha256: str | None = None,
) -> str:
    env = Environment(
        loader=FileSystemLoader(_template_dir()),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )
    tpl = env.get_template("results.md.j2")
    ctx: dict[str, Any] = {
        "s": summary,
        "breakdown": breakdown,
        "gpu_active_ceiling": gpu_active_ceiling,
        "manifest_sha256": manifest_sha256,
        "pct": lambda x: f"{x * 100:.1f}%",
    }
    return tpl.render(**ctx)


def write_results(text: str, out: str | Path) -> Path:
    out = Path(out)
    out.write_text(text)
    return out
