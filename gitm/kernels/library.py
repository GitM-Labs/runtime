"""Load and validate the intervention library YAML."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import Path

import yaml

from gitm.kernels.spec import InterventionSpec


def _library_path() -> Path:
    return Path(__file__).parent / "library.yaml"


def load_library(path: Path | str | None = None, *, workload: str | None = None) -> list[InterventionSpec]:
    """Load and validate every entry in the library."""
    p = Path(path) if path is not None else _library_path()
    if not p.exists():
        raise FileNotFoundError(
            f"intervention library not found at {p}; candidate coverage is unavailable"
        )
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("interventions"), list):
        raise ValueError(
            f"intervention library {p} has no interventions list; candidate coverage is unavailable"
        )
    entries = raw["interventions"]
    if not entries:
        raise ValueError(
            f"intervention library {p} is empty; candidate coverage is unavailable"
        )
    specs = [InterventionSpec.model_validate(e) for e in entries]
    if workload is not None:
        specs = [s for s in specs if workload in s.applicability.workloads]
    return specs


def parse_skips(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalise skip patterns from a flag, an env var, or both.

    Takes a comma- or whitespace-separated string (how an env var or a
    Kubernetes manifest carries a list) or an iterable of patterns (how a
    repeated CLI flag arrives), and returns them deduplicated in the order
    given. Empty fragments drop out, so a trailing comma is not a pattern that
    matches nothing in particular.
    """
    if raw is None:
        return ()
    items = [raw] if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for item in items:
        for part in str(item).replace(",", " ").split():
            if part not in out:
                out.append(part)
    return tuple(out)


def skipped_by(spec: InterventionSpec, patterns: Iterable[str]) -> str | None:
    """The first pattern that excludes ``spec``, or ``None`` to keep it.

    Returns the pattern rather than a bool so a run can record *why* a lever
    never ran. A lever that is simply absent from a report is indistinguishable
    from one that was tried and failed, which is the confusion the report is
    supposed to remove.

    Matched against the lever's name and against the knob or knobs it sets, with
    shell globs and without regard to case. Both handles earn their place: the
    name is what a report prints and what an operator reads back, while the knob
    is what actually hangs a model, and one knob can be reached by several
    levers. ``speculative*`` and ``num_speculative_tokens`` therefore both
    exclude n-gram speculative decoding.

    Swept levers expand into one candidate per grid point, named
    ``<lever>_x2``, and this is applied before that expansion — so excluding a
    lever excludes every point of its sweep. To drop a single point, exclude it
    by its expanded name after the fact; nothing asks for that today.
    """
    handles = [spec.name, spec.knob, *spec.knobs]
    for pattern in patterns:
        needle = pattern.lower()
        for handle in handles:
            if handle and fnmatch.fnmatch(str(handle).lower(), needle):
                return pattern
    return None
