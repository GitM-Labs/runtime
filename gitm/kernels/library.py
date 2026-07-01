"""Load and validate the intervention library YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from gitm.kernels.spec import InterventionSpec


def _library_path() -> Path:
    return Path(__file__).parent / "library.yaml"


def load_library(path: Path | str | None = None, *, workload: str | None = None) -> list[InterventionSpec]:
    """Load and validate every entry in the library."""
    p = Path(path) if path is not None else _library_path()
    if not p.exists():
        return []
    with p.open() as fh:
        raw = yaml.safe_load(fh) or {}
    entries = raw.get("interventions", [])
    specs = [InterventionSpec.model_validate(e) for e in entries]
    if workload is not None:
        specs = [s for s in specs if workload in s.applicability.workloads]
    return specs
