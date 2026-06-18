"""Resolve data locations.

GITM's canonical data store is **S3**. Datasets, plus the durable copy of run
outputs, traces, and telemetry, all live under an ``s3://`` root. Datasets are
far too large to hold on local disk wholesale (the AlphaFold2 DBs alone are
~2.2 TB), so the local filesystem is treated only as *bounded scratch*: the
active run's working set is staged in, used, and evicted. Nothing here ever
assumes a dataset lives on local disk.

Two roots:

* ``$GITM_S3_ROOT`` — ``s3://bucket/prefix``, the canonical store. Datasets at
  ``<s3_root>/datasets/<name>/``; durable run/trace/telemetry archives at
  ``<s3_root>/{runs,traces,telemetry}/``.
* ``$GITM_SCRATCH`` — a local *ephemeral* directory for the active run's
  outputs and staged inputs (small: a run writes here, then the durable copy is
  synced to S3). Defaults to ``~/.cache/gitm``. Never holds datasets at rest.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SCRATCH = "~/.cache/gitm"

# Local scratch subdirectories. Note: no ``datasets`` — datasets are never
# materialized wholesale on local disk; they are staged on demand into
# ``staging/`` from S3 for the duration of a run, then evicted.
_SCRATCH_SUBDIRS = ("runs", "traces", "telemetry", "staging")


def s3_root(override: str | None = None) -> str | None:
    """Return the canonical ``s3://`` root, or ``None`` if unconfigured.

    Resolution order: explicit ``override``, then ``$GITM_S3_ROOT``. Returned
    without a trailing slash. Returns ``None`` when neither is set so callers
    can degrade gracefully (e.g. a local run with no archival). Callers that
    *require* S3 — anything touching datasets — should use :func:`dataset_uri`
    or :func:`require_s3_root`, which raise with a clear message instead.
    """
    raw = override or os.environ.get("GITM_S3_ROOT")
    if not raw:
        return None
    return raw.rstrip("/")


def require_s3_root(override: str | None = None) -> str:
    """Like :func:`s3_root` but raise if no canonical store is configured."""
    root = s3_root(override)
    if root is None:
        raise RuntimeError(
            "No S3 root configured. GITM datasets live in S3 and are never "
            "stored on local disk. Set $GITM_S3_ROOT, e.g.\n"
            "    export GITM_S3_ROOT=s3://gitm-data/prod"
        )
    return root


def dataset_uri(name: str, *, s3_root_override: str | None = None) -> str:
    """Canonical ``s3://`` URI for a dataset.

    ``dataset_uri("hft/hft_1b_seed42")`` -> ``s3://.../datasets/hft/hft_1b_seed42``.
    """
    return f"{require_s3_root(s3_root_override)}/datasets/{name.strip('/')}"


def durable_uri(kind: str, run_id: str, *, s3_root_override: str | None = None) -> str:
    """Canonical ``s3://`` archive URI for a run output.

    ``kind`` is one of ``runs``, ``traces``, ``telemetry`` — the durable
    destination a scratch artifact is synced to once the run completes.
    """
    if kind not in ("runs", "traces", "telemetry"):
        raise ValueError(f"unknown durable artifact kind: {kind!r}")
    return f"{require_s3_root(s3_root_override)}/{kind}/{run_id}"


def scratch_root(override: str | None = None) -> Path:
    """Return the local scratch directory as an absolute Path.

    Resolution order: explicit ``override``, then ``$GITM_SCRATCH``, then
    ``~/.cache/gitm``. Ephemeral — holds the active run's outputs and staged
    working set only, never datasets at rest. Created if absent.
    """
    raw = override or os.environ.get("GITM_SCRATCH") or DEFAULT_SCRATCH
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sub in _SCRATCH_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def traces_dir(override: str | None = None) -> Path:
    return scratch_root(override) / "traces"


def runs_dir(override: str | None = None) -> Path:
    return scratch_root(override) / "runs"


def telemetry_dir(override: str | None = None) -> Path:
    return scratch_root(override) / "telemetry"


def staging_dir(override: str | None = None) -> Path:
    """Local landing zone for datasets staged in from S3 for the active run."""
    return scratch_root(override) / "staging"
