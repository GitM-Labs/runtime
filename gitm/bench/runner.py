"""Execute one work-unit and emit a ``BaselineRun`` JSON.

The harness contract is deliberately tiny so any of the three benchmark pairs
can satisfy it in whatever language their kernels live in: run the work-unit,
print **one JSON object** on stdout (the last such line wins) carrying at least
``metric_value``. Optionally include ``stall_breakdown`` (a list of
:class:`~gitm.bench.schema.StallPhase` dicts), ``gpu_name`` and ``device_count``.

This module wraps that line with provenance — git sha, gitm version, the
dataset manifest's own sha256 — so the resulting ``<name>_baseline_N.json`` is
reproducible by construction.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

from gitm.bench.manifest import manifest_digest
from gitm.bench.schema import BaselineRun, BenchConfig, StallPhase


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _last_json_line(stdout: str) -> dict:
    """Return the last parseable JSON object printed by the harness."""
    found: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "metric_value" in obj:
            found = obj
    if found is None:
        raise ValueError(
            "harness emitted no JSON line containing 'metric_value' on stdout"
        )
    return found


def build_command(config: BenchConfig, seed: int) -> list[str]:
    """Substitute ``{seed}`` / ``{warm_window_s}`` and split into argv.

    Uses literal replacement rather than ``str.format`` so a harness command may
    contain other braces (JSON, shell expansion) without tripping a KeyError.
    """
    cmd = (
        config.work_unit.command
        .replace("{seed}", str(seed))
        .replace("{warm_window_s}", str(config.warm_window_s))
    )
    return shlex.split(cmd)


def run_seed(
    config: BenchConfig,
    seed: int,
    *,
    manifest_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> BaselineRun:
    """Run the work-unit for one seed and return a populated ``BaselineRun``.

    ``manifest_path`` (or ``config_dir`` + the config's manifest name) is hashed
    into the run so a baseline is provably tied to the exact dataset bytes.
    """
    from gitm import __version__

    argv = build_command(config, seed)
    started = time.time_ns()
    proc = subprocess.run(argv, capture_output=True, text=True)
    ended = time.time_ns()
    if proc.returncode != 0:
        raise RuntimeError(
            f"work-unit failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )

    payload = _last_json_line(proc.stdout)

    provenance_warnings: list[str] = []
    manifest_sha = None
    resolved_manifest = _resolve_manifest(config, manifest_path, config_dir)
    if resolved_manifest and Path(resolved_manifest).exists():
        manifest_sha = manifest_digest(resolved_manifest)
    else:
        provenance_warnings.append(
            f"dataset manifest unavailable at {resolved_manifest!s}; baseline cannot be signed off"
        )

    git_sha = _git_sha()
    if git_sha == "unknown":
        provenance_warnings.append("git SHA unavailable; baseline cannot be signed off")
    if not payload.get("gpu_name"):
        provenance_warnings.append("harness omitted gpu_name")
    if payload.get("device_count") is None:
        provenance_warnings.append("harness omitted device_count; defaulted to 1")

    breakdown = [StallPhase.model_validate(p) for p in payload.get("stall_breakdown", [])]

    return BaselineRun(
        benchmark=config.name,
        seed=seed,
        vendor=config.vendor,
        metric=config.metric,
        metric_value=float(payload["metric_value"]),
        warm_window_s=config.warm_window_s,
        git_sha=git_sha,
        gitm_version=__version__,
        harness_commit=payload.get("harness_commit"),
        manifest_sha256=manifest_sha,
        provenance_warnings=provenance_warnings,
        gpu_name=payload.get("gpu_name", ""),
        device_count=int(payload.get("device_count", 1)),
        started_at_ns=started,
        ended_at_ns=ended,
        stall_breakdown=breakdown,
    )


def _resolve_manifest(
    config: BenchConfig,
    manifest_path: str | Path | None,
    config_dir: str | Path | None,
) -> str | Path | None:
    if manifest_path is not None:
        return manifest_path
    if config_dir is not None:
        return Path(config_dir) / config.dataset.manifest
    return None


def write_run(run: BaselineRun, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run.model_dump_json(indent=2) + "\n")
    return out
