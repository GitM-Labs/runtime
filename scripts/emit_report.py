"""Emit a verification report — the fingerprint two people compare.

Captures the *deterministic* identity of a run (git SHA, pinned package
versions, CUDA/GPU, dataset manifest sha256s) plus any benchmark metrics found
in scratch. Two reports are compared with ``compare_results.py``: the
deterministic fields must match exactly; performance numbers are checked within
the benchmark's spread tolerance (and only across the same GPU SKU).

    python scripts/emit_report.py --out verify_report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PKGS = ["pydantic", "jinja2", "PyYAML", "numpy", "statsmodels", "pyarrow",
         "pandas", "torch", "cudf-cu12", "cupy-cuda12x"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in _PKGS:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def _gpu() -> dict:
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    nvcc = _run(["nvcc", "--version"])
    cuda = None
    if nvcc:
        for tok in nvcc.split():
            if tok.startswith("V") and tok[1:2].isdigit():
                cuda = tok[1:]
    return {
        "name": name.splitlines()[0] if name else None,
        "driver": driver.splitlines()[0] if driver else None,
        "cuda_toolkit": cuda,
    }


def _committed_manifests() -> dict:
    """sha256 of each benchmark's committed manifest.yaml (dataset identity)."""
    import hashlib

    out = {}
    for mani in sorted((REPO / "benchmarks").glob("*/manifest.yaml")):
        out[mani.parent.name] = hashlib.sha256(mani.read_bytes()).hexdigest()[:16]
    return out


def _cupti_shim_built() -> bool:
    try:
        from gitm.tracer._cupti import available

        return available()
    except Exception:
        return False


def build_report() -> dict:
    from gitm import __version__

    return {
        "schema": "gitm.verify_report/v1",
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "gitm_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "gpu": _gpu(),
        "cupti_shim_built": _cupti_shim_built(),
        "dataset_manifests": _committed_manifests(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit the GITM verification fingerprint.")
    p.add_argument("--out", type=Path, default=None, help="Write JSON here (default: stdout).")
    args = p.parse_args(argv)

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
