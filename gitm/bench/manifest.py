"""Streaming sha256 dataset manifests — the "frozen dataset" contract.

A manifest pins a dataset to bytes: every file with its sha256 and byte count.
Anyone with ``manifest.yaml`` can re-fetch identical bytes and prove they did.
This is the one-liner run to (re)generate each dataset manifest.

Datasets are TB-scale (HFT alone is ~600 GB compressed per seed), so hashing
streams in fixed chunks and never loads a file into memory. Generation happens
on whatever box staged or generated the data; the resulting ``manifest.yaml``
is committed to ``benchmarks/<name>/`` and travels with the repo. The bytes
themselves stay in ``$GITM_S3_ROOT/datasets/<name>/`` — never on local disk at
rest (see :mod:`gitm._paths`).
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA = "gitm.bench.manifest/v1"
_CHUNK = 1 << 20  # 1 MiB — stream, never slurp a TB file into RAM.


def sha256_file(path: str | Path, *, chunk: int = _CHUNK) -> tuple[str, int]:
    """Return ``(hex_digest, byte_count)`` for a file, streamed in chunks."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def _iter_files(root: Path, exclude: Iterable[str]) -> Iterator[Path]:
    patterns = tuple(exclude)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            continue
        yield p


# Files that should never end up in a dataset manifest.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "manifest.yaml",
    "*.tmp",
    "*.partial",
    ".DS_Store",
    "**/.ipynb_checkpoints/**",
)


def build_manifest(
    root: str | Path,
    benchmark: str,
    *,
    dataset_root: str | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> dict:
    """Walk ``root`` and return a manifest dict (sorted, deterministic).

    ``root`` is the local directory holding the staged/generated dataset.
    ``dataset_root`` records the canonical S3 sub-path (defaults to ``benchmark``)
    so a consumer knows where the bytes live.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root not found: {root}")

    files = []
    total_bytes = 0
    for p in _iter_files(root, exclude):
        digest, n = sha256_file(p)
        files.append(
            {"path": p.relative_to(root).as_posix(), "sha256": digest, "bytes": n}
        )
        total_bytes += n

    from gitm import __version__

    return {
        "schema": SCHEMA,
        "benchmark": benchmark,
        "dataset_root": dataset_root or benchmark,
        "generated_by_gitm": __version__,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,  # already sorted by path via _iter_files
    }


def write_manifest(manifest: dict, out: str | Path) -> Path:
    out = Path(out)
    out.write_text(yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False))
    return out


def load_manifest(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} manifest")
    return data


def manifest_digest(path: str | Path) -> str:
    """sha256 of the manifest file itself — pinned into each ``BaselineRun``."""
    digest, _ = sha256_file(path)
    return digest


@dataclass
class VerifyResult:
    ok: bool
    checked: int
    missing: list[str]
    mismatched: list[str]  # path: reason
    extra: list[str]  # present on disk, absent from manifest

    def summary(self) -> str:
        if self.ok:
            return f"OK — {self.checked} files match manifest"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.mismatched:
            parts.append(f"{len(self.mismatched)} mismatched")
        if self.extra:
            parts.append(f"{len(self.extra)} unexpected")
        return "FAIL — " + ", ".join(parts)


def verify_manifest(
    manifest: str | Path | dict,
    root: str | Path,
    *,
    check_extra: bool = True,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> VerifyResult:
    """Re-hash the dataset at ``root`` and compare against ``manifest``.

    This is the integrity check ``make verify`` runs and the first thing the
    reproducibility test does on a clean box before trusting a baseline.
    """
    man = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    root = Path(root)

    expected = {f["path"]: f for f in man["files"]}
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0

    for rel, rec in expected.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        digest, n = sha256_file(p)
        checked += 1
        if n != rec["bytes"]:
            mismatched.append(f"{rel}: bytes {n} != {rec['bytes']}")
        elif digest != rec["sha256"]:
            mismatched.append(f"{rel}: sha256 mismatch")

    extra: list[str] = []
    if check_extra:
        on_disk = {p.relative_to(root).as_posix() for p in _iter_files(root, exclude)}
        extra = sorted(on_disk - set(expected))

    ok = not (missing or mismatched or extra)
    return VerifyResult(
        ok=ok, checked=checked, missing=missing, mismatched=mismatched, extra=extra
    )
