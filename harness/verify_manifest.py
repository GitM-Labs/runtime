"""Verify benchmarks/kitti/manifest.yaml against live data.

Usage:
    python harness/verify_manifest.py

Exits 0 if every file in the manifest exists and its sha256 matches.
Exits 1 and prints a failure table otherwise.

Pass --fast to skip sha256 re-hashing (existence-only check).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "kitti" / "manifest.yaml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(fast: bool = False) -> list[str]:
    if not MANIFEST_PATH.exists():
        return [f"manifest not found: {MANIFEST_PATH}"]

    with MANIFEST_PATH.open() as fh:
        manifest = yaml.safe_load(fh)

    training = Path(manifest["kitti_object"]["root"])
    failures: list[str] = []

    frames = manifest.get("frames", [])
    print(f"Verifying {len(frames)} frames (fast={fast})…")

    for i, frame in enumerate(frames):
        fid = frame["id"]
        if i % 500 == 0:
            print(f"  {i}/{len(frames)}")

        for key in ("velodyne", "calib", "label"):
            entry = frame.get(key)
            if entry is None:
                failures.append(f"[{fid}] {key}: manifest entry is null")
                continue

            path = training / entry["path"]
            if not path.exists():
                failures.append(f"[{fid}] {key}: file not found: {path}")
                continue

            actual_bytes = path.stat().st_size
            if actual_bytes != entry["bytes"]:
                failures.append(
                    f"[{fid}] {key}: size mismatch (manifest={entry['bytes']}, actual={actual_bytes})"
                )
                continue

            if not fast:
                actual_sha = sha256_file(path)
                if actual_sha != entry["sha256"]:
                    failures.append(
                        f"[{fid}] {key}: sha256 mismatch"
                    )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify KITTI manifest.")
    parser.add_argument(
        "--fast", action="store_true", help="Skip sha256 re-hash (existence + size only)."
    )
    args = parser.parse_args()

    failures = verify(fast=args.fast)

    if failures:
        print(f"\nFAIL — {len(failures)} issue(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print("\nPASS — all files verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
