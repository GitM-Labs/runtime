"""Build the combined benchmark manifest (manifest.jsonl).

Consumes one or more dataset "source" modules -- each exposing
``iter_rows(data_root) -> Iterator[dict]`` -- chains their rows, validates the
shared schema, guards against duplicate frame IDs, and streams everything to
``$GITM_DATA_ROOT/manifest.jsonl`` as line-delimited JSON.

Each source is responsible for its own dataset-specific path resolution; this
orchestrator only knows the common row schema and how to write it. Adding a new
dataset (e.g. nuScenes) is a one-line change to SOURCES below.

Rows are written one at a time so memory stays flat regardless of manifest size
(~42k rows for the full nuScenes v1.0-trainval + KITTI batch).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterator

# Make sibling source modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitti_source  # noqa: E402
import nuscenes_source  # noqa: E402

# Every row must carry exactly these keys. Sources that deviate are a bug.
REQUIRED_KEYS = frozenset({"scene_id", "frame_id", "lidar_path", "gt_path"})

# Output location relative to GITM_DATA_ROOT
# (GITM_DATA_ROOT/manifest.jsonl -> /workspace/edge/data/manifest.jsonl).
MANIFEST_REL = Path("manifest.jsonl")

# Registered dataset sources. nuScenes runs first so its (slower, devkit-backed)
# load surfaces config/path errors before KITTI's cheap filesystem walk.
SOURCES: list[tuple[str, Callable[[str], Iterator[dict]]]] = [
    ("nuscenes", nuscenes_source.iter_rows),
    ("kitti", kitti_source.iter_rows),
]


def _validate_row(row: dict, source_name: str) -> None:
    """Raise if a row does not match the shared schema exactly."""
    keys = set(row.keys())
    if keys != REQUIRED_KEYS:
        missing = REQUIRED_KEYS - keys
        extra = keys - REQUIRED_KEYS
        raise ValueError(
            f"[{source_name}] row has wrong keys "
            f"(missing={sorted(missing)}, extra={sorted(extra)}): {row!r}"
        )
    for key in REQUIRED_KEYS:
        value = row[key]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"[{source_name}] field {key!r} must be a non-empty string: {row!r}"
            )


def build_manifest(data_root: str, out_path: Path) -> dict:
    """Write the combined manifest and return per-source counts.

    Args:
        data_root: Absolute path to GITM_DATA_ROOT.
        out_path: Destination .jsonl file.

    Returns:
        Dict mapping source name -> row count, plus a "total" key.

    Raises:
        ValueError: On a malformed row or a duplicate frame_id across sources.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    seen_frame_ids: set[str] = set()

    # Write to a temp file first, then atomically rename, so a crash mid-write
    # never leaves a half-written manifest in place.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for source_name, iter_rows in SOURCES:
            source_count = 0
            for row in iter_rows(data_root):
                _validate_row(row, source_name)

                frame_id = row["frame_id"]
                if frame_id in seen_frame_ids:
                    raise ValueError(
                        f"Duplicate frame_id {frame_id!r} "
                        f"(collision while adding source {source_name!r})"
                    )
                seen_frame_ids.add(frame_id)

                f.write(json.dumps(row) + "\n")
                source_count += 1

            counts[source_name] = source_count

    tmp_path.replace(out_path)

    counts["total"] = sum(counts.values())
    return counts


def _main() -> None:
    data_root = os.environ.get("GITM_DATA_ROOT")
    if not data_root:
        raise SystemExit("GITM_DATA_ROOT is not set in the environment.")

    out_path = Path(data_root) / MANIFEST_REL
    counts = build_manifest(data_root, out_path)

    print(f"Wrote manifest: {out_path}")
    for source_name, _ in SOURCES:
        print(f"  {source_name}: {counts[source_name]} rows")
    print(f"  total: {counts['total']} rows")


if __name__ == "__main__":
    _main()
