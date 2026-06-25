"""KITTI Object Detection manifest source.

Walks the already-extracted KITTI Object Detection training split and yields
one manifest row per keyframe in the schema shared with the nuScenes source:

    {scene_id, frame_id, lidar_path, gt_path}

Paths in each row are RELATIVE to GITM_DATA_ROOT so the manifest is portable
across machines (local dev box, GPU box, clean-box re-run).

This module ONLY yields rows. It does not download, does not write the JSONL,
and does not compute hashes. A separate orchestrator consumes iter_rows() and
writes the manifest; a separate script handles input hashing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# KITTI Object Detection canonical layout, relative to GITM_DATA_ROOT
# (GITM_DATA_ROOT/kitti/training -> /workspace/edge/data/kitti/training).
KITTI_ROOT = Path("kitti/training")
VELODYNE_SUBDIR = "velodyne"  # NNNNNN.bin  (lidar point clouds)
LABEL_SUBDIR = "label_2"      # NNNNNN.txt  (ground truth boxes, camera frame)

SCENE_ID = "kitti"            # KITTI Object has no scenes; flat namespace.
FRAME_PREFIX = "kitti_"       # Prefix so frame_ids never collide with nuScenes.


def iter_rows(data_root: str) -> Iterator[dict]:
    """Yield one manifest row per KITTI training keyframe.

    Args:
        data_root: Absolute path to GITM_DATA_ROOT. Used to (a) locate the
            KITTI files and (b) make the emitted paths relative to it.

    Yields:
        dict with exactly the keys: scene_id, frame_id, lidar_path, gt_path.
        lidar_path and gt_path are POSIX-style paths relative to data_root.

    Raises:
        FileNotFoundError: if the velodyne directory is missing, or if any
            lidar file lacks a sibling label file (fail loud, never skip).
    """
    root = Path(data_root)
    velodyne_dir = root / KITTI_ROOT / VELODYNE_SUBDIR
    label_dir = root / KITTI_ROOT / LABEL_SUBDIR

    if not velodyne_dir.is_dir():
        raise FileNotFoundError(
            f"KITTI velodyne directory not found: {velodyne_dir}. "
            f"Expected extracted KITTI Object data under {root / KITTI_ROOT}."
        )

    # Sort by stem for deterministic, reproducible row order. Determinism
    # matters for the clean-box re-run: identical input -> identical manifest.
    lidar_files = sorted(velodyne_dir.glob("*.bin"), key=lambda p: p.stem)

    if not lidar_files:
        raise FileNotFoundError(f"No .bin lidar files found in {velodyne_dir}.")

    for lidar_file in lidar_files:
        stem = lidar_file.stem  # e.g. "000123"
        label_file = label_dir / f"{stem}.txt"

        # Fail loud on a missing label. A lidar file with no matching label
        # signals a broken extraction (e.g. the testing/ split leaking in, or
        # archives that did not merge correctly) -- not something to skip.
        if not label_file.is_file():
            raise FileNotFoundError(
                f"Missing label for frame {stem}: expected {label_file}. "
                f"Every training lidar file must have a label_2 sibling."
            )

        yield {
            "scene_id": SCENE_ID,
            "frame_id": f"{FRAME_PREFIX}{stem}",
            "lidar_path": lidar_file.relative_to(root).as_posix(),
            "gt_path": label_file.relative_to(root).as_posix(),
        }


def _main() -> None:
    """Standalone smoke test: print the first 3 rows and the total count."""
    data_root = os.environ.get("GITM_DATA_ROOT")
    if not data_root:
        raise SystemExit("GITM_DATA_ROOT is not set in the environment.")

    count = 0
    for row in iter_rows(data_root):
        if count < 3:
            print(row)
        count += 1

    print(f"\nTotal KITTI rows: {count}")


if __name__ == "__main__":
    _main()
