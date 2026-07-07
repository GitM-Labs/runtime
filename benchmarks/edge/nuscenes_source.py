"""nuScenes manifest source.

Reads the frozen nuScenes metadata via the official devkit and yields one
manifest row per keyframe (sample) in the schema shared with the KITTI source:

    {scene_id, frame_id, lidar_path, gt_path}

GT treatment differs from KITTI by necessity. KITTI has a per-frame label file
(label_2/NNNNNN.txt); nuScenes keeps every box for the whole dataset in a single
table, sample_annotation.json. So instead of a per-frame path we do:

  * frame_id embeds the SAMPLE TOKEN. Downstream code recovers a frame's boxes
    by filtering sample_annotation.json on sample_token == <that token>.
  * gt_path points at that one shared sample_annotation.json. It is therefore
    IDENTICAL for every nuScenes row (still a non-empty string, so it passes
    build_manifest's per-field validation).

The version is read from GITM_NUSCENES_VERSION at call time, defaulting to
"v1.0-trainval" (the full batch). Set the env var to override (e.g.
"v1.0-mini" for early iteration). Both mini and trainval carry real
annotations and back a valid gt_path; only v1.0-test ships an empty
sample_annotation.json, so test cannot back a gt_path.

DISK-LEAN LAYOUT: this source only ever reads the keyframe LIDAR_TOP blobs
(under samples/LIDAR_TOP/) and the metadata tables. Camera, radar, and the
non-keyframe sweeps/ blobs are never touched, so a nuScenes tree pruned to just
metadata + samples/LIDAR_TOP/ is sufficient here.

Paths in each row are RELATIVE to GITM_DATA_ROOT so the manifest is portable
across machines (local dev box, GPU box, clean-box re-run).

This module ONLY yields rows. It does not download, does not write the JSONL,
and does not compute hashes. A separate orchestrator consumes iter_rows() and
writes the manifest; a separate script handles input hashing.

NOTE: unlike the streaming JSONL writer (which keeps memory flat), the nuScenes
devkit loads all metadata tables into RAM for the duration of iter_rows() --
notably sample_annotation (~1.4M rows on trainval). That is the cost of using
the devkit.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# nuScenes canonical layout, relative to GITM_DATA_ROOT
# (GITM_DATA_ROOT/nuscenes -> /workspace/edge/data/nuscenes).
NUSCENES_ROOT = Path("nuscenes")

# Env var naming the metadata version to read. Defaults to the full batch;
# the resolved version is still printed/recorded, so the clean-box re-run keeps
# a deliberate, observable choice rather than a hidden one.
VERSION_ENV = "GITM_NUSCENES_VERSION"
DEFAULT_VERSION = "v1.0-trainval"

LIDAR_CHANNEL = "LIDAR_TOP"            # keyframe lidar sensor channel
ANNOTATION_FILE = "sample_annotation.json"
SCENE_ID_FALLBACK = "nuscenes"        # only used if a scene has no name (shouldn't happen)
FRAME_PREFIX = "nuscenes_"            # prefix so frame_ids never collide with KITTI


def _resolve_version() -> str:
    """Return the nuScenes version from the environment, or the default."""
    return os.environ.get(VERSION_ENV) or DEFAULT_VERSION


def iter_rows(data_root: str) -> Iterator[dict]:
    """Yield one manifest row per nuScenes keyframe (sample).

    Args:
        data_root: Absolute path to GITM_DATA_ROOT. Used to (a) locate the
            nuScenes data + metadata and (b) make the emitted paths relative.

    Yields:
        dict with exactly the keys: scene_id, frame_id, lidar_path, gt_path.
        lidar_path and gt_path are POSIX-style paths relative to data_root.
        frame_id is "nuscenes_<sample_token>"; gt_path is the shared
        sample_annotation.json (identical across all nuScenes rows).

    Raises:
        FileNotFoundError: if the nuScenes root, the version metadata dir, the
            annotation table, or any keyframe lidar blob is missing
            (fail loud, never skip -- a missing blob means a broken extraction).
        ValueError: if a keyframe is missing its LIDAR_TOP entry.
    """
    # Lazy import: keeps build_manifest importable (and the KITTI-only path
    # runnable) on boxes that do not have the nuScenes devkit installed.
    from nuscenes.nuscenes import NuScenes

    version = _resolve_version()

    root = Path(data_root)
    nuscenes_root = root / NUSCENES_ROOT

    if not nuscenes_root.is_dir():
        raise FileNotFoundError(
            f"nuScenes root not found: {nuscenes_root}. "
            f"Expected extracted nuScenes data under {nuscenes_root}."
        )

    version_dir = nuscenes_root / version
    if not version_dir.is_dir():
        raise FileNotFoundError(
            f"nuScenes metadata dir not found: {version_dir}. "
            f"Check {VERSION_ENV}={version!r} or extract the {version} tables."
        )

    gt_file = version_dir / ANNOTATION_FILE
    if not gt_file.is_file():
        raise FileNotFoundError(
            f"Annotation table not found: {gt_file}. "
            f"Every nuScenes row points its gt_path at this file."
        )
    # Same gt_path for every row; compute once.
    gt_path = gt_file.relative_to(root).as_posix()

    nusc = NuScenes(
        version=version, dataroot=str(nuscenes_root), verbose=False
    )

    # Deterministic, reproducible row order for the clean-box re-run:
    # sort by (scene name, timestamp, sample token). Token is the final
    # tiebreaker so the order is fully determined by frozen input.
    keyed = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        scene_name = scene.get("name") or SCENE_ID_FALLBACK
        keyed.append((scene_name, sample["timestamp"], sample["token"], sample))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))

    for scene_name, _ts, sample_token, sample in keyed:
        if LIDAR_CHANNEL not in sample["data"]:
            raise ValueError(
                f"Sample {sample_token} has no {LIDAR_CHANNEL} entry; "
                f"expected every keyframe to carry a top lidar sweep."
            )

        sd = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
        lidar_rel = sd["filename"]  # e.g. "samples/LIDAR_TOP/....pcd.bin"

        # Fail loud on a missing blob -- mirrors the KITTI source. A keyframe
        # whose lidar file is absent signals a broken/partial extraction, not
        # something to silently skip. (Only samples/LIDAR_TOP/ is required;
        # sweeps/ and other sensors are never read.)
        lidar_file = nuscenes_root / lidar_rel
        if not lidar_file.is_file():
            raise FileNotFoundError(
                f"Missing lidar blob for sample {sample_token}: {lidar_file}."
            )

        yield {
            "scene_id": scene_name,
            "frame_id": f"{FRAME_PREFIX}{sample_token}",
            "lidar_path": (NUSCENES_ROOT / lidar_rel).as_posix(),
            "gt_path": gt_path,
        }


def _main() -> None:
    """Standalone smoke test: print the first 3 rows and the total count."""
    data_root = os.environ.get("GITM_DATA_ROOT")
    if not data_root:
        raise SystemExit("GITM_DATA_ROOT is not set in the environment.")

    print(f"nuScenes version: {_resolve_version()}")
    count = 0
    for row in iter_rows(data_root):
        if count < 3:
            print(row)
        count += 1

    print(f"\nTotal nuScenes rows: {count}")


if __name__ == "__main__":
    _main()
