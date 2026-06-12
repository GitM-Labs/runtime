"""Build the nuScenes/edge yaml freeze manifest (sha256 + bytes per artifact).

Computes hashes live so the manifest always matches the data on this pod and
can be re-verified. Mirrors the HFT manifest schema (a list of entries each
carrying path / bytes / sha256), but pins the frozen *artifacts* (metadata
tables, infos pkls, config, checkpoint, the jsonl work-list) rather than
per-keyframe files — per-frame addressing lives in manifest.jsonl.

Usage:
    python build_edge_yaml_manifest.py \\
        --data-root  /workspace/edge/data \\
        --nuscenes-root /workspace/edge/OpenPCDet/data/nuscenes/v1.0-trainval \\
        --openpcdet  /workspace/edge/OpenPCDet \\
        --out        /workspace/edge/data/edge_manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml


META_TABLES = [
    "attribute.json", "calibrated_sensor.json", "category.json", "ego_pose.json",
    "instance.json", "log.json", "map.json", "sample.json",
    "sample_annotation.json", "sample_data.json", "scene.json", "sensor.json",
    "visibility.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: Path, role: str, missing: list[str]) -> dict | None:
    if not path.exists():
        missing.append(str(path))
        return None
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(data_root: Path, nuscenes_root: Path, openpcdet: Path) -> dict:
    artifacts: list[dict] = []
    missing: list[str] = []

    meta_dir = nuscenes_root / "v1.0-trainval"
    for name in META_TABLES:
        e = entry(meta_dir / name, "metadata", missing)
        if e:
            artifacts.append(e)

    for name in [
        "nuscenes_infos_10sweeps_train.pkl",
        "nuscenes_infos_10sweeps_val.pkl",
        "nuscenes_dbinfos_10sweeps_withvelo.pkl",
    ]:
        e = entry(nuscenes_root / name, "infos", missing)
        if e:
            artifacts.append(e)

    for rel in [
        "tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml",
        "tools/cfgs/dataset_configs/nuscenes_dataset.yaml",
    ]:
        e = entry(openpcdet / rel, "config", missing)
        if e:
            artifacts.append(e)

    e = entry(openpcdet / "checkpoints" / "cbgs_pp_centerpoint_nds6070.pth",
              "checkpoint", missing)
    if e:
        artifacts.append(e)

    e = entry(data_root / "manifest.jsonl", "worklist", missing)
    if e:
        artifacts.append(e)

    return {
        "dataset": "nuscenes_v1.0-trainval",
        "scenes": {"train": 700, "val": 150},
        "keyframes": {"train": 28130, "val": 6019},
        "devkit_version": "1.2.0",
        "openpcdet_commit": "233f849829b6ac19afb8af8837a0246890908755",
        "source_patch": "numpy_alias_fixes.patch",
        "note": (
            "Freeze record. Per-keyframe work-list is manifest.jsonl. Paths are "
            "absolute on the pod that generated this; re-run the builder after a "
            "migration rather than trusting paths across pods."
        ),
        "artifacts": artifacts,
        "_missing": missing,  # non-empty => incomplete; investigate before trust
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build nuScenes/edge yaml freeze manifest.")
    p.add_argument("--data-root", default="/workspace/edge/data")
    p.add_argument("--nuscenes-root",
                   default="/workspace/edge/OpenPCDet/data/nuscenes/v1.0-trainval")
    p.add_argument("--openpcdet", default="/workspace/edge/OpenPCDet")
    p.add_argument("--out", default="/workspace/edge/data/edge_manifest.yaml")
    args = p.parse_args(argv)

    manifest = build(Path(args.data_root), Path(args.nuscenes_root), Path(args.openpcdet))
    Path(args.out).write_text(yaml.safe_dump(manifest, sort_keys=False))

    n = len(manifest["artifacts"])
    miss = manifest["_missing"]
    print(f"Wrote {args.out}: {n} artifacts pinned.")
    if miss:
        print(f"WARNING: {len(miss)} expected artifact(s) missing:")
        for m in miss:
            print(f"  - {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
