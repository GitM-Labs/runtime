"""Generate benchmarks/kitti/manifest.yaml from staged KITTI data.

Usage:
    python harness/gen_kitti_manifest.py [--root KITTI_TRAINING_DIR]

Default root on RunPod: /workspace/edge/kitti/training

Preferred alternative (uses the shared bench tooling):
    python -m gitm.bench manifest build \\
        --root /workspace/edge/kitti \\
        --benchmark edge \\
        --out benchmarks/edge/manifest.yaml

This script writes a frame-level manifest (sha256 per velodyne/calib/label_2
file) to benchmarks/kitti/manifest.yaml. The gitm.bench command above writes
the canonical edge dataset manifest to benchmarks/edge/manifest.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_OUT = REPO_ROOT / "benchmarks" / "kitti" / "manifest.yaml"
EXPECTED_FRAMES = 7481
DEFAULT_TRAINING_ROOT = "/workspace/edge/kitti/training"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate KITTI sha256 manifest.")
    parser.add_argument(
        "--root",
        default=DEFAULT_TRAINING_ROOT,
        help=f"KITTI object training dir (default: {DEFAULT_TRAINING_ROOT})",
    )
    parser.add_argument(
        "--out",
        default=str(MANIFEST_OUT),
        help="Output manifest path (default: benchmarks/kitti/manifest.yaml)",
    )
    args = parser.parse_args(argv)

    training = Path(args.root)
    for subdir in ("velodyne", "calib", "label_2"):
        d = training / subdir
        if not d.is_dir():
            print(f"ERROR: expected directory missing: {d}", file=sys.stderr)
            print(
                "  Set --root to the KITTI training dir "
                "(contains velodyne/, calib/, label_2/).",
                file=sys.stderr,
            )
            return 1

    frame_ids = sorted(
        p.stem for p in (training / "velodyne").glob("*.bin")
    )
    if len(frame_ids) != EXPECTED_FRAMES:
        print(
            f"WARNING: expected {EXPECTED_FRAMES} frames, found {len(frame_ids)}",
            file=sys.stderr,
        )

    print(f"Hashing {len(frame_ids)} frames from {training} …")
    print("(this takes a few minutes; progress every 500 frames)")

    frames: list[dict] = []
    for i, fid in enumerate(frame_ids):
        if i % 500 == 0:
            print(f"  {i}/{len(frame_ids)}")

        vel = training / "velodyne" / f"{fid}.bin"
        cal = training / "calib" / f"{fid}.txt"
        lab = training / "label_2" / f"{fid}.txt"

        entry: dict = {"id": fid}
        for key, path in (("velodyne", vel), ("calib", cal), ("label", lab)):
            if not path.exists():
                print(f"WARNING: missing {path}", file=sys.stderr)
                entry[key] = None
                continue
            entry[key] = {
                "path": str(path.relative_to(training)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        frames.append(entry)

    manifest = {
        "kitti_object": {
            "root": str(training),
            "n_frames": len(frames),
            "generated_by": "harness/gen_kitti_manifest.py",
        },
        "frames": frames,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        yaml.dump(manifest, fh, default_flow_style=False, sort_keys=False)

    print(f"\nWrote {out} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
