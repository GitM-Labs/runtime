"""Smoke test: run 10 KITTI frames through WorkUnit and verify detections.

Done when: all 10 frames produce output dicts with no exceptions.
This is the "inference runs end-to-end on 10 frames" gate for the
PointPillars inference wrapper task.

Usage:
    python harness/smoke_kitti.py \\
        --cfg  /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \\
        --ckpt $GITM_DATA_ROOT/checkpoints/kitti/pointpillar_7728.pth

Reads the 10 first frames from benchmarks/kitti/manifest.yaml.
Exits 0 on pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "kitti" / "manifest.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KITTI WorkUnit smoke test.")
    parser.add_argument("--cfg", required=True, help="pointpillar.yaml path.")
    parser.add_argument("--ckpt", required=True, help="pointpillar_7728.pth path.")
    parser.add_argument("--n-frames", type=int, default=10, help="Frames to test.")
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Manifest path (default: benchmarks/kitti/manifest.yaml).",
    )
    args = parser.parse_args(argv)

    import yaml

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print(
            "  Run 'python harness/gen_kitti_manifest.py' first "
            "(requires GITM_DATA_ROOT and staged data).",
            file=sys.stderr,
        )
        return 1

    with manifest_path.open() as fh:
        manifest = yaml.safe_load(fh)

    root = Path(manifest["kitti_object"]["root"])
    frames = manifest["frames"][: args.n_frames]
    if len(frames) < args.n_frames:
        print(
            f"WARNING: manifest has {len(frames)} frames, fewer than requested {args.n_frames}.",
            file=sys.stderr,
        )

    from gitm.benchmarks.kitti.workunit import WorkUnit

    print(f"Loading WorkUnit from {args.ckpt} …")
    try:
        unit = WorkUnit.from_checkpoint(cfg_path=args.cfg, ckpt_path=args.ckpt)
    except ImportError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"Running {len(frames)} frames …\n")
    failures: list[str] = []
    for i, frame in enumerate(frames):
        fid = frame["id"]
        vel_entry = frame.get("velodyne")
        if vel_entry is None:
            failures.append(f"frame {fid}: no velodyne entry in manifest")
            continue

        vel_path = root / vel_entry["path"]
        if not vel_path.exists():
            failures.append(f"frame {fid}: file not found: {vel_path}")
            continue

        try:
            result = unit.run(vel_path)
        except Exception as exc:
            failures.append(f"frame {fid}: inference raised {type(exc).__name__}: {exc}")
            continue

        if result.t_total_s <= 0:
            failures.append(f"frame {fid}: t_total_s={result.t_total_s} <= 0")
            continue

        print(
            f"  [{i+1:02d}/{len(frames)}] {fid}  "
            f"{result.n_detections:2d} detections  "
            f"{result.t_total_s*1000:.1f} ms  "
            f"(load {result.t_load_s*1000:.1f} ms  "
            f"vox {result.t_preprocess_s*1000:.1f} ms  "
            f"gpu {result.t_inference_s*1000:.1f} ms  "
            f"post {result.t_postprocess_s*1000:.1f} ms)"
        )

    if failures:
        print(f"\nFAIL — {len(failures)} error(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"\nPASS — {len(frames)} frames completed without error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
