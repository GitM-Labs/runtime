"""Edge/robotics dataset pipeline: nuScenes + KITTI acquisition.

Drives the two downloads, then hands off to the shared keyframe builder
(`gitm.bench edge-manifest`, exposed as `make keyframes`) which flattens both
into the combined `manifest.jsonl`.

* **nuScenes v1.0 full** — 1 000 scenes, ~40 k keyframes. Fetched via the
  official nuScenes devkit (requires an account + dataset agreement), so the
  real download runs on a staging box, not a laptop.
* **KITTI object** — 7 481 training frames, lidar + labels, from pinned URLs.

`--smoke` synthesises a tiny dataset with both shapes — a minimal nuScenes
metadata tree (scene/sample/sample_data JSON + placeholder lidar bins) and a
KITTI `velodyne/`+`label_2/` pair — so `make keyframes`, `make manifest`, and
`make reproduce` all run locally without the real datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KITTI_URLS = {
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
    "label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
}
NUSCENES_FULL_SCENES = 1000
KITTI_TRAIN_FRAMES = 7481


def verify_counts(edge_root: str | Path) -> dict:
    """Count what landed — the 'verify counts' step in the day cards."""
    edge_root = Path(edge_root)
    nusc_meta = edge_root / "nuscenes" / "v1.0-trainval"
    n_scenes = 0
    if (nusc_meta / "scene.json").is_file():
        n_scenes = len(json.loads((nusc_meta / "scene.json").read_text()))
    velo = edge_root / "kitti" / "training" / "velodyne"
    n_kitti = len(list(velo.glob("*.bin"))) if velo.is_dir() else 0
    return {"nuscenes_scenes": n_scenes, "kitti_frames": n_kitti}


# --- real acquisition (staging box) -----------------------------------------


def download_nuscenes(out_dir: str | Path, *, version: str = "v1.0-trainval") -> Path:
    """Download nuScenes via the official devkit (credentialed)."""
    try:
        import nuscenes  # noqa: F401
    except Exception as exc:  # pragma: no cover - devkit absent on laptop
        raise RuntimeError(
            "nuscenes-devkit not installed / not credentialed — the full "
            "download runs on the staging box. Use --smoke for a local fixture."
        ) from exc
    raise NotImplementedError(
        f"Invoke the nuScenes devkit download for {version} into {Path(out_dir)}."
    )


def download_kitti(out_dir: str | Path) -> list[Path]:
    """Download + unzip KITTI object velodyne + labels from pinned URLs."""
    import urllib.request
    import zipfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, url in KITTI_URLS.items():
        zpath = out_dir / f"{name}.zip"
        urllib.request.urlretrieve(url, zpath)  # noqa: S310 - pinned source
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
        written.append(zpath)
    return written


# --- smoke fixtures (laptop) -------------------------------------------------


def write_smoke(out_dir: str | Path, *, n_scenes: int = 2, frames_per_scene: int = 3,
                n_kitti: int = 5) -> dict:
    """Write a tiny nuScenes-shaped + KITTI-shaped dataset for the local loop."""
    out_dir = Path(out_dir)

    # --- nuScenes metadata tree ---
    meta = out_dir / "nuscenes" / "v1.0-trainval"
    meta.mkdir(parents=True, exist_ok=True)
    lidar_dir = out_dir / "nuscenes" / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    scenes, samples, sample_data = [], [], []
    for s in range(n_scenes):
        stok = f"SCENE{s}"
        sname = f"scene-{s:04d}"
        scenes.append({"token": stok, "name": sname})
        for f in range(frames_per_scene):
            samp = f"SAMP_{s}_{f}"
            fn = f"samples/LIDAR_TOP/{sname}__LIDAR_TOP__{f}.pcd.bin"
            (out_dir / "nuscenes" / fn).write_bytes(b"\x00" * 16)
            samples.append({"token": samp, "scene_token": stok})
            sample_data.append({"sample_token": samp, "is_key_frame": True, "filename": fn})
    (meta / "scene.json").write_text(json.dumps(scenes))
    (meta / "sample.json").write_text(json.dumps(samples))
    (meta / "sample_data.json").write_text(json.dumps(sample_data))

    # --- KITTI object tree ---
    velo = out_dir / "kitti" / "training" / "velodyne"
    labels = out_dir / "kitti" / "training" / "label_2"
    velo.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n_kitti):
        (velo / f"{i:06d}.bin").write_bytes(b"\x00" * 16)
        (labels / f"{i:06d}.txt").write_text("Car 0.00 0 0.0 0 0 0 0 0 0 0 0 0 0 0\n")

    counts = verify_counts(out_dir)
    counts["nuscenes_keyframes"] = len(sample_data)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Edge dataset pipeline (nuScenes + KITTI).")
    p.add_argument("--out", type=Path, required=True, help="Edge dataset staging dir.")
    p.add_argument("--smoke", action="store_true", help="Synthesize a tiny local dataset.")
    p.add_argument("--step", choices=["nuscenes", "kitti", "all"], default="all")
    args = p.parse_args(argv)

    if args.smoke:
        counts = write_smoke(args.out)
        print(f"smoke edge dataset in {args.out}: {counts}")
        print("next: `make keyframes` to build manifest.jsonl")
        return 0

    if args.step in ("kitti", "all"):
        download_kitti(args.out / "kitti")
    if args.step in ("nuscenes", "all"):
        download_nuscenes(args.out / "nuscenes")
    print(f"counts: {verify_counts(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
