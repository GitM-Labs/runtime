# Edge/robotics benchmark — datasets

> Owner: Kevin — dataset + reproducibility. Fill the TODOs as data lands.

Pipeline driver: [`fetch.py`](fetch.py) (`download_nuscenes` / `download_kitti`
/ `verify_counts`). nuScenes needs the credentialed devkit, so the full download
runs on the staging box; `make smoke` synthesizes a tiny local dataset to drive
the keyframe build + freeze/verify/reproduce loop.

## Sources
- **nuScenes v1.0 full** — 1 000 scenes, ~40 000 keyframes, lidar + per-frame
  ego-pose. Download via the official nuScenes devkit.
- **KITTI raw + object** — 7 481 training frames, lidar + camera (pinned URLs in
  `fetch.KITTI_URLS`).

## Fields & units
Two manifests:
- **`manifest.yaml`** — sha256 + byte count per raw file (the freeze).
- **`manifest.jsonl`** — the flat keyframe work-list the harness iterates, one
  row per keyframe: `{scene_id, frame_id, lidar_path, gt_path, source}`,
  ~47 k rows. Built by `make keyframes` (→ `gitm.bench edge-manifest`), which
  reads nuScenes metadata JSON + KITTI `velodyne/` + `label_2/` directly (no
  devkit dependency). For nuScenes rows, `gt_path` carries the sample token
  (the annotation lookup key); for KITTI it is the `label_2/<id>.txt` path.

## Scale target
~47 k keyframes combined (nuScenes ~40 k + KITTI 7 481).
<!-- TODO: record exact row count from `make keyframes` output. -->

## Seed protocol
`{42, 43, 44}` vary the run (sampling/ordering), not the bytes.
Layout: `$GITM_S3_ROOT/datasets/edge/{nuscenes,kitti}/` + `manifest.jsonl`.

## Freeze & verify
```bash
make keyframes     # -> manifest.jsonl (the keyframe work-list)
make manifest      # -> manifest.yaml  (sha256 freeze; also runs keyframes)
make verify        # re-hash and confirm byte-identical
```
Manifest sha256: <!-- TODO: paste after first freeze -->
Keyframe count: <!-- TODO: paste from `make keyframes` -->
