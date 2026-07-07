# Edge/robotics benchmark — datasets

> Owner: Kevin — dataset + reproducibility. Fill the TODOs as data lands.

Pipeline driver: [`fetch.py`](fetch.py) (`download_nuscenes` / `download_kitti`
/ `verify_counts`). nuScenes needs the credentialed devkit, so the full download
runs on the staging box; `make smoke` synthesizes a tiny local dataset to drive
the keyframe build + freeze/verify/reproduce loop.

---

## Sources

- **nuScenes v1.0 full** — 1 000 scenes, ~40 000 keyframes, lidar + per-frame
  ego-pose. Download via the official nuScenes devkit, or pull the cached
  archives from the AWS Registry of Open Data
  (`s3://motional-nuscenes/public/`, region `ap-northeast-1`,
  `--no-sign-request`). Extracted tree: `{maps,samples,sweeps,v1.0-trainval}`.
- **KITTI raw + object** — 7 481 training frames, lidar + camera (pinned URLs in
  `fetch.KITTI_URLS`).

---

## The two manifests

The benchmark carries **two** manifest files with different jobs:

- **`manifest.yaml`** — the *freeze*: sha256 + byte count per raw file.
- **`manifest.jsonl`** — the flat keyframe work-list the harness iterates, one
  row per keyframe, ~47 k rows.

`manifest.yaml` is documented under **Freeze & verify** below.
`manifest.jsonl` is specified in full next.

---

## `manifest.jsonl` — combined keyframe manifest

A line-delimited JSON index over both 3D object-detection datasets. Each line is
one **keyframe**: a single lidar sweep plus a pointer to its ground-truth
annotations. The manifest contains **no point data and no boxes** — only
relative paths into the on-disk dataset trees. It is the portable, deterministic
spine that downstream loaders walk.

### Location & generation

- **Output path:** `$GITM_DATA_ROOT/datasets/edge/manifest.jsonl`
- **Built by:** `build_manifest.py`, which chains `nuscenes_source.iter_rows()`
  then `kitti_source.iter_rows()`.
- **Make interface:** `make keyframes` (→ `gitm.bench edge-manifest`).

Required environment:

| Variable | Required by | Purpose |
|---|---|---|
| `GITM_DATA_ROOT` | all | Absolute path all manifest paths are relative to. |
| `GITM_NUSCENES_VERSION` | nuScenes source | Which metadata set to read, e.g. `v1.0-mini`, `v1.0-trainval`. No default — fails loud if unset. |

Build:

```bash
export GITM_DATA_ROOT=/abs/path/to/data
export GITM_NUSCENES_VERSION=v1.0-trainval
python build_manifest.py
```

The writer streams one row at a time (flat memory) into a `.tmp` file, then
atomically renames — so a crash never leaves a half-written manifest. This
flat-memory guarantee is the *writer's*; the nuScenes devkit still loads all its
metadata tables into RAM (notably ~1.4M `sample_annotation` rows) while
producing rows.

### Row schema

Every row has **exactly** these four keys, each a **non-empty string**. Any
deviation is rejected at build time.

| Key | Meaning |
|---|---|
| `scene_id` | Grouping label. nuScenes: the scene name (e.g. `scene-0061`). KITTI: always the literal `kitti` (KITTI Object has no scenes). |
| `frame_id` | Globally unique keyframe ID. Source-prefixed to prevent cross-dataset collisions. Also encodes how to find the labels (see below). |
| `lidar_path` | POSIX path to the lidar point cloud, **relative to `GITM_DATA_ROOT`**. |
| `gt_path` | POSIX path to the ground-truth annotation file, relative to `GITM_DATA_ROOT`. |


Example rows (one per source):

```json
{"scene_id": "scene-0061", "frame_id": "nuscenes_e93e98b63d3b40209056d129dc53ceee", "lidar_path": "datasets/edge/nuscenes/samples/LIDAR_TOP/n015-2018-...pcd.bin", "gt_path": "datasets/edge/nuscenes/v1.0-trainval/sample_annotation.json"}
{"scene_id": "kitti", "frame_id": "kitti_000123", "lidar_path": "datasets/edge/kitti/training/velodyne/000123.bin", "gt_path": "datasets/edge/kitti/training/label_2/000123.txt"}
```

To get an absolute path, join with the root:
`os.path.join(os.environ["GITM_DATA_ROOT"], row["lidar_path"])`.

### `frame_id` structure

The prefix is the discriminator. Strip it to get the source-native key.

- `kitti_<stem>` — `<stem>` is the zero-padded KITTI frame number (`000123`).
- `nuscenes_<sample_token>` — `<sample_token>` is the nuScenes **sample token**
  (32-hex-char string). This token is the key you use to recover the frame's
  boxes from the shared annotation table.

### Retrieving the point cloud

Both sources point `lidar_path` at a raw little-endian `float32` binary blob,
but the layouts differ:

- **KITTI** (`velodyne/NNNNNN.bin`): shape `(N, 4)` — `x, y, z, reflectance`,
  in the Velodyne lidar coordinate frame.
- **nuScenes** (`samples/LIDAR_TOP/*.pcd.bin`): shape `(N, 5)` —
  `x, y, z, intensity, ring_index`, in the LIDAR_TOP sensor frame.

```python
import numpy as np, os

root = os.environ["GITM_DATA_ROOT"]
path = os.path.join(root, row["lidar_path"])
cols = 4 if row["frame_id"].startswith("kitti_") else 5
points = np.fromfile(path, dtype=np.float32).reshape(-1, cols)
```

### Retrieving annotations

Branch on the `frame_id` prefix.

#### KITTI — per-frame label file

`gt_path` is a unique text file for that frame. Read it directly; one line per
object. The detection-relevant columns are:

`type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, h, w, l, x, y, z, rotation_y`

- `type` — class string (`Car`, `Pedestrian`, `Cyclist`, …, plus `DontCare`).
- `truncated` (0–1 float), `occluded` (0–3 int) — quality flags.
- `alpha` — observation angle, $[-\pi, \pi]$.
- `bbox_*` — 2D box in **image pixels**.
- `h, w, l` — 3D box dimensions in metres.
- `x, y, z` — 3D box center, in the **camera** coordinate frame (not lidar).
- `rotation_y` — yaw about the camera Y-axis, $[-\pi, \pi]$.

```python
def kitti_boxes(root, gt_path):
    objs = []
    with open(os.path.join(root, gt_path)) as f:
        for line in f:
            p = line.split()
            objs.append({
                "type": p[0],
                "dims_hwl": tuple(map(float, p[8:11])),
                "loc_xyz": tuple(map(float, p[11:14])),
                "rotation_y": float(p[14]),
            })
    return objs
```

Caveat: KITTI 3D boxes live in the **camera** frame, while the lidar points live
in the **Velodyne** frame. Aligning them needs the calibration matrices
(`calib/NNNNNN.txt`), which this manifest does **not** index.

#### nuScenes — filter the shared table by token

**Every** nuScenes row's `gt_path` points at the *same* file —
`…/<version>/sample_annotation.json` — which holds the boxes for the entire
dataset. The per-frame selector is the sample token embedded in `frame_id`:

```python
import json

def nuscenes_boxes(root, row, table_cache={}):
    token = row["frame_id"].removeprefix("nuscenes_")
    gt = os.path.join(root, row["gt_path"])
    if gt not in table_cache:                 # load the big table once
        with open(gt) as f:
            table_cache[gt] = json.load(f)
    return [a for a in table_cache[gt] if a["sample_token"] == token]
```

Performance: `sample_annotation.json` for `v1.0-trainval` is ~1.4M records.
Re-scanning it per frame is O(frames × annotations). Build a
`sample_token -> [annotations]` dict once and reuse it.

Each annotation record carries the geometry directly:

- `translation` — box center `[x, y, z]` in the **global** map frame (metres).
- `size` — `[width, length, height]` (metres).
- `rotation` — orientation as a quaternion `[w, x, y, z]` in the global frame.
- `instance_token`, `num_lidar_pts`, `num_radar_pts`, `prev`, `next`, etc.

Two things the raw table does **not** give you:

1. **Class label.** `sample_annotation.json` has no category string. Resolve it
   via `instance_token` → `instance.json` → `category_token` → `category.json`.
   (The official devkit hides this join behind
   `nusc.get('sample_annotation', token)['category_name']`; raw-JSON consumers
   must do it themselves.)
2. **A usable coordinate frame.** Boxes are in the **global** frame; the lidar
   points are in the **sensor** frame. Transforming between them requires
   `ego_pose.json` and `calibrated_sensor.json` — again, not indexed by this
   manifest.

### Version & split notes (nuScenes)

- `v1.0-mini` and `v1.0-trainval` both ship real annotations, so both back a
  valid `gt_path`.
- `v1.0-test` ships an **empty** `sample_annotation.json` — it cannot back a
  ground-truth path. Do not build a labelled manifest from the test split.
- The version is part of `gt_path` itself (`…/v1.0-trainval/…`), so a manifest
  records which split it was built against.

### Guarantees the build enforces

- **Schema:** each row has exactly the four keys, each a non-empty string;
  otherwise the build raises.
- **Uniqueness:** duplicate `frame_id` across sources is a hard error. The
  `kitti_` / `nuscenes_` prefixes make collisions structurally impossible
  between the two sources.
- **Determinism:** identical frozen input → byte-identical manifest. KITTI rows
  are sorted by frame stem; nuScenes rows by
  `(scene_name, timestamp, sample_token)`. This supports clean-box re-runs.
- **Fail loud, never skip:** a missing lidar blob (either source) or a missing
  KITTI label file aborts the build. A gap signals a broken/partial extraction,
  not a row to silently drop.
- **Ordering:** nuScenes rows come first, then KITTI — chosen so the slower,
  devkit-backed load surfaces config/path errors before KITTI's cheap
  filesystem walk runs.

---

## Scale target

~47 k keyframes combined (nuScenes ~40 k + KITTI 7 481).
<!-- TODO: record exact row count from `make keyframes` output. -->

---

## Seed protocol

`{42, 43, 44}` vary the run (sampling/ordering), not the bytes.
Layout: `$GITM_DATA_ROOT/datasets/edge/{nuscenes,kitti}/` + `manifest.jsonl`.

---

## Freeze & verify

```bash
make keyframes     # -> manifest.jsonl (the keyframe work-list)
make manifest      # -> manifest.yaml  (sha256 freeze; also runs keyframes)
make verify        # re-hash and confirm byte-identical
```

`manifest.yaml` records sha256 + byte count per raw file.

Manifest sha256: <!-- TODO: paste after first freeze -->
Keyframe count: <!-- TODO: paste from `make keyframes` -->