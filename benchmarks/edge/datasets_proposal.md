# Additional edge/robotics datasets — proposal

> Author: Karthik — for review by Adit before adding to spec.

Current scope: nuScenes v1.0 + KITTI Object (~47k keyframes combined).
Below are the next candidates ranked by signal value for the Git.M invariants.

---

## Tier 1 — High signal, worth adding

### Waymo Open Dataset (v2.0)
- **Size:** ~1,000 segments, 200k frames, 5-beam lidar (top + 4 side)
- **Why it matters:** Much denser point clouds (64-beam top lidar vs KITTI's 64-beam but wider
  range + higher annotation quality). Significantly harder for the backbone — GPU active % likely
  higher, which tightens the stream-concurrency signal.
- **Concern:** Requires a data access agreement (Google form, ~1 week turnaround).
  Also non-commercial only — verify with Adit before committing.
- **Manifest rows:** ~200k (5x current KITTI). Build time ~20 min.
- **Blocker:** License approval.

### Argoverse 2 (Sensor Dataset)
- **Size:** 1,000 scenarios, ~30k frames, 2x lidar (spinning + forward-facing).
- **Why it matters:** Two asynchronous lidar streams per frame — interesting for concurrency
  invariant because merging two streams before voxelization introduces a sync point.
  Good test of whether stream-concurrency signal carries to multi-lidar setups.
- **Download:** Open access via S3 (`s3://argoai-argoverse2/...`). No license gate.
- **Manifest rows:** ~30k. Adds ~30% to current combined manifest.
- **Blocker:** None. Could add this week.

---

## Tier 2 — Useful if we want breadth

### ONCE (One Million Scenes)
- **Size:** ~1M frames, single 40-beam lidar.
- **Why it matters:** Volume — more frames = tighter convergence bounds and better
  steady-state GPU utilization measurements. Useful for validating that the 2%
  convergence requirement holds at scale.
- **Download:** Open access (Chinese hosting, slow downloads). May need mirror.
- **Blocker:** Download bandwidth on RunPod. Otherwise no license gate.

### PandaSet
- **Size:** ~16k frames, dual lidar (mechanical + solid-state).
- **Why it matters:** Solid-state lidar has a fundamentally different point density
  pattern (non-uniform angular resolution). Tests whether the voxelization step
  behaves differently under non-uniform inputs.
- **Download:** Open access (free sign-up, direct download).
- **Blocker:** None.

---

## Tier 3 — Lower priority

### SemanticKITTI (KITTI odometry sequences with semantic labels)
- **Size:** Same lidar as KITTI Object but sequential (not individual frames).
  22 sequences, ~43k scans.
- **Why it matters:** Sequential frames are much more cache-friendly — useful
  as a control condition to isolate the I/O cache locality effect.
- **Blocker:** None. Builds on top of existing KITTI download.

### nuScenes-lidarseg
- **Same data as nuScenes v1.0** but with per-point semantic labels.
  No new lidar frames; adds annotation load to the post-processing step.
- **Why it matters:** Tests sync_stall_pct sensitivity to heavier post-processing.

---

## Recommendation

Add **Argoverse 2** first — no license gate, open S3, meaningful new signal
(multi-lidar sync point). After that, pursue **Waymo** if the license approval
clears, since it's the most widely used benchmark for 3D detection and having
it in the manifest would make the benchmark credible to external readers.

Skip ONCE for now (download pain) and SemanticKITTI/PandaSet unless we need
more control conditions.
