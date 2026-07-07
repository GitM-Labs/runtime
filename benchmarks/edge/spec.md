# Edge/robotics benchmark — spec

> Scope: baseline + profiling + spec doc.

## 1. Input definition
nuScenes v1.0 full + KITTI (raw + object), flattened to a ~47 k-row keyframe
work-list `manifest.jsonl` (see [datasets.md](datasets.md)). Bytes in
`$GITM_S3_ROOT/datasets/edge/`, frozen by [manifest.yaml](manifest.yaml).

## 2. Work unit
One keyframe through:
`voxelization → 3D backbone → BEV head → NMS → mAP accumulation`.
Baseline harness: **OpenPCDet CenterPoint on PointPillars** — pinned commit,
pinned config hash. Phases above are the rows of the stall table.

- OpenPCDet commit: `233f849829b6ac19afb8af8837a0246890908755`
- pointpillar.yaml sha256: `170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8`

## 3. Success metric
`frames_per_second` over a 5 000-frame warm window across nuScenes + KITTI
combined. Three seeds must agree within 2 %. Auxiliary (regression sentinel,
**not** a target): scene-level mAP.

## 4. Expected stall profile
Matches `[expected_stall]` in [bench.toml](bench.toml):

| | CPU | Data-stall | Sync | GPU active |
| --- | --- | --- | --- | --- |
| Expected | ~5 % | 20–35 % | 10–20 % | 50–65 % |

Data-stall is lidar decode + host voxelization; sync is NMS serialization.
**Load-bearing for the stream-concurrency invariant:** the timeline must show
host voxelization overlapping device inference. Confirm that overlap is visible
in the nsys timeline — it is what the deviation monitor's stream-concurrency
check keys off.
