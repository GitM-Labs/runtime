# KITTI edge benchmark spec

## Section 1: Input definition

Datasets used:
- KITTI 3D Object Detection: 7,481 training frames, Velodyne lidar + calibration + labels
- Data location: `$GITM_DATA_ROOT/kitti/training/`
- Directory layout:
  - `velodyne/`   -- 000000.bin to 007480.bin  (float32 XYZI point clouds)
  - `calib/`      -- 000000.txt to 007480.txt  (camera-lidar calibration)
  - `label_2/`    -- 000000.txt to 007480.txt  (3D bounding box annotations)
- Manifest: `benchmarks/kitti/manifest.yaml`
  - Every file sha256-verified. Pass/fail gated by `python harness/verify_manifest.py`.
  - Generate: `python harness/gen_kitti_manifest.py --root $GITM_DATA_ROOT/kitti/training`

## Section 2: Work unit

One frame processed end-to-end through:

    voxelization -> 3D backbone (PointPillars) -> BEV head -> NMS -> detections

Model: OpenPCDet PointPillars (KITTI config)

- OpenPCDet commit: `233f849829b6ac19afb8af8837a0246890908755`
- Config (pointpillar.yaml) sha256: `170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8`
- Checkpoint: `pointpillar_7728.pth`
- Checkpoint sha256: `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`

Stage breakdown per frame:
1. Load .bin (np.fromfile) -- CPU / data stall
2. Voxelization + H2D copy -- CPU / data stall
3. Backbone + BEV head -- GPU active
4. NMS + box assembly -- CPU / sync stall

Implementation: `gitm.benchmarks.kitti.WorkUnit`

## Section 3: Success metric

- Top-line metric: `frames_per_second` (timed warm window)
- Warm-up: 100 frames discarded before timing begins
- Disk pre-warm: all frames read once before GPU warmup (eliminates OS page cache
  locality as a seed-ordering confound)
- Timed window: 7,381 frames (all training frames minus warmup)
- Convergence: 6 seeds (42-47) must agree within 2% fps spread
- GPU saturation check: GPU active % must be < 85%
- Auxiliary: `total_detections` per run (regression sentinel, not a target)

Baseline result (fill after running `bash harness/run_baselines.sh`):

| Seed | fps | GPU active % | Data stall % | Sync % | CPU % | Compute headroom % |
|------|-----|--------------|-------------|--------|-------|-------------------|
| 42   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| 43   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| 44   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| 45   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| 46   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| 47   | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| Mean | TBD | TBD          | TBD         | TBD    | TBD   | TBD               |
| Stddev | TBD | TBD        | TBD         | TBD    | TBD   | --                |

6-seed fps spread: TBD -- within 2%: TBD

## Section 4: Expected stall profile

| Category | What it is | Expected % | Measured % |
|----------|-----------|------------|------------|
| Data stall | lidar .bin decode + host-side voxelization + H2D copy | 20-35% | TBD |
| Sync stall | NMS serialization on CPU | 10-20% | TBD |
| GPU active | backbone + BEV head forward pass | 50-65% | TBD |
| CPU overhead | Python dispatch, dataloader | ~5% | TBD |

**Critical check:** GPU active must be < 85%. If saturated, flag for review same day
for 500-frame shard fallback.

**Stream-concurrency check (nsys):** host-side voxelization of frame N+1 should
overlap device-side backbone inference on frame N. Capture nsys timeline and
commit screenshot to `benchmarks/kitti/results.md`. If overlap is absent, the
stream-concurrency invariant has no signal -- flag for review immediately.

## Section 5: GPU headroom (runtime integration)

Measured via `gitm.optimizer.headroom_kernel_rank.gpu_headroom()` using NVML
samples collected at 5 Hz during the timed window.

| Metric | Expected | Measured |
|--------|----------|---------|
| Compute headroom (100 - mean util) | >35% | TBD |
| Memory free at peak | >10 GB | TBD |

Per-stage spread (p50/p95 latency per stage across all frames):

| Stage | mean ms | p50 ms | p95 ms | % of frame |
|-------|---------|--------|--------|------------|
| load  | TBD | TBD | TBD | TBD |
| preprocess (voxelize + H2D) | TBD | TBD | TBD | TBD |
| inference (backbone + BEV + NMS) | TBD | TBD | TBD | TBD |
| postprocess (D2H) | TBD | TBD | TBD | TBD |

Stage spread is emitted as `stage_spread` in each baseline JSON and as
`stage_spread_report.txt` alongside it.

## Environment

- Machine: RunPod y4xbh7yws2e4tu-64410cb0 (2 TB persistent /workspace)
- GPU: TBD
- Driver: TBD
- CUDA: TBD
- OpenPCDet commit: 233f849829b6ac19afb8af8837a0246890908755
- Date: TBD
