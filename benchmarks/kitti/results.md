# KITTI edge benchmark results

## Baseline measurements

| Run | Seed | fps | GPU active % | Data stall % | Sync % | CPU % |
|-----|------|-----|--------------|-------------|--------|-------|
| Baseline 1 | 42 | TBD | TBD | TBD | TBD | TBD |
| Baseline 2 | 43 | TBD | TBD | TBD | TBD | TBD |
| Baseline 3 | 44 | TBD | TBD | TBD | TBD | TBD |
| Mean | — | TBD | TBD | TBD | TBD | TBD |
| Stddev | — | TBD | TBD | TBD | TBD | TBD |

3-seed fps spread: TBD% — within 2%: TBD

NVML cross-check (mean utilization): TBD%

## Stream-concurrency verification

Host-side voxelization overlaps device-side backbone inference: **TBD**

<!-- Insert nsys timeline screenshot below after capturing:
     nsys profile --output $GITM_DATA_ROOT/profiles/kitti_smoke \
       python harness/smoke_kitti.py --cfg $OPENPCDET_CFG --ckpt $OPENPCDET_CKPT --n-frames 200
     Open the .nsys-rep in Nsight Systems GUI, zoom in on a few consecutive frames,
     look for CPU voxelization bar overlapping GPU backbone bar.
     Screenshot → benchmarks/kitti/concurrency_timeline.png
-->

![nsys concurrency timeline](concurrency_timeline.png)

Screenshot shows: CPU track (voxelization of frame N+1) running simultaneously
with GPU track (backbone + BEV head on frame N).

If overlap is absent: message Adit before proceeding. The stream-concurrency
invariant has no signal for this workload and the benchmark needs review.

## Notes

- Machine: TBD
- GPU: TBD
- Driver version: TBD
- CUDA version: TBD
- OpenPCDet commit: TBD
- Config sha256: TBD
- Checkpoint sha256: 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2
- Date: TBD
