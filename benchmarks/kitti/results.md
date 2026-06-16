# KITTI edge benchmark results

> **Status: baselines pending compute.** All harness code (manifest gen,
> WorkUnit, 6-seed baseline runner, disk pre-warm convergence fix, GPU-headroom
> + stage-spread integration, predicted execution graph, `run_under_runtime`
> wiring) is complete and verified locally. The `TBD` cells below are filled
> automatically by `harness/fill_results.py` after a GPU run of
> `bash harness/pod_setup_and_run.sh`. Awaiting a live CUDA pod to execute the
> ~90 min 6-seed run.

## Baseline measurements

| Run | Seed | fps | GPU active % | Data stall % | Sync % | CPU % | Compute headroom % |
|-----|------|-----|--------------|-------------|--------|-------|-------------------|
| Baseline 1 | 42 | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 2 | 43 | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 3 | 44 | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 4 | 45 | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 5 | 46 | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 6 | 47 | TBD | TBD | TBD | TBD | TBD | TBD |
| Mean | -- | TBD | TBD | TBD | TBD | TBD | TBD |
| Stddev | -- | TBD | TBD | TBD | TBD | TBD | -- |

6-seed fps spread: TBD% -- within 2%: TBD

GPU headroom (compute_headroom_pct = 100 - mean NVML util): TBD%
Memory free at peak: TBD GB

## Stream-concurrency verification

Host-side voxelization overlaps device-side backbone inference: **TBD**

<!-- Insert nsys timeline screenshot below after capturing:
     nsys profile --output $GITM_DATA_ROOT/profiles/kitti_smoke \
       python harness/smoke_kitti.py --cfg $OPENPCDET_CFG --ckpt $OPENPCDET_CKPT --n-frames 200
     Open the .nsys-rep in Nsight Systems GUI, zoom in on a few consecutive frames,
     look for CPU voxelization bar overlapping GPU backbone bar.
     Screenshot -> benchmarks/kitti/concurrency_timeline.png
-->

![nsys concurrency timeline](concurrency_timeline.png)

Screenshot shows: CPU track (voxelization of frame N+1) running simultaneously
with GPU track (backbone + BEV head on frame N).

If overlap is absent: message Adit before proceeding. The stream-concurrency
invariant has no signal for this workload and the benchmark needs review.

## Notes

- Machine: RunPod y4xbh7yws2e4tu-64410cb0
- GPU: TBD
- Driver version: TBD
- CUDA version: TBD
- OpenPCDet commit: 233f849829b6ac19afb8af8837a0246890908755
- Config sha256: 170a9ffe76cfd8509d1044cfbcf1cbd44c5d320fda81bf0089a8d5efaf1c91c8
- Checkpoint sha256: 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2
- Date: TBD
