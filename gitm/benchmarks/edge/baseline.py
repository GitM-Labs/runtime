"""nuScenes CenterPoint-PointPillar baseline runner.

Runs convergent baselines over the val split (6019 keyframes, multi-sweep),
records FPS + NVML utilization + a (caveated) stall breakdown, and writes a
JSON result file per run. The JSON contract matches the KITTI runner, so the
existing convergence checker in harness/run_baselines.sh grades it unchanged.

Usage:
    python -m gitm.benchmarks.edge.baseline \\
        --cfg     /workspace/edge/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml \\
        --ckpt    /workspace/edge/OpenPCDet/checkpoints/cbgs_pp_centerpoint_nds6070.pth \\
        --seed    42 \\
        --frames  5100 \\
        --output  $GITM_DATA_ROOT/runs/edge_baseline_1.json

Frame sourcing: the val infos (loaded by NuScenesWorkUnit's NuScenesDataset)
are the work-list, because multi-sweep accumulation needs the per-keyframe
sweep paths + transforms that live in the infos, not in manifest.jsonl. The
jsonl manifest remains the keyframe freeze record; --manifest is recorded in
the output for provenance and (optionally) cross-checked for count.

Stall breakdown methodology (proxy without CUPTI), per WorkUnit stage timing:
    data_stall_pct = load (10-sweep) + host voxelization
    sync_stall_pct = D2H + box assembly
    gpu_active_pct = forward (backbone + BEV head + NMS; + dyn voxelization)
    cpu_pct        = 100 - the above
See workunit.py STAGE-TIMING CAVEAT: with the dyn config the data_stall /
gpu_active split is architecture-shifted and not spec-conformant until the
voxelization placement is resolved. FPS and convergence are valid regardless.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

WARM_FRAMES = 100        # discarded before the timing window
NVML_SAMPLE_HZ = 5
GPU_ACTIVE_WARN_PCT = 85.0


def _sample_nvml(samples: list[float], stop_event: threading.Event) -> None:
    """Background thread: sample GPU 0 utilization at NVML_SAMPLE_HZ."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        interval = 1.0 / NVML_SAMPLE_HZ
        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            samples.append(float(util))
            time.sleep(interval)
        pynvml.nvmlShutdown()
    except Exception:
        pass  # no GPU or pynvml absent -> nvml_mean_pct null in output


def _convergence_check(results: list[dict]) -> tuple[bool, float]:
    """Return (within_2pct, spread_pct) over the runs' frames_per_second."""
    if len(results) < 2:
        return True, 0.0
    fps_vals = [r["frames_per_second"] for r in results]
    spread = (max(fps_vals) - min(fps_vals)) / max(fps_vals) * 100
    return spread <= 2.0, spread


def _manifest_nuscenes_count(manifest_path: Path) -> int | None:
    """Count nuScenes rows in the jsonl manifest, for a provenance cross-check."""
    if not manifest_path or not manifest_path.exists():
        return None
    n = 0
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "nuscenes" in str(row.get("lidar_path", "")).lower():
                n += 1
    return n


def run_baseline(
    cfg_path: str | Path,
    ckpt_path: str | Path,
    seed: int,
    n_frames: int,
    output_path: str | Path,
    data_root: str | Path,
    manifest_path: str | Path | None = None,
    max_sweeps: int = 10,
) -> dict[str, Any]:
    """Run one baseline over n_frames val keyframes and write JSON.

    n_frames includes the WARM_FRAMES discarded from the front, so the timed
    window is n_frames - WARM_FRAMES (e.g. --frames 5100 -> 5000-frame window).
    """
    import random

    import numpy as np

    from gitm.benchmarks.edge.workunit import NuScenesWorkUnit

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path) if manifest_path else None

    unit = NuScenesWorkUnit.from_checkpoint(
        cfg_path=cfg_path, ckpt_path=ckpt_path,
        data_root=data_root, max_sweeps=max_sweeps,
    )

    n_total = len(unit)
    if n_total == 0:
        raise RuntimeError(
            "NuScenesDataset loaded 0 val infos — check INFO_PATH/test split."
        )
    if n_frames > n_total:
        print(f"  note: requested {n_frames} > {n_total} available; clamping.")
        n_frames = n_total
    if n_frames < WARM_FRAMES + 1:
        raise ValueError(
            f"n_frames={n_frames} too small; need > {WARM_FRAMES} for warm window"
        )

    # Seed both the frame-order shuffle and numpy (sweep selection inside
    # get_lidar_with_sweeps) so each run is reproducible.
    rng = random.Random(seed)
    np.random.seed(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    run_indices = indices[:n_frames]

    print(f"Warming up ({WARM_FRAMES} frames)…")
    for idx in run_indices[:WARM_FRAMES]:
        unit.run(idx)

    warm_indices = run_indices[WARM_FRAMES:]
    nvml_samples: list[float] = []
    stop_event = threading.Event()
    nvml_thread = threading.Thread(
        target=_sample_nvml, args=(nvml_samples, stop_event), daemon=True
    )
    nvml_thread.start()

    data_stall_fracs: list[float] = []
    sync_stall_fracs: list[float] = []
    gpu_active_fracs: list[float] = []
    total_detections = 0

    t_wall_start = time.perf_counter()
    for i, idx in enumerate(warm_indices):
        if i % 500 == 0:
            print(f"  {i}/{len(warm_indices)}")
        result = unit.run(idx)
        data_stall_fracs.append(result.data_stall_frac)
        sync_stall_fracs.append(result.sync_stall_frac)
        gpu_active_fracs.append(result.gpu_active_frac)
        total_detections += result.n_detections
    t_wall_end = time.perf_counter()

    stop_event.set()
    nvml_thread.join(timeout=5)

    n_warm = len(warm_indices)
    elapsed = t_wall_end - t_wall_start
    fps = n_warm / elapsed

    data_stall_pct = sum(data_stall_fracs) / n_warm * 100
    sync_stall_pct = sum(sync_stall_fracs) / n_warm * 100
    gpu_active_pct = sum(gpu_active_fracs) / n_warm * 100
    cpu_pct = max(0.0, 100.0 - data_stall_pct - sync_stall_pct - gpu_active_pct)
    nvml_mean = sum(nvml_samples) / len(nvml_samples) if nvml_samples else None

    manifest_count = _manifest_nuscenes_count(manifest_path) if manifest_path else None

    output = {
        "frames_per_second": round(fps, 3),
        "seed": seed,
        "n_frames_warm_window": n_warm,
        "gpu_active_pct": round(gpu_active_pct, 2),
        "data_stall_pct": round(data_stall_pct, 2),
        "sync_stall_pct": round(sync_stall_pct, 2),
        "cpu_pct": round(cpu_pct, 2),
        "nvml_mean_util_pct": round(nvml_mean, 2) if nvml_mean is not None else None,
        "total_detections": total_detections,
        "elapsed_s": round(elapsed, 3),
        "max_sweeps": max_sweeps,
        # Flags so downstream readers don't treat the stall split as final.
        "stall_breakdown_valid": False,
        "stall_breakdown_note": (
            "dyn config voxelizes on GPU; data_stall/gpu_active split is "
            "architecture-shifted and not spec-conformant until VFE placement "
            "is resolved. FPS + convergence are valid."
        ),
        "dataset": "nuscenes_v1.0-trainval_val",
        "n_val_infos": n_total,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_nuscenes_rows": manifest_count,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(
        f"\nResult: {fps:.1f} fps | GPU active {gpu_active_pct:.1f}% "
        f"(NVML {nvml_mean:.1f}%)" if nvml_mean is not None
        else f"\nResult: {fps:.1f} fps | GPU active {gpu_active_pct:.1f}%"
    )
    print(f"Wrote {output_path}")

    if nvml_mean is not None and nvml_mean > GPU_ACTIVE_WARN_PCT:
        print(
            f"\nWARNING: NVML util={nvml_mean:.1f}% > {GPU_ACTIVE_WARN_PCT}% — "
            "near-saturated. Flag for review / consider 500-frame fallback. "
            "(Expected with dyn GPU-voxelization; see voxelization decision.)"
        )

    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="nuScenes CenterPoint-PointPillar baseline.")
    p.add_argument("--cfg", required=True, help="path to cbgs_dyn_pp_centerpoint.yaml")
    p.add_argument("--ckpt", required=True, help="path to cbgs_pp_centerpoint_nds6070.pth")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--frames", type=int, default=5100,
                   help=f"total frames incl. {WARM_FRAMES} warmup (window = frames - warmup)")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--data-root", default="/workspace/edge/OpenPCDet/data/nuscenes",
                   help="NuScenesDataset root (VERSION is appended)")
    p.add_argument("--manifest", default=None, help="manifest.jsonl for provenance")
    p.add_argument("--max-sweeps", type=int, default=10)
    args = p.parse_args(argv)

    run_baseline(
        cfg_path=args.cfg, ckpt_path=args.ckpt, seed=args.seed,
        n_frames=args.frames, output_path=args.output,
        data_root=args.data_root, manifest_path=args.manifest,
        max_sweeps=args.max_sweeps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
