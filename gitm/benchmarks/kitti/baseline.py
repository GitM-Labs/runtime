"""KITTI PointPillars baseline runner.

Runs convergent baselines over the 7,481 training frames, records the
stall breakdown and GPU headroom, and writes a JSON result file per run.

Usage (on the RunPod GPU box):

    python -m gitm.benchmarks.kitti.baseline \\
        --cfg        /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \\
        --ckpt       $GITM_DATA_ROOT/checkpoints/kitti/pointpillar_7728.pth \\
        --data-root  $GITM_DATA_ROOT \\
        --seed       42 \\
        --frames     7481 \\
        --output     $GITM_DATA_ROOT/runs/kitti_baseline_1.json

Stall breakdown methodology (proxy without CUPTI):
    data_stall_pct  = mean fraction of frame time on file-load + voxelization
    sync_stall_pct  = mean fraction of frame time on NMS (CPU-serialized)
    gpu_active_pct  = mean fraction of frame time in GPU backbone + BEV head
    cpu_pct         = 100 - data_stall_pct - sync_stall_pct - gpu_active_pct

GPU headroom (via NVML + gitm.optimizer.headroom_kernel_rank):
    compute_headroom_pct = 100 - mean NVML utilization
    mem_free_at_peak_gb  = free VRAM at peak memory pressure

NVML utilization is sampled at 5 Hz throughout the warm window.
Kernel-level ROI (kernel_roi) requires CUPTI traces — not captured here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


WARM_FRAMES = 100  # discard before timing window
NVML_SAMPLE_HZ = 5

# KITTI canonical layout relative to GITM_DATA_ROOT, matching kitti_source.py
_KITTI_VELODYNE_SUBDIR = Path("kitti") / "training" / "velodyne"

_STAGE_NAMES = ["load", "preprocess", "inference", "postprocess"]


def _stage_spread(
    load_s: list[float],
    preprocess_s: list[float],
    inference_s: list[float],
    postprocess_s: list[float],
) -> dict:
    """Compute p50/p95 latency and % of frame for each pipeline stage.

    Surfaces the runtime's kernel-level spread signal: high p95/p50 ratio on a
    stage means long-tail invocations that are good intervention targets.
    Mirrors the shape of headroom_kernel_rank.render_roi_table() but for
    PointPillars pipeline stages rather than CUDA kernel families.
    """
    import statistics

    totals = [l + p + i + q for l, p, i, q in
              zip(load_s, preprocess_s, inference_s, postprocess_s)]
    mean_total = statistics.mean(totals) if totals else 1.0

    def _stats(xs: list[float]) -> dict:
        s = sorted(xs)
        n = len(s)
        p50 = s[int(0.50 * (n - 1))] if n else 0.0
        p95 = s[int(0.95 * (n - 1))] if n else 0.0
        mean = statistics.mean(xs) if xs else 0.0
        return {
            "mean_ms": round(mean * 1000, 3),
            "p50_ms": round(p50 * 1000, 3),
            "p95_ms": round(p95 * 1000, 3),
            "p95_p50_ratio": round(p95 / p50, 2) if p50 > 0 else None,
            "mean_pct": round(mean / mean_total * 100, 2) if mean_total > 0 else 0.0,
        }

    return {
        "load": _stats(load_s),
        "preprocess": _stats(preprocess_s),
        "inference": _stats(inference_s),
        "postprocess": _stats(postprocess_s),
    }


def render_stage_spread(spread: dict) -> str:
    """Human-readable stage spread table (stdout + saved as stage_spread_report.txt)."""
    lines = ["PointPillars stage spread (p50 / p95 latency per stage):", ""]
    hdr = f"{'stage':<14} {'mean_ms':>9} {'p50_ms':>9} {'p95_ms':>9} {'p95/p50':>8} {'% frame':>8}"
    lines += [hdr, "-" * len(hdr)]
    for stage in _STAGE_NAMES:
        s = spread.get(stage, {})
        ratio = s.get("p95_p50_ratio")
        lines.append(
            f"{stage:<14} {s.get('mean_ms', 0):>9.3f} {s.get('p50_ms', 0):>9.3f} "
            f"{s.get('p95_ms', 0):>9.3f} {ratio if ratio else 'N/A':>8} "
            f"{s.get('mean_pct', 0):>7.1f}%"
        )
    lines += ["", "High p95/p50 ratio = long-tail invocations = intervention target."]
    return "\n".join(lines)


def _load_frame_paths(data_root: Path) -> list[Path]:
    """Return sorted velodyne .bin paths from $GITM_DATA_ROOT/kitti/training/velodyne/.

    Sorted by stem (000000, 000001, ...) for deterministic, reproducible ordering
    across machines — same order as kitti_source.py's iter_rows().
    """
    velodyne_dir = data_root / _KITTI_VELODYNE_SUBDIR
    if not velodyne_dir.is_dir():
        raise FileNotFoundError(
            f"KITTI velodyne directory not found: {velodyne_dir}\n"
            f"Expected extracted KITTI Object data at {data_root}/kitti/training/\n"
            f"Ask Adit for the S3 path, or set GITM_DATA_ROOT correctly."
        )
    paths = sorted(velodyne_dir.glob("*.bin"), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"No .bin lidar files found in {velodyne_dir}.")
    return paths


def _sample_nvml(samples: list[dict], stop_event: threading.Event) -> None:
    """Background thread: sample GPU 0 util + memory at NVML_SAMPLE_HZ.

    Each sample is a dict with util_pct, mem_used_bytes, mem_total_bytes —
    the format expected by gitm.optimizer.headroom_kernel_rank.gpu_headroom().
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        interval = 1.0 / NVML_SAMPLE_HZ
        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            samples.append({
                "util_pct": float(util),
                "mem_used_bytes": int(mem.used),
                "mem_total_bytes": int(mem.total),
            })
            time.sleep(interval)
        pynvml.nvmlShutdown()
    except Exception:
        pass  # no GPU or pynvml absent — headroom fields will be null in output


def _convergence_check(results: list[dict]) -> tuple[bool, float]:
    """Return (within_2pct, spread_pct) for a list of run results."""
    if len(results) < 2:
        return True, 0.0
    fps_vals = [r["frames_per_second"] for r in results]
    spread = (max(fps_vals) - min(fps_vals)) / max(fps_vals) * 100
    return spread <= 2.0, spread


def run_baseline(
    cfg_path: str | Path,
    ckpt_path: str | Path,
    seed: int,
    n_frames: int,
    output_path: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one baseline over n_frames KITTI frames and write JSON to output_path.

    Args:
        cfg_path:    Path to pointpillar.yaml (OpenPCDet config).
        ckpt_path:   Path to pointpillar_7728.pth checkpoint.
        seed:        Random seed controlling frame traversal order.
        n_frames:    Total frames to process (WARM_FRAMES discarded + timing window).
        output_path: Destination JSON file.
        data_root:   GITM_DATA_ROOT — root of kitti/training/velodyne/. Falls back
                     to the GITM_DATA_ROOT env var if None.

    Returns:
        Result dict (same content as the JSON file).
    """
    import random

    import numpy as np

    from gitm.benchmarks.kitti.workunit import WorkUnit

    if data_root is None:
        env = os.environ.get("GITM_DATA_ROOT")
        if not env:
            raise RuntimeError(
                "data_root not provided and GITM_DATA_ROOT env var is not set."
            )
        data_root = env

    data_root = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_paths = _load_frame_paths(data_root)

    rng = random.Random(seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    run_paths = shuffled[:n_frames]

    total = len(run_paths)
    if total < WARM_FRAMES + 1:
        raise ValueError(
            f"n_frames={n_frames} is too small; need > {WARM_FRAMES} for warm window"
        )

    # Disk pre-warm: read all timing-window frames into the OS page cache before
    # any GPU warmup or measurement. Without this, seeds that shuffle frames into
    # a non-sequential (cache-cold) access pattern report lower fps than seeds
    # that happen to be cache-warm from prior runs — causing spurious divergence.
    # KITTI training velodyne is ~750 MB, comfortably fits in page cache.
    print(f"Pre-warming disk cache ({len(run_paths)} frames)…")
    for path in run_paths:
        np.fromfile(str(path), dtype=np.float32)

    unit = WorkUnit.from_checkpoint(cfg_path=cfg_path, ckpt_path=ckpt_path)

    print(f"GPU warmup ({WARM_FRAMES} frames)…")
    for path in run_paths[:WARM_FRAMES]:
        unit.run(path)

    # Timed warm window
    warm_paths = run_paths[WARM_FRAMES:]
    nvml_samples: list[dict] = []
    stop_event = threading.Event()
    nvml_thread = threading.Thread(
        target=_sample_nvml, args=(nvml_samples, stop_event), daemon=True
    )
    nvml_thread.start()

    t_wall_start = time.perf_counter()

    data_stall_fracs: list[float] = []
    sync_stall_fracs: list[float] = []
    gpu_active_fracs: list[float] = []
    total_detections = 0

    # Per-frame stage timings for spread analysis
    t_load_list: list[float] = []
    t_preprocess_list: list[float] = []
    t_inference_list: list[float] = []
    t_postprocess_list: list[float] = []

    for i, path in enumerate(warm_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(warm_paths)}")
        result = unit.run(path)
        data_stall_fracs.append(result.data_stall_frac)
        sync_stall_fracs.append(result.sync_stall_frac)
        gpu_active_fracs.append(result.gpu_active_frac)
        total_detections += result.n_detections
        t_load_list.append(result.t_load_s)
        t_preprocess_list.append(result.t_preprocess_s)
        t_inference_list.append(result.t_inference_s)
        t_postprocess_list.append(result.t_postprocess_s)

    t_wall_end = time.perf_counter()
    stop_event.set()
    nvml_thread.join(timeout=5)

    n_warm = len(warm_paths)
    elapsed = t_wall_end - t_wall_start
    fps = n_warm / elapsed

    data_stall_pct = sum(data_stall_fracs) / n_warm * 100
    sync_stall_pct = sum(sync_stall_fracs) / n_warm * 100
    gpu_active_pct = sum(gpu_active_fracs) / n_warm * 100
    cpu_pct = max(0.0, 100.0 - data_stall_pct - sync_stall_pct - gpu_active_pct)

    # Stage spread — kernel-level spread signal for PointPillars pipeline stages
    spread = _stage_spread(t_load_list, t_preprocess_list, t_inference_list, t_postprocess_list)
    spread_report = render_stage_spread(spread)
    print("\n" + spread_report)

    # GPU headroom from NVML samples via gitm.optimizer.headroom_kernel_rank
    headroom_dict: dict[str, Any] = {}
    try:
        from gitm.optimizer.headroom_kernel_rank import gpu_headroom

        h = gpu_headroom(nvml_samples)
        if h is not None:
            headroom_dict = {
                "compute_headroom_pct": round(h.compute_headroom_pct, 2),
                "mean_util_pct": round(h.mean_util_pct, 2),
                "peak_util_pct": round(h.peak_util_pct, 2),
                "mem_free_at_peak_gb": round(h.mem_free_at_peak_bytes / 1e9, 3),
                "mem_total_gb": round(h.mem_total_bytes / 1e9, 3),
                "nvml_n_samples": h.n_samples,
            }
    except Exception:
        pass  # graceful degradation if headroom_kernel_rank unavailable

    import platform
    import socket

    output: dict[str, Any] = {
        "frames_per_second": round(fps, 3),
        "seed": seed,
        "n_frames_warm_window": n_warm,
        "gpu_active_pct": round(gpu_active_pct, 2),
        "data_stall_pct": round(data_stall_pct, 2),
        "sync_stall_pct": round(sync_stall_pct, 2),
        "cpu_pct": round(cpu_pct, 2),
        **headroom_dict,
        "stage_spread": spread,
        "total_detections": total_detections,
        "elapsed_s": round(elapsed, 3),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "data_root": str(data_root),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_path.write_text(json.dumps(output, indent=2))
    # Save stage spread report alongside the JSON for easy review
    report_path = output_path.with_name(output_path.stem + "_stage_spread.txt")
    report_path.write_text(spread_report)
    print(
        f"\nResult: {fps:.1f} fps | GPU active {gpu_active_pct:.1f}% "
        f"| data stall {data_stall_pct:.1f}% | wrote {output_path}"
    )
    if headroom_dict:
        print(
            f"Headroom: compute {headroom_dict['compute_headroom_pct']:.1f}% free "
            f"| mem {headroom_dict['mem_free_at_peak_gb']:.1f} GB free at peak"
        )

    if headroom_dict.get("mean_util_pct", 0) > 85.0:
        print(
            f"\nWARNING: NVML util={headroom_dict['mean_util_pct']:.1f}% > 85% — "
            "flag Adit: workload may be near-saturated. Consider the 500-frame fallback."
        )

    return output


def _check_convergence_across_runs(run_paths: list[Path]) -> None:
    """Read existing baseline JSONs and report convergence."""
    results = []
    for p in run_paths:
        if p.exists():
            with p.open() as fh:
                results.append(json.load(fh))
    ok, spread = _convergence_check(results)
    if len(results) >= 2:
        status = "PASS" if ok else "FAIL"
        print(
            f"Convergence ({len(results)} runs): {status} — spread={spread:.2f}% "
            f"(threshold 2.0%)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one KITTI PointPillars baseline.")
    parser.add_argument("--cfg", required=True, help="OpenPCDet config .yaml path.")
    parser.add_argument("--ckpt", required=True, help="PointPillars checkpoint .pth path.")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("GITM_DATA_ROOT"),
        help="GITM_DATA_ROOT — root of kitti/training/velodyne/. "
             "Defaults to $GITM_DATA_ROOT env var.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--frames", type=int, default=7481, help="Frames to process.")
    parser.add_argument("--output", required=True, help="Output .json path.")
    parser.add_argument(
        "--check-convergence",
        nargs="+",
        metavar="JSON",
        help="After writing, check convergence against these existing baseline JSONs.",
    )
    args = parser.parse_args(argv)

    if not args.data_root:
        parser.error("--data-root is required (or set GITM_DATA_ROOT env var)")

    run_baseline(
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        seed=args.seed,
        n_frames=args.frames,
        output_path=args.output,
        data_root=args.data_root,
    )

    if args.check_convergence:
        all_paths = [Path(args.output)] + [Path(p) for p in args.check_convergence]
        _check_convergence_across_runs(all_paths)

    return 0


if __name__ == "__main__":
    sys.exit(main())
