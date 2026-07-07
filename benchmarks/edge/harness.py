"""Edge/robotics baseline harness — 3D perception via OpenPCDet PointPillars.

The work unit is one keyframe through
``voxelization -> 3D backbone -> BEV head -> NMS``, run with OpenPCDet's
PointPillars config (pinned commit + config hash). The metric is
``frames_per_second`` over a warm window of frames drawn from the combined
``manifest.jsonl`` (nuScenes + KITTI).

As with biotech, there is no CPU equivalent — OpenPCDet *is* the workload — so
this is framework-integration code for a GPU box. The per-frame inference sits
behind a ``Runner`` seam (``infer(frame) -> {...}``) so the scaffolding
(manifest iteration, warm-window timing, contract emission) is testable here
with an injected fake; the real runner is built by :func:`load_openpcdet_runner`.

Prints the one-line harness contract: ``metric_value`` = frames/sec, plus device
info and an optional mean-mAP auxiliary (regression sentinel, not a target).

Environment variables (real runner only):
    OPENPCDET_CFG   — path to pointpillar.yaml config
                      (default: /workspace/edge/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml)
    OPENPCDET_CKPT  — path to pointpillar_7728.pth checkpoint
                      (default: /workspace/edge/checkpoints/pointpillar_7728.pth)
"""

from __future__ import annotations

import argparse
import json
import logging as _logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

OPENPCDET_COMMIT = "v0.6.0"  # pinned; config hash pinned in datasets.md

_OPENPCDET_DEFAULT_CFG = (
    "/workspace/edge/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
)
_OPENPCDET_DEFAULT_CKPT = "/workspace/edge/checkpoints/pointpillar_7728.pth"


class Frame(dict):
    """One manifest.jsonl row: scene_id, frame_id, lidar_path, gt_path, source."""


class Runner(Protocol):
    name: str

    def infer(self, frame: Frame, stage: Path) -> dict: ...


class _OpenPCDetRunner:
    """Real OpenPCDet PointPillars runner for KITTI (and nuScenes) lidar.

    Per-frame timing is returned as ``_t_*`` keys so :func:`run` can aggregate
    a stall breakdown without an external profiler.
    """

    name = "openpcdet-pointpillars"

    def __init__(self, model: Any, dataset: Any, class_names: list[str]) -> None:
        self._model = model
        self._dataset = dataset
        self._class_names = class_names

    def infer(self, frame: Frame, stage: Path) -> dict:
        import numpy as np
        import torch
        from pcdet.models import load_data_to_gpu

        source = frame.get("source", "kitti")
        lidar_rel = frame["lidar_path"]

        if source == "kitti":
            full_path = stage / "kitti" / lidar_rel
            n_feat = 4  # x, y, z, intensity
        elif source == "nuscenes":
            full_path = stage / "nuscenes" / lidar_rel
            n_feat = 5  # x, y, z, intensity, ring_index
        else:
            full_path = stage / lidar_rel
            n_feat = 4

        if not full_path.exists():
            raise FileNotFoundError(f"lidar file not found: {full_path}")

        # Stage 1 — file I/O
        t0 = time.perf_counter()
        pts = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, n_feat)
        if n_feat == 5:
            # KITTI-trained PointPillars expects 4 features; drop ring_index
            pts = pts[:, :4]
        t1 = time.perf_counter()

        # Stage 2 — host voxelization + H2D copy
        data_dict = self._dataset.prepare_data(
            data_dict={"points": pts, "frame_id": frame["frame_id"]}
        )
        data_dict = self._dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        t2 = time.perf_counter()

        # Stage 3 — GPU: backbone + BEV head + NMS (iou3d_nms_cuda inside forward).
        # NMS is a CUDA kernel, so t_inference_s includes it. Use nsys to split
        # backbone vs NMS at the kernel level if needed for the sync-stall column.
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(data_dict)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Stage 4 — D2H copy (negligible; a few microseconds)
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        t4 = time.perf_counter()

        return {
            "n_detections": int(len(scores)),
            "_t_load_s": t1 - t0,
            "_t_preprocess_s": t2 - t1,
            "_t_inference_s": t3 - t2,
            "_t_postprocess_s": t4 - t3,
            "_t_total_s": t4 - t0,
        }


def load_openpcdet_runner(*, config_hash: str | None = None) -> _OpenPCDetRunner:
    """Build the real OpenPCDet PointPillars runner. GPU-only.

    Reads OPENPCDET_CFG / OPENPCDET_CKPT from the environment, falling back to
    the default RunPod paths. Raises RuntimeError if torch/pcdet are absent so
    the caller gets a clear message instead of an AttributeError deep in the stack.
    """
    try:
        import torch  # noqa: F401
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network
    except Exception as exc:  # pragma: no cover - framework absent on laptop
        raise RuntimeError(
            "OpenPCDet/torch not importable — the edge harness runs on a GPU box "
            "with OpenPCDet installed (its CUDA ops compiled). The dataset + "
            "reproducibility loop is exercised via the CPU smoke harness instead.\n"
            "Install on RunPod:\n"
            "  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -e /workspace/edge/OpenPCDet"
        ) from exc

    cfg_path = Path(os.environ.get("OPENPCDET_CFG", _OPENPCDET_DEFAULT_CFG))
    ckpt_path = os.environ.get("OPENPCDET_CKPT", _OPENPCDET_DEFAULT_CKPT)

    # OpenPCDet resolves _BASE_CONFIG_ entries (e.g.
    # cfgs/dataset_configs/kitti_dataset.yaml) relative to the tools/ dir that
    # holds the top-level cfgs/ folder, not relative to the config file. chdir
    # there for the load, then restore — mirrors gitm.benchmarks.kitti.workunit
    # so the harness works regardless of the caller's CWD.
    resolved_parts = cfg_path.resolve().parts
    if "cfgs" in resolved_parts:
        tools_dir = Path(*resolved_parts[: resolved_parts.index("cfgs")])
    else:
        tools_dir = cfg_path.resolve().parent
    prev_cwd = os.getcwd()
    try:
        os.chdir(tools_dir)
        cfg_from_yaml_file(str(cfg_path.resolve()), cfg)
    finally:
        os.chdir(prev_cwd)

    class _InferenceDataset(DatasetTemplate):
        def __init__(self) -> None:
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=list(cfg.CLASS_NAMES),
                training=False,
                root_path=Path("."),
                logger=_logging.getLogger("gitm.edge"),
            )

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> dict:
            raise NotImplementedError("Use runner.infer() directly.")

    dataset = _InferenceDataset()
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    # OpenPCDet's load_params_from_file calls logger.info() unconditionally, so
    # a None logger raises AttributeError — pass a real one.
    model.load_params_from_file(
        filename=str(ckpt_path), logger=_logging.getLogger("gitm.edge"), to_cpu=True
    )
    model.cuda().eval()

    return _OpenPCDetRunner(model, dataset, list(cfg.CLASS_NAMES))


def iter_frames(manifest: Path, *, warm: int) -> Iterator[Frame]:
    """Yield up to ``warm`` keyframes from manifest.jsonl (file order)."""
    n = 0
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Frame(json.loads(line))
            n += 1
            if n >= warm:
                return


def run(stage: Path, *, warm: int, runner: Runner) -> dict:
    """Run the warm window of frames through ``runner`` and return the payload."""
    manifest = stage / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"missing {manifest} — build it with `make keyframes` (gitm.bench edge-manifest)"
        )

    n = 0
    maps: list[float] = []
    timings: list[dict] = []
    t0 = time.perf_counter()
    for frame in iter_frames(manifest, warm=warm):
        result = runner.infer(frame, stage)
        n += 1
        if "map" in result:
            maps.append(float(result["map"]))
        if "_t_total_s" in result:
            timings.append(result)
    elapsed = max(time.perf_counter() - t0, 1e-9)

    if n == 0:
        raise RuntimeError(f"no frames in {manifest}")

    payload: dict = {
        "metric_value": n / elapsed,  # frames per second
        "n_frames": n,
        "mean_map": (sum(maps) / len(maps)) if maps else None,
        "harness_commit": f"openpcdet-{OPENPCDET_COMMIT}",
    }
    if timings:
        payload["stall_breakdown"] = [_build_stall_phase(timings, elapsed)]
    return payload


def _build_stall_phase(timings: list[dict], wall_clock_s: float) -> dict:
    """Aggregate per-frame timing dicts into a single StallPhase-compatible dict.

    Fractions sum to 1.0 within frame time. The ``cpu`` field captures residual
    Python overhead not attributed to load/preprocess/inference/postprocess.
    Note: NMS runs inside model.forward() as iou3d_nms_cuda, so it is included in
    ``gpu_active`` here. Use nsys to separate backbone from NMS at kernel level.
    """
    t_load = sum(t["_t_load_s"] for t in timings)
    t_pre = sum(t["_t_preprocess_s"] for t in timings)
    t_inf = sum(t["_t_inference_s"] for t in timings)
    t_post = sum(t["_t_postprocess_s"] for t in timings)
    total = max(sum(t["_t_total_s"] for t in timings), 1e-9)

    data_stall = min(1.0, (t_load + t_pre) / total)
    gpu_active = min(1.0, t_inf / total)
    sync = min(1.0, t_post / total)
    cpu = max(0.0, 1.0 - data_stall - gpu_active - sync)

    return {
        "phase": "all",
        "cpu": round(cpu, 4),
        "data_stall": round(data_stall, 4),
        "sync": round(sync, 4),
        "gpu_active": round(gpu_active, 4),
        "throughput": len(timings) / wall_clock_s,
        "wall_clock_s": round(wall_clock_s, 3),
    }


def _device_info() -> tuple[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return "cpu", 0


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(description="Edge perception harness (OpenPCDet).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-frames", type=int, default=5000)
    p.add_argument("--stage", type=Path, default=None)
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    runner = runner or load_openpcdet_runner()
    gpu_name, device_count = _device_info()

    payload = run(stage, warm=args.warm_frames, runner=runner)
    payload.update({"gpu_name": gpu_name, "device_count": device_count})

    print(f"[edge harness:{getattr(runner, 'name', '?')}] "
          f"{payload['n_frames']} frames, {payload['metric_value']:.1f} fps")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
