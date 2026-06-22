"""CenterPoint-PointPillar WorkUnit for nuScenes object detection.

One WorkUnit = one keyframe processed end-to-end:
    load keyframe + 9 sweeps (ego-motion compensated) -> voxelize
        -> backbone + BEV head -> NMS -> detections

Mirrors gitm.benchmarks.kitti.workunit, with three nuScenes differences:
  * Multi-sweep input. Points are 10 accumulated sweeps (1 keyframe + 9 prior),
    each transformed into the keyframe frame via its precomputed
    transform_matrix, with a per-point time_lag in the 5th feature column.
    This is sourced from OpenPCDet's NuScenesDataset.get_lidar_with_sweeps so
    the points are byte-identical to the Phase 2 evaluation.
  * 5 point features: [x, y, z, intensity, time]. Note the raw .pcd.bin 5th
    column is *ring* and is discarded by OpenPCDet; the 5th feature is the
    computed sweep time_lag (0.0 for the keyframe).
  * 10 nuScenes classes (taken from cfg.CLASS_NAMES, not hard-coded).

STAGE-TIMING CAVEAT (load-bearing for the stall breakdown):
    Stage boundaries assume host-side voxelization. With the dyn config
    (DynamicPillarVFE), voxelization runs on the GPU inside model.forward(),
    so t_preprocess_s is near-empty and that work lands in t_inference_s.
    The stall breakdown derived from these stages is therefore only
    spec-conformant (20-35% data-stall / 50-65% GPU) under a static-VFE
    (PillarVFE) config. Until the voxelization-placement decision is made,
    treat data_stall/gpu_active from this WorkUnit as architecture-caveated.
    FPS (t_total_s) is unaffected and valid regardless.

OpenPCDet is a soft dependency — this module imports without it. Calling
from_checkpoint() raises ImportError with install instructions if absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# SHA256 of cbgs_pp_centerpoint_nds6070.pth (OpenPCDet nuScenes
# CenterPoint-PointPillar checkpoint). Confirm with:
#   sha256sum cbgs_pp_centerpoint_nds6070.pth
CHECKPOINT_SHA256 = "955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a"

# SHA256 of the pinned tools/cfgs/dataset_configs/nuscenes_dataset.yaml.
CONFIG_SHA256 = "a9a561d3da8a05e0537da55c15817b19dd5ff60c968a562d9475bc71094e1036"

# Default data root passed to NuScenesDataset. The dataset appends VERSION
# (v1.0-trainval), so this is the dir whose child v1.0-trainval/ holds
# samples/, sweeps/, the metadata, and the infos pkls.
DEFAULT_DATA_ROOT = "/workspace/edge/OpenPCDet/data/nuscenes"

# 1 keyframe + 9 sweeps = 10 accumulated point sets (matches the
# nuscenes_infos_10sweeps_*.pkl generation).
DEFAULT_MAX_SWEEPS = 10


@dataclass
class WorkUnitResult:
    frame_id: str
    n_detections: int
    detections: list[dict[str, Any]] = field(default_factory=list)

    t_load_s: float = 0.0        # get_lidar_with_sweeps: 10-file load + transforms
    t_preprocess_s: float = 0.0  # voxelization (host, static VFE) + H2D copy
    t_inference_s: float = 0.0   # GPU: (dyn voxelization +) backbone + BEV head + NMS
    t_postprocess_s: float = 0.0 # D2H copy + box assembly
    t_total_s: float = 0.0

    @property
    def data_stall_frac(self) -> float:
        """Fraction of frame time on point load + host voxelization.

        With the dyn config, voxelization is on the GPU, so this reflects
        only the multi-sweep load. See the module STAGE-TIMING CAVEAT.
        """
        if self.t_total_s <= 0:
            return 0.0
        return (self.t_load_s + self.t_preprocess_s) / self.t_total_s

    @property
    def sync_stall_frac(self) -> float:
        if self.t_total_s <= 0:
            return 0.0
        return self.t_postprocess_s / self.t_total_s

    @property
    def gpu_active_frac(self) -> float:
        if self.t_total_s <= 0:
            return 0.0
        return self.t_inference_s / self.t_total_s


class NuScenesWorkUnit:
    """Runs one nuScenes keyframe (10-sweep) through CenterPoint-PointPillar."""

    def __init__(self, model: Any, dataset: Any, class_names: list[str],
                 max_sweeps: int) -> None:
        self._model = model
        self.dataset = dataset
        self._class_names = class_names
        self._max_sweeps = max_sweeps

    def __len__(self) -> int:
        return len(self.dataset)

    @classmethod
    def from_checkpoint(
        cls,
        cfg_path: str | Path,
        ckpt_path: str | Path,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        max_sweeps: int = DEFAULT_MAX_SWEEPS,
    ) -> NuScenesWorkUnit:
        """Load OpenPCDet config + checkpoint + NuScenesDataset (val split).

        Raises ImportError with install instructions if OpenPCDet is absent.
        """
        try:
            import logging
            import os

            import torch  # noqa: F401 — confirm GPU torch before continuing
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
            from pcdet.models import build_network
        except ImportError as exc:
            raise ImportError(
                "OpenPCDet not installed / importable. On the GPU box:\n"
                "  cd /workspace/edge/OpenPCDet && pip install -e . --no-build-isolation\n"
                "  pip install nuscenes-devkit 'numpy==1.26.4' torch_scatter\n"
                "and apply numpy_alias_fixes.patch on top of the pinned commit."
            ) from exc

        cfg_path = Path(cfg_path)

        # OpenPCDet resolves _BASE_CONFIG_ entries relative to the tools/ dir
        # holding the top-level cfgs/ folder. chdir there for config load.
        parts = cfg_path.resolve().parts
        if "cfgs" in parts:
            tools_dir = Path(*parts[: parts.index("cfgs")])
        else:
            tools_dir = cfg_path.resolve().parent
        prev_cwd = os.getcwd()
        try:
            os.chdir(tools_dir)
            cfg_from_yaml_file(str(cfg_path.resolve()), cfg)
        finally:
            os.chdir(prev_cwd)

        logger = logging.getLogger("gitm.edge")

        # Real NuScenesDataset in test mode -> loads the val infos (6019) and
        # provides get_lidar_with_sweeps for faithful multi-sweep loading.
        dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=list(cfg.CLASS_NAMES),
            training=False,
            root_path=Path(data_root),
            logger=logger,
        )

        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(
            filename=str(ckpt_path), logger=logger, to_cpu=True
        )
        model.cuda()
        model.eval()
        return cls(
            model=model,
            dataset=dataset,
            class_names=list(cfg.CLASS_NAMES),
            max_sweeps=max_sweeps,
        )

    def run(self, index: int) -> WorkUnitResult:
        """Process one keyframe (by val-infos index) end-to-end.

        index: position into the dataset's val infos (0 .. len(self)-1).
        """
        import torch
        from pcdet.models import load_data_to_gpu

        info = self.dataset.infos[index]
        frame_id = info.get("token", str(index))
        t0 = time.perf_counter()

        # Stage 1: load keyframe + (max_sweeps-1) sweeps, ego-motion
        # compensated, with per-point time in column 5. Reuses OpenPCDet's
        # own accumulation so points match Phase 2 exactly.
        points = self.dataset.get_lidar_with_sweeps(
            index, max_sweeps=self._max_sweeps
        )
        t1 = time.perf_counter()

        # Stage 2: voxelization (host for static VFE; near-noop for dyn) + H2D.
        data_dict = self.dataset.prepare_data(
            data_dict={"points": points, "frame_id": frame_id}
        )
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        t2 = time.perf_counter()

        # Stage 3: (dyn voxelization +) backbone + BEV head + NMS in forward().
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(data_dict)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Stage 4: D2H copy + box assembly.
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()  # 1-indexed
        t4 = time.perf_counter()

        n = len(scores)
        detections = [
            {
                "name": self._class_names[int(labels[i]) - 1],
                "score": float(scores[i]),
                "box3d": boxes[i].tolist(),
            }
            for i in range(n)
        ]

        return WorkUnitResult(
            frame_id=frame_id,
            n_detections=n,
            detections=detections,
            t_load_s=t1 - t0,
            t_preprocess_s=t2 - t1,
            t_inference_s=t3 - t2,
            t_postprocess_s=t4 - t3,
            t_total_s=t4 - t0,
        )

    def run_batch(self, indices: list) -> list[WorkUnitResult]:
        """Process a batch of keyframes in ONE forward pass (detections only).

        Collates B keyframes (each with its 10-sweep accumulation) and launches
        the network once, amortizing per-kernel launch overhead. Per-frame
        equivalent to :meth:`run` in eval mode (BatchNorm running stats; conv is
        per-sample), so detections match the single-frame path within float
        tolerance — the batching intervention's correctness gate. Stage timings
        are omitted (the A/B times wall-clock around this call).
        """
        if not indices:
            return []

        import torch
        from pcdet.models import load_data_to_gpu

        frame_ids: list[str] = []
        data_dicts = []
        for index in indices:
            info = self.dataset.infos[index]
            fid = info.get("token", str(index))
            frame_ids.append(fid)
            points = self.dataset.get_lidar_with_sweeps(index, max_sweeps=self._max_sweeps)
            data_dicts.append(
                self.dataset.prepare_data(data_dict={"points": points, "frame_id": fid})
            )
        batch = self.dataset.collate_batch(data_dicts)
        load_data_to_gpu(batch)
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(batch)
        torch.cuda.synchronize()

        results: list[WorkUnitResult] = []
        for fid, pred in zip(frame_ids, pred_dicts, strict=True):
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            boxes = pred["pred_boxes"].cpu().numpy()
            n = len(scores)
            results.append(
                WorkUnitResult(
                    frame_id=fid,
                    n_detections=n,
                    detections=[
                        {
                            "name": self._class_names[int(labels[i]) - 1],
                            "score": float(scores[i]),
                            "box3d": boxes[i].tolist(),
                        }
                        for i in range(n)
                    ],
                )
            )
        return results
