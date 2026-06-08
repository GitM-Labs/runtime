"""PointPillars WorkUnit for KITTI object detection.

One WorkUnit = one frame processed end-to-end:
    load .bin → voxelize → backbone + BEV head → NMS → detections

Stages are timed individually so the baseline runner can compute the
stall breakdown without CUPTI.

Usage (on the RunPod GPU box once OpenPCDet is installed):

    unit = WorkUnit.from_checkpoint(
        cfg_path="/path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
        ckpt_path="$GITM_DATA_ROOT/checkpoints/kitti/pointpillar_7728.pth",
    )
    result = unit.run("/path/to/velodyne/000042.bin")
    print(result.n_detections, result.t_total_s)

OpenPCDet is a soft dependency — this module imports without it.
Calling from_checkpoint() raises ImportError with install instructions
if OpenPCDet is absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# SHA256 of pointpillar_7728.pth (OpenPCDet KITTI PointPillars checkpoint).
# Confirm with: sha256sum pointpillar_7728.pth
CHECKPOINT_SHA256 = "c9c84e5cf1059b84fb37a4d47f8e58fc16b22e2c3e9ddf47ed59700d7b0e9ccd"

# SHA256 of the pinned tools/cfgs/kitti_models/pointpillar.yaml.
# Fill in after cloning OpenPCDet at the pinned commit:
#   sha256sum tools/cfgs/kitti_models/pointpillar.yaml
CONFIG_SHA256 = ""


@dataclass
class WorkUnitResult:
    frame_id: str
    n_detections: int
    detections: list[dict[str, Any]] = field(default_factory=list)

    t_load_s: float = 0.0        # file I/O: np.fromfile
    t_preprocess_s: float = 0.0  # voxelization + H2D copy
    t_inference_s: float = 0.0   # GPU: backbone + BEV head (wall after sync)
    t_postprocess_s: float = 0.0 # NMS + box assembly
    t_total_s: float = 0.0

    @property
    def data_stall_frac(self) -> float:
        """Fraction of frame time spent on data loading + voxelization."""
        if self.t_total_s <= 0:
            return 0.0
        return (self.t_load_s + self.t_preprocess_s) / self.t_total_s

    @property
    def sync_stall_frac(self) -> float:
        """Fraction of frame time spent on NMS (CPU-serialized post-processing)."""
        if self.t_total_s <= 0:
            return 0.0
        return self.t_postprocess_s / self.t_total_s

    @property
    def gpu_active_frac(self) -> float:
        """Fraction of frame time spent in GPU backbone + BEV head."""
        if self.t_total_s <= 0:
            return 0.0
        return self.t_inference_s / self.t_total_s


class WorkUnit:
    """Runs one KITTI Velodyne frame through PointPillars end-to-end."""

    def __init__(self, model: Any, dataset: Any, class_names: list[str]) -> None:
        self._model = model
        self._dataset = dataset
        self._class_names = class_names

    @classmethod
    def from_checkpoint(
        cls,
        cfg_path: str | Path,
        ckpt_path: str | Path,
    ) -> "WorkUnit":
        """Load OpenPCDet config + checkpoint and return a ready WorkUnit.

        Raises ImportError with install instructions if OpenPCDet is not found.
        """
        try:
            import logging

            import torch  # noqa: F401 — confirm GPU torch before continuing
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network
        except ImportError as exc:
            raise ImportError(
                "OpenPCDet not installed. On the RunPod dev box:\n"
                "  git clone https://github.com/open-mmlab/OpenPCDet.git\n"
                "  cd OpenPCDet && pip install -e .\n"
                "Then pull the checkpoint:\n"
                "  # Ask Adit for the checkpoint URL or pull from S3"
            ) from exc

        # OpenPCDet resolves _BASE_CONFIG_ entries (e.g.
        # cfgs/dataset_configs/kitti_dataset.yaml) relative to the current
        # working directory, which is the tools/ dir holding the top-level
        # cfgs/ folder. Temporarily chdir there so config loading works no
        # matter where the caller runs from, then restore the CWD.
        import os

        cfg_path = Path(cfg_path)
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

        # Minimal dataset wrapper for single-frame inference.
        # DatasetTemplate.prepare_data handles the voxelization pipeline;
        # we only need root_path to satisfy the base class __init__.
        class _InferenceDataset(DatasetTemplate):
            def __init__(self) -> None:
                super().__init__(
                    dataset_cfg=cfg.DATA_CONFIG,
                    class_names=list(cfg.CLASS_NAMES),
                    training=False,
                    root_path=Path("."),
                    logger=logging.getLogger("gitm.kitti"),
                )

            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int) -> dict:
                raise NotImplementedError("Use WorkUnit.run() directly")

        dataset = _InferenceDataset()
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(
            filename=str(ckpt_path),
            logger=logging.getLogger("gitm.kitti"),
            to_cpu=True,
        )
        model.cuda()
        model.eval()
        return cls(model=model, dataset=dataset, class_names=list(cfg.CLASS_NAMES))

    def run(self, velodyne_path: str | Path) -> WorkUnitResult:
        """Process one frame and return detections + per-stage timing.

        velodyne_path: path to a KITTI Velodyne .bin file.
        """
        import numpy as np
        import torch
        from pcdet.models import load_data_to_gpu

        frame_id = Path(velodyne_path).stem
        t0 = time.perf_counter()

        # Stage 1: load point cloud (N, 4) float32 — x, y, z, intensity
        points = np.fromfile(str(velodyne_path), dtype=np.float32).reshape(-1, 4)
        t1 = time.perf_counter()

        # Stage 2: voxelization + H2D copy
        data_dict = self._dataset.prepare_data(
            data_dict={"points": points, "frame_id": frame_id}
        )
        data_dict = self._dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        t2 = time.perf_counter()

        # Stage 3: backbone + BEV head + NMS — all inside model.forward().
        # OpenPCDet's PointPillars post-processing module runs GPU-accelerated NMS
        # via iou3d_nms_cuda before returning, so t_inference_s includes NMS.
        # For NMS-separate timing, use nsys to split at the kernel level.
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(data_dict)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Stage 4: D2H copy of result tensors (negligible; a few microseconds).
        # t_postprocess_s is near-zero here — real sync stall (NMS) is in t_inference_s.
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].cpu().numpy()   # (N, 7)
        scores = pred["pred_scores"].cpu().numpy()  # (N,)
        labels = pred["pred_labels"].cpu().numpy()  # (N,) 1-indexed
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
