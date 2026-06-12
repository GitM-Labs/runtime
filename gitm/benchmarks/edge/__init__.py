"""GITM edge (nuScenes) benchmark — CenterPoint-PointPillar baseline.

Mirrors gitm.benchmarks.kitti but for the nuScenes CenterPoint-PointPillar
(dyn / GPU-voxelization) baseline, with multi-sweep (keyframe + 9 sweeps)
point accumulation sourced from OpenPCDet's NuScenesDataset.
"""

from gitm.benchmarks.edge.workunit import NuScenesWorkUnit, WorkUnitResult

__all__ = ["NuScenesWorkUnit", "WorkUnitResult"]
