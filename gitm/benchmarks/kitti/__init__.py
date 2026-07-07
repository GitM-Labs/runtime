"""KITTI PointPillars benchmark harness.

    from gitm.benchmarks.kitti import WorkUnit, run_baseline
"""

from gitm.benchmarks.kitti.baseline import run_baseline
from gitm.benchmarks.kitti.workunit import WorkUnit

__all__ = ["WorkUnit", "run_baseline"]
