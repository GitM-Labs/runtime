"""Behavioral Compiler — emits a predicted execution graph.

This is the **predicted** half of the residual equation. Roofline-based
per-operation predictions: max of compute-bound and memory-bound time.
GQA-aware for decode. Wrong in detail, right in shape — residuals isolate
the optimization-worthy kernels.
"""

from __future__ import annotations

from gitm.planner.graph import Graph, PredictedNode, predict_graph
from gitm.planner.kitti_graph import KittiGraph, KittiNode, predict_kitti_graph, render_kitti_graph
from gitm.planner.roofline import HardwareSpec, ModelSpec, BatchConfig, roofline

__all__ = [
    "Graph",
    "PredictedNode",
    "predict_graph",
    "KittiGraph",
    "KittiNode",
    "predict_kitti_graph",
    "render_kitti_graph",
    "HardwareSpec",
    "ModelSpec",
    "BatchConfig",
    "roofline",
]
