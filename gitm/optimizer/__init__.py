"""Optimizer — operates on the residuals between observed and predicted.

This package contains the deviation monitor (residuals against 3 invariants),
causal attribution (Granger + DR), counterfactual replay, qualification gate,
and the provenance report writer.
"""

from __future__ import annotations

from gitm.optimizer.attribution import RankedHypotheses, attribute
from gitm.optimizer.headroom import (
    GpuHeadroom,
    KernelROI,
    gpu_headroom,
    kernel_family,
    kernel_roi,
    live_gpu_headroom,
    render_roi_table,
)
from gitm.optimizer.invariants import INVARIANTS, Invariant, Violation
from gitm.optimizer.monitor import Residuals, check_invariants, residuals
from gitm.optimizer.qualification import QualificationResult, qualify
from gitm.optimizer.replay import predict_delta
from gitm.optimizer.report import Claim, Provenance, write_report

__all__ = [
    "Residuals",
    "residuals",
    "check_invariants",
    "Invariant",
    "INVARIANTS",
    "Violation",
    "RankedHypotheses",
    "attribute",
    "predict_delta",
    "QualificationResult",
    "qualify",
    "Claim",
    "Provenance",
    "write_report",
    "kernel_roi",
    "KernelROI",
    "render_roi_table",
    "gpu_headroom",
    "GpuHeadroom",
    "kernel_family",
    "live_gpu_headroom",
]