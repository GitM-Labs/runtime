"""Customer profiler intake — import external traces into gitm's Trace schema.

Supported sources today: Nsight Systems (``.nsys-rep`` / exported ``.sqlite``)
and PyTorch profiler chrome-trace JSON (``.json`` / ``.json.gz``).
"""

from __future__ import annotations

from gitm.importers.analyze import analyze_paths
from gitm.importers.detect import DetectedFormat, detect_format

__all__ = ["DetectedFormat", "analyze_paths", "detect_format"]
