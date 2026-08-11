"""Auto-revert: yank an applied change when it regresses vs baseline.

The detect half of day-one's detect-revert-page. After a change is applied,
throughput (or any higher-is-better metric) is sampled over a short window; if
the windowed mean drops below baseline by more than ``tolerance``, ``observe``
signals a revert. A full window is required before any decision so a single
noisy sample can't trip it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AutoRevertDecision:
    should_revert: bool
    reason: str
    windowed_mean: float | None = None
    relative_delta: float | None = None  # (mean - baseline) / baseline


class AutoRevert:
    def __init__(self, baseline: float, *, tolerance: float = 0.0, window: int = 5) -> None:
        if (
            isinstance(baseline, bool)
            or not isinstance(baseline, int | float)
            or not math.isfinite(float(baseline))
            or baseline <= 0
        ):
            raise ValueError(f"baseline must be finite and > 0, got {baseline!r}")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, int | float)
            or not math.isfinite(float(tolerance))
            or tolerance < 0
        ):
            raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.baseline = float(baseline)
        self.tolerance = float(tolerance)  # allowed fractional drop before reverting
        self._w: deque[float] = deque(maxlen=window)

    def observe(self, value: float) -> AutoRevertDecision:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"observation must be a finite number, got {value!r}")
        self._w.append(float(value))
        if len(self._w) < self._w.maxlen:
            return AutoRevertDecision(False, "warming up (need a full window)")
        mean = sum(self._w) / len(self._w)
        rel = (mean - self.baseline) / self.baseline
        if rel < -self.tolerance:
            return AutoRevertDecision(
                True, f"regression {rel:+.1%} beyond tolerance -{self.tolerance:.1%}",
                windowed_mean=mean, relative_delta=rel,
            )
        return AutoRevertDecision(
            False, f"within tolerance ({rel:+.1%})", windowed_mean=mean, relative_delta=rel,
        )
