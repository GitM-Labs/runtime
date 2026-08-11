"""Shared timing trust predicates."""

from __future__ import annotations

import math


def require_positive_duration(duration_s: float, *, context: str) -> float:
    """Return a usable duration or refuse to fabricate a throughput denominator."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise RuntimeError(
            f"{context} timing unavailable: expected a finite positive duration, "
            f"got {duration_s!r}"
        )
    return duration_s


def require_positive_work(value: int | float, *, context: str) -> int | float:
    """Refuse throughput or speedup claims over an empty work unit."""
    if not math.isfinite(float(value)) or value <= 0:
        raise RuntimeError(f"{context} work coverage unavailable: expected > 0, got {value!r}")
    return value
