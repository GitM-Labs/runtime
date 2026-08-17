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


def require_timing_partition(
    total_s: float, components_s: dict[str, float], *, context: str
) -> dict[str, float]:
    """Return component fractions plus ``unattributed`` or refuse overlap.

    Benchmark stall breakdowns are sign-off evidence. Repairing a zero total or
    independently clamping overlapping phase timers would turn broken evidence
    into a plausible partition, so validate the complete partition in one place.
    """
    total = require_positive_duration(total_s, context=context)
    invalid = {
        name: value
        for name, value in components_s.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid:
        raise RuntimeError(f"{context} timing unavailable: invalid components {invalid}")
    assigned = math.fsum(components_s.values())
    tolerance = max(1e-12, total * 1e-9)
    if assigned > total + tolerance:
        detail = ", ".join(f"{name}={value:.6g}s" for name, value in components_s.items())
        raise RuntimeError(
            f"{context} timing attribution overlaps: {detail}, total={total:.6g}s; "
            "refusing to clamp"
        )
    fractions = {name: value / total for name, value in components_s.items()}
    fractions["unattributed"] = max(0.0, total - assigned) / total
    return fractions
