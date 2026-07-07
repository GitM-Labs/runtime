"""The 3 invariants the deviation monitor checks.

See ``docs/invariants.md`` for the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InvariantId = Literal["kernel_time", "memory_traffic", "stream_concurrency"]


@dataclass(frozen=True)
class Invariant:
    id: InvariantId
    band_width: float  # used by severity normalization
    tier: int  # 1 always, 2 conditional, 3 aspirational


INVARIANTS: tuple[Invariant, ...] = (
    Invariant("kernel_time", band_width=0.4, tier=1),
    Invariant("memory_traffic", band_width=0.4, tier=1),
    Invariant("stream_concurrency", band_width=1.0, tier=2),
)


@dataclass
class Violation:
    invariant: InvariantId
    node_op: str  # which predicted op deviated
    layer: int | None
    residual: float
    severity: float  # normalized in [0, 1]
    detail: str = ""

"""
The three invariants live in different units:

kernel-time residual: dimensionless fraction of time (+0.3 = 30% slower than predicted)
memory-traffic residual: dimensionless fraction of bytes (+0.3 = 30% more traffic than predicted)
stream-concurrency residual: fraction of kernels that serialized when they shouldn't have

"""
