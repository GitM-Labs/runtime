"""The bottleneck vocabulary, in one place.

    normalize_bound("launch") -> IDLE_STALL

Two vocabularies describe the same idea today and neither imports the other. The
roofline predicts a per-node ``bound`` of ``compute | memory | launch``
(:mod:`gitm.planner.roofline`), while the trace-level classifier and the headroom
split both speak ``idle_stall | memory_bound | compute_bound``
(:mod:`gitm.agents.autoresearch`, :mod:`gitm.optimizer.headroom`). The strings
were repeated at three call sites and had already drifted once — ``monitor``
documented ``bound`` as ``compute | memory`` when ``launch`` occurs in practice.

This module owns the class names so a fourth consumer cannot invent a fourth
spelling, and gives the one mapping between the two vocabularies.

It lives under ``gitm.optimizer`` rather than ``gitm.agents`` because the
dependency runs agents -> optimizer; ``autoresearch`` re-exports these names so
its own vocabulary stays a single definition rather than a copy.
"""

from __future__ import annotations

__all__ = [
    "IDLE_STALL",
    "MEMORY_BOUND",
    "COMPUTE_BOUND",
    "BOUND_CLASSES",
    "ROOFLINE_BOUNDS",
    "normalize_bound",
]

#: The GPU was not doing the op's work — scheduling gaps, launch overhead.
IDLE_STALL = "idle_stall"
#: Data movement is the limit.
MEMORY_BOUND = "memory_bound"
#: Arithmetic is the limit.
COMPUTE_BOUND = "compute_bound"

BOUND_CLASSES = (IDLE_STALL, MEMORY_BOUND, COMPUTE_BOUND)

#: What the roofline emits per node (:func:`gitm.planner.roofline` sets this from
#: whichever of t_compute / t_memory / t_launch dominates).
ROOFLINE_BOUNDS = ("compute", "memory", "launch")

#: ``launch -> idle_stall`` because both name time the GPU spent not doing the
#: op's work. The other two are the same concept under two spellings.
_TO_CLASS = {
    "compute": COMPUTE_BOUND,
    "memory": MEMORY_BOUND,
    "launch": IDLE_STALL,
}


def normalize_bound(roofline_bound: str | None) -> str | None:
    """A roofline ``bound`` as a :data:`BOUND_CLASSES` member.

    ``None`` in, ``None`` out — an op with no prediction has no bound, and
    guessing one would invent an attribution. An unrecognized string is also
    ``None`` rather than a default: a new roofline bound should show up as
    missing, not silently land in whichever class was chosen as the fallback.
    """
    if roofline_bound is None:
        return None
    return _TO_CLASS.get(roofline_bound)
