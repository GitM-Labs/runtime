"""Node-level rollup over per-device imported traces.

Computes device skew, collective communication share, and exposed (non-overlapped)
communication time. Pure timestamp math — no invented counters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gitm.optimizer.metrics import _merge_intervals
from gitm.tracer.schema import Trace

# Collective kernel name patterns (case-insensitive). One module-level table;
# each entry is a substring match against the demangled/exported kernel name.
# Comment: covers NCCL device kernels and common collective op labels in chrome
# traces. Unknown collective libraries (e.g. proprietary) will not match —
# zero hits on multi-device is reported as inconclusive, not zero communication.
_COMM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("nccl", "NCCL library prefix (any nccl* kernel)"),
    ("ncclkernel", "NCCL device kernel entry points"),
    ("allreduce", "AllReduce collective"),
    ("reducescatter", "ReduceScatter collective"),
    ("allgather", "AllGather collective"),
    ("broadcast", "Broadcast collective"),
    ("sendrecv", "point-to-point SendRecv"),
    # also match spaced / underscored chrome labels
    ("all_reduce", "AllReduce underscored label"),
    ("reduce_scatter", "ReduceScatter underscored label"),
    ("all_gather", "AllGather underscored label"),
    ("send_recv", "SendRecv underscored label"),
)

_COMM_RES = tuple(
    (re.compile(re.escape(pat), re.IGNORECASE), note) for pat, note in _COMM_PATTERNS
)
# Fast path: lowercased substrings (avoids 11 regexes × N kernels on huge traces).
_COMM_SUBSTR = tuple(pat.lower() for pat, _ in _COMM_PATTERNS)

# Skew above this (busy_fraction max − min) triggers the straggler sentence.
SKEW_THRESHOLD = 0.05


def is_comm_kernel(name: str) -> bool:
    """True if kernel ``name`` matches a collective pattern."""
    if not name:
        return False
    low = name.lower()
    return any(s in low for s in _COMM_SUBSTR)


@dataclass
class DeviceCommStats:
    device_id: int
    busy_ns: int
    comm_ns: int
    exposed_comm_ns: int  # comm that does not overlap non-comm kernels
    comm_share_of_busy: float  # comm_ns / busy_ns
    exposed_comm_share_of_wall: float  # exposed_comm_ns / duration_ns


@dataclass
class NodeRollup:
    n_devices: int
    device_busy: dict[int, float]  # device_id → busy_fraction
    device_wall_s: dict[int, float]
    device_ceiling: dict[int, float]
    skew: float  # max busy − min busy
    has_straggler: bool
    has_collective: bool
    comm_inconclusive: bool  # multi-device but zero comm kernels matched
    per_device_comm: list[DeviceCommStats] = field(default_factory=list)
    node_ceiling_distance: float = 0.0  # duration-weighted mean
    total_exposed_comm_share: float = 0.0  # mean exposed/wall across devices
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_devices": self.n_devices,
            "device_busy": {str(k): v for k, v in self.device_busy.items()},
            "device_wall_s": {str(k): v for k, v in self.device_wall_s.items()},
            "device_ceiling": {str(k): v for k, v in self.device_ceiling.items()},
            "skew": self.skew,
            "has_straggler": self.has_straggler,
            "has_collective": self.has_collective,
            "comm_inconclusive": self.comm_inconclusive,
            "node_ceiling_distance": self.node_ceiling_distance,
            "total_exposed_comm_share": self.total_exposed_comm_share,
            "per_device_comm": [
                {
                    "device_id": c.device_id,
                    "comm_share_of_busy": c.comm_share_of_busy,
                    "exposed_comm_share_of_wall": c.exposed_comm_share_of_wall,
                    "busy_ns": c.busy_ns,
                    "comm_ns": c.comm_ns,
                    "exposed_comm_ns": c.exposed_comm_ns,
                }
                for c in self.per_device_comm
            ],
            "caveats": list(self.caveats),
        }


def _interval_len(spans: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in _merge_intervals(spans))


def _subtract_overlap(
    base: list[tuple[int, int]], mask: list[tuple[int, int]]
) -> int:
    """Length of ``base`` intervals not covered by ``mask`` (exposed time).

    Two-pointer sweep over sorted merges — O(n + m), not O(n·m).
    """
    if not base:
        return 0
    merged_base = _merge_intervals(base)
    if not mask:
        return sum(e - s for s, e in merged_base)
    merged_mask = _merge_intervals(mask)
    exposed = 0
    j = 0
    m_len = len(merged_mask)
    for b0, b1 in merged_base:
        cursor = b0
        # Advance mask pointer to the first interval that can overlap [b0, b1).
        while j < m_len and merged_mask[j][1] <= cursor:
            j += 1
        k = j
        while k < m_len:
            m0, m1 = merged_mask[k]
            if m0 >= b1:
                break
            if m0 > cursor:
                exposed += min(m0, b1) - cursor
            cursor = max(cursor, m1)
            if cursor >= b1:
                break
            k += 1
        if cursor < b1:
            exposed += b1 - cursor
    return exposed


def device_comm_stats(trace: Trace) -> DeviceCommStats:
    """Comm share + exposed comm for a single-device trace."""
    kernels = [e for e in trace.events if getattr(e, "kind", None) == "kernel"]
    # Prefer the device_id on events; fall back to 0.
    dev = kernels[0].device_id if kernels else 0
    wall = max(trace.duration_ns, 1)
    comm = [(k.start_ns, k.end_ns) for k in kernels if is_comm_kernel(k.name)]
    non_comm = [(k.start_ns, k.end_ns) for k in kernels if not is_comm_kernel(k.name)]
    busy_ns = _interval_len([(k.start_ns, k.end_ns) for k in kernels])
    comm_ns = _interval_len(comm)
    exposed_ns = _subtract_overlap(comm, non_comm)
    return DeviceCommStats(
        device_id=dev,
        busy_ns=busy_ns,
        comm_ns=comm_ns,
        exposed_comm_ns=exposed_ns,
        comm_share_of_busy=(comm_ns / busy_ns) if busy_ns > 0 else 0.0,
        exposed_comm_share_of_wall=exposed_ns / wall,
    )


def build_node_rollup(
    per_device: list[tuple[Trace, float, float]],
    *,
    multi_device_file: bool,
) -> NodeRollup:
    """Roll up per-device (trace, busy_fraction, ceiling_distance) into a node summary.

    ``per_device`` entries must each be a single-device Trace. ``multi_device_file``
    is True when the source file contained more than one device (drives the
    inconclusive-comm message).
    """
    if not per_device:
        return NodeRollup(
            n_devices=0,
            device_busy={},
            device_wall_s={},
            device_ceiling={},
            skew=0.0,
            has_straggler=False,
            has_collective=False,
            comm_inconclusive=False,
        )

    device_busy: dict[int, float] = {}
    device_wall: dict[int, float] = {}
    device_ceiling: dict[int, float] = {}
    comm_stats: list[DeviceCommStats] = []
    any_comm = False
    weight_sum = 0.0
    weighted_ceiling = 0.0

    for trace, busy_frac, ceiling in per_device:
        kernels = [e for e in trace.events if getattr(e, "kind", None) == "kernel"]
        dev = kernels[0].device_id if kernels else 0
        # Prefer unique keying by actual device id.
        device_busy[dev] = busy_frac
        wall_s = trace.duration_ns / 1e9
        device_wall[dev] = wall_s
        device_ceiling[dev] = ceiling
        cs = device_comm_stats(trace)
        # Ensure device_id matches.
        cs.device_id = dev
        comm_stats.append(cs)
        if cs.comm_ns > 0:
            any_comm = True
        weight_sum += max(wall_s, 0.0)
        weighted_ceiling += ceiling * max(wall_s, 0.0)

    busies = list(device_busy.values())
    skew = (max(busies) - min(busies)) if busies else 0.0
    has_straggler = skew > SKEW_THRESHOLD
    n_devices = len(device_busy)
    multi = multi_device_file or n_devices > 1
    comm_inconclusive = multi and not any_comm
    node_ceiling = (weighted_ceiling / weight_sum) if weight_sum > 0 else 0.0
    mean_exposed = (
        sum(c.exposed_comm_share_of_wall for c in comm_stats) / len(comm_stats)
        if comm_stats
        else 0.0
    )

    caveats: list[str] = []
    if multi:
        caveats.append(
            "Cross-device dependency attribution requires captured telemetry; "
            "this report identifies skew and exposed communication, not their root cause."
        )
        caveats.append(
            "Communication classification is name-based; unknown collective "
            "libraries are not counted."
        )

    return NodeRollup(
        n_devices=n_devices,
        device_busy=device_busy,
        device_wall_s=device_wall,
        device_ceiling=device_ceiling,
        skew=round(skew, 6),
        has_straggler=has_straggler,
        has_collective=any_comm,
        comm_inconclusive=comm_inconclusive,
        per_device_comm=comm_stats,
        node_ceiling_distance=round(node_ceiling, 6),
        total_exposed_comm_share=round(mean_exposed, 6),
        caveats=caveats,
    )
