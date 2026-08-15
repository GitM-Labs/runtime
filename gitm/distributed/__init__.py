"""Multi-process capture support: rank topology and rank-scoped correlation.

A capture of a tensor- or expert-parallel deployment comprises one activity
shard per rank. This package supplies the two facilities a single-process tracer
does not require:

:mod:`gitm.distributed.topology`
    Association of process identifier, CUDA device ordinal and rank ordinal,
    derived from the records themselves rather than from environment ordering.

:mod:`gitm.distributed.correlate`
    Correlation of activity records to NVTX ranges, performed within each
    process, since the ``correlation_id`` on which correlation depends is
    assigned per process and is not unique across a merged capture.

Dependency direction
--------------------
This package imports nothing from other GITM subpackages. Shard discovery, the
shard-filename convention and the trace-output environment remain owned by
:mod:`gitm.tracer.injection`, which passes shard locations inward; device
enumeration is performed here against NVML and ``CUDA_VISIBLE_DEVICES`` directly.
Callers in :mod:`gitm.tracer` import from this package, not the reverse.
"""

from gitm.distributed.correlate import (
    RANK_KEYS,
    correlate_by_rank,
    correlate_kernels_to_ranges,
    correlation_id_collisions,
    parse_range_name,
)
from gitm.distributed.topology import (
    Rank,
    Topology,
    group_records_by_pid,
    topology_from_records,
    topology_from_shards,
    visible_gpu_count,
)

__all__ = [
    "RANK_KEYS",
    "Rank",
    "Topology",
    "correlate_by_rank",
    "correlate_kernels_to_ranges",
    "correlation_id_collisions",
    "group_records_by_pid",
    "parse_range_name",
    "topology_from_records",
    "topology_from_shards",
    "visible_gpu_count",
]
