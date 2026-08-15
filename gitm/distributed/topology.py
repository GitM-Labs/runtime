"""Rank topology for multi-process CUPTI captures.

Under tensor or expert parallelism each rank executes in an independent process
with its own injected collector and its own activity shard. Trace records
therefore originate from several processes, and their interpretation requires an
explicit association between operating-system process identifier, CUDA device
ordinal, and rank ordinal.

The collector does not record this association. Its records contain no process
identifier, and the shard merge performed by the tracer concatenates all shards
into a single sequence, discarding the shard filename from which the process
identifier is derivable. Two downstream operations are incorrect in its absence:

1. CUPTI assigns correlation identifiers per process. Concatenation of shards
   admits identifier collisions between ranks executing equivalent work; see
   :mod:`gitm.distributed.correlate`.
2. Per-rank comparison requires a rank label on each record. Without one, the
   distribution of a metric across ranks cannot be recovered from a merged
   trace.

Device association is derived from the ``device_id`` values present in the
records a process emitted, rather than from ``CUDA_VISIBLE_DEVICES`` ordering.
The latter is not a reliable index of rank assignment, since launchers may
reorder ranks relative to device ordinals.

This package holds no dependency on :mod:`gitm.tracer` or any other GITM
subpackage. Shard locations and the shard-filename convention are owned by the
tracer, which supplies them to :func:`group_records_by_pid`; nothing here reads
the trace-output environment or parses a shard name.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rank:
    """A single collecting process and its associated CUDA device.

    Attributes
    ----------
    pid
        Operating-system process identifier of the collecting process.
    device_id
        CUDA device ordinal attributed to the process, or ``None`` when the
        process emitted no device activity within the capture window. The
        ``None`` case is distinct from device 0: a process may hold a CUDA
        context without executing kernels, and assigning it an ordinal would
        assert an association the records do not support.
    local_rank
        Ordinal assigned by :func:`topology_from_records`, stable across
        captures of the same deployment.
    n_records
        Count of collector records attributed to the process within the window.
    shard
        Path to the shard file, where the caller supplied one.
    """

    pid: int
    device_id: int | None
    local_rank: int
    n_records: int = 0
    shard: Path | None = None

    @property
    def collected_device_work(self) -> bool:
        """Whether the process emitted device activity attributable to a device."""
        return self.device_id is not None and self.n_records > 0


@dataclass(frozen=True)
class Topology:
    """The set of collecting ranks in one capture, and the visible device count."""

    ranks: tuple[Rank, ...] = ()
    visible_gpus: int = 0
    gpu_source: str = "unknown"

    @property
    def world_size(self) -> int:
        """Number of ranks that emitted device activity."""
        return sum(1 for r in self.ranks if r.collected_device_work)

    @property
    def devices(self) -> tuple[int, ...]:
        """Distinct device ordinals represented in the capture, ascending."""
        return tuple(sorted({r.device_id for r in self.ranks if r.device_id is not None}))

    def rank_for_pid(self, pid: int) -> Rank | None:
        """The rank record for ``pid``, or ``None`` if the process did not collect."""
        return next((r for r in self.ranks if r.pid == pid), None)

    @property
    def covers_all_visible_gpus(self) -> bool:
        """Whether every visible device produced records.

        A false result indicates a partial capture: the injection variable
        reached a proper subset of the worker processes. The resulting trace is
        internally consistent but describes only part of the job, and aggregate
        quantities computed from it understate the whole by an unknown factor.
        """
        return self.visible_gpus > 0 and len(self.devices) == self.visible_gpus

    def summary(self) -> str:
        """Human-readable description of the topology, one line per rank."""
        if not self.ranks:
            return "no collecting processes found"
        parts = [
            f"rank {r.local_rank}: pid {r.pid}, "
            + (f"device {r.device_id}" if r.device_id is not None else "no device work")
            + f", {r.n_records} records"
            for r in self.ranks
        ]
        head = (
            f"{self.world_size} collecting rank(s) across {len(self.devices)} device(s); "
            f"{self.visible_gpus} GPU(s) visible ({self.gpu_source})"
        )
        if not self.covers_all_visible_gpus and self.visible_gpus:
            head += "  [PARTIAL: one or more visible devices produced no records]"
        return "\n".join([head, *("  " + p for p in parts)])


def _cuda_visible_devices() -> tuple[int, str] | None:
    """Device count implied by ``CUDA_VISIBLE_DEVICES``, if it constrains the set.

    Returns ``None`` when the variable is unset, which delegates to driver
    enumeration. An empty value denotes zero visible devices and is distinct
    from unset. Entries may be ordinals or GPU UUIDs; only the count is derived
    here, since UUID form carries no ordinal to map onto.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return len(entries), "CUDA_VISIBLE_DEVICES"


def visible_gpu_count() -> tuple[int, str]:
    """Return ``(count, source)`` for the devices available to this process.

    ``CUDA_VISIBLE_DEVICES`` takes precedence where set, since it constrains the
    set below what the driver reports. Otherwise the count comes from NVML.

    Returns ``(0, "unavailable")`` when neither source can be read. Topology
    construction remains valid in that case, with
    :attr:`Topology.covers_all_visible_gpus` reporting ``False`` rather than
    asserting coverage it cannot establish.
    """
    masked = _cuda_visible_devices()
    if masked is not None:
        return masked

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount()), "nvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0, "unavailable"


def group_records_by_pid(
    shards: dict[int, Path | str],
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> dict[int, list[dict]]:
    """Read collector shards and group their records by originating process.

    Parameters
    ----------
    shards
        Mapping of process identifier to shard path. The caller owns shard
        discovery and the shard-filename convention; this function only reads
        what it is given.
    start_ns, end_ns
        Inclusive bounds in the collector's clock domain. Records outside the
        window are discarded.

    Returns
    -------
    dict[int, list[dict]]
        Undecoded records per process. Correlation must be performed per process
        on these raw records prior to any merge, so decoding is deliberately not
        performed here.

    Notes
    -----
    A malformed trailing line, produced when a process is terminated mid-write,
    is skipped. Losing the final record of a shard is preferable to failing the
    capture, which would discard every complete record alongside it.
    """
    out: dict[int, list[dict]] = {}
    for pid, path in shards.items():
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        bucket = out.setdefault(pid, [])
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            ts = rec.get("start_ns")
            if not isinstance(ts, int):
                continue
            if start_ns is not None and ts < start_ns:
                continue
            if end_ns is not None and ts > end_ns:
                continue
            bucket.append(rec)
    return out


def topology_from_records(
    by_pid: dict[int, list[dict]], *, shards: dict[int, Path] | None = None
) -> Topology:
    """Construct a :class:`Topology` from records grouped by process.

    Device attribution uses the modal ``device_id`` across a process's records
    rather than the first observed value. A worker process emits incidental
    activity on peer devices through peer-to-peer copies and collective
    operations, and a minority of such records must not determine the rank's
    device assignment.

    Rank ordinals are assigned by device ordinal and then by process identifier.
    This ordering is stable across captures of the same deployment, which is a
    precondition for comparing a given rank between traces. Processes with no
    device attribution are ordered last, so that they do not occupy ordinals
    that would otherwise be held by ranks carrying device work.
    """
    entries: list[tuple[int | None, int, int]] = []
    for pid, recs in by_pid.items():
        counts: dict[int, int] = {}
        for r in recs:
            dev = r.get("device_id")
            if isinstance(dev, int):
                counts[dev] = counts.get(dev, 0) + 1
        if counts:
            dev = max(counts, key=lambda d: (counts[d], -d))
            entries.append((dev, pid, len(recs)))
        else:
            entries.append((None, pid, len(recs)))

    entries.sort(key=lambda e: (e[0] is None, e[0] if e[0] is not None else 0, e[1]))

    n, src = visible_gpu_count()
    return Topology(
        ranks=tuple(
            Rank(
                pid=pid,
                device_id=dev,
                local_rank=i,
                n_records=n_rec,
                shard=(shards or {}).get(pid),
            )
            for i, (dev, pid, n_rec) in enumerate(entries)
        ),
        visible_gpus=n,
        gpu_source=src,
    )


def topology_from_shards(
    shards: dict[int, Path | str],
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[dict[int, list[dict]], Topology]:
    """Read shards and construct the topology in one pass.

    Returns the grouped records alongside the topology, since callers correlating
    the records need both and re-reading the shards to obtain the second would
    admit a window in which they disagree.
    """
    by_pid = group_records_by_pid(shards, start_ns, end_ns)
    paths = {pid: Path(p) for pid, p in shards.items()}
    return by_pid, topology_from_records(by_pid, shards=paths)
