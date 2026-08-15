"""Correlation of CUPTI activity records to NVTX ranges, scoped by process.

Correlation associates a device-side kernel record with the host-side launch
that issued it, and thence with the innermost enclosing NVTX range, yielding the
operation and layer identity that name-based classification cannot recover. See
``docs/kernel_identity.md``.

The chain spans three record kinds on two clock domains. A kernel's own
timestamps are device-clock and must never be compared against a range's
host-clock window, since asynchronous execution routinely places a kernel's
completion long after its enclosing range has been popped::

    kernel   (device clock)  correlation_id=X             [start_ns, end_ns]
        |  same correlation_id
    runtime  (host clock)    cudaLaunchKernel  id=X       [start_ns, end_ns], thread_id
        |  host-timestamp containment, same thread_id
    marker   (host clock)    NVTX range "L{layer}/{op}"   [start_ns, end_ns], thread_id

Record contract, as emitted by the collector::

    kernel   {kind:"kernel",  correlation_id:int, start_ns:int, end_ns:int, ...}
    runtime  {kind:"runtime", correlation_id:int, start_ns:int, end_ns:int,
              thread_id:int}
    marker   {kind:"marker",  name:str, start_ns:int, end_ns:int, thread_id:int}

A ``marker`` record is one fully-resolved push/pop range, with start and end
already paired.

Process scoping
---------------
``correlation_id`` is assigned by CUPTI per process, numbered from a low origin
in each. Under tensor or expert parallelism the identifier is therefore not
unique across a capture: ranks executing equivalent work issue equivalent launch
sequences and allocate overlapping identifier ranges. Correlating a merged
record sequence resolves each identifier to whichever record was indexed last,
so range attribution is drawn from an arbitrary rank. ``thread_id``, applied as
a secondary constraint, is likewise process-scoped and provides no cross-process
discrimination.

The resulting error is silent: every kernel receives a syntactically valid
``range_op`` and ``range_layer``, wrong only in which rank supplied it.
:func:`correlate_by_rank` removes it by partitioning records by originating
process before correlation and merging only afterwards, confining identifier
resolution to the scope in which identifiers are unique.
"""

from __future__ import annotations

import re

from gitm.distributed.topology import Rank, Topology, topology_from_records

_RANGE_NAME_RE = re.compile(r"^L(\d+)/(.+)$")

#: Keys added to every record returned by :func:`correlate_by_rank`.
RANK_KEYS = ("pid", "local_rank")


def parse_range_name(name: str) -> tuple[str, int | None]:
    """Parse an NVTX range name into ``(op, layer)``.

    ``"L3/qkv_proj"`` yields ``("qkv_proj", 3)``. A range carrying no layer
    prefix, such as ``"lm_head"`` which executes once rather than per layer,
    yields ``(name, None)``.
    """
    m = _RANGE_NAME_RE.match(name)
    if m:
        return m.group(2), int(m.group(1))
    return name, None


def correlate_kernels_to_ranges(records: list[dict]) -> list[dict]:
    """Correlate the records of a **single process**.

    Returns every ``kind == "kernel"`` record as a shallow copy carrying
    ``range_op`` and ``range_layer``. Both are ``None`` where no match exists:
    no runtime record bears the kernel's ``correlation_id``, or no marker range
    contains that runtime record. Input order is preserved. Runtime and marker
    records are consumed to build the correlation index and are not returned.

    Containment is evaluated on the runtime record's host window against the
    marker's host window, matched on ``thread_id`` — never on the kernel's own
    device-clock window, and never across threads. Where nested ranges both
    contain the runtime window, the innermost, being the one of smallest span,
    is selected.

    Callers holding records from more than one process must use
    :func:`correlate_by_rank`; this function assumes ``correlation_id`` is
    unique within ``records``, which holds only within a process.
    """
    runtime_by_corr: dict[int, dict] = {}
    markers: list[dict] = []
    kernels: list[dict] = []

    for r in records:
        kind = r.get("kind")
        if kind == "kernel":
            kernels.append(r)
        elif kind == "runtime":
            cid = r.get("correlation_id")
            if cid is not None:
                runtime_by_corr[cid] = r
        elif kind == "marker":
            markers.append(r)

    out: list[dict] = []
    for k in kernels:
        enriched = dict(k)
        enriched["range_op"] = None
        enriched["range_layer"] = None

        rt = runtime_by_corr.get(k.get("correlation_id"))
        if rt is not None:
            best: dict | None = None
            best_span: int | None = None
            for m in markers:
                if m.get("thread_id") != rt.get("thread_id"):
                    continue
                if m["start_ns"] <= rt["start_ns"] and rt["end_ns"] <= m["end_ns"]:
                    span = m["end_ns"] - m["start_ns"]
                    if best_span is None or span < best_span:
                        best, best_span = m, span
            if best is not None:
                op, layer = parse_range_name(best["name"])
                enriched["range_op"] = op
                enriched["range_layer"] = layer

        out.append(enriched)

    return out


def correlation_id_collisions(by_pid: dict[int, list[dict]]) -> dict[int, list[int]]:
    """Return correlation identifiers observed in more than one process.

    Maps each colliding identifier to the sorted process identifiers that
    emitted it. An empty result indicates that merged correlation would have
    produced the same attribution as partitioned correlation. A non-empty result
    quantifies the cross-rank contamination that partitioning avoids.
    """
    owners: dict[int, set[int]] = {}
    for pid, recs in by_pid.items():
        for r in recs:
            cid = r.get("correlation_id")
            if isinstance(cid, int):
                owners.setdefault(cid, set()).add(pid)
    return {cid: sorted(pids) for cid, pids in owners.items() if len(pids) > 1}


def correlate_by_rank(
    by_pid: dict[int, list[dict]], topology: Topology | None = None
) -> list[dict]:
    """Correlate each process's records independently and return their union.

    Parameters
    ----------
    by_pid
        Raw collector records grouped by originating process.
    topology
        Rank assignment used to label records. Derived from ``by_pid`` when
        omitted. Processes absent from the supplied topology are labelled with
        ``local_rank == -1`` rather than dropped.

    Returns
    -------
    list[dict]
        Enriched kernel records ordered by ``start_ns``, each carrying
        ``range_op`` and ``range_layer`` from correlation together with ``pid``
        and ``local_rank`` identifying its origin.

    Notes
    -----
    Correlation is delegated per partition to
    :func:`correlate_kernels_to_ranges`. Single-process semantics are unchanged
    and remain defined in one place; this function supplies only the partition
    within which those semantics are valid.

    The returned ordering is global by ``start_ns``. Records from different
    processes share a clock domain only insofar as CUPTI timestamps are
    system-wide; where cross-process skew is material, analysis should group by
    ``local_rank`` before comparing timings.
    """
    topo = topology if topology is not None else topology_from_records(by_pid)
    out: list[dict] = []
    for pid, recs in by_pid.items():
        rank: Rank | None = topo.rank_for_pid(pid)
        local_rank = rank.local_rank if rank is not None else -1
        for enriched in correlate_kernels_to_ranges(recs):
            enriched["pid"] = pid
            enriched["local_rank"] = local_rank
            out.append(enriched)
    out.sort(key=lambda r: r.get("start_ns", 0))
    return out
