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

    enclosing = _innermost_enclosing(markers, runtime_by_corr.values())

    out: list[dict] = []
    for k in kernels:
        enriched = dict(k)
        enriched["range_op"] = None
        enriched["range_layer"] = None

        rt = runtime_by_corr.get(k.get("correlation_id"))
        if rt is not None:
            m = enclosing.get(id(rt))
            if m is not None:
                op, layer = parse_range_name(m["name"])
                enriched["range_op"] = op
                enriched["range_layer"] = layer

        out.append(enriched)

    return out


# Event phases for the sweep below. Ordering at equal timestamps is semantic, not
# cosmetic: a range that opens exactly at a launch's start does contain it, and a
# range that closes exactly at that instant does not.
_PHASE_OPEN, _PHASE_QUERY, _PHASE_CLOSE = 0, 1, 2


def _innermost_enclosing(markers, runtimes) -> dict[int, dict]:
    """``{id(runtime_record): innermost marker fully containing it}``.

    Replaces a nested scan of every marker per kernel. That scan is O(k·m), and
    the shapes are not small: a 4,096-step capture instrumented at
    ``L{layer}/{op}`` granularity emits on the order of 800k markers against 9.5M
    kernels, which is 7.8e12 comparisons — about a day of CPU. This is
    O((n+m) log(n+m)), ~2.4e8 operations on the same input.

    The algorithm is a per-thread sweep. NVTX ranges on a single thread are
    pushed and popped as a stack, so at any instant the open ranges *are* a
    stack, ordered outermost to innermost. Walking events in time order and
    maintaining that stack means the innermost range containing a launch is at
    or near its top.

    "Near", not "at": containment requires the marker to cover the launch's
    ``end`` as well as its ``start``, and an asynchronous launch can outlive the
    range that issued it. So the stack is walked down from the top until a range
    covers the whole window. Nesting depth is a handful, so this is effectively
    constant per query rather than a second linear scan.

    Threads are handled separately throughout. A launch on one thread must never
    be attributed to a range pushed on another; grouping first also keeps each
    stack a genuine stack, which interleaved threads would not be.
    """
    by_thread_markers: dict[object, list[dict]] = {}
    for m in markers:
        by_thread_markers.setdefault(m.get("thread_id"), []).append(m)

    by_thread_runtimes: dict[object, list[dict]] = {}
    for rt in runtimes:
        by_thread_runtimes.setdefault(rt.get("thread_id"), []).append(rt)

    out: dict[int, dict] = {}
    for thread, thread_markers in by_thread_markers.items():
        queries = by_thread_runtimes.get(thread)
        if not queries:
            continue

        # Third sort key breaks timestamp ties so the stack reflects nesting.
        # An op range opened by the same instrumentation call as its enclosing
        # layer range shares its start exactly; without this the outer can land
        # on top of the inner and every launch inside resolves to the layer
        # instead of the op. Opens are ordered by descending end (outermost
        # first, since it closes last); closes by descending start (innermost
        # first, since it opened last).
        events: list[tuple[int, int, int, dict]] = []
        for m in thread_markers:
            events.append((m["start_ns"], _PHASE_OPEN, -m["end_ns"], m))
            events.append((m["end_ns"], _PHASE_CLOSE, -m["start_ns"], m))
        for rt in queries:
            events.append((rt["start_ns"], _PHASE_QUERY, 0, rt))
        events.sort(key=lambda e: e[:3])

        stack: list[dict] = []
        for _t, phase, _tie, payload in events:
            if phase == _PHASE_OPEN:
                stack.append(payload)
            elif phase == _PHASE_CLOSE:
                # Pop by identity rather than blindly: a malformed capture with
                # crossed ranges would otherwise desynchronise the stack and
                # mis-attribute everything after it.
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] is payload:
                        del stack[i]
                        break
            else:
                end = payload["end_ns"]
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i]["end_ns"] >= end:
                        out[id(payload)] = stack[i]
                        break
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
