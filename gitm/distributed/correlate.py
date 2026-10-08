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

CUDA graphs
-----------
The chain above names the op only for an eagerly launched kernel. A graph
replay is one ``cudaGraphLaunch``, and every kernel it runs carries that one
launch's ``correlation_id``, so the range enclosing the launch is whatever
encloses the replay: a step, a layer group, or nothing. Taking it as the op
would hand every kernel in the replay the same, wrong identity. A kernel with a
nonzero ``graph_id`` therefore gets that range only as ``launch_range`` (a
time-series label), and ``range_op``/``range_layer`` come from the graph node
instead::

    kernel      graph_node_id=N
        |  same graph_node_id (following cloned_from to the captured node)
    graph_node  {kind:"graph_node", graph_node_id:int, name:str,
                 cloned_from:int | None}

A ``graph_node`` record names the innermost NVTX range open on the capturing
thread when the node was created, which is how Nsight Systems projects ranges
onto replayed kernels (``cupti_core.c``, NVTX + RESOURCE callbacks). Records
sharing an id are merged; along a clone chain the name nearest the captured
node wins. Without a node map a graph kernel's op is ``None`` and falls back to
name classification.

ROCm (docs/rocm_correlation.md)
-------------------------------
Eager kernels arrive with ``range_id``, the rocTX range open on the launching
thread at enqueue, and join the marker with that ``marker_id`` directly;
containment is the fallback and a cross-check. Replayed kernels arrive with
``graph_id``/``graph_node_id`` = ``(exec, ordinal)``; a ``graph_exec`` record
links the exec to the capture whose ``graph_node`` records carry the ranges.
Position is only an identity if the replay runs in capture order, so each
replay is validated (distinct ordinals, within ``n_nodes``, each record
matching its node's signature) and refused whole on any mismatch.

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
from collections import Counter
from dataclasses import dataclass, field, fields

from gitm.distributed.topology import Rank, Topology, topology_from_records

_RANGE_NAME_RE = re.compile(r"^L(\d+)/(.+)$")
_ATTR_TOKEN = re.compile(r"[A-Za-z0-9_.:+-]+")

#: Keys added to every record returned by :func:`correlate_by_rank`.
RANK_KEYS = ("pid", "local_rank")

# ROCm node-id layout, mirrored by GITM_NODE_* in rocm_inject.c. Ordinals are
# stored +1 so no id is 0 ("not a graph"); the flag keeps capture ids apart.
NODE_ORDINAL_BITS = 32
NODE_SEQ_BITS = 31
NODE_CAPTURE_FLAG = 1 << 63
_ORD_MASK = (1 << NODE_ORDINAL_BITS) - 1
_SEQ_MASK = (1 << NODE_SEQ_BITS) - 1
#: graph_id of a replay whose exec was never seen instantiated.
GRAPH_UNTRACKED = _SEQ_MASK
#: ROCm runs captured memcpy/memset nodes as these blit kernels (or SDMA copies).
ROCCLR_BLIT_PREFIX = "__amd_rocclr_"
#: Share of unnamed graph kernels at which the capture itself is suspect.
UNNAMED_SHARE = 0.5


def exec_node_id(exec_seq: int, ordinal: int) -> int:
    return ((exec_seq & _SEQ_MASK) << NODE_ORDINAL_BITS) | ((ordinal + 1) & _ORD_MASK)


def capture_node_id(capture_seq: int, ordinal: int) -> int:
    return NODE_CAPTURE_FLAG | exec_node_id(capture_seq, ordinal)


def node_ordinal(node_id: int) -> int:
    return (node_id & _ORD_MASK) - 1


def split_range_annotations(name: str) -> tuple[str, dict[str, str] | None]:
    """``"L3/moe_routed#wave=2"`` -> ``("L3/moe_routed", {"wave": "2"})``.

    rocTX has no payload, so attributes ride on the name and must come off
    before it is parsed. Malformed pairs are dropped; vLLM's ``{...}`` names
    are never split.
    """
    if not name or name.startswith("{") or "#" not in name:
        return name, None
    base, _, tail = name.partition("#")
    attrs = {}
    for part in tail.split(","):
        k, eq, v = (s.strip() for s in part.partition("="))
        if eq and _ATTR_TOKEN.fullmatch(k) and _ATTR_TOKEN.fullmatch(v):
            attrs[k] = v
    return base, attrs or None


@dataclass
class CorrelationReport:
    """How identity was recovered or lost; every failure here is otherwise silent."""

    kernels: int = 0
    #: kernels by mechanism: "range_id" | "containment" | "graph_node" | "none"
    identity: Counter = field(default_factory=Counter)
    graph_kernels: int = 0
    graph_refused: Counter = field(default_factory=Counter)
    #: replayed kernels whose node was captured outside every range
    graph_unnamed: int = 0
    stamp_containment_disagree: int = 0
    stamp_unresolved: int = 0
    memcpys: int = 0
    memcpys_labelled: int = 0

    def merge(self, other: CorrelationReport) -> None:
        for f in fields(self):
            mine = getattr(self, f.name)
            if isinstance(mine, Counter):
                mine.update(getattr(other, f.name))
            else:
                setattr(self, f.name, mine + getattr(other, f.name))

    def problems(self) -> list[str]:
        out = []
        if self.graph_refused:
            out.append(f"{sum(self.graph_refused.values())} of {self.graph_kernels} "
                       f"graph-replayed kernel(s) refused node identity "
                       f"({dict(self.graph_refused)})")
        if self.graph_kernels and self.graph_unnamed >= UNNAMED_SHARE * self.graph_kernels:
            out.append(f"{self.graph_unnamed} of {self.graph_kernels} graph-replayed kernel(s) "
                       "ran from nodes captured outside every range — were ranges pushed "
                       "during graph capture? (torch.compile can trace forward hooks away)")
        if self.stamp_containment_disagree:
            out.append(f"{self.stamp_containment_disagree} kernel(s): stamped range and "
                       "host-time containment disagree (stamp used)")
        return out


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
    contains that runtime record. Input order is preserved. Runtime, marker and
    graph_node records are consumed to build the correlation index and are not
    returned.

    A kernel with a nonzero ``graph_id`` also carries ``launch_range``, the raw
    name of the range around its graph launch, and takes ``range_op`` and
    ``range_layer`` from its graph node only (module docstring, "CUDA graphs").

    Containment is evaluated on the runtime record's host window against the
    marker's host window, matched on ``thread_id`` — never on the kernel's own
    device-clock window, and never across threads. Where nested ranges both
    contain the runtime window, the innermost, being the one of smallest span,
    is selected.

    Callers holding records from more than one process must use
    :func:`correlate_by_rank`; this function assumes ``correlation_id`` is
    unique within ``records``, which holds only within a process.
    """
    return correlate_records(records)[0]


def correlate_records(records: list[dict]) -> tuple[list[dict], list[dict], CorrelationReport]:
    """Kernels and memcpys of one process, enriched, plus a :class:`CorrelationReport`.

    Kernels additionally carry ``range_attrs`` and ``identity``; memcpys carry
    ``launch_range``, a step label that survives graph replay.
    """
    runtime_by_corr: dict[int, dict] = {}
    markers: list[dict] = []
    kernels: list[dict] = []
    memcpys: list[dict] = []
    graph_nodes: dict[int, dict] = {}
    n_nodes: dict[int, int] = {}
    for r in records:
        kind = r.get("kind")
        if kind == "kernel":
            kernels.append(r)
        elif kind == "memcpy":
            memcpys.append(r)
        elif kind == "runtime" and r.get("correlation_id") is not None:
            runtime_by_corr[r["correlation_id"]] = r
        elif kind == "marker":
            markers.append(r)
        elif kind == "graph_node" and r.get("graph_node_id") is not None:
            prior = graph_nodes.get(r["graph_node_id"], {})
            graph_nodes[r["graph_node_id"]] = {
                **prior, **{k: v for k, v in r.items() if v not in (None, "", 0)}}
        elif kind == "graph_exec" and r.get("graph_id") and isinstance(r.get("n_nodes"), int):
            gid, cap = r["graph_id"], r.get("capture_id")
            n_nodes[gid] = r["n_nodes"]
            # ROCm's form of an instantiated clone: exec node k -> capture node k.
            for i in range(r["n_nodes"] if cap else 0):
                graph_nodes.setdefault(exec_node_id(gid, i), {
                    "graph_node_id": exec_node_id(gid, i), "cloned_from": capture_node_id(cap, i)})

    markers_by_id = {m["marker_id"]: m for m in markers if m.get("marker_id")}
    enclosing = _innermost_enclosing(markers, runtime_by_corr.values())
    refused = _refused_replays(kernels, memcpys, graph_nodes, n_nodes)
    report = CorrelationReport()

    def stamped(rec: dict | None) -> dict | None:
        rid = rec.get("range_id") if rec else None
        if not rid:
            return None
        m = markers_by_id.get(rid)
        report.stamp_unresolved += m is None
        return m

    def contained(rec: dict) -> dict | None:
        rt = runtime_by_corr.get(rec.get("correlation_id"))
        return enclosing.get(id(rt)) if rt is not None else None

    out: list[dict] = []
    for k in kernels:
        k = dict(k)
        rt = runtime_by_corr.get(k.get("correlation_id"))
        if not k.get("graph_id") and rt is not None and rt.get("graph_launch"):
            # A replay the ordinal stamp missed: still never takes the launch range.
            k["graph_id"], k["graph_node_id"] = GRAPH_UNTRACKED, None
        k.update(range_op=None, range_layer=None, range_attrs=None, identity=None)
        report.kernels += 1

        if k.get("graph_id"):
            report.graph_kernels += 1
            launch = stamped(rt) or contained(k)
            k["launch_range"] = launch["name"] if launch else None
            reason = ("untracked_launch" if k["graph_id"] == GRAPH_UNTRACKED
                      else refused.get((k["graph_id"], k.get("correlation_id"))))
            named = None
            if reason:
                report.graph_refused[reason] += 1
            else:
                chain = _node_chain(graph_nodes, k.get("graph_node_id"))
                named = next((n for n in reversed(chain) if n.get("name")), None)
                if not chain:
                    report.graph_refused["no_node"] += 1
                elif named is None:
                    report.graph_unnamed += 1
            name = named["name"] if reason is None and named else None
            attrs = named.get("attrs") if name else None
            source = "graph_node"
        else:
            by_stamp, by_host = stamped(k), contained(k)
            if by_stamp and by_host and by_stamp["name"] != by_host["name"]:
                report.stamp_containment_disagree += 1
            chosen = by_stamp or by_host
            name = chosen["name"] if chosen else None
            attrs = chosen.get("attrs") if chosen else None
            source = "range_id" if by_stamp else "containment"

        if name:
            k["range_op"], k["range_layer"] = parse_range_name(name)
            k["range_attrs"], k["identity"] = attrs, source
        report.identity[k["identity"] or "none"] += 1
        out.append(k)

    copies = []
    for c in memcpys:
        m = stamped(c) or contained(c)
        copies.append({**c, "launch_range": m["name"] if m else None})
        report.memcpys += 1
        report.memcpys_labelled += m is not None
    return out, copies, report


def _node_chain(graph_nodes: dict[int, dict], node_id: int | None) -> list[dict]:
    """Nodes from ``node_id`` along ``cloned_from``; empty if it leaves the map or cycles."""
    chain: list[dict] = []
    seen: set[int] = set()
    while node_id:
        node = graph_nodes.get(node_id)
        if node is None or node_id in seen:
            return []
        seen.add(node_id)
        chain.append(node)
        node_id = node.get("cloned_from")
    return chain


def _signature_mismatch(rec: dict, node: dict) -> str | None:
    """Why ``rec`` can't be a replay of ``node``. Fields absent on either side
    (all of them, for CUPTI nodes) are not compared."""
    kind = node.get("node_kind")
    is_copy = rec.get("kind") == "memcpy"
    if kind in ("memcpy", "memset"):
        blit = is_copy or str(rec.get("name") or "").startswith(ROCCLR_BLIT_PREFIX)
        return None if blit else "copy_node_ran_kernel"
    if kind == "kernel":
        if is_copy:
            return "kernel_node_ran_copy"
        nk, rk = node.get("kernel_id"), rec.get("kernel_id")
        if nk and rk and nk != rk:
            return "kernel_id"
        for dim in ("grid", "block"):
            a, b = node.get(dim), rec.get(dim)
            if a and b and list(a) != list(b):
                return dim
    return None


def _refused_replays(kernels, memcpys, graph_nodes, n_nodes) -> dict[tuple, str]:
    """``{(graph_id, correlation_id): reason}`` for replays whose node join fails.

    One mismatch refuses the whole replay: an ordinal shift misplaces everything
    after it.
    """
    replays: dict[tuple, list[dict]] = {}
    for r in (*kernels, *memcpys):
        if r.get("graph_id"):
            replays.setdefault((r["graph_id"], r.get("correlation_id")), []).append(r)
    out: dict[tuple, str] = {}
    for key, recs in replays.items():
        ids = [r["graph_node_id"] for r in recs if r.get("graph_node_id")]
        if len(ids) != len(set(ids)):
            out[key] = "duplicate_ordinal"
        elif key[0] in n_nodes and any(node_ordinal(i) >= n_nodes[key[0]] for i in ids):
            out[key] = "ordinal_out_of_range"
        else:
            for r in recs:
                chain = _node_chain(graph_nodes, r.get("graph_node_id"))
                why = _signature_mismatch(r, chain[-1]) if chain else None
                if why:
                    out[key] = f"signature_{why}"
                    break
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
