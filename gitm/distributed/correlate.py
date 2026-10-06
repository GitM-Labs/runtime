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
onto replayed kernels. The CUPTI collector does not emit these yet (it needs
CUPTI's graph-node resource callbacks), so there a graph kernel's op is ``None``
and falls back to name classification, rather than silently taking the step's.

Stamped ranges (ROCm)
---------------------
The ROCm collector (``rocm_inject.c``) does better than containment for eager
launches. rocprofiler-sdk's external-correlation request service asks the tool,
synchronously on the enqueuing thread, for a value to stamp on every dispatch,
copy and HIP API record; the tool answers with the id of the innermost rocTX
range open on that thread. Those records arrive with ``range_id``, and the
paired marker carries the same ``marker_id``::

    kernel   range_id=R                     (stamped at enqueue, launching thread)
        |  same id
    marker   {kind:"marker", marker_id:R, name, start_ns, end_ns, thread_id}

No clock comparison and no thread matching is involved, so neither the async
end-time hazard nor a launch issued from a helper thread can misplace it. When
both joins are available they are compared, and a disagreement is counted in
the :class:`CorrelationReport` rather than resolved silently; the stamp wins,
because it is the one taken at the moment of the launch. A stamp whose marker
is not in the capture (pushed before the window armed) falls back to
containment, which usually fails for the same reason and leaves ``None``.

HIP graphs (ROCm)
-----------------
ROCm 7.2 has no graph-node id on a dispatch record. The collector builds one
from the documented recipe (rocprofiler-sdk ``callback_tracing.h``, HIP graph
domain): it tracks ``hipGraphLaunch`` ENTER/EXIT per thread and stamps each
dispatch or copy enqueued inside it with ``(exec, ordinal)``. The capture side
records, for every launch made while a thread is stream-capturing, a
``graph_node`` carrying the range open at that moment, its position in the
capture, and a signature; ``hipStreamEndCapture`` + ``hipGraphInstantiate*``
link the capture to the executable graph::

    kernel      graph_id=E, graph_node_id=node_id(E, k)
        |  graph_exec {kind:"graph_exec", graph_id:E, capture_id:C, n_nodes:N}
    graph_node  {graph_node_id: capture_node_id(C, k), name, node_kind,
                 kernel_id, grid, block}

Position is only an identity if replay order equals capture order, which holds
for a single-stream capture and is not promised for a forked one. So the join
is *validated*, per replay (kernels sharing one launch ``correlation_id``):
ordinals must be distinct and within ``n_nodes``, and each kernel must match
its node's signature — the same symbol (``kernel_id``) and launch geometry for
a kernel node; a ROCclr blit (``__amd_rocclr_*``) or a copy record for a
memcpy/memset node, since ROCm executes those as blit kernels or SDMA copies.
One mismatch refuses the whole replay — an ordinal shift misplaces everything
after it — and the refusal is counted, never guessed past.

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
from dataclasses import dataclass, field

from gitm.distributed.topology import Rank, Topology, topology_from_records

_RANGE_NAME_RE = re.compile(r"^L(\d+)/(.+)$")

#: Keys added to every record returned by :func:`correlate_by_rank`.
RANK_KEYS = ("pid", "local_rank")

# ── ROCm graph-node id layout ───────────────────────────────────────────────
#
# Mirrors GITM_NODE_* in rocm_inject.c. Ids are nonzero by construction: 0 means
# "not a graph launch" throughout this codebase (CUPTI's convention, and
# ``_opt_graph_id`` decodes it to None), so the zero-based replay ordinal is
# stored as ordinal + 1. The capture flag keeps capture-side and replay-side ids
# disjoint, so one ``graph_nodes`` map can hold both.
NODE_ORDINAL_BITS = 32
NODE_SEQ_BITS = 31
NODE_CAPTURE_FLAG = 1 << 63
_ORD_MASK = (1 << NODE_ORDINAL_BITS) - 1
_SEQ_MASK = (1 << NODE_SEQ_BITS) - 1

#: ``graph_id`` for a replayed kernel whose executable the collector never saw
#: instantiated (or whose stamp did not fire): still a replay, so never given
#: its launch's range as an op, but no node can be named. Mirrors
#: GITM_EXEC_UNTRACKED in rocm_inject.c.
GRAPH_UNTRACKED = _SEQ_MASK

#: Prefix of ROCclr's blit kernels (``__amd_rocclr_copyBuffer``,
#: ``__amd_rocclr_fillBufferAligned``, ...): how ROCm executes a captured
#: memcpy/memset node when it does not go to an SDMA engine.
ROCCLR_BLIT_PREFIX = "__amd_rocclr_"


def exec_node_id(exec_seq: int, ordinal: int) -> int:
    """Replay-side node id for zero-based ``ordinal`` of executable ``exec_seq``."""
    return ((exec_seq & _SEQ_MASK) << NODE_ORDINAL_BITS) | ((ordinal + 1) & _ORD_MASK)


def capture_node_id(capture_seq: int, ordinal: int) -> int:
    """Capture-side node id for zero-based ``ordinal`` of capture ``capture_seq``."""
    return NODE_CAPTURE_FLAG | exec_node_id(capture_seq, ordinal)


def node_ordinal(node_id: int) -> int:
    """Zero-based ordinal of either kind of node id."""
    return (node_id & _ORD_MASK) - 1


def split_range_annotations(name: str) -> tuple[str, dict[str, str] | None]:
    """``"L3/moe_routed#phase=expert,wave=2"`` -> ``("L3/moe_routed", {...})``.

    The attribute channel for ranges. NVTX has a typed payload for this; rocTX
    has none (``roctxRangePushA`` takes one string), so the vendor-neutral
    carrier is a suffix on the name — and it must come off *before*
    :func:`parse_range_name` and op resolution see the name, or
    ``moe_routed#wave=2`` becomes an op no rule knows. Everything from the first
    ``#`` is the annotation; pairs that do not parse are dropped, and an
    annotation with no valid pair yields ``None``. vLLM's dict-repr range names
    (``{...}``) are never split: their contents are not ours.
    """
    if not name or name.startswith("{") or "#" not in name:
        return name, None
    base, _, tail = name.partition("#")
    attrs: dict[str, str] = {}
    for part in tail.split(","):
        k, eq, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if eq and k and v and _ATTR_TOKEN.fullmatch(k) and _ATTR_TOKEN.fullmatch(v):
            attrs[k] = v
    return base, attrs or None


_ATTR_TOKEN = re.compile(r"[A-Za-z0-9_.:+-]+")


#: Share of graph kernels from unnamed nodes above which the capture, not the
#: model, is the likelier explanation (CorrelationReport.graph_unnamed).
UNNAMED_SHARE = 0.5


@dataclass
class CorrelationReport:
    """How each kernel got (or did not get) its identity, for one capture.

    The point of counting is that every failure mode here is otherwise silent:
    a refused replay, a stamp whose range never reached the capture, and a stamp
    that disagrees with containment all produce well-formed records.
    """

    kernels: int = 0
    #: kernels by the mechanism that named them: "range_id" | "containment" |
    #: "graph_node" | "none".
    identity: Counter = field(default_factory=Counter)
    graph_kernels: int = 0
    #: graph kernels refused, by reason.
    graph_refused: Counter = field(default_factory=Counter)
    #: graph kernels whose node exists but was captured outside every range.
    #: A few is normal (collectives, blits); most of them means the range
    #: instrumentation did not run during capture — torch.compile can trace
    #: forward hooks away — and every replay is anonymous without any refusal.
    graph_unnamed: int = 0
    #: eager kernels where stamp and containment both resolved and disagreed.
    stamp_containment_disagree: int = 0
    #: kernels whose stamp named a range absent from the capture.
    stamp_unresolved: int = 0
    memcpys: int = 0
    memcpys_labelled: int = 0

    def merge(self, other: CorrelationReport) -> None:
        for name in ("kernels", "graph_kernels", "graph_unnamed", "stamp_containment_disagree",
                     "stamp_unresolved", "memcpys", "memcpys_labelled"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.identity.update(other.identity)
        self.graph_refused.update(other.graph_refused)

    def problems(self) -> list[str]:
        """Findings worth a warning: each one means identity was lost or doubted."""
        out = []
        if self.graph_refused:
            n = sum(self.graph_refused.values())
            out.append(f"{n} of {self.graph_kernels} graph-replayed kernel(s) refused "
                       f"node identity ({dict(self.graph_refused)})")
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
    """Kernels and memcpys of a **single process**, enriched, plus a report.

    Kernels come back as in :func:`correlate_kernels_to_ranges`, additionally
    carrying ``range_attrs`` (the range's annotations) and ``identity`` (which
    mechanism named it — see :class:`CorrelationReport`). Memcpys come back with
    ``launch_range``: the range around the copy's issue, a step-boundary label
    that survives graph replay because vLLM's per-step host<->device copies run
    outside the graph. Input order is preserved within each list.
    """
    runtime_by_corr: dict[int, dict] = {}
    markers: list[dict] = []
    markers_by_id: dict[int, dict] = {}
    kernels: list[dict] = []
    memcpys: list[dict] = []
    graph_nodes: dict[int, dict] = {}
    graph_execs: list[dict] = []

    for r in records:
        kind = r.get("kind")
        if kind == "kernel":
            kernels.append(r)
        elif kind == "memcpy":
            memcpys.append(r)
        elif kind == "runtime":
            cid = r.get("correlation_id")
            if cid is not None:
                runtime_by_corr[cid] = r
        elif kind == "marker":
            markers.append(r)
            mid = r.get("marker_id")
            if mid:
                markers_by_id[mid] = r
        elif kind == "graph_node":
            nid = r.get("graph_node_id")
            if nid is not None:
                graph_nodes[nid] = r
        elif kind == "graph_exec":
            graph_execs.append(r)

    # Replay-side node ids resolve to their capture node through the exec's
    # capture link — the ROCm form of an instantiated clone.
    n_nodes: dict[int, int] = {}
    for ex in graph_execs:
        gid, cap, n = ex.get("graph_id"), ex.get("capture_id"), ex.get("n_nodes")
        if not gid or not isinstance(n, int):
            continue
        n_nodes[gid] = n
        if cap:
            for i in range(n):
                graph_nodes.setdefault(exec_node_id(gid, i), {
                    "graph_node_id": exec_node_id(gid, i),
                    "cloned_from": capture_node_id(cap, i)})

    enclosing = _innermost_enclosing(markers, runtime_by_corr.values())
    report = CorrelationReport()

    def stamped(rec: dict) -> dict | None:
        rid = rec.get("range_id")
        if not rid:
            return None
        m = markers_by_id.get(rid)
        if m is None:
            report.stamp_unresolved += 1
        return m

    def contained(rec: dict) -> dict | None:
        rt = runtime_by_corr.get(rec.get("correlation_id"))
        return enclosing.get(id(rt)) if rt is not None else None

    refused = _refused_replays(kernels, memcpys, graph_nodes, n_nodes)

    out: list[dict] = []
    for k in kernels:
        enriched = dict(k)
        if not k.get("graph_id"):
            rt = runtime_by_corr.get(k.get("correlation_id"))
            if rt is not None and rt.get("graph_launch"):
                # Launched by a graph replay the stamp did not mark: the
                # replay's range is not this kernel's op (module docstring).
                enriched["graph_id"] = GRAPH_UNTRACKED
                enriched["graph_node_id"] = None
                k = enriched
        enriched["range_op"] = None
        enriched["range_layer"] = None
        enriched["range_attrs"] = None
        enriched["identity"] = None
        report.kernels += 1

        if k.get("graph_id"):
            report.graph_kernels += 1
            rt = runtime_by_corr.get(k.get("correlation_id"))
            launch = (stamped(rt) if rt is not None else None) or contained(k)
            enriched["launch_range"] = launch["name"] if launch else None
            reason = ("untracked_launch" if k["graph_id"] == GRAPH_UNTRACKED else
                      refused.get((k.get("graph_id"), k.get("correlation_id"))))
            raw = None
            if reason is not None:
                report.graph_refused[reason] += 1
            else:
                raw = _graph_node_range(graph_nodes, k.get("graph_node_id"))
                if raw is None:
                    if _capture_node(graph_nodes, k.get("graph_node_id")) is None:
                        report.graph_refused["no_node"] += 1
                    else:
                        # The node exists but was captured outside every range:
                        # not a refusal, but counted (see graph_unnamed).
                        report.graph_unnamed += 1
            # Node names arrive normalized from the decoder (pair_markers), so
            # only the annotation needs splitting here.
            name, attrs = split_range_annotations(raw) if raw else (None, None)
            source = "graph_node" if name else None
        else:
            by_stamp, by_host = stamped(k), contained(k)
            if by_stamp is not None and by_host is not None and by_stamp is not by_host \
                    and by_stamp["name"] != by_host["name"]:
                report.stamp_containment_disagree += 1
            chosen = by_stamp or by_host
            name = chosen["name"] if chosen else None
            attrs = chosen.get("attrs") if chosen else None
            source = ("range_id" if by_stamp is not None else
                      "containment" if by_host is not None else None)

        if name:
            enriched["range_op"], enriched["range_layer"] = parse_range_name(name)
            enriched["range_attrs"] = attrs
            enriched["identity"] = source
        report.identity[enriched["identity"] or "none"] += 1
        out.append(enriched)

    out_copies: list[dict] = []
    for c in memcpys:
        enriched = dict(c)
        report.memcpys += 1
        m = stamped(c) or contained(c)
        enriched["launch_range"] = m["name"] if m else None
        report.memcpys_labelled += m is not None
        out_copies.append(enriched)

    return out, out_copies, report


def _signature_mismatch(rec: dict, node: dict) -> str | None:
    """Why ``rec`` cannot be the replay of capture ``node``, or None if it can.

    Only fields both sides carry are compared: the CUPTI collector's nodes carry
    none of them, and a check that fails on missing data would refuse every
    NVIDIA replay for want of a field it never had.
    """
    kind = node.get("node_kind")
    is_blit = rec.get("kind") == "memcpy" or str(rec.get("name") or "").startswith(
        ROCCLR_BLIT_PREFIX)
    if kind in ("memcpy", "memset"):
        return None if is_blit else "copy_node_ran_kernel"
    if kind == "kernel":
        if rec.get("kind") == "memcpy":
            return "kernel_node_ran_copy"
        nk, rk = node.get("kernel_id"), rec.get("kernel_id")
        if nk and rk and nk != rk:
            return "kernel_id"
        for dim in ("grid", "block"):
            a, b = node.get(dim), rec.get(dim)
            if a and b and list(a) != list(b):
                return dim
    return None


def _capture_node(graph_nodes: dict[int, dict], node_id: int | None) -> dict | None:
    """The node at the end of the clone chain (the one with the signature)."""
    seen: set[int] = set()
    node = None
    while node_id and node_id not in seen:
        seen.add(node_id)
        node = graph_nodes.get(node_id)
        if node is None or not node.get("cloned_from"):
            return node
        node_id = node["cloned_from"]
    return None


def _refused_replays(kernels, memcpys, graph_nodes, n_nodes) -> dict[tuple, str]:
    """``{(graph_id, correlation_id): reason}`` for replays whose node join fails.

    A replay is the set of graph records sharing one launch's correlation id.
    Checks, in order: a node id seen twice in one replay (the per-dispatch stamp
    did not advance — the hazard of an experimental service on an older
    runtime); an ordinal past the executable's node count; a record that does
    not match its node's signature.
    """
    replays: dict[tuple, list[dict]] = {}
    for r in (*kernels, *memcpys):
        if r.get("graph_id"):
            replays.setdefault((r["graph_id"], r.get("correlation_id")), []).append(r)
    out: dict[tuple, str] = {}
    for key, recs in replays.items():
        ids = [r.get("graph_node_id") for r in recs if r.get("graph_node_id")]
        if len(ids) != len(set(ids)):
            out[key] = "duplicate_ordinal"
            continue
        n = n_nodes.get(key[0])
        if n is not None and any(node_ordinal(i) >= n for i in ids):
            out[key] = "ordinal_out_of_range"
            continue
        for r in recs:
            node = _capture_node(graph_nodes, r.get("graph_node_id"))
            why = _signature_mismatch(r, node) if node is not None else None
            if why:
                out[key] = f"signature_{why}"
                break
    return out


def _graph_node_range(graph_nodes: dict[int, dict], node_id: int | None) -> str | None:
    """The range a graph node was captured under, following clones to the original.

    Instantiating or cloning a graph gives its nodes new ids, and which one a
    kernel record reports is not something to assume. Walking ``cloned_from``
    resolves either. A chain that cycles or leaves the map resolves to ``None``.
    """
    seen: set[int] = set()
    while node_id and node_id not in seen:
        seen.add(node_id)
        node = graph_nodes.get(node_id)
        if node is None:
            return None
        if node.get("name"):
            return node["name"]
        node_id = node.get("cloned_from")
    return None


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
