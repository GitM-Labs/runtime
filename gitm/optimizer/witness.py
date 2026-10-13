"""Witness checks: which named causes left their signal in a capture, and how much time.

For each entry of :mod:`gitm.optimizer.primitives` this answers ``fired``,
``not_fired`` or ``cannot_tell``, and where the data supports it the time associated
with the cause. Per ``(pid, device)`` partition the account closes::

    busy + row 11 + row 10 + G1 + remainder = window     (window = the kernel span)

Rows 22 and 25 sit inside busy and are not in that sum. Everything here is read from the trace: the server's
``/metrics`` counters (rows 2 to 5) are not read by this build. A time here is
*associated* with a cause. It is not the time removing the cause would recover.

A priced row carries an attribution range beside its time: the time only its own
records explain, to the time they could explain at all. Where a sync record, a
memcpy record and a launch-scale gap sit over the same time, the trace does not say
which one owns it, and the split used is a fixed precedence.

A row below its threshold is ``absent`` only when the trace header reports zero
dropped records and nothing else could hide the row's records (see
:mod:`gitm.optimizer.pricing`). A header with no count, as every imported trace has,
leaves such a row not identifiable.

The capture must come from a server started under gitm's injection library; nothing
here diagnoses an unmodified running server. Validated on synthetic data only.

    python -m gitm.optimizer.witness RUN_DIR [--all] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from gitm.optimizer import pricing
from gitm.optimizer import primitives as prim
from gitm.optimizer.deviation import classify_op
from gitm.optimizer.metrics import (
    _LAUNCH_LATENCY_NS,
    _merge_intervals,
    _overlap_sweep,
    _stall_ns_per_gap,
    _strict_json,
    _write_atomic,
)

__all__ = ["Header", "Report", "WitnessInputError", "account", "run"]

SCHEMA_VERSION = 1

_PRECONDITION = (
    "Capture precondition: the server was started under gitm's injection library; "
    "nothing here diagnoses an unmodified running server."
)
_VALIDATION = "validated on synthetic data only; never run on a real capture"
_REPORTED = ("10", "11", "22", "25", "24", "G1", "G2", "G3", "G4")
_COLLECTIVE_ROWS = {"tp_all_reduce": "22", "moe_all_to_all": "25"}
_REMAINDER_NAME = "unexplained idle (waiting on the host, or no work)"
#: The sources the trace schema allows. Anything else is not trusted to mean any of them.
_SOURCES = ("cupti", "rocprof", "nsys-import", "torch-import", "none")
#: Collectors that emit no synchronization records at all (ROCm has no such activity
#: kind; the PyTorch profiler importer builds none).
_NO_SYNC_SOURCES = ("rocprof", "torch-import")
_NO_SYNC_NEXT = "capture with the CUPTI collector, which records sync calls"
_NO_COPY_NEXT = "export the trace with its memcpy table, or capture with the CUPTI collector"
_ALL = {"pid": None, "device_id": None, "all": True}
_RUN = {"pid": None, "device_id": None}


class WitnessInputError(Exception):
    """The trace is missing or cannot be parsed. Data problems never raise this."""


@dataclass(frozen=True)
class Header:
    """The trace header fields the checks read."""

    source: str = "cupti"
    device_count: int = 1
    vendor: str = ""
    #: Fields a trace file's header did not state; their defaults are not evidence.
    missing: tuple[str, ...] = ()
    #: Records the collector reports lost; ``-1`` when some were lost and it could not
    #: say how many; ``None`` when it did not count.
    dropped_records: int | None = None


@dataclass
class Report:
    schema_version: int
    thresholds_version: str
    generated_at: str
    source: str
    validation: str
    inputs: dict[str, list[str]]
    partitions: list[dict[str, Any]]
    results: list[dict[str, Any]]
    baselines: list[dict[str, Any]]
    run_presence: dict[str, str]
    records_unplaced: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _strict_json(asdict(self))


#: What a malformed line raises: bad JSON, a missing or wrong-typed field, Infinity
#: where an integer belongs, or nesting deep enough to exhaust the decoder.
_MALFORMED = (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError)
#: Timestamps past this cannot be nanoseconds, and would overflow the float seconds.
_MAX_NS = 2**62
#: A header line, a printed name: bounded so a hostile file cannot cost memory.
_MAX_HEADER_BYTES = 1_000_000
_MAX_EVENT_BYTES = 1_000_000
#: More (pid, device) partitions than any capture has: each one costs a full set of rows.
_MAX_PARTITIONS = 1024
#: Distinct sync or memcpy labels a row splits its time by; the rest are folded together.
_MAX_LABELS = 32


class _Rec(NamedTuple):
    start_ns: int
    end_ns: int


class _Event(NamedTuple):
    kind: str
    start_ns: int
    end_ns: int
    stream_id: int
    device_id: int
    pid: int | None
    #: kernel name, sync kind, or "src->dst"
    label: str
    #: CUDA context, when the collector reports one; streams are numbered per context
    context_id: int | None = None


# ── reading ──────────────────────────────────────────────────────────────────


#: CUpti_ActivitySynchronizationType, for the split of row 10 by what the host called.
_SYNC_TYPE = {1: "event_synchronize", 2: "stream_wait_event", 3: "stream_synchronize",
              4: "context_synchronize"}


def _sync_label(sync_kind: Any, sync_type: Any) -> Any:
    """The collector's own sync type when it kept one, else the folded ``sync_kind``."""
    if not isinstance(sync_type, int) or isinstance(sync_type, bool):
        return sync_kind
    # A type this table does not know keeps its number, not a folded kind's name.
    return _SYNC_TYPE.get(sync_type, f"type_{sync_type}")


def _normalize(events: Iterable[Any]) -> Iterator[_Event]:
    """In-memory events (pydantic or the importers' slotted classes) as plain tuples.

    An object missing a field comes out with a timestamp of ``None``, which placement
    counts as an event it cannot use.
    """
    for e in events:
        if isinstance(e, _Event):
            yield e
            continue
        kind = getattr(e, "kind", None)
        if kind not in ("kernel", "sync", "memcpy"):
            continue
        try:
            if kind == "kernel":
                label = e.name
            elif kind == "sync":
                label = _sync_label(e.sync_kind, getattr(e, "sync_type", None))
            else:
                label = f"{e.src}->{e.dst}"
            # The importers' lightweight events have no pid or context attribute.
            yield _Event(kind, e.start_ns, e.end_ns, e.stream_id, e.device_id,
                         getattr(e, "pid", None), label, getattr(e, "context_id", None))
        except AttributeError:
            yield _Event(kind, None, None, 0, 0, None, "")  # type: ignore[arg-type]


def _read_trace(path: Path) -> tuple[Header, Iterator[_Event]]:
    """Header and a line-by-line event stream; no full ``Trace`` is built."""
    if not path.is_file():
        # A FIFO or a device would block or never end.
        raise WitnessInputError(f"cannot read {path}: not a regular file")
    try:
        # A stray non-UTF-8 byte becomes U+FFFD, so the bad line fails as JSON below.
        fh = path.open(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise WitnessInputError(f"cannot read {path}: {exc}") from exc
    try:
        line = fh.readline(_MAX_HEADER_BYTES)
        if len(line) == _MAX_HEADER_BYTES:
            raise ValueError("header line too long")
        raw = json.loads(line)["_header"]
        device_count = raw.get("device_count")
        if device_count in (None, ""):
            device_count = 1  # named in ``missing`` below
        elif not _is_id(device_count):
            raise TypeError("device_count is not an integer")
        dropped = raw.get("dropped_records")
        header = Header(
            source=str(raw.get("source", "cupti")),
            device_count=device_count,
            vendor=str(raw.get("vendor", "")),
            missing=tuple(k for k in ("vendor", "device_count", "source")
                          if raw.get(k) in (None, "")),
            # Anything that is not a count (or -1, lost and uncounted) is no count.
            dropped_records=dropped if _is_id(dropped) and dropped >= -1 else None,
        )
    except (*_MALFORMED, OSError) as exc:
        fh.close()
        raise WitnessInputError(f"{path} has no trace header on its first line") from exc
    if "source" in header.missing:
        # The source decides how sync records are placed; a default would be a guess.
        fh.close()
        raise WitnessInputError(f"{path} header does not state its source; expected one of "
                                + ", ".join(_SOURCES))
    if header.source not in _SOURCES:
        fh.close()
        raise WitnessInputError(
            f"{path} header names an unknown source {header.source[:40]!r}; expected one of "
            + ", ".join(_SOURCES))

    def events() -> Iterator[_Event]:
        with fh:
            lineno = 1
            while True:
                lineno += 1
                try:
                    line = fh.readline(_MAX_EVENT_BYTES)
                except OSError as exc:
                    raise WitnessInputError(f"cannot read {path}: {exc}") from exc
                if not line:
                    return
                if len(line) == _MAX_EVENT_BYTES and not line.endswith("\n"):
                    raise WitnessInputError(f"{path} line {lineno} is longer than "
                                            f"{_MAX_EVENT_BYTES} characters")
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    kind = e.get("kind", "kernel")
                    if kind == "kernel":
                        label = e["name"]
                    elif kind == "sync":
                        label = _sync_label(e["sync_kind"], e.get("sync_type"))
                    elif kind == "memcpy":
                        label = f"{e['src']}->{e['dst']}"
                    else:
                        continue
                    if not isinstance(label, str):
                        raise TypeError("label is not a string")
                    start, end = e["start_ns"], e["end_ns"]
                    if not (_is_ns(start) and _is_ns(end)):
                        raise TypeError("timestamp is not an integer nanosecond value")
                    # A default or a coerced value is not evidence of where a record ran.
                    stream, device = e["stream_id"], e["device_id"]
                    pid, context = e.get("pid"), e.get("context_id")
                    if not (_is_id(stream) and _is_id(device) and (pid is None or _is_id(pid))
                            and (context is None or _is_id(context))):
                        raise TypeError("stream, device, pid or context is not an integer")
                    event = _Event(kind, start, end, stream, device, pid, label, context)
                except _MALFORMED as exc:
                    raise WitnessInputError(
                        f"{path} line {lineno} is not a trace event (truncated file?)"
                    ) from exc
                yield event

    return header, events()


# ── rows ─────────────────────────────────────────────────────────────────────

_LIBRARY_REVIEW: dict[str, Any] | None = None


def _lever(name: str) -> dict[str, Any]:
    global _LIBRARY_REVIEW
    if _LIBRARY_REVIEW is None:
        try:
            from gitm.kernels.library import load_library

            _LIBRARY_REVIEW = {s.name: s.review for s in load_library()}
        except Exception:  # noqa: BLE001 - an unreadable library must not lose a fired row
            _LIBRARY_REVIEW = {}
    review = _LIBRARY_REVIEW.get(name, "not found in library")
    if review is not None and not isinstance(review, (str, int, float, bool)):
        review = str(review)
    return {"name": name, "review": review, "note": "applicability not checked"}


def _candidate_levers(p: prim.Primitive) -> dict[str, Any]:
    if p.is_group:
        return {
            "candidate_levers": [
                {"member": m, "member_name": prim.get(m).name,
                 "levers": [_lever(n) for n in prim.get(m).levers]}
                for m in p.members
            ],
            "levers_note": "the group cannot tell which member applies",
        }
    out: dict[str, Any] = {"candidate_levers": [_lever(n) for n in p.levers]}
    if not p.levers:
        out["levers_note"] = "no lever in the library; code or ops fix"
    return out


def _row(
    p: prim.Primitive,
    partition: dict[str, Any],
    state: str,
    *,
    observation: dict[str, Any] | None = None,
    seconds: float | None = None,
    share: float | None = None,
    price_reason: str | None = None,
    completeness: str | None = None,
    reason: str | None = None,
    what: str | None = None,
    why: str | None = None,
    next_step: str | None = None,
    assumptions: Iterable[str] = (),
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": p.id,
        "name": p.name,
        "kind": "group" if p.is_group else ("member" if p.group else "cause"),
        "form": p.form,
        "scope": p.scope,
        "partition": dict(partition),
        "state": state,
        "reason": reason,
        "what": what,
        "why": why,
        "next": next_step if next_step is not None else (p.remedy if reason else None),
        "observation": observation,
        "seconds": seconds,
        "share": share,
        "price_kind": "associated" if seconds is not None else None,
        "price_reason": price_reason,
        "completeness": completeness,
        "assumptions": list(assumptions),
        "fire_at": p.fire_at,
        "fire_unit": p.fire_unit,
        "thresholds_version": prim.THRESHOLDS_VERSION,
    }
    row.update(extra)
    if state == "fired":
        row.update(_candidate_levers(p))
    return row


def _cannot(p: prim.Primitive, partition: dict[str, Any], reason: str, what: str, why: str,
            next_step: str | None = None, **extra: Any) -> dict[str, Any]:
    return _row(p, partition, "cannot_tell", reason=reason, what=what, why=why,
                next_step=next_step, **extra)


_LOSSY_NEXT = "re-capture with a shorter flush period (GITM_TRACE_FLUSH_MS) until no record is dropped"
_UNCOUNTED_NEXT = "capture with a collector that writes dropped_records to the trace header"
_LOSS_NEXT = {"lossy": _LOSSY_NEXT, "unknown": _UNCOUNTED_NEXT}
#: The collectors that count what they drop. An importer cannot, whatever its header says.
_COUNTING_SOURCES = ("cupti", "rocprof")
#: What settles a row that rests on an assumption.
_ASSUMPTION_NEXT = {
    "other_pid_active_on_device": "trace one process per device",
    "no_sync_records_in_source": _NO_SYNC_NEXT,
    "no_memcpy_records_in_source": _NO_COPY_NEXT,
    "collective_classified_by_name_only": "NVTX op identity, or the op classifier checked "
                                          "against a real multi-device capture",
    "misrouted_kernel_not_checked": "NVTX op identity, to check for misrouted kernels",
}
_CONFOUNDED_NEXT = ("runtime API records, which say which host call each overlapping record "
                    "came from")


def _completeness(header: Header) -> tuple[str, str | None]:
    """Whether the capture holds every record: ``complete``, ``lossy`` or ``unknown``.

    With it, why not: a missing kernel record reads as gap time, a missing sync or
    memcpy record as idle.
    """
    dropped = header.dropped_records
    if header.source not in _COUNTING_SOURCES:
        return "unknown", f"a {str(header.source)[:40]} trace cannot count dropped records"
    if dropped is None or not _is_id(dropped) or dropped < -1:
        return "unknown", f"the {header.source} trace header carries no dropped-record count"
    if dropped == -1:
        return "lossy", "records were lost at capture, count unknown"
    if dropped:
        return "lossy", f"{dropped} record(s) were dropped at capture"
    return "complete", None


def _share_row(p: prim.Primitive, partition: dict[str, Any], ns: int, window_ns: int,
               header: Header, assumptions: Iterable[str], slices: Any = None,
               bounds: tuple[int, int] | None = None, shares_with: Iterable[str] = (),
               **extra: Any) -> dict[str, Any]:
    """A priced trace row. ``slices`` are its per-slice totals in ns.

    ``bounds`` are the ends of its attribution range in ns: the time only this row's
    records explain, and the time they could explain at all. ``shares_with`` are the
    rows that claim some of the same time. With no bounds the range is the point.
    """
    completeness, note = _completeness(header)
    seconds = ns / 1e9
    share = seconds / (window_ns / 1e9)
    facts, facts_reason = pricing._slice_facts(slices, window_ns)
    low, high = bounds or (ns, ns)

    def fires(amount: int) -> bool:
        # On the integers: a share of exactly the threshold is below it as a float quotient.
        return Fraction(amount, window_ns) >= Fraction(str(p.fire_at))

    return _row(
        p, partition, "fired" if fires(ns) else "not_fired",
        observation={"kind": "share", "value": share, "unit": "share of window"},
        seconds=seconds, share=share,
        completeness=completeness, assumptions=assumptions, completeness_note=note,
        interval={"low_s": low / 1e9, "high_s": high / 1e9, "low_share": low / window_ns,
                  "high_share": high / window_ns, "low_fires": fires(low),
                  "high_fires": fires(high)},
        shares_with=list(shares_with),
        slices=facts, slices_reason=facts_reason, **extra,
    )


def _claim_bounds(per_gap: Any, gaps: list[tuple[int, int]], syncs: list[_Rec]
                  ) -> dict[str, tuple[tuple[int, int], list[str]]]:
    """For rows 11, 10 and G1: the time only that row can claim and all it can claim,
    and the rows that claim some of the same time.

    A piece of gap time is claimed by row 11 if a memcpy record covers it, by row 10
    if a sync record does, and by G1 if its whole gap is launch-scale. The split in
    ``per_gap`` gives each piece to one row by a fixed precedence; where two rows
    claim a piece, the trace does not say which is right.
    """
    copy, sync_only, launch = (per_gap[:, c] for c in (1, 2, 3))
    length = per_gap[:, 1:].sum(axis=1)
    merged = _merge_intervals([(r.start_ns, r.end_ns) for r in syncs if r.end_ns > r.start_ns])
    under_sync = np.array(_overlap_sweep(merged, sorted(gaps)), dtype=per_gap.dtype)
    both = under_sync - sync_only  # under a sync record and a memcpy record
    long_gap = length > _LAUNCH_LATENCY_NS
    overlap = {("10", "11"): both.sum() > 0, ("10", "G1"): under_sync[~long_gap].sum() > 0,
               ("11", "G1"): copy[~long_gap].sum() > 0}

    def others(row: str) -> list[str]:
        return [a if b == row else b for (a, b), hit in overlap.items() if hit and row in (a, b)]

    return {
        "11": ((int((copy - both)[long_gap].sum()), int(copy.sum())), others("11")),
        "10": ((int(sync_only[long_gap].sum()), int(under_sync.sum())), others("10")),
        "G1": ((int(launch.sum()), int(length[~long_gap].sum())), others("G1")),
    }


def _remainder(partition: dict[str, Any], shared: bool, seconds: float | None,
               window_s: float | None, reason: str | None = None,
               why: str | None = None, next_step: str | None = None,
               assumes: Iterable[str] = ()) -> dict[str, Any]:
    name = _REMAINDER_NAME
    if shared:
        name = name[:-1] + ", or another traced process on this device)"
    return {
        "id": "remainder", "name": name, "kind": "remainder",
        "definition": "gap time not assigned to any built cause",
        "scope": "partition", "partition": dict(partition),
        "state": None, "reason": reason, "why": why,
        "next": next_step if reason else prim.get("G2").remedy,
        "seconds": seconds,
        "share": None if seconds is None or not window_s else seconds / window_s,
        "assumptions": [*(["other_pid_active_on_device"] if shared else []), *assumes],
        "thresholds_version": prim.THRESHOLDS_VERSION,
    }


# ── trace checks ─────────────────────────────────────────────────────────────


@dataclass
class _Partition:
    pid: int | None
    device_id: int
    #: start, end, (context, stream), name id
    kernels: list[tuple[int, int, tuple[int | None, int], int]] = field(default_factory=list)
    memcpys: list[tuple[int, int, str]] = field(default_factory=list)
    syncs: list[tuple[int, int, str]] = field(default_factory=list)
    invalid: str | None = None
    #: whether kernels lacked a context id: {True}, {False}, or both (a fault)
    contexts: set[bool] = field(default_factory=set)
    #: memcpy or sync records with no pid on a device several pids share (rows 10, 11, G1);
    #: one list per device, shared by its partitions and judged against each window
    pidless: list[tuple[int, int]] = field(default_factory=list)
    #: every memcpy or sync record in the capture that matched no partition; one list,
    #: shared by all partitions, and judged against each window (rows 10, 11, G1)
    unplaced: list[tuple[int, int]] = field(default_factory=list)
    #: sync records that could be this partition's and cannot be placed (rows 10, G1): each
    #: entry is their spans (a list shared with the other candidate partitions), why, and
    #: the next step. One that covers none of the window decides nothing.
    doubtful_syncs: list[tuple[list[tuple[int, int]], str, str]] = field(default_factory=list)

    @property
    def key(self) -> dict[str, Any]:
        return {"pid": self.pid, "device_id": self.device_id}


def _gaps(busy: list[tuple[int, int]], w0: int, w1: int) -> list[tuple[int, int]]:
    """Complement of sorted, disjoint ``busy`` inside the window ``[w0, w1)``."""
    gaps, cursor = [], w0
    for s, e in busy:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < w1:
        gaps.append((cursor, w1))
    return gaps


def _exposed_spans(base: list[tuple[int, int]], mask: list[tuple[int, int]]
                   ) -> list[tuple[int, int]]:
    """The parts of ``base`` that ``mask`` does not cover, as sorted disjoint intervals."""
    out: list[tuple[int, int]] = []
    covered = _merge_intervals(list(mask))
    j, m = 0, len(covered)
    for b0, b1 in _merge_intervals(list(base)):
        cursor = b0
        while j < m and covered[j][1] <= cursor:
            j += 1
        k = j
        while k < m and covered[k][0] < b1 and cursor < b1:
            if covered[k][0] > cursor:
                out.append((cursor, covered[k][0]))
            cursor = max(cursor, covered[k][1])
            k += 1
        if cursor < b1:
            out.append((cursor, b1))
    return out


def _stream_overlap(kernels: list[tuple[int, int, tuple[int | None, int], int]]) -> bool:
    last_end: dict[tuple[int | None, int], int] = {}
    def key(k: tuple[int, int, tuple[int | None, int], int]) -> tuple[int, int, int]:
        return (-1 if k[2][0] is None else k[2][0], k[2][1], k[0])

    # A zero-length kernel occupies no time on its stream; with one left in, two kernels
    # sharing a start would be judged by the order they arrived in.
    for start, end, stream, _ in sorted((k for k in kernels if k[1] > k[0]), key=key):
        if start < last_end.get(stream, start):
            return True
        last_end[stream] = max(end, last_end.get(stream, end))
    return False


def _by_label(gaps: list[tuple[int, int]], records: list[tuple[int, int, str]],
              under: list[_Rec]) -> dict[str, float]:
    """Seconds of gap time under each label's records and not under ``under``.

    Labels that overlap each count. A file chooses its labels, so only the
    ``_MAX_LABELS`` with the most records are kept apart and each is cut to a
    printable length; the rest are one entry.
    """
    # Grouped by the name that is printed: two labels cut to one name are one entry,
    # their records merged, not two amounts added.
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s, e, label in records:
        groups[_safe(label)].append((s, e))
    keep = sorted(groups, key=lambda k: (-len(groups[k]), k))[:_MAX_LABELS]
    shown = {k: groups[k] for k in sorted(keep)}
    if len(groups) > len(keep):
        kept = set(keep)
        fold = f"other ({len(groups) - len(keep)} labels)"
        while fold in shown:  # a file may name a label anything, this included
            fold += "*"
        shown[fold] = [r for k, recs in groups.items() if k not in kept for r in recs]
    mask = _merge_intervals([(r.start_ns, r.end_ns) for r in under if r.end_ns > r.start_ns])
    return {name: sum(_overlap_sweep(_exposed_spans(recs, mask), gaps)) / 1e9
            for name, recs in shown.items()}


def _partition_rows(part: _Partition, header: Header, shared: bool, no_sync: str | None,
                    no_copy: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Window info and the rows of one ``(pid, device)`` partition.

    The window is the kernel span, first kernel start to last kernel end. Records
    are clipped to it, so one stray record far from the work cannot dilute a share.
    ``no_sync`` and ``no_copy`` say why this capture cannot show synchronization or
    memcpy records, if so: time under a record the source drops is then somewhere in
    the rows that remain, which carry that as an assumption.
    """
    key = part.key
    r10, r11, g1 = prim.get("10"), prim.get("11"), prim.get("G1")
    # A zero-length kernel has no busy time; left in, it would split the gap around it
    # into launch-scale pieces, and one far from the work would stretch the window.
    live = [(k[0], k[1]) for k in part.kernels if k[1] > k[0]]
    spans = live or [(k[0], k[1]) for k in part.kernels]
    w0, w1 = min(s for s, _ in spans), max(e for _, e in spans)
    window_ns = w1 - w0
    busy = _merge_intervals(live)
    busy_ns = sum(e - s for s, e in busy)
    info = {**key, "window_start_ns": w0, "window_end_ns": w1, "window_s": window_ns / 1e9,
            "busy_s": busy_ns / 1e9, "n_kernels": len(part.kernels),
            "closure_asserted": False}
    assumptions = ["other_pid_active_on_device"] if shared else []
    unseen = [a for a, why in (("no_sync_records_in_source", no_sync),
                               ("no_memcpy_records_in_source", no_copy)) if why]

    def all_cannot(reason: str, what: str, why: str, next_step: str) -> list[dict[str, Any]]:
        return [_cannot(p, key, reason, what, why, next_step) for p in (r11, r10, g1)] + [
            _remainder(key, shared, None, None, reason, why, next_step)]

    def no_channel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # A missing channel outranks a short window: a longer capture would not help.
        if no_copy:
            rows[0] = _cannot(r11, key, "channel_missing", "no memcpy records to read", no_copy,
                              _NO_COPY_NEXT)
        if no_sync:
            rows[1] = _cannot(r10, key, "channel_missing", "no synchronization records to read",
                              no_sync, _NO_SYNC_NEXT, sync_record_s=None, sync_record_count=None)
        return rows

    if part.invalid:
        return info, all_cannot("invalid_capture", "the partition's events are inconsistent",
                                part.invalid, "re-capture; the trace is not usable as written")

    def in_window(spans: list[tuple[int, int]]) -> int:
        # A record with no duration, or wholly outside the window, covers no gap.
        return sum(1 for s, e in spans if e > s and e > w0 and s < w1)

    if in_window(part.pidless):
        return info, all_cannot(
            "invalid_capture", "gap time could not be assigned",
            "memcpy/sync records carry no pid and several traced pids share this device",
            "re-capture with the current tracer, which keeps pid on memcpy and sync records")
    if n_unplaced := in_window(part.unplaced):
        return info, all_cannot(
            "invalid_capture", "gap time could not be assigned",
            f"{n_unplaced} memcpy/sync record(s) inside the window match no traced partition "
            "(a pid or device with no kernels), so the gaps they cover are unknown",
            "re-capture so every record's pid and device belong to a traced process")

    def sliced(per_gap: Any, column: int) -> Any:
        # A gap is classified whole, so it belongs whole to the slice it starts in.
        return pricing._slices_by_start(per_gap[:, 0], per_gap[:, column], w0, window_ns)

    too_short = (
        "too_short", f"the window is {window_ns / 1e9:.3f} s",
        f"a window under {prim.MIN_WINDOW_NS / 1e9:.0f} s is not evaluated",
        "capture a longer window")
    window_s = window_ns / 1e9
    gaps = _gaps(busy, w0, w1)
    copies = [_Rec(s, e) for s, e, _ in part.memcpys]
    # Unplaceable sync records are an invalid capture, which outranks a short window.
    # The smallest by text, so the reason given does not depend on which record came first.
    sync_no_device = min(((why, nxt) for spans, why, nxt in part.doubtful_syncs
                          if in_window(spans)), default=None)
    if sync_no_device:
        why, nxt = sync_no_device
        if window_ns < prim.MIN_WINDOW_NS:
            row11 = _cannot(r11, key, *too_short)
        else:
            per_gap = _stall_ns_per_gap(gaps, copies, [])
            row11 = _share_row(r11, key, int(per_gap[:, 1].sum()), window_ns,
                               header, assumptions, sliced(per_gap, 1),
                               *_claim_bounds(per_gap, gaps, [])["11"],
                               by_direction=_by_label(gaps, part.memcpys, []))
        return info, [
            row11,
            _cannot(r10, key, "invalid_capture", "sync records could not be placed", why, nxt,
                    sync_record_s=None, sync_record_count=None),
            _cannot(g1, key, "invalid_capture", "gap time could not be split", why, nxt),
            _remainder(key, shared, None, None, "invalid_capture", why, nxt),
        ]
    if window_ns < prim.MIN_WINDOW_NS:
        return info, no_channel(all_cannot(*too_short))

    # With no sync channel, time the host spent in a sync stays in G1 or the remainder.
    placed = [] if no_sync else part.syncs
    syncs = [_Rec(s, e) for s, e, _ in placed]
    per_gap = _stall_ns_per_gap(gaps, copies, syncs)
    transfer_ns, sync_ns, launch_ns, idle_ns = (int(v) for v in per_gap[:, 1:].sum(axis=0))
    bounds = _claim_bounds(per_gap, gaps, syncs)
    info["closure_asserted"] = True
    merged = _merge_intervals([(r.start_ns, r.end_ns) for r in syncs])
    # Time under a copy the source drops reads as sync, launch or idle time.
    after_copies = [*assumptions, *(["no_memcpy_records_in_source"] if no_copy else [])]
    rows = no_channel([
        _share_row(r11, key, transfer_ns, window_ns, header, assumptions,
                   sliced(per_gap, 1), *bounds["11"],
                   by_direction=_by_label(gaps, part.memcpys, [])),
        _share_row(
            r10, key, sync_ns, window_ns, header, after_copies, sliced(per_gap, 2),
            *bounds["10"],
            by_sync_kind=_by_label(gaps, placed, copies),
            sync_record_s=_overlap_sweep(merged, [(w0, w1)])[0] / 1e9,
            sync_record_count=sum(1 for r in syncs if r.end_ns > w0 and r.start_ns < w1)),
        _share_row(g1, key, launch_ns, window_ns, header, [*assumptions, *unseen],
                   sliced(per_gap, 3), *bounds["G1"]),
        _remainder(key, shared, idle_ns / 1e9, window_s, assumes=unseen),
    ])
    return info, rows


def _device_rows(device_id: int, parts: list[_Partition],
                 names: list[str], header: Header, n_devices: int) -> list[dict[str, Any]]:
    """Rows 22 and 25 for one device, pids pooled.

    ``n_devices`` is the larger of the header's count and the devices with kernels:
    the header counts what one process could see, and may be missing.
    """
    key = {"pid": None, "device_id": device_id}
    entries = [prim.get(i) for i in _COLLECTIVE_ROWS.values()]
    bad = next((p.invalid for p in parts if p.invalid), None)
    if bad:
        return [_cannot(p, key, "invalid_capture", "the device's events are inconsistent", bad,
                        "re-capture; the trace is not usable as written") for p in entries]
    # A zero-length kernel is kept for its name and never stretches the window.
    kernels = [k for p in parts for k in p.kernels if k[1] > k[0]]
    w0, w1 = min(k[0] for k in kernels), max(k[1] for k in kernels)
    window_ns = w1 - w0
    # Classified once per distinct name, not once per kernel.
    op_of: dict[str, str | None] = {}
    spans: dict[str, list[tuple[int, int]]] = {op: [] for op in _COLLECTIVE_ROWS}
    compute: list[tuple[int, int]] = []
    for start, end, _, name_id in kernels:
        name = names[name_id]
        if name not in op_of:
            op = classify_op(name)
            op_of[name] = op if op in _COLLECTIVE_ROWS else None
        op = op_of[name]
        (spans[op] if op else compute).append((start, end))
    # A missing channel outranks a short window: a longer capture would not help.
    if not any(spans.values()) and n_devices > 1:
        return [_cannot(
            p, key, "channel_missing", "no collective kernel names recognised",
            f"the capture shows {n_devices} devices, so collectives are "
            "expected, and no kernel name matched the op classifier") for p in entries]
    if window_ns < prim.MIN_WINDOW_NS:
        return [_cannot(p, key, "too_short", f"the device window is {window_ns / 1e9:.3f} s",
                        f"a window under {prim.MIN_WINDOW_NS / 1e9:.0f} s is not evaluated",
                        "capture a longer window") for p in entries]
    # A kernel is a collective here because its name matched a substring, and the
    # classifier has never been checked against a real capture: one it does not know
    # reads as compute, and a compute kernel it mistakes is priced. Only a row with no
    # match on a capture that states exactly one device rests on nothing: one device
    # has no collectives to miss. A header count below 1 is capture's "probe failed".
    several = n_devices > 1 or header.device_count < 1 or "device_count" in header.missing
    rows = []
    for op, row_id in _COLLECTIVE_ROWS.items():
        p = prim.get(row_id)
        # The two collective classes never hide each other: only compute is the mask.
        exposed = _exposed_spans(spans[op], compute)
        rows.append(_share_row(p, key, sum(e - s for s, e in exposed), window_ns,
                               header,
                               # Pids are pooled: another process's collectives count here.
                               [*(["other_pid_active_on_device"] if len(parts) > 1 else []),
                                *(["collective_classified_by_name_only"]
                                  if several or spans[op] else [])],
                               pricing._slices_by_overlap(exposed, w0, window_ns),
                               device_window_s=window_ns / 1e9, n_kernels=len(spans[op])))
    return rows


def _safe(value: Any, limit: int = 80) -> str:
    """Text from a file made safe to print: no control or non-ASCII characters, bounded."""
    raw = str(value)
    text = raw[:limit].encode("unicode_escape").decode("ascii")
    return text + "..." if len(raw) > limit else text


def _is_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_ns(value: Any) -> bool:
    """An integer that can be a nanosecond timestamp (floats and NaN are not)."""
    return _is_id(value) and -_MAX_NS < value < _MAX_NS


def _place(events: Iterable[_Event], header: Header
           ) -> tuple[str | None, list[_Partition], list[str], int,
                      Counter]:
    """Sort events into partitions. Returns the capture-level fault, if any.

    The last value counts the memcpy and sync records seen, by kind.
    """
    parts: dict[tuple[int | None, int], _Partition] = {}
    names: list[str] = []
    name_ids: dict[str, int] = {}
    records: list[_Event] = []
    streams: dict[tuple[int | None, int], tuple[int | None, int]] = {}
    #: the partitions whose kernels ran in each (pid, context), and in each context
    context_parts: dict[tuple[int | None, int], set[tuple[int | None, int]]] = defaultdict(set)
    by_context: dict[int, set[tuple[int | None, int]]] = defaultdict(set)
    n_events = reversed_events = out_of_range = 0
    for e in events:
        n_events += 1
        if not (_is_ns(e.start_ns) and _is_ns(e.end_ns) and _is_id(e.stream_id)
                and _is_id(e.device_id) and (e.pid is None or _is_id(e.pid))
                and (e.context_id is None or _is_id(e.context_id))
                and isinstance(e.label, str)):
            out_of_range += 1
            continue
        if e.end_ns < e.start_ns:
            reversed_events += 1
        if e.kind != "kernel":
            records.append(e)
            continue
        name_id = name_ids.get(e.label)
        if name_id is None:
            name_id = name_ids[e.label] = len(names)
            names.append(e.label)
        part = parts.get((e.pid, e.device_id))
        if part is None:
            part = parts[(e.pid, e.device_id)] = _Partition(e.pid, e.device_id)
        # Streams are numbered per context: the overlap check keys on both. One shared
        # tuple per stream, not one per kernel.
        stream = (e.context_id, e.stream_id)
        stream = streams.setdefault(stream, stream)
        part.contexts.add(e.context_id is None)
        if e.context_id is not None:
            context_parts[(e.pid, e.context_id)].add((e.pid, e.device_id))
            by_context[e.context_id].add((e.pid, e.device_id))
        part.kernels.append((e.start_ns, e.end_ns, stream, name_id))

    ordered = [parts[k] for k in sorted(parts, key=lambda k: (k[1], -1 if k[0] is None else k[0]))]
    n_records = Counter(e.kind for e in records)
    fault = None
    pids = {k[0] for k in parts}
    if not _is_id(header.device_count):
        fault = "the trace header's device_count is not a number"
    elif header.source not in _SOURCES:
        fault = (f"the trace header names an unknown source {str(header.source)[:40]!r}, "
                 "so its records cannot be trusted to mean what a known source's do")
    elif header.source == "none":
        fault = "the trace header says source is none: no event collector ran"
    elif n_events == 0:
        fault = "the trace has no events"
    elif not parts:
        fault = "the trace has events but no kernel event"
    elif len(parts) > _MAX_PARTITIONS:
        fault = (f"the trace names {len(parts)} (pid, device) partitions, more than the "
                 f"{_MAX_PARTITIONS} any capture holds; its pid or device fields are not usable")
    elif out_of_range:
        fault = (f"{out_of_range} event(s) have timestamps that cannot be nanoseconds, or a "
                 "name, stream, device, pid or context of the wrong type")
    elif reversed_events:
        fault = f"{reversed_events} event(s) end before they start"
    elif None in pids and len(pids) > 1:
        fault = "some kernel events carry a pid and some do not"
    if fault:
        return fault, ordered, names, 0, n_records

    for part in ordered:
        if all(k[1] <= k[0] for k in part.kernels):
            part.invalid = "no kernel in the partition has a duration, so it has no window"
        elif len(part.contexts) > 1:
            # A kernel with no context could be on any of them: overlap cannot be checked.
            part.invalid = "some kernels carry a context id and some do not"
        elif _stream_overlap(part.kernels):
            part.invalid = "two kernels overlap on the same stream"

    pids_on_device: dict[int, set[int | None]] = defaultdict(set)
    devices_of_pid: dict[int | None, set[int]] = defaultdict(set)
    for pid, device_id in parts:
        pids_on_device[device_id].add(pid)
        devices_of_pid[pid].add(device_id)

    unplaced: list[tuple[int, int]] = []
    doubt: dict[tuple[str, int | None], list[tuple[int, int]]] = {}
    pidless_on: dict[int, list[tuple[int, int]]] = {}

    def pidless(device_id: int, span: tuple[int, int]) -> None:
        """Keep a record with no pid on a device several pids share, once for the device."""
        spans = pidless_on.get(device_id)
        if spans is None:
            spans = pidless_on[device_id] = []
            for pid in pids_on_device[device_id]:
                parts[(pid, device_id)].pidless = spans
        spans.append(span)

    def doubtful(targets: list[_Partition], kind: str, pid: int | None, why: str, nxt: str,
                 span: tuple[int, int]) -> None:
        """Keep a sync record no one partition can claim, once, for all that might."""
        spans = doubt.get((kind, pid))
        if spans is None:
            spans = doubt[(kind, pid)] = []
            for part in targets:
                part.doubtful_syncs.append((spans, why, nxt))
        spans.append(span)

    for e in records:
        span = (e.start_ns, e.end_ns)
        if e.kind == "sync" and header.source == "cupti":
            # A live CUPTI sync record's device_id is not real; it follows its pid.
            if e.pid is not None and e.pid in devices_of_pid:
                targets = [parts[(e.pid, d)] for d in devices_of_pid[e.pid]]
            elif e.pid is None:
                targets = ordered
            else:
                unplaced.append(span)
                continue
            if e.context_id is not None and not any(True in t.contexts for t in targets):
                # Those kernels all name their context, so the record's own context says
                # which device it waited on, or that none was traced.
                owners = (by_context.get(e.context_id, ()) if e.pid is None
                          else context_parts.get((e.pid, e.context_id), ()))
                if len(owners) == 1:
                    targets = [parts[next(iter(owners))]]
                elif not owners:
                    doubtful(targets, "context", e.pid,
                             "a sync record inside the window names a context that has no "
                             "kernels in the capture, so the device it waited on is unknown",
                             "trace every context the process uses", span)
                    continue
            if len(targets) == 1:
                targets[0].syncs.append((e.start_ns, e.end_ns, e.label))
            elif e.pid is None:
                # On a device several pids share, a record with no pid is the stronger
                # fault: all four gap rows, not only the sync ones.
                for device_id in sorted({t.device_id for t in targets}):
                    if len(pids_on_device[device_id]) > 1:
                        pidless(device_id, span)
                doubtful(targets, "pidless", None,
                         "sync records carry neither a pid nor a device, and the capture has "
                         "several partitions",
                         "re-capture with the current tracer, which keeps pid on sync records",
                         span)
            else:
                doubtful(targets, "devices", e.pid,
                         "sync records carry no device and this pid has kernels on several "
                         "devices",
                         "keep the device on sync records at capture, or trace one device per "
                         "process", span)
            continue
        on_device = pids_on_device.get(e.device_id)
        if not on_device:
            unplaced.append(span)
            continue
        if e.pid is None and on_device != {None}:
            if len(on_device) > 1:
                pidless(e.device_id, span)
                continue
            part = parts[(next(iter(on_device)), e.device_id)]
        else:
            part = parts.get((e.pid, e.device_id))
            if part is None:
                unplaced.append(span)
                continue
        (part.memcpys if e.kind == "memcpy" else part.syncs).append(
            (e.start_ns, e.end_ns, e.label))
    for part in ordered:
        part.unplaced = unplaced
    return None, ordered, names, len(unplaced), n_records


# ── the account ──────────────────────────────────────────────────────────────


def _drop_row(header: Header) -> dict[str, Any]:
    """Row 24, from the header's dropped-record count. A misrouted kernel is not checked.

    So the row can say records were dropped and can never be ruled out.
    """
    p = prim.get("24")
    state, note = _completeness(header)
    if state == "unknown":
        return _cannot(p, _RUN, "channel_missing", "no dropped-record count to read", note,
                       _UNCOUNTED_NEXT)
    counted = header.dropped_records >= 0
    return _row(
        p, _RUN, "fired" if state == "lossy" else "not_fired",
        observation=({"kind": "count", "value": header.dropped_records,
                      "unit": "dropped records"} if counted else None),
        price_reason=((f"unpriced: {note}; " if not counted else "unpriced: ")
                      + "the time under a dropped record is not known"
                      if state == "lossy" else None),
        next_step=_LOSS_NEXT.get(state, p.remedy), completeness=state, completeness_note=note,
        assumptions=["misrouted_kernel_not_checked"])


def _fixed_rows() -> list[dict[str, Any]]:
    return [_cannot(p, _RUN, p.fixed_reason, "not evaluated in this build", p.fixed_detail)
            for p in prim.REGISTRY.values() if p.fixed_reason]


def _group_rows(scope: str, key: dict[str, Any]) -> list[dict[str, Any]]:
    """Unbuilt groups and every group member, for one instance of ``scope``."""
    rows = []
    for p in prim.REGISTRY.values():
        if p.scope != scope:
            continue
        if p.id in prim.NOT_BUILT:
            rows.append(_cannot(p, key, "not_built", "this check is not built", p.witness,
                                p.remedy))
        elif p.group:
            rows.append(_cannot(
                p, key, "undivided_group", f"reported as part of {p.group}",
                f"the trace cannot split {p.group} ({prim.get(p.group).name}) into its members"))
    return rows


def account(
    events: Iterable[Any],
    header: Header,
) -> Report:
    """Run every built check on events in memory.

    ``events`` are trace events (pydantic, the importers' slotted classes, or the
    tuples the file reader yields) in any order.
    """
    fault, parts, names, unplaced, n_records = _place(_normalize(events), header)

    results: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    if fault:
        for p in prim.REGISTRY.values():
            if p.scope == "run" or p.baseline:
                continue
            if p.id in prim.NOT_BUILT or p.group:
                # These keep their own reason: the trace would not have settled them.
                results.extend(r for r in _group_rows(p.scope, _ALL) if r["id"] == p.id)
            else:
                results.append(_cannot(p, _ALL, "invalid_capture", "the trace is not usable",
                                       fault, "re-capture; see the reason"))
        results.append(_remainder(_ALL, False, None, None, "invalid_capture", fault,
                                  "re-capture; see the reason"))
    else:
        by_device: dict[int, list[_Partition]] = defaultdict(list)
        for part in parts:
            by_device[part.device_id].append(part)
        n_devices = max(header.device_count, len(by_device))
        no_sync = None
        if header.source in _NO_SYNC_SOURCES:
            no_sync = f"{header.source} traces carry no synchronization records"
        elif header.source == "nsys-import" and not n_records["sync"]:
            no_sync = ("this nsys-import trace has no synchronization record at all, so the "
                       "export may not include that table")
        no_copy = None
        if header.source in ("nsys-import", "rocprof") and not n_records["memcpy"]:
            # The importer returns nothing when the export lacks the memcpy table, and
            # the ROCm collector does not check that its memory-copy service started.
            no_copy = (f"this {header.source} trace has no memcpy record at all, so its source "
                       "may not have recorded them")
        for device_id, group in by_device.items():
            shared = len(group) > 1
            for part in group:
                info, rows = _partition_rows(part, header, shared, no_sync, no_copy)
                infos.append(info)
                results.extend(rows)
                results.extend(_group_rows("partition", part.key))
            results.extend(_device_rows(device_id, group, names, header, n_devices))
            results.extend(_group_rows("device", {"pid": None, "device_id": device_id}))

    results.append(_drop_row(header))
    results.extend(_fixed_rows())
    for row in results:
        if row["state"] is None:
            continue
        row.update(pricing._status(row))
        if row["status"] == "estimated" and row["completeness"] == "unknown":
            # The one assumption an `estimated` row may carry (see pricing): the header
            # has no drop count, and a missing kernel record would read as gap time.
            row["assumptions"].append("no_dropped_events")
        if row["status"] == pricing._NOT_IDENTIFIABLE and row["next"] is None:
            # What would settle the reason the status gives, in the order it gives them.
            steps = [_ASSUMPTION_NEXT[a] for a in row["assumptions"] if a in _ASSUMPTION_NEXT]
            loss = _LOSS_NEXT.get(row["completeness"])
            unpriced = row["state"] == "fired" and row["seconds"] is None
            if row["status_reason"] == "confounded":
                row["next"] = _CONFOUNDED_NEXT
            else:
                row["next"] = ("; ".join(dict.fromkeys(steps)) if steps and not unpriced
                               else (None if unpriced else loss) or prim.get(row["id"]).remedy)

    presence: dict[str, str] = {}
    for row in results:
        state = row["state"]
        if state is None:
            continue
        seen = presence.get(row["id"])
        if state == "fired" or seen is None or (state == "cannot_tell" and seen == "not_fired"):
            presence[row["id"]] = state

    return Report(
        schema_version=SCHEMA_VERSION,
        thresholds_version=prim.THRESHOLDS_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=header.source,
        validation=_VALIDATION,
        inputs={"found": [], "missing": []},
        partitions=infos,
        results=results,
        baselines=[{"id": p.id, "name": p.name, "note": "baseline, not checked"}
                   for p in prim.REGISTRY.values() if p.baseline],
        run_presence=presence,
        records_unplaced=unplaced,
    )


def run(run_dir: str | Path) -> Report:
    """Run the checks on a capture directory. Raises only :class:`WitnessInputError`."""
    run_dir = Path(run_dir)
    header, events = _read_trace(run_dir / "trace.jsonl")
    report = account(events, header)
    report.inputs = {"found": ["trace.jsonl"], "missing": []}
    return report


# ── output ───────────────────────────────────────────────────────────────────


def _where(partition: dict[str, Any]) -> str:
    if partition.get("all"):
        return "all"
    if partition["device_id"] is None:
        return "run"
    pid = "" if partition["pid"] is None else f"pid {partition['pid']} "
    return f"{pid}gpu {partition['device_id']}"


def _other_forms(row_id: str) -> str:
    """For a group whose members span forms, the forms besides its own."""
    p = prim.get(row_id)
    others = sorted({prim.get(m).form for m in p.members} - {p.form})
    return f" (also {', '.join(others)})" if others else ""


def _render(report: Report, show_all: bool = False) -> str:
    """The text table. ``show_all`` adds group members, fixed-state rows and baselines."""
    reported = [r for r in report.results if r["id"] in _REPORTED]
    counts = Counter(r["state"] for r in reported)
    statuses = Counter(r["status"] for r in reported)
    remedies = Counter(r["next"] for r in reported
                       if r["state"] == "cannot_tell" and r["reason"] != "not_built" and r["next"])
    lines = [
        _PRECONDITION,
        f"{counts['fired']} fired, {counts['not_fired']} not fired, "
        f"{counts['cannot_tell']} cannot tell"
        + (f"; most common next step: {remedies.most_common(1)[0][0]}" if remedies else ""),
        "status: " + ", ".join(f"{statuses[k]} {k}" for k in prim.STATUSES),
        f"source {report.source}; thresholds {report.thresholds_version}; {report.validation}",
        "inputs found: " + ", ".join(report.inputs["found"] or ["none"])
        + "; missing: " + ", ".join(report.inputs["missing"] or ["none"]),
    ]
    for info in report.partitions:
        lines.append(
            f"{_where(info)}: window {info['window_s']:.3f} s, busy {info['busy_s']:.3f} s"
            + ("" if info["closure_asserted"] else " (account does not close: see rows)"))
    lines.append("")
    for r in reported + [r for r in report.results if r["id"] == "remainder"]:
        head = f"{_where(r['partition']):<16} {r['id']:<9} {r['state'] or '-':<11} {r['name']}"
        if r["seconds"] is not None:
            head += f": {r['seconds']:.4f} s ({r['share']:.2%} of window"
            if r.get("interval"):
                head += (f", associated; range {r['interval']['low_s']:.4f} to "
                         f"{r['interval']['high_s']:.4f} s)")
            else:
                head += ", associated)" if r.get("price_kind") else ")"
        elif r.get("observation"):
            head += f": {r['observation']['value']:.4g} {r['observation']['unit']}"
        lines.append(head)
        if r.get("status"):
            reason, detail = r["status_reason"], r["status_detail"]
            if r["state"] == "cannot_tell" or (detail or "").startswith(f"{reason}:"):
                why = reason if r["state"] == "cannot_tell" else detail
            else:
                why = ": ".join(x for x in (reason, detail) if x)
            lines.append(f"    status: {r['status']}" + (f" ({why})" if why else ""))
            if r["state"] != "cannot_tell" and r["status"] == pricing._NOT_IDENTIFIABLE:
                lines.append(f"    Next: {r['next']}")
            f = r.get("slices")
            if r["status"] == "estimated" and f and f["busiest_tenth_share"] is not None:
                lines.append(
                    f"    where: in {f['with_time']} of {f['n']} one-second slices; largest "
                    f"{f['max_slice_s']:.4f} s; busiest tenth holds "
                    f"{f['busiest_tenth_share']:.0%}")
            elif r["status"] == "estimated":
                lines.append(f"    where: not given ({r.get('slices_reason') or 'no time'})")
        if r["state"] == "cannot_tell" or (r["state"] is None and r["reason"]):
            lines.append(f"    {r['reason']}: " + "; ".join(
                _safe(s, 300) for s in (r.get("what"), r.get("why")) if s)
                + (f". Next: {r['next']}" if r.get("next") else ""))
        if r.get("price_reason"):
            lines.append(f"    {r['price_reason']}")
        if r.get("assumptions"):
            lines.append("    assumes: " + ", ".join(r["assumptions"]))
        if r["state"] == "fired":
            levers = r["candidate_levers"]
            if levers and "member" in levers[0]:
                for m in levers:
                    names = ", ".join(f"{v['name']} (review: {v['review'] or 'none yet'})" for v in m["levers"])
                    lines.append(f"    candidate levers for row {m['member']} "
                                 f"({m['member_name']}): {names or 'none in the library'}")
                lines.append("    applicability not checked")
            elif levers:
                lines.append("    candidate levers: " + ", ".join(
                    f"{v['name']} (review: {v['review'] or 'none yet'})" for v in levers))
            lines.append(f"    {r.get('levers_note') or 'applicability not checked'}")
    # Each row once under its form; no figure is summed across a form.
    lines += ["", "By form (a group is listed under its own form; its members' other forms follow):"]
    for form in prim.FORMS:
        mine = [r for r in reported if r["form"] == form]
        if not mine:
            continue
        lines.append(f"  {form}: " + "; ".join(
            f"{r['id']} [{_where(r['partition'])}] {r['status']}" + _other_forms(r["id"])
            for r in mine))
    if show_all:
        lines += ["", "Group members (the group is the reported unit):"]
        for r in report.results:
            if r.get("kind") == "member":
                lines.append(f"  {_where(r['partition']):<16} {r['id']:<4} {r['name']}: {r['what']}")
        lines += ["", "Not evaluated in this build:"]
        for r in report.results:
            if r["id"] not in _REPORTED and r.get("kind") == "cause":
                lines.append(f"  {r['id']:<4} {r['name']}: {r['reason']} ({r['why']}). "
                             f"Next: {r['next']}")
        lines += ["", "Baselines, not checked:"]
        lines += [f"  {b['id']:<4} {b['name']}" for b in report.baselines]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gitm.optimizer.witness", description=_PRECONDITION)
    ap.add_argument("run_dir", type=Path, help="capture directory holding trace.jsonl")
    ap.add_argument("--json", action="store_true", help="print witness.json instead of the table")
    ap.add_argument("--all", action="store_true", dest="show_all",
                    help="also print group members, rows not evaluated in this build, and "
                         "baselines (witness.json always holds every row)")
    args = ap.parse_args(argv)
    out = args.run_dir / "witness.json"
    try:
        report = run(args.run_dir)
    except WitnessInputError as exc:
        stale = f"; the existing {out} is stale and was left untouched" if out.exists() else ""
        print(f"witness checks did not run: {exc}. Point RUN_DIR at a capture directory "
              f"with a complete trace.jsonl{stale}", file=sys.stderr)
        return 2
    text = json.dumps(report.to_dict(), indent=2, allow_nan=False)
    print(text if args.json else _render(report, args.show_all))
    try:
        _write_atomic(out, text + "\n")
    except OSError as exc:
        print(f"cannot write {out}: {exc}. The result above was not saved; make "
              f"{args.run_dir} writable and re-run", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
