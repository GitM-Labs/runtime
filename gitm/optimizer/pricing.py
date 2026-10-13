"""Where a price sits inside the capture, and the brief's status for a witness row.

A price is time *associated* with a cause in this capture. It is arithmetic on the
records, so its status describes this capture and nothing else:

    fired      priced, no unverified assumption              estimated (this capture)
    fired      priced, but not over the threshold alone      not identifiable (confounded)
    fired      priced, resting on an unverified assumption   not identifiable (structural)
    fired      priced, records known to be dropped           not identifiable (structural)
    fired      no price                                      not identifiable (unpriced)
    not_fired  complete, no assumption                       absent
    not_fired  but its records cover the threshold           not identifiable (confounded)
    not_fired  anything less                                 not identifiable (structural)
    cannot_tell                                              not identifiable (the presence reason)

The interval on a priced row is its attribution range, in this capture: from the
time only this row's records explain to the time they could explain at all. Rows 10,
11 and G1 can claim the same piece of gap time (a sync record and a memcpy record
over one gap; either over a launch-scale gap), and the point value gives each piece
to one of them by precedence. So a row is ``estimated`` only when the low end alone
reaches the threshold, ``absent`` only when the high end stays under it, and
``confounded`` in between: the records cannot say which of the rows it names owns
the time. The range is exact arithmetic on the records. It is not an error bar: it
does not cover timestamp error, tracer overhead or dropped records, and it says
nothing of other windows or of the time removing the cause would recover.

``estimated`` says how much time this capture holds under the witness. It does not say
the same would be seen in another window: one capture cannot show that. Beside it a
row carries plain facts about where the time sat, over 1 s slices of the window: how
many slices held any of it, the largest slice, and the share of the total in the
busiest tenth of slices. There is no band for other windows and no verdict on
evenness. Both were built and withdrawn: a bootstrap band over slices only means
something for other windows, and each check for evenness let through a shape the
next review found.

When ``absent`` may be said. A row's time is counted over every record in the window,
not sampled, so a row below its threshold is ruled out at that threshold exactly when
no record that would have raised it can be missing. That needs all of:

* the trace header reports zero dropped records (``completeness`` is ``complete``).
  A missing sync, memcpy or collective record hides the row's time, and a missing
  kernel record turns launch-scale gaps into idle;
* the source records the row's kind at all, else the row is ``cannot_tell``;
* a source that counts: ``cupti`` or ``rocprof``. An imported trace cannot count,
  whatever its header says;
* no assumption on the row: another traced pid on the device, or a kind the source
  drops, leaves the split of the gap time unverified; rows 22 and 25 rest on a
  kernel name classifier never checked against a real capture, unless the header
  states exactly one device and no name matched; row 24 does not check misroute.

The same assumptions keep a fired row from ``estimated``, so rows 22 and 25 show
their time and are not identifiable until the classifier is checked.

``absent`` is then "under the row's ``fire_at`` in this capture". Like ``estimated`` it says
nothing of other windows, and nothing of a cause that leaves no record of this kind.
The drop count is the collector's: records still in its buffer when the capture
closed are not in it, which only a real capture can size. The reader also treats a
traced process with nothing on disk for the window as unknown, and the capturing
process may itself be one: until a real capture shows what an idle process writes,
``absent`` on an injected capture is unconfirmed and may never be reached.

Without a drop count (every imported trace, and captures from an older collector) an
``estimated`` row carries the assumption ``no_dropped_events``: a missing kernel
record would read as gap time. It is the one assumption an ``estimated`` row may
carry. A count above zero is that assumption known false: the row is priced and not
identifiable, since the lost records could move the time either way.

Nothing here is public.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from gitm.optimizer import primitives as prim

_NOT_IDENTIFIABLE = "not_identifiable_from_this_data"


def _n_slices(window_ns: int) -> int:
    return window_ns // prim.SLICE_NS


def _too_long(window_ns: int) -> bool:
    return _n_slices(window_ns) > prim.MAX_SLICES


def _slices_by_start(starts: np.ndarray, amounts: np.ndarray, w0: int,
                     window_ns: int) -> np.ndarray | None:
    """Per-slice totals (ns) when each amount belongs whole to the slice it starts in.

    Used for gap time: a gap is classified as a whole, so it is not cut at a slice
    boundary. Amounts starting in the final partial slice are dropped. ``None`` when
    the window has more slices than are ever allocated.
    """
    if _too_long(window_ns):
        return None
    n = _n_slices(window_ns)
    if n == 0:
        return np.zeros(0)
    # No overflow: the window is at most MAX_SLICES slices, so starts - w0 is small.
    idx = (np.asarray(starts, dtype=np.int64) - w0) // prim.SLICE_NS
    keep = (idx >= 0) & (idx < n)
    return np.bincount(idx[keep], weights=np.asarray(amounts, dtype=np.float64)[keep],
                       minlength=n)


def _slices_by_overlap(spans: Iterable[tuple[int, int]], w0: int, window_ns: int
                       ) -> np.ndarray | None:
    """Per-slice totals (ns) of intervals, each cut at the slice boundaries it crosses."""
    if _too_long(window_ns):
        return None
    n = _n_slices(window_ns)
    totals = np.zeros(n)
    size = prim.SLICE_NS
    for start, end in spans:
        first = max((start - w0) // size, 0)
        last = min((end - 1 - w0) // size, n - 1)
        for i in range(first, last + 1):
            lo, hi = w0 + i * size, w0 + (i + 1) * size
            totals[i] += min(end, hi) - max(start, lo)
    return totals


def _slice_facts(slices: np.ndarray | None, window_ns: int) -> tuple[dict[str, Any] | None,
                                                                    str | None]:
    """Where the row's time sat across the window's full 1 s slices, or why that is not given."""
    if slices is None:
        return None, (f"window too long: {_n_slices(window_ns)} slices of "
                      f"{prim.SLICE_NS / 1e9:.0f} s, at most {prim.MAX_SLICES}")
    n = len(slices)
    if n == 0:
        return None, "window shorter than one slice"
    total = float(slices.sum())
    busiest = np.sort(slices)[::-1][: max(1, -(-n // 10))]  # the top tenth, rounded up
    return {
        "slice_s": prim.SLICE_NS / 1e9,
        "n": n,
        "with_time": int(np.count_nonzero(slices)),
        "max_slice_s": float(slices.max()) / 1e9,
        # The slices' own total: the final partial slice is not in it.
        "slice_sum_s": total / 1e9,
        "busiest_tenth_share": float(busiest.sum()) / total if total > 0 else None,
    }, None


def _status(row: dict[str, Any]) -> dict[str, Any]:
    """The status fields for one row: ``status``, ``status_reason``, ``status_detail``."""
    state = row["state"]

    def out(status: str, reason: str | None, detail: str | None) -> dict[str, Any]:
        return {"status": status, "status_reason": reason, "status_detail": detail}

    if state == "cannot_tell":
        return out(_NOT_IDENTIFIABLE, row["reason"], row.get("why"))
    complete = row.get("completeness") == "complete"
    note = row.get("completeness_note")
    interval = row.get("interval") or {}
    shared = ", ".join(row.get("shares_with") or ()) or "none"
    if state == "not_fired":
        if row.get("assumptions"):
            return out(_NOT_IDENTIFIABLE, "structural",
                       "below the threshold, but that rests on an unverified assumption: "
                       + ", ".join(row["assumptions"]))
        if not complete:
            return out(_NOT_IDENTIFIABLE, "structural",
                       "below the threshold, but it cannot be ruled out"
                       + (f": {note}" if note else ""))
        if interval.get("high_fires"):
            return out(_NOT_IDENTIFIABLE, "confounded",
                       f"below the threshold by the split used, but its records cover up to "
                       f"{interval['high_s']:.4f} s, which row(s) {shared} can also claim")
        return out("absent", None, f"ruled out at or above {row['fire_at']:g} "
                   f"{row['fire_unit']} in this capture, which reports no dropped record")
    # fired
    if row.get("seconds") is None:
        return out(_NOT_IDENTIFIABLE, "unpriced", row.get("price_reason"))
    if row.get("assumptions"):
        return out(_NOT_IDENTIFIABLE, "structural",
                   "priced, but the time rests on an unverified assumption: "
                   + ", ".join(row["assumptions"]))
    if row.get("completeness") == "lossy":
        return out(_NOT_IDENTIFIABLE, "structural",
                   f"priced, but {note}, so the time could be higher or lower")
    if interval and not interval["low_fires"]:
        return out(_NOT_IDENTIFIABLE, "confounded",
                   f"priced, but only {interval['low_s']:.4f} s is under this row's records "
                   f"alone; row(s) {shared} can claim the rest")
    return out("estimated", None, "the time in this capture; says nothing of other windows")
