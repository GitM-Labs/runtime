"""Kernel identity via NVTX range correlation — see ``docs/kernel_identity.md``.

Attaches an exact ``op``/``layer`` to a kernel dict when the capture has NVTX
range instrumentation, instead of guessing from the mangled kernel name
(:func:`gitm.optimizer.deviation.classify_op`).

The chain uses three record kinds on two different clocks — a kernel's own
timestamps are device-clock and must never be compared directly against a
range's host-clock window (async execution means a kernel routinely finishes
long after its range popped):

    kernel   (device clock)  correlation_id=X            [start_ns, end_ns]
        |  same correlation_id
    runtime  (host clock)    cudaLaunchKernel  id=X       [start_ns, end_ns], thread_id
        |  host-timestamp containment, same thread_id
    marker   (host clock)    NVTX range "L{layer}/{op}"   [start_ns, end_ns], thread_id

Dict contract this module expects (not yet emitted by ``cupti_shim.c`` — the
shim today only emits ``kernel``/``memcpy``/``sync``; ``runtime``/``marker``
are the pending CUPTI activity kinds ``RUNTIME`` and ``MARKER``):

    runtime  {kind:"runtime", correlation_id:int, start_ns:int, end_ns:int,
              thread_id:int}
    marker   {kind:"marker", name:str, start_ns:int, end_ns:int, thread_id:int}

A ``marker`` record here is one fully-resolved push/pop range (start/end
already paired) — folding raw NVTX push/pop activity records into ranges is
shim/decoder work tracked separately in the design doc.
"""

from __future__ import annotations

import re

_RANGE_NAME_RE = re.compile(r"^L(\d+)/(.+)$")


def parse_range_name(name: str) -> tuple[str, int | None]:
    """Parse an NVTX range name into ``(op, layer)``.

    ``"L3/qkv_proj"`` -> ``("qkv_proj", 3)``. A range with no layer prefix
    (e.g. ``"lm_head"``, which runs once, not per-layer) -> ``(name, None)``.
    """
    m = _RANGE_NAME_RE.match(name)
    if m:
        return m.group(2), int(m.group(1))
    return name, None


def correlate_kernels_to_ranges(records: list[dict]) -> list[dict]:
    """Return every ``kind == "kernel"`` dict from ``records``, enriched.

    Each returned dict is a shallow copy of the input kernel dict with
    ``range_op``/``range_layer`` keys added (both ``None`` when no match is
    found — no runtime record for the kernel's ``correlation_id``, or no
    marker range whose host window contains that runtime record). Order is
    preserved. ``runtime``/``marker`` records are consumed only to build the
    correlation index; they are not returned.

    Containment is checked on the *runtime* record's host window against the
    marker's host window, matched on ``thread_id`` — never the kernel's own
    (device-clock) window, and never across threads. When multiple markers
    contain the runtime window (nested ranges), the innermost (smallest span)
    wins.
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
            best_span = None
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
