"""Canonical request-trace shapes — one format every load source normalizes into.

Three contracts:

* :class:`CanonicalRequest` — one request. The unit every adapter emits and every
  output mode consumes. A frozen slotted dataclass, not a pydantic model:
  a trace is millions of these and they are hot data, not configuration. The
  *validation* that would justify pydantic already happens in the adapter, where
  a rejected row can be attributed to a named :class:`DropReason` instead of
  raising.
* :class:`TraceMeta` — provenance for a whole trace: where the bytes came from,
  their sha256, how many rows were read, how many survived, and **what was
  dropped and why**. This is the part that gets serialized into a result row, so
  it is a pydantic model with ``extra="forbid"``.
* :class:`Trace` — meta + requests, and the reason the two cannot be separated.
  Nothing downstream accepts a bare list of requests: a stream you cannot trace
  back to bytes is not evidence, and the playbook (deliverable 4) keys on this
  provenance.

Units, stated once and never re-derived: **``arrival_s`` is seconds offset from
the trace's first row**, not an epoch and not the source's native unit. Adapters
convert; :attr:`TraceMeta.raw_time_unit` records what they converted from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: Schema identity, in the style of ``gitm.bench.manifest.SCHEMA``. Bump on any
#: field change that a consumer could misread as the old meaning.
SCHEMA = "gitm.traffic.trace/v1"


class DropReason(str, Enum):
    """Why an adapter refused a raw row.

    Every rejection is one of these — an adapter may not drop a row silently, and
    the counts travel with the trace in :attr:`TraceMeta.drops`. The first seven
    are *defects* in the source; :attr:`FILTERED_OUT` is a selection the caller
    asked for and is counted separately so a narrow filter never reads as dirty
    data.
    """

    MALFORMED_ROW = "malformed_row"  # unparseable line / wrong column count
    MISSING_FIELD = "missing_field"  # column present in the header, empty in the row
    NON_NUMERIC = "non_numeric"  # a length or timestamp that is not a number
    NEGATIVE_VALUE = "negative_value"  # negative tokens or a negative timestamp
    ZERO_INPUT_TOKENS = "zero_input_tokens"  # nothing to prefill
    ZERO_OUTPUT_TOKENS = "zero_output_tokens"  # nothing to decode
    NON_MONOTONIC_ARRIVAL = "non_monotonic_arrival"  # timestamp went backwards
    FILTERED_OUT = "filtered_out"  # excluded by a caller-supplied filter, not a defect


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    """One request, in the only shape the harness fires.

    ============== ============================= =========================================
    field          type / unit                   when the source lacks it
    ============== ============================= =========================================
    arrival_s      float, **seconds from trace   fatal. No adapter may synthesize arrivals;
                   start** (not epoch)           a source without timing is not a trace.
    input_tokens   int, tokens, ``> 0``          fatal — the row is dropped, never defaulted.
    output_tokens  int | None, tokens            ``None`` = "generate to the regime's sampled
                                                 length". Recorded in ``TraceMeta.notes``;
                                                 a trace of ``None`` cannot be replayed
                                                 as-is, only parameterized.
    session_id     str | None                    ``None`` — no conversation identity.
                                                 ``TraceMeta.has_session_identity`` is False
                                                 and multi-turn experiments must not use it.
    prefix_blocks  tuple[int, ...], block ids    ``()`` — no prefix identity.
                   in prompt order               ``TraceMeta.has_prefix_identity`` is False
                                                 and prefix-cache experiments must not use it.
    ============== ============================= =========================================

    ``prefix_blocks`` is a *chain*, not a single hash, because partial sharing is
    the whole point: two requests share a prefix exactly as far as their leading
    block ids agree. One hash of the whole chain would only ever match identical
    prompts, which is the case that does not need measuring. Each id stands for
    :attr:`TraceMeta.prefix_block_tokens` tokens.

    ``source_e2e_latency_s`` is the **source system's** end-to-end
    submission-to-final-response time, when the trace records one (BurstGPT_3's
    ``Elapsed time``). It is carried so the adapter does not destroy real data at
    the boundary, and it is named at length so it cannot be mistaken for
    something it is not:

    * it is **end-to-end**, not TTFT, and not ITL;
    * it was measured on **someone else's hardware, model and load**.

    **Never compare it against a measured TTFT/ITL, and never promote a playbook
    row against it.** Its legitimate use is bounding think-time between turns of
    a session, where only the source's own timeline matters.
    """

    arrival_s: float
    input_tokens: int
    output_tokens: int | None = None
    session_id: str | None = None
    prefix_blocks: tuple[int, ...] = ()
    source_e2e_latency_s: float | None = None


class TraceMeta(BaseModel):
    """Provenance for one trace. Without it a trace cannot be replayed.

    ``rows_read`` counts raw records seen; ``rows_emitted`` counts survivors;
    ``drops`` maps :class:`DropReason` values to counts. The three must reconcile
    (:meth:`Trace.__post_init__` checks it), so "we dropped some bad rows" is
    never a hand-wave.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: str = SCHEMA
    source: str  # adapter name, e.g. "burstgpt"
    path: str  # the file that was read
    sha256: str  # of the raw bytes — pins the trace to bytes, per gitm.bench.manifest
    source_url: str | None = None  # where the raw file came from, when known

    rows_read: int = 0
    rows_emitted: int = 0
    drops: dict[str, int] = Field(default_factory=dict)

    span_s: float = 0.0  # last arrival minus first, seconds
    raw_time_unit: str = "s"  # what the source's timestamps were before conversion

    prefix_block_tokens: int | None = None  # tokens each prefix-block id stands for
    has_prefix_identity: bool = False
    has_session_identity: bool = False
    #: How much session identity there actually is. ``has_session_identity`` only
    #: says *some* row carried one; in BurstGPT_3 that is true while 90 % of rows
    #: are single-shot API traffic with no conversation at all. A multi-turn
    #: experiment needs the counts, not the flag, to decide whether the trace can
    #: carry it.
    session_rows: int = 0
    sessions: int = 0

    notes: list[str] = Field(default_factory=list)

    @property
    def dropped(self) -> int:
        return sum(self.drops.values())

    @property
    def defects(self) -> int:
        """Dropped rows that were *bad data*, excluding caller-requested filtering."""
        return sum(v for k, v in self.drops.items() if k != DropReason.FILTERED_OUT.value)

    def summary(self) -> str:
        drops = ", ".join(f"{k}={v}" for k, v in sorted(self.drops.items())) or "none"
        return (
            f"{self.source}: {self.rows_emitted}/{self.rows_read} rows over "
            f"{self.span_s:.1f}s (drops: {drops})"
        )


@dataclass(frozen=True)
class Trace:
    """A normalized trace: provenance plus requests, inseparable by construction.

    ``requests`` are held in arrival order. The whole trace is materialized —
    ponytail: fine to a few million rows (BurstGPT_1 is ~1.4 M), and the memory
    ceiling is `slots` dataclasses at roughly 100 B each. If a source arrives that
    does not fit, make the adapters yield and give ``Trace`` a streaming sibling;
    nothing above this line assumes random access except the quantile fits.
    """

    meta: TraceMeta
    requests: list[CanonicalRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.meta.rows_emitted != len(self.requests):
            raise ValueError(
                f"TraceMeta.rows_emitted={self.meta.rows_emitted} disagrees with "
                f"{len(self.requests)} requests — provenance must reconcile"
            )
        if self.meta.rows_read != self.meta.rows_emitted + self.meta.dropped:
            raise ValueError(
                f"rows_read={self.meta.rows_read} != emitted={self.meta.rows_emitted} "
                f"+ dropped={self.meta.dropped} — a row went missing unattributed"
            )

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def arrivals(self) -> list[float]:
        return [r.arrival_s for r in self.requests]

    @property
    def input_tokens(self) -> list[int]:
        return [r.input_tokens for r in self.requests]

    @property
    def output_tokens(self) -> list[int]:
        """Output lengths, with ``None`` excluded — callers must check the count."""
        return [r.output_tokens for r in self.requests if r.output_tokens is not None]

    def rate_rps(self) -> float:
        return len(self.requests) / self.meta.span_s if self.meta.span_s > 0 else 0.0
