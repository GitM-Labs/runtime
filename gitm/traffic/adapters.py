"""Source adapters: raw trace file in, :class:`~gitm.traffic.schema.Trace` out.

Two sources for v1. Both formats were read off the real published files, not off
a paper:

* **BurstGPT** — real Azure OpenAI traffic, the burstiness reference.
  ``Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type``,
  timestamps in **seconds**. No prefix or session identity.
  https://github.com/HPMLL/BurstGPT
* **Mooncake** — Kimi production serving traces. JSONL,
  ``{"timestamp", "input_length", "output_length", "hash_ids"}``, timestamps in
  **milliseconds**, ``hash_ids`` are **512-token** cache blocks (vLLM's own
  ``--timed-trace-chunk-hash-size`` help names 512 for the Moonshot traces; on
  the published slice ``len(hash_ids) * 512`` covers ``input_length`` exactly,
  never over-covering by a whole block).
  https://github.com/kvcache-ai/Mooncake

Both adapters share the same discipline: **nothing is defaulted, everything
rejected is counted**. Real data is the work here — 7.9 % of the BurstGPT rows
carry zero input *and* zero output tokens, and a loader that quietly kept them
would put empty prefills into every regime fit.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from gitm.bench.manifest import sha256_file
from gitm.traffic.schema import CanonicalRequest, DropReason, Trace, TraceMeta

#: Tokens per ``hash_ids`` entry in the published Mooncake traces.
MOONCAKE_BLOCK_TOKENS = 512

#: The six columns every BurstGPT release has carried. Required.
_BURSTGPT_COLUMNS = (
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
)

#: Columns BurstGPT_3 (release v2.0) adds, at positions 1 and 2 — inserted, not
#: appended, which is why the reader goes by column *name* and never by index.
#: ``BurstGPT_without_fails_3.csv`` carries the same eight.
_BURSTGPT_SESSION = "Session ID"
_BURSTGPT_ELAPSED = "Elapsed time"
_BURSTGPT_OPTIONAL = (_BURSTGPT_SESSION, _BURSTGPT_ELAPSED)


class _Collector:
    """Accumulates survivors and drop counts so an adapter never loses a row.

    ``last_ts`` tracks the newest *parsed* raw timestamp, which is what the
    monotonicity check compares against: a row that was dropped for some other
    defect still tells us where the file's clock had reached.
    """

    def __init__(self) -> None:
        self.requests: list[CanonicalRequest] = []
        self.drops: Counter[str] = Counter()
        self.rows_read = 0
        self.last_ts: float | None = None

    def drop(self, reason: DropReason) -> None:
        self.drops[reason.value] += 1

    def check_lengths(self, inp: int, out: int | None) -> DropReason | None:
        """Shared length validation — every adapter routes through here.

        One place, so a new source cannot invent a different definition of
        "unusable row" and quietly widen the envelope.
        """
        if inp < 0 or (out is not None and out < 0):
            return DropReason.NEGATIVE_VALUE
        if inp == 0:
            return DropReason.ZERO_INPUT_TOKENS
        if out == 0:
            return DropReason.ZERO_OUTPUT_TOKENS
        return None

    def check_monotonic(self, ts: float) -> DropReason | None:
        if self.last_ts is not None and ts < self.last_ts:
            return DropReason.NON_MONOTONIC_ARRIVAL
        return None


def _finish(
    coll: _Collector,
    *,
    source: str,
    path: Path,
    source_url: str | None,
    raw_time_unit: str,
    prefix_block_tokens: int | None,
    has_prefix_identity: bool,
    has_session_identity: bool,
    notes: list[str],
    session_rows: int = 0,
    sessions: int = 0,
) -> Trace:
    arrivals = [r.arrival_s for r in coll.requests]
    span = (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
    digest, nbytes = sha256_file(path)
    if any(r.output_tokens is None for r in coll.requests):
        notes = [*notes, "some requests carry no output length; replay-as-is is not valid"]
    meta = TraceMeta(
        source=source,
        path=str(path),
        sha256=digest,
        source_url=source_url,
        rows_read=coll.rows_read,
        rows_emitted=len(coll.requests),
        drops=dict(coll.drops),
        span_s=span,
        raw_time_unit=raw_time_unit,
        prefix_block_tokens=prefix_block_tokens,
        has_prefix_identity=has_prefix_identity,
        has_session_identity=has_session_identity,
        session_rows=session_rows,
        sessions=sessions,
        notes=[*notes, f"raw file {nbytes} bytes"],
    )
    return Trace(meta=meta, requests=coll.requests)


def read_burstgpt(
    path: str | Path,
    *,
    model: str | None = None,
    log_type: str | None = None,
    max_rows: int | None = None,
    source_url: str | None = None,
) -> Trace:
    """Read a BurstGPT CSV into canonical form. Handles every published layout.

    Two layouts exist. ``BurstGPT_1`` / ``_2`` carry six columns;
    **``BurstGPT_3`` (release v2.0) carries eight**, inserting ``Session ID`` and
    ``Elapsed time`` at positions 1 and 2 — *inserted*, not appended. So the
    reader goes by column **name**: the six core columns are required, the two
    extras are used when present, and an unrecognized extra column is recorded in
    ``TraceMeta.notes`` rather than rejected. A future ``BurstGPT_4`` that adds a
    column will load rather than raise.

    ``Session ID`` is populated **only for ``Conversation log`` rows** — in the
    published v3 file every ``API log`` row has it empty, and those are 90 % of
    the trace. An empty session id is therefore *by design and not a defect*:
    the row is emitted with ``session_id=None``. Dropping them would discard most
    of a real v3 trace. ``TraceMeta.session_rows`` / ``.sessions`` report how much
    conversation identity actually survived, which is what a multi-turn
    experiment must check — the boolean flag alone would say "yes" on a trace
    that is 90 % single-shot.

    ``Elapsed time`` becomes ``CanonicalRequest.source_e2e_latency_s``. Read that
    field's docstring before using it: it is end-to-end latency on the *source*
    system, not TTFT, and not ours. An unparseable value is treated as absent
    (the row survives; the count lands in ``TraceMeta.notes``), because an
    optional annotation being junk is no reason to throw away a valid request.

    ``model`` (``"ChatGPT"`` / ``"GPT-4"``) and ``log_type`` (``"Conversation
    log"`` / ``"API log"``) select a subset; excluded rows count as
    ``FILTERED_OUT``, never as defects.

    The arrival clock is anchored on the **first row read**, before any filtering,
    so narrowing the selection shifts which requests appear but never shifts when
    they appear.
    """
    path = Path(path)
    coll = _Collector()
    t0: float | None = None
    sessions: set[str] = set()
    session_rows = 0
    bad_elapsed = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file, not a BurstGPT CSV")
        columns = [h.strip() for h in header]
        missing = [c for c in _BURSTGPT_COLUMNS if c not in columns]
        if missing:
            raise ValueError(
                f"{path}: not a BurstGPT CSV — missing column(s) {missing}; "
                f"header was {columns!r}"
            )
        idx = {name: i for i, name in enumerate(columns)}
        has_session = _BURSTGPT_SESSION in idx
        has_elapsed = _BURSTGPT_ELAPSED in idx
        unknown = [c for c in columns if c not in _BURSTGPT_COLUMNS + _BURSTGPT_OPTIONAL]

        for row in reader:
            if max_rows is not None and coll.rows_read >= max_rows:
                break
            coll.rows_read += 1

            if len(row) != len(columns):
                coll.drop(DropReason.MALFORMED_ROW)
                continue
            cells = [c.strip() for c in row]
            ts_raw = cells[idx["Timestamp"]]
            row_model = cells[idx["Model"]]
            inp_raw = cells[idx["Request tokens"]]
            out_raw = cells[idx["Response tokens"]]
            row_log = cells[idx["Log Type"]]

            # The timestamp is parsed first and alone. A row whose *lengths* are
            # junk still says where the file's clock had reached, and folding both
            # parses into one try lets a bad length hide a backwards jump from the
            # monotonicity check entirely.
            if not ts_raw:
                coll.drop(DropReason.MISSING_FIELD)
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                coll.drop(DropReason.NON_NUMERIC)
                continue
            if t0 is None:
                t0 = ts
            reason = coll.check_monotonic(ts)
            coll.last_ts = ts if coll.last_ts is None else max(coll.last_ts, ts)
            if reason is not None:
                coll.drop(reason)
                continue

            if not inp_raw or not out_raw:
                coll.drop(DropReason.MISSING_FIELD)
                continue
            try:
                inp = int(inp_raw)
                out = int(out_raw)
            except ValueError:
                coll.drop(DropReason.NON_NUMERIC)
                continue
            if (model is not None and row_model != model) or (
                log_type is not None and row_log != log_type
            ):
                coll.drop(DropReason.FILTERED_OUT)
                continue
            reason = coll.check_lengths(inp, out)
            if reason is not None:
                coll.drop(reason)
                continue

            # Optional columns. Absent or blank is normal, never a drop: v3
            # leaves Session ID empty on every API-log row by design.
            session_id = cells[idx[_BURSTGPT_SESSION]] if has_session else ""
            if session_id:
                sessions.add(session_id)
                session_rows += 1
            elapsed: float | None = None
            if has_elapsed and cells[idx[_BURSTGPT_ELAPSED]]:
                try:
                    elapsed = float(cells[idx[_BURSTGPT_ELAPSED]])
                except ValueError:
                    bad_elapsed += 1

            coll.requests.append(
                CanonicalRequest(
                    arrival_s=ts - t0,
                    input_tokens=inp,
                    output_tokens=out,
                    session_id=session_id or None,
                    source_e2e_latency_s=elapsed,
                )
            )

    notes = [f"columns: {', '.join(columns)}"]
    if session_rows:
        notes.append(
            f"session identity on {session_rows}/{len(coll.requests)} emitted rows "
            f"({len(sessions)} sessions) — the rest are single-shot API traffic"
        )
    else:
        notes.append("no session identity in this layout (BurstGPT_1/_2)")
    notes.append("BurstGPT carries no prefix identity in any layout")
    if bad_elapsed:
        notes.append(f"{bad_elapsed} unparseable '{_BURSTGPT_ELAPSED}' values read as absent")
    if unknown:
        notes.append(f"unrecognized columns ignored: {', '.join(unknown)}")
    if model or log_type:
        notes.append(f"filtered: model={model!r} log_type={log_type!r}")
    return _finish(
        coll,
        source="burstgpt",
        path=path,
        source_url=source_url,
        raw_time_unit="s",
        prefix_block_tokens=None,
        has_prefix_identity=False,
        has_session_identity=bool(session_rows),
        session_rows=session_rows,
        sessions=len(sessions),
        notes=notes,
    )


def read_mooncake(
    path: str | Path,
    *,
    block_tokens: int = MOONCAKE_BLOCK_TOKENS,
    max_rows: int | None = None,
    source_url: str | None = None,
    time_scale: float = 0.001,
    source: str = "mooncake",
) -> Trace:
    """Read a Mooncake JSONL trace into canonical form.

    ``block_tokens`` is how many tokens one ``hash_ids`` entry stands for. It is
    **512 for the published Moonshot traces** and getting it wrong is silent: the
    replay path expands each block to that many tokens, so a wrong value produces
    prompts that are a clean multiple too short while every count still looks
    right. :func:`gitm.traffic.replay.write_timed_trace` checks the coverage and
    refuses rather than truncating.

    ``time_scale`` converts the source's timestamps to seconds (0.001 for
    Mooncake's milliseconds). :func:`gitm.traffic.replay.read_timed_trace` reuses
    this reader at ``time_scale=1.0`` — the emitted replay file is the same shape,
    so re-parsing it needs one parameter, not a second parser that can drift.
    """
    path = Path(path)
    coll = _Collector()
    t0: float | None = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if max_rows is not None and coll.rows_read >= max_rows:
                break
            coll.rows_read += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                coll.drop(DropReason.MALFORMED_ROW)
                continue
            if not isinstance(rec, dict):
                coll.drop(DropReason.MALFORMED_ROW)
                continue
            # Timestamp first and alone — see the note in read_burstgpt.
            if "timestamp" not in rec:
                coll.drop(DropReason.MISSING_FIELD)
                continue
            try:
                ts_raw = float(rec["timestamp"])
            except (TypeError, ValueError):
                coll.drop(DropReason.NON_NUMERIC)
                continue
            if t0 is None:
                t0 = ts_raw
            reason = coll.check_monotonic(ts_raw)
            coll.last_ts = ts_raw if coll.last_ts is None else max(coll.last_ts, ts_raw)
            if reason is not None:
                coll.drop(reason)
                continue

            if not {"input_length", "output_length"} <= rec.keys():
                coll.drop(DropReason.MISSING_FIELD)
                continue
            try:
                inp = int(rec["input_length"])
                out = int(rec["output_length"])
            except (TypeError, ValueError):
                coll.drop(DropReason.NON_NUMERIC)
                continue
            reason = coll.check_lengths(inp, out)
            if reason is not None:
                coll.drop(reason)
                continue

            blocks = rec.get("hash_ids") or []
            if not isinstance(blocks, list) or any(not isinstance(b, int) for b in blocks):
                coll.drop(DropReason.MALFORMED_ROW)
                continue

            coll.requests.append(
                CanonicalRequest(
                    arrival_s=(ts_raw - t0) * time_scale,
                    input_tokens=inp,
                    output_tokens=out,
                    prefix_blocks=tuple(blocks),
                )
            )

    has_prefix = any(r.prefix_blocks for r in coll.requests)
    return _finish(
        coll,
        source=source,
        path=path,
        source_url=source_url,
        raw_time_unit="ms" if time_scale == 0.001 else "s",
        prefix_block_tokens=block_tokens if has_prefix else None,
        has_prefix_identity=has_prefix,
        has_session_identity=False,
        notes=[f"hash_ids read as {block_tokens}-token cache blocks"],
    )


#: Adapter registry — name to reader. Keeps the CLI and the selftest from
#: growing an if-chain per source.
ADAPTERS = {"burstgpt": read_burstgpt, "mooncake": read_mooncake}
