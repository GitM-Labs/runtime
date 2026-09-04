# Traffic replay v1 — what is frozen, and why

Deliverable 1 of the validation infrastructure. The load generator for every
experiment: real production traces normalized into one format, fired through the
harness, and tagged with the workload regime every result row is keyed on.

**Library:** `gitm/traffic/`. **CPU-only** — nothing here needs a GPU, and only
the final firing needs a server.

```bash
python -m gitm.traffic --selftest                                  # the check
python -m gitm.traffic --describe burstgpt <BurstGPT_1.csv>        # meta + regime
python -m gitm.traffic --replay  mooncake <trace.jsonl> --out replay.jsonl \
                                 --model Qwen/Qwen3.6-35B-A3B-FP8  # + validation
python -m gitm.traffic --sweep   burstgpt <BurstGPT_1.csv>         # the grid
```

## 1. Sources

| source | format | timestamps | session | prefix identity | fixture |
|---|---|---|---|---|---|
| **BurstGPT_1/_2** | CSV, 6 columns: `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type` | **seconds** | none | none | `fixtures/burstgpt_slice.csv` |
| **BurstGPT_3** | CSV, 8 columns: `Timestamp,`**`Session ID,Elapsed time,`**`Model,Request tokens,Response tokens,Total tokens,Log Type` | **seconds** | `Session ID` (UUID) | none | `fixtures/burstgpt3_slice.csv` |
| **Mooncake** | JSONL, `{timestamp, input_length, output_length, hash_ids}` | **milliseconds** | none | `hash_ids`, **512-token** blocks | `fixtures/mooncake_slice.jsonl` |

All three formats were read off the real published files, not off a paper.
Sources: `HPMLL/BurstGPT` — `data/BurstGPT_1.csv` on `main`, and
`BurstGPT_3.csv` from **release v2.0** (the `_3` files exist only there; `main`
carries `_1` alone) — and `kvcache-ai/Mooncake`
(`FAST25-release/traces/conversation_trace.jsonl`).

**BurstGPT_3 inserts its two columns at positions 1 and 2 — it does not append
them.** A positional reader does not merely miss them; it reads `Session ID` as
the model and `Elapsed time` as the request length. `read_burstgpt` therefore
goes by column **name**: the six core columns are required, `Session ID` and
`Elapsed time` are used when present, and an unrecognized extra column is
recorded in `TraceMeta.notes` rather than rejected — so a future `BurstGPT_4`
loads instead of raising. `BurstGPT_without_fails_3.csv` carries the same eight.

Real fixtures are the **first 400 rows of each real file**, unmodified, plus
three hand-authored dirty files that exercise every drop reason. All six are
pinned by sha256 in `manifest.yaml` (`gitm.bench.manifest/v1`, the same contract
the other benchmarks use).

## 2. The canonical schema

`gitm.traffic.schema`. Per request: `arrival_s` (**seconds offset from trace
start**, never an epoch), `input_tokens`, `output_tokens`, `session_id`,
`prefix_blocks`. Absent-from-source behaviour is documented per field in the
docstring and is *never* a silent default — a row that cannot supply a required
field is dropped under a named reason.

`prefix_blocks` is a **chain**, not one hash: two requests share a prefix exactly
as far as their leading block ids agree, and a single digest of the whole chain
would only match identical prompts — the case that does not need measuring.

`TraceMeta` carries the raw file's sha256, `rows_read`, `rows_emitted` and the
per-reason drop counts. **`Trace` refuses to exist unless those reconcile**
(`rows_read == rows_emitted + dropped`), so "we dropped some bad rows" can never
be a hand-wave.

## 3. What the real data actually contains

Pinned in `gitm/traffic/_selftest.py`, measured not assumed:

| | BurstGPT_1 slice | BurstGPT_3 slice | Mooncake slice |
|---|---|---|---|
| rows read / emitted | 400 / **383** | 400 / **399** | 400 / 400 |
| drops | `zero_input_tokens=17` | `zero_input_tokens=1` | none |
| span | 37,269 s | 41,453 s | 141.0 s |
| session rows / sessions | — | **393 / 134** | — |
| regime label | `prod/io1/in256/out128/burst-poisson/copen` | `prod/io2/in256/out64/burst-poisson/copen` | `prod/io32/in8k/out256/burst-hi/copen` |
| input p50 / p95 | 353 / 1,638 | — | 9,075 / 49,904 |
| output p50 / p95 | 238 / 841 | — | 370 / 662 |
| burstiness (D @ 1 s) | 1.01 | — | 6.74 |

### Session identity is partial by design, so it is counted, not flagged

In the full `BurstGPT_3` file, `Session ID` is populated on **exactly** the
`Conversation log` rows and empty on **exactly** the `API log` rows — 528 with,
5,115 without, in the first 5,643. So **an empty `Session ID` is normal data, not
a defect**: the row is emitted with `session_id=None`. Dropping them would throw
away 90 % of a real v3 trace.

That is also why `TraceMeta` carries `session_rows` and `sessions` and not just
`has_session_identity`. The boolean says "yes" on a trace that is 90 % single-shot;
a multi-turn experiment needs the counts to decide whether the trace can carry it.
In the committed slice: 393 of 399 emitted rows across **134 conversations**,
longest 24 turns.

`Elapsed time` becomes `CanonicalRequest.source_e2e_latency_s` — deliberately
long-named. It is the **source system's end-to-end submission-to-final-response
time on OpenAI's hardware**: not TTFT, not ITL, and not ours. It must never be
compared against a measured latency or used to promote a playbook row. It is
carried rather than discarded because losing real data at the adapter boundary is
unrecoverable; its one legitimate use is bounding think-time between turns of a
session. An unparseable value is read as *absent* (the row survives, the count
lands in `TraceMeta.notes`) — an optional annotation being junk is no reason to
throw away a valid request.

**7.9 % of the full BurstGPT file carries zero input *and* zero output tokens**
(4.3 % of this slice). A loader that kept them would put empty prefills into
every regime fit. This is the "handle real data" requirement firing on the first
source, not a hypothetical.

## 4. Regime axes

`gitm.traffic.regime`. Prefill/decode token ratio, input and output length
distributions, arrival burstiness, concurrency — plus `source_kind`.

Burstiness is the **index of dispersion** (variance/mean of arrival counts per
1 s bin), not the CV of interarrival times. Both are 1 for a Poisson process, but
two traces with the same mean rate and different bunching collapse to one CV, and
bunching is the axis the customer's traffic actually varies on.

`source_kind ∈ {production, synthetic, scoreboard}` is a schema field, not a
naming convention. **Artificial Analysis's fixed-length workload is a
`scoreboard` regime** and its label starts `board/`, so it can never be read as
production traffic after being copied into a spreadsheet.

`Regime.label()` is bucketed on purpose — two runs of the same workload must
produce the same label, and raw quantiles never repeat. Out-of-envelope sampled
points are suffixed `/xenv`.

## 5. Replay mode — no custom load generator

vLLM's bench-serve has a native `timed_trace` dataset that consumes exactly
`{timestamp, input_length, output_length, hash_ids}` JSONL and, under
`--self-timed`, schedules **each request at its own timestamp**. That is faithful
replay, already written and maintained, so `gitm.traffic.replay` writes that file
and builds that command line. Nothing here fires traffic itself.

Two silent failures this module exists to make loud:

1. **Block coverage.** `timed_trace` expands each `hash_ids` entry to
   `--timed-trace-chunk-hash-size` tokens and *stops when the ids run out*. Pass
   vLLM's default of 16 against Mooncake's 512-token blocks and every prompt is
   32× short while every count still looks right. `write_timed_trace` checks
   `len(blocks) * block_tokens >= input_tokens` per request and **refuses the
   whole file** rather than truncating.
2. **Sources with no prefix identity.** BurstGPT has no `hash_ids`; an empty list
   produces a *zero-length* prompt. Blocks are therefore **synthesized** — a
   fresh, globally unique run per request, so lengths are honoured and **no
   prefix sharing is invented that the source never had**. `ReplayPlan`
   records `prefix_synthesized=True`, and a prefix-cache experiment must reject
   such a plan.

## 6. Parameterized mode

`gitm.traffic.parameterize`. Fits each trace along the regime axes, then samples
the grid — including points beyond any single trace, because the customer's next
hour is never the trace's next hour.

- Length distributions are fitted **empirically** (101-point quantile grid,
  inverse-CDF sampling). No parametric family: production length distributions
  are multi-modal — short chat turns and long document prompts in one trace — and
  a lognormal fit would smear the modes and quietly move the prefill/decode ratio
  the whole exercise turns on.
- Arrival burstiness is generated by drawing **per-bin counts from a negative
  binomial**, whose index of dispersion is `1 + m/r` and so can be set directly to
  the target. Poisson is the `D = 1` case. `D < 1` clamps to Poisson and says so.
- Every point outside the fitted envelope is `in_envelope=False` and labelled
  `/xenv`. An extrapolation that cannot be told from a measurement is worse than
  no extrapolation.
- A synthetic trace's identity is the **sha256 of its generating parameters** —
  same digest, same trace, reproducible without storing the file.

## 7. Validation — the deliverable, shown

`gitm.traffic.validate` compares the replayed stream against the source on
arrival timing, both length distributions, mean rate and burstiness, and renders
the check table beside both arrival-rate profiles. Any mismatch is **explained in
prose**, not just printed.

What is compared is the **file bench-serve will actually read**: the emitter
writes it, `read_timed_trace` reads it back, and `compare` puts it beside the
adapter's output. So "the pipeline preserves the trace" is a measurement of the
artifact, not an argument about the code.

Two standards, named rather than implied:

| | `REPLAY_THRESHOLDS` | `SAMPLED_THRESHOLDS` |
|---|---|---|
| KS (arrival, lengths) | 0.001 | 0.15 |
| mean rate | 1e-6 | 0.25 |
| burstiness | 1e-6 | 0.60 |
| request count | exact | 0.35 |
| arrival timeline compared | **yes** | **no** |

A replay must *reproduce* the trace — the emitter is a format change, so every
statistic comes back identical. A parameterized sample is a *draw* from the
envelope: it reproduces the rate and the dispersion by construction and the
timeline by nothing, so comparing timelines there would only measure that a
sample is not a copy.

Arrival times are compared at **microsecond resolution**
(`ARRIVAL_RESOLUTION_S`). Sub-microsecond replay fidelity is meaningless next to
millisecond network jitter, and without the quantization the KS reports the
6e-16 s residue of `5999 / 1000` as a real distribution gap — which it did, on
the Mooncake fixture, before this was added.

Measured, on the committed fixtures: **every replay check returns exactly 0.0**
for both adapters.

## 8. The check

```bash
python -m gitm.traffic --selftest        # 10 checks, 2 real traces, 7 drop reasons
python -m pytest tests/test_traffic.py -q   # the same assertions, as pytest cases
```

The assertions live in `gitm/traffic/_selftest.py` and both entry points call
them, so the runnable check named here and the one CI runs cannot drift apart.

## 9. Not in v1, and why

- **Adapters beyond these two.** Azure LLM inference, the Prism provider traces,
  TraceLab, ShareGPT/LMSYS lengths — "later" in the brief. Two adapters are what
  proves the canonical schema is real; the third would only prove it again.
- **Session-aware *firing*.** The adapter is session-aware; the replay path
  cannot be. vLLM's `timed_trace` format has **no session field**, so conversation
  identity stops at the emitter — pinned by
  `check_session_trace_replay_understates_reuse`, not left as folklore. Sessions
  are available for analysis and regime characterization today. Making them flow
  would mean deriving prefix blocks from session membership, i.e. asserting how
  much each turn re-sends: an *invented* cache hit, which is the one thing this
  module will not do. Consequence, stated on every such plan: prefix reuse on a
  BurstGPT_3 replay is **understated, never overstated** — a floor, not an
  estimate.
- **A streaming `Trace`.** The whole trace is materialized; fine to a few million
  rows (BurstGPT_1 is ~1.4 M). Named ceiling, not an oversight.
- **Firing at a real endpoint.** vLLM is not installed on the authoring box. The
  emitted file and the argv are built against vLLM's current `timed_trace`
  contract, read from source; the first run against a live server is the
  outstanding confirmation.
- **Concurrency as a fitted axis.** It is an offered-load *setting*, not a
  property of a trace: it is carried on `Regime` and set by the caller.
