# D1 - Traffic Replay Experiment

Library map (`runtime/gitm/traffic/`):

| Module                                          | Role                                                           |
| ----------------------------------------------- | -------------------------------------------------------------- |
| `schema.py`                                   | `CanonicalRequest`, `TraceMeta`, `Trace`, `DropReason` |
| `adapters.py`                                 | `read_burstgpt`, `read_mooncake`                           |
| `regime.py`                                   | `Regime.from_trace`, `index_of_dispersion`, buckets        |
| `parameterize.py`                             | `fit`, `sample_trace`, `grid`                            |
| `replay.py`                                   | `write_timed_trace`, `ReplayPlan.bench_serve_argv`         |
| `validate.py`                                 | `compare`, `REPLAY_THRESHOLDS` / `SAMPLED_THRESHOLDS`    |
| `results.py`                                  | seam 3`join_result`, `BenchRun`                            |
| `runner.py`                                   | `check_vllm`, `run_replay`, `RunResult`                  |
| `gui.py` / `__main__.py` / `_selftest.py` | viewer, CLI, pinned checks                                     |

---

## What D1 is for

### What makes D1 a controlled experiment

Hold the **workload fixed**; change only the treatment (engine config / knobs / model). Control comes from:

1. **Pinned source** — raw file `sha256`, adapter drops, kept request set
2. **Fixed shape** — same arrivals, input/output lengths, (when real) prefix blocks
3. **Known plan** — request count, total input tokens, span, `chunk_hash_size=512`
4. **Reconcile after fire** — completed / failed / tokens / duration must match the plan

A throughput or TTFT delta is then attributable to the treatment, not to accidentally sending a different load.

### What is getting replayed (and what is not)

**Replayed:** the *shape* of production traffic from public traces — not the original prompts.

| Replayed                                               | Not replayed                                        |
| ------------------------------------------------------ | --------------------------------------------------- |
| When each request arrives                              | User text / real token IDs                          |
| How long each input/output is (source-reported counts) | Guaranteed same tokenizer as the target model       |
| Mooncake: shared prefix-block identity                 | BurstGPT: real prefix sharing (ids are synthesized) |
| Session IDs for analysis (BurstGPT_3)                  | Session-aware firing through`timed_trace`         |

Mechanically: D1 emits `timed_trace` JSONL; vLLM `bench serve --self-timed` fires synthetic pre-tokenized prompts sized from `hash_ids × 512` on that schedule.

**Why that shape:** fixed-length / constant-rate benches miss the regimes that matter (bursty short chat vs long prefill + prefix reuse). Replaying these traces lets D2/D4 say “this knob helped under *this* traffic,” with provenance.

**D1 role (short):** raw public traces → normalized requests + provenance → controlled replay → measured result under realistic arrival/length (and optionally prefix) shape.

```text
raw trace → normalized requests → controlled replay → measured result
```

---

## Pipeline (first → last)

```text
0. purpose / controlled experiment
1. raw CSV/JSONL
2. adapter → CanonicalRequest[] + TraceMeta + DropReason
3. Regime.from_trace() / Regime.label()
4. optional: fit envelope → sample grid (parameterize, incl. /xenv)
5. write timed_trace JSONL + ReplayPlan
6. round-trip / validate.compare                         (seam 1)
7. vLLM bench serve --self-timed                         (seam 2)
8. join_result → BenchRun (reconcile + regime)           (seam 3)
9. D2 / D3 / D4 consume plan + regime + results
10. CLI / GUI / selftests (entry points)
```

---

## Seams (1–3)

A **seam** is a boundary where *our* code hands off to *vLLM* (or reads something back). Each one can fail silently, so D1 tests them explicitly.

| Seam | Name      | Boundary                                     | Status (as verified)             |
| ---- | --------- | -------------------------------------------- | -------------------------------- |
| 1    | format    | our JSONL ↔`timed_trace` reader           | closed — round-trip exactly 0.0 |
| 2    | execution | argv we build ↔ live`vllm bench serve`    | closed — session 8, vLLM 0.28.0 |
| 3    | results   | bench-serve result JSON ↔ our plan + regime | closed — session 9              |

### Seam 1 — format (round-trip, no vLLM process)

**Boundary:** our emitter ↔ the file `timed_trace` would read.

**What it does:** `write_timed_trace` → read JSONL back → compare to the plan:

- request count
- total input tokens
- span (last − first timestamp)
- plan records `chunk_hash_size=512`

**Pass:** `all exact (0.0)` — the artifact is lossless and speaks vLLM’s field names (`timestamp`, `input_length`, `output_length`, `hash_ids`).

No server runs. This only proves we wrote the right file. Detail: Step 6.

### Seam 2 — execution (real fire)

**Boundary:** the argv we build ↔ a live `vllm bench serve`.

**What it does:** launch bench-serve with `--dataset-name timed_trace`, `--self-timed`, `--timed-trace-chunk-hash-size 512`, etc., against a real endpoint.

**Pass:** the command actually runs (flags exist, dataset accepted, pacing happens). Verified on vLLM 0.28.0: span paced to ~12.0 s against a 12.000 s plan (independent server clock ~11.996 s across POSTs) — not “dump everything instantly.” Detail: Step 7.

### Seam 3 — results (reconcile + join regime)

**Boundary:** bench-serve’s result JSON ↔ our plan + regime.

**What it does:** `results.join_result` checks transport / workload-shape (completed, failures, input tokens, paced span), drops misleading `request_rate` / `burstiness` under `--self-timed`, keep-lists metrics via `KEPT_METRICS`, and **joins the regime** onto the result. `BenchRun.promotable` requires reconcile + zero failures. Detail: Step 8.

### How they chain

```text
emit file  →  [seam 1] file == plan
fire argv  →  [seam 2] vLLM actually runs that file on schedule
read JSON  →  [seam 3] metrics match plan; regime attached
```

- Seam 1 without 2/3 = “format looks right.”
- Seam 2 without 3 = “it ran.”
- Seam 3 = “the number is about the workload we intended.”

---

## Notes carried forward

- BurstGPT is CSV; Mooncake is JSONL.
- BurstGPT is the burstiness reference; Mooncake supplies prefix-cache relationships.
- Written before D4 existed (superseded 2026-09-02, session 4). D4 now lives at `runtime/gitm/playbook/` with `benchmarks/playbook/examples.json`. Built schema differs from early examples: separate `model` / `model_revision`, delta key `delta` (not `measured_delta`), `regime` is the whole `Regime` object (not a pipe-joined label), and every row has `evidence` (`measured` → `illustrative`).
- Shared config-capture schema still missing (risk R1). D4’s `EnvCapture` is a named subset marked `pending-config-capture` until real types land.

---

# Step 1 — Raw traces

## What the raw files contain vs what D1 computes

| Source         | Per-request fields used                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| BurstGPT CSV   | `Timestamp`, `Request tokens`, `Response tokens` (plus Model / Log Type / Total tokens on `_1`) |
| Mooncake JSONL | `timestamp`, `input_length`, `output_length`, `hash_ids`                                        |

No `p50`, no `D`, no regime string in the files. After the adapter: a list of `CanonicalRequest`s. Percentiles, burstiness, I/O ratio, rate, and label come from `Regime.from_trace()` (`runtime/gitm/traffic/regime.py`).

```text
raw CSV/JSONL
  → adapter (CanonicalRequest list + TraceMeta)
  → Regime.from_trace()   # percentiles, D, io_ratio, rate
  → Regime.label()        # bucketed string for logs / exact gates
```

---

## Privacy: why no raw prompts

A raw prompt is human-readable input; raw tokens are encoded IDs — either can leak customer data.

```text
prompt:  "Summarize this customer support conversation..."
tokens:  [791, 2034, 17, 5632, ...]
```

Both datasets publish only derived metadata (lengths, arrivals, and for Mooncake `hash_ids`) so researchers can reproduce load *shape* without reconstructing content.

---

## BurstGPT source

### Raw format (`BurstGPT_1`)

```csv
Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
```

```csv
5,ChatGPT,472,18,490,Conversation log
45,ChatGPT,1087,230,1317,Conversation log
118,GPT-4,417,276,693,Conversation log
```

- `Timestamp`: seconds from midnight of the trace’s first day (not Unix epoch).
- `Request` / `Response` tokens: input / output counts.
- `Total tokens`: read, not used as an independent truth.
- `Model`, `Log Type`: optional filters.

Adapter anchors time at the first row:

```python
CanonicalRequest(arrival_s=0.0, input_tokens=472, output_tokens=18,
                 session_id=None, prefix_blocks=())
CanonicalRequest(arrival_s=40.0, input_tokens=1087, output_tokens=230,
                 session_id=None, prefix_blocks=())
```

No `hash_ids` / prefix identity. Session identity is also absent on `_1`.

### BurstGPT_3 extras

`BurstGPT_3.csv` (and `BurstGPT_without_fails_3.csv`) insert two columns (not append):

- `Session ID` — conversation membership (populated only on `Conversation log` rows; empty on `API log` by design — ~90% of the published v3 file)
- `Elapsed time` — source-system submission-to-final-response seconds → `CanonicalRequest.source_e2e_latency_s` (not TTFT; not our hardware — do not compare to measured TTFT/ITL or promote against it)

**Implemented:** `read_burstgpt` selects columns by **name**. The six core columns are required; Session ID / Elapsed time are used when present; unknown extras are noted, not rejected. Empty Session ID → `session_id=None` (not a defect). `TraceMeta.session_rows` / `.sessions` report how much conversation identity survived — use those for multi-turn, not `has_session_identity` alone.

**Still true:** session-aware *firing* cannot flow through `timed_trace` (no session field). Sessions are for analysis / regime; do not invent prefix reuse from turn membership.

### What BurstGPT can supply to Regime

| Axis                 | How                                                                            |
| -------------------- | ------------------------------------------------------------------------------ |
| I/O ratio            | `sum(input) / sum(output)` — token-volume proxy, not prefill/decode latency |
| Length distributions | input/output p50, p95                                                          |
| Arrival rate         | kept requests / span                                                           |
| Burstiness           | 1 s bin index of dispersion                                                    |
| Concurrency          | **not** from the trace → `None` / `copen`                           |
| Source kind          | assigned provenance →`production`                                           |

Concurrency would need a configured cap or completion intervals; BurstGPT_3 elapsed time could estimate in-flight count but current D1 does not.

---

## Mooncake source

### Raw format

One JSON object per line:

```json
{
  "timestamp": 27482,
  "input_length": 6955,
  "output_length": 52,
  "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]
}
```

- `timestamp`: relative arrival in **milliseconds** (published one-hour trace spans 0…3,600,000 ms).
- Lengths: reported token counts; no prompts for privacy.
- `hash_ids`: remapped IDs for chained **512-token** prefix blocks (each ID includes prior-block history).

`read_mooncake(..., time_scale=0.001)` converts ms → s (default). `read_timed_trace` reuses the same reader at `time_scale=1.0` so emit and re-parse cannot drift apart. Anchors at the first raw row:

```python
CanonicalRequest(
    arrival_s=27.482,
    input_tokens=6955,
    output_tokens=52,
    session_id=None,
    prefix_blocks=(46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354),
)
```

Fixture rows with `timestamp: 0` all become `arrival_s=0.0` (immediate burst).

### `hash_ids` semantics

Not prompt IDs; not joinable into one `prefix_hash`. Equality of leading IDs is the only meaningful operation:

```text
A: [46, 47, …, 57, 2353, 2354]
B: [46, 47, …, 57, 2366]
→ 12 matching blocks × 512 = 6,144 reusable prefix tokens
```

```text
same leading IDs  → shared reusable prefix
different next ID → prefix diverges here
```

IDs are remapped opaque identifiers — equality is the meaningful operation; no text or token recovery. Adapter can preserve cache-reuse **metadata**; whether the live stack actually hits prefix cache must still be verified (open concern in source).

### Lengths vs BurstGPT

Conceptually: `input_length` ≈ Request tokens, `output_length` ≈ Response tokens. Comparable as workload-size signals, not as same-tokenizer guarantees.

---

## BurstGPT vs Mooncake

|              | BurstGPT                                             | Mooncake                            |
| ------------ | ---------------------------------------------------- | ----------------------------------- |
| Origin       | Azure OpenAI ChatGPT/GPT-4                           | Kimi production                     |
| Format       | CSV                                                  | JSONL                               |
| Core fields  | timing, model, in/out lengths                        | timing, in/out lengths,`hash_ids` |
| Prefix cache | none                                                 | chained block identity              |
| Session      | `_3` has Session ID; firing path does not carry it | no session in D1 Mooncake adapter   |

```text
BurstGPT: when did independent requests arrive, and how big were they?
Mooncake: that, plus which prefixes were shared across requests?
```

Both provide enough to model a serving workload:

- when requests arrived
- how large each input was
- how many output tokens each request generated
- patterns across many requests: average rate, bursts, and input/output-size distributions

Neither publishes raw prompts or token IDs (privacy).

**Does BurstGPT provide hash IDs?** No — no `hash_ids`, prefix hashes, or other prefix-cache identity. Newer BurstGPT `Session ID` says requests share a conversation; that does **not** prove token prefixes match. Mooncake’s `hash_ids` are why Mooncake is appropriate for prefix-cache experiments and basic BurstGPT is not.

**Prompt ID** is not a published field. Closest concepts: BurstGPT Session ID (conversation), Mooncake `hash_ids` (prefix blocks). Cache reuse needs *how much* of a prefix matches, not whole-prompt equality.

---

---

# Step 2 — Adapter: canonical form and drops

## Common D1 schema

Every source becomes this shape before replay or benchmarking. Schema id: `gitm.traffic.trace/v1`.

`CanonicalRequest` is a frozen slotted dataclass (hot path — traces can be millions of rows). Validation happens in the adapter with named `DropReason`s, not via pydantic on each request.

```python
CanonicalRequest(
    arrival_s: float,                  # seconds after first raw request (fatal if missing)
    input_tokens: int,                 # > 0; never defaulted
    output_tokens: int | None,         # None = "sample length later" — cannot replay-as-is
    session_id: str | None,
    prefix_blocks: tuple[int, ...],
    source_e2e_latency_s: float | None # BurstGPT_3 Elapsed time only; NOT our TTFT
)

TraceMeta(
    schema_id, source, path, sha256, source_url,
    rows_read, rows_emitted, drops,
    span_s, raw_time_unit,
    prefix_block_tokens, has_prefix_identity,
    has_session_identity, session_rows, sessions,  # counts matter more than the bool
    notes,
)
```

A `Trace` must carry both (`Trace.__post_init__` enforces `rows_read == rows_emitted + dropped` and `rows_emitted == len(requests)`). D4 claims without checksum, filter details, and drop counts are not defensible provenance. Trace is fully materialized (fine through ~BurstGPT_1 scale); streaming deferred.

**`output_tokens is None`:** means generate to a regime-sampled length. `write_timed_trace` **refuses** the whole file if any output is missing — use `parameterize.sample_trace` instead.

**`source_e2e_latency_s`:** source-system end-to-end latency only. Never compare to measured TTFT/ITL; never promote a playbook row against it. Legitimate use: think-time bounds between session turns on the *source* timeline.

Implemented field is `prefix_blocks: tuple[int, ...]`, not a single `prefix_hash`. One whole-prompt hash only answers equal/not-equal; the chain distinguishes:

```text
no cache reuse
partial prefix reuse
complete prompt reuse
```

```text
Mooncake hash_ids → D1 prefix_blocks → TraceMeta records prefix identity
  → D4 can restrict a prefix-caching playbook row to compatible traffic
```

---

## Invalid rows and DropReason

Bad rows are skipped at adapter-read time **before** `CanonicalRequest` creation; `TraceMeta.drops` increments the precise reason. Trace must reconcile:

```text
rows_read == rows_emitted + total_dropped
```

| Reason                    | Trigger                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| `malformed_row`         | wrong CSV width; invalid JSON / non-object / bad`hash_ids`                               |
| `missing_field`         | required timestamp or lengths blank (optional`_3` Session ID / Elapsed time do not drop) |
| `non_numeric`           | timestamp or lengths unparseable (malformed optional Elapsed time → absent, noted)        |
| `negative_value`        | negative timestamp or token counts                                                         |
| `zero_input_tokens`     | no prefill work                                                                            |
| `zero_output_tokens`    | no decode work                                                                             |
| `non_monotonic_arrival` | timestamp earlier than newest previously seen                                              |
| `filtered_out`          | excluded by caller filter (`model` / `log_type`) — **not a defect**             |

`TraceMeta.defects` sums drops excluding `filtered_out`, so a narrow filter never reads as dirty data.

Timestamp is parsed and the clock advanced **before** length checks and before `FILTERED_OUT`, so a bad-length row still updates `last_ts` and cannot mask a later backward timestamp. On BurstGPT, the arrival clock is anchored on the **first row read** (before filtering) — narrowing which requests appear never shifts when they appear.

Collector helpers (adapter-local):

```python
class _Collector:
    def drop(self, reason: DropReason) -> None:
        self.drops[reason.value] += 1

    def check_lengths(self, inp: int, out: int | None) -> DropReason | None:
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
```

Parse order (Mooncake-shaped; BurstGPT is the same idea on CSV columns):

```python
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
```

Example BurstGPT_1 slice provenance shape:

```json
{
  "source": "burstgpt",
  "raw_time_unit": "s",
  "sha256": "…",
  "rows_read": 400,
  "rows_emitted": 368,
  "drops": { "zero_input_tokens": 16, "zero_output_tokens": 16 },
  "has_session_identity": false,
  "has_prefix_identity": false
}
```

So D1 can report, for example:

```text
400 rows read
383 usable requests emitted
17 dropped: zero_input_tokens=17
```

rather than quietly changing the workload distribution.

---

---

# Step 3 — Tag the regime

## Regime

A **regime** answers: under what kind of traffic was this result produced? Computed from the whole kept trace, not one request.

Without it, D4 could only say: “setting `X` gave an 8% improvement.”
With it, D4 can say: “setting `X` gave an 8% improvement for bursty, input-heavy, open-loop production-like traffic.”

It is an object with numeric axes — not one enum and not only the concatenated label. The slash label is a lossy derived string for logs and exact-gate matching. D4 distance uses raw numbers (label buckets alone would treat 1023 vs 1025 as different and 1025 vs 2047 as the same).

```python
Regime(
    source_kind=SourceKind.PRODUCTION,
    trace="mooncake",
    requests=400,
    rate_rps=2.837,
    io_ratio=39.09,
    input_p50=9075,
    input_p95=49904,
    output_p50=370,
    output_p95=662,
    burstiness=6.74,
    bin_s=1.0,
    burstiness_defined=True,   # False if span==0 or <2 arrivals
    concurrency=None,
    in_envelope=True,
    notes=[],                  # e.g. "no output lengths; io_ratio is 0"
)
```

Built by `Regime.from_trace`. `label()` builds the slash string; `_bucket_tokens` / `_bucket_ratio` / `_bucket_burst` do the floor-to-power-of-two / burst bands:

```python
def label(self) -> str:
    parts = [
        {"production": "prod", "synthetic": "syn", "scoreboard": "board"}[
            self.source_kind.value
        ],
        _bucket_ratio(self.io_ratio),
        f"in{_bucket_tokens(self.input_p50)}",
        f"out{_bucket_tokens(self.output_p50)}",
        _bucket_burst(self.burstiness),
        f"c{self.concurrency}" if self.concurrency else "copen",
    ]
    if not self.in_envelope:
        parts.append("xenv")
    return "/".join(parts)

def _bucket_tokens(n: float) -> str:
    if n < 1:
        return "0"
    exp = int(np.floor(np.log2(n)))
    v = 1 << exp
    return f"{v // 1024}k" if v >= 1024 else str(v)

def _bucket_ratio(r: float) -> str:
    if r <= 0:
        return "io0"
    exp = int(np.floor(np.log2(r)))
    return f"io{2**exp}" if exp >= 0 else f"io1-{2 ** -exp}"

def _bucket_burst(d: float) -> str:
    if d < 0.8:
        return "burst-flat"
    if d < 1.5:
        return "burst-poisson"
    if d < 5.0:
        return "burst-mod"
    return "burst-hi"
```

### Field definitions

#### Length percentiles

`input_p50` / `input_p95` (and output analogs) — percentiles of per-request token lengths across kept requests.

```text
input lengths = [r.input_tokens for each kept request]
input_p50 = 50th percentile (median)
input_p95 = 95th percentile
```

BurstGPT_1: half the kept prompts ≤ **353** tokens; 95% ≤ **1,638**. Mooncake: median **9,075**, p95 **49,904**.

Example intuition:

```text
inputs: 100, 200, 500, 2,000
p50 input: roughly 350
p95 input: near the long-request end
```

These are workload-shape axes for D3/D4. D4 distance uses the raw numbers, not label buckets.

#### Burstiness — How (`index_of_dispersion`, bin = `DEFAULT_BIN_S = 1.0`)

1. Take the span of the trace (last arrival − first).
2. Slice that span into **1 s** bins.
3. Count how many requests land in each bin.
4. Compute:

```text
D = Var(counts) / Mean(counts)
```

| D    | Meaning                        |
| ---- | ------------------------------ |
| ≈ 1 | Poisson-like (random arrivals) |
| ≪ 1 | flatter / paced                |
| ≫ 1 | bursty (clumps)                |

BurstGPT lands near **1.01** (`burst-poisson`); Mooncake at **6.74** (`burst-hi`). Same mean rate can still differ a lot in bunching; this metric keeps that difference.

“`@1s`” means the bin width is 1 second. A 60 s bin would smooth Mooncake toward ~1 and hide the burst.

Same mean rate, different bunching:

```text
second:     0   1   2   3   4
requests:   2   3   1  30   4

smooth:  10, 10, 10, 10, 10 requests/sec
bursty:   0,  0, 50,  0,  0 requests/sec
```

Both average 10 req/s in the second pair; the bursty one stresses queues, batching, TTFT, and tails.

Label bands:

```text
burst-flat     → D < 0.8      → paced / unusually regular
burst-poisson  → 0.8 ≤ D < 1.5 → ordinary random arrivals
burst-mod      → 1.5 ≤ D < 5   → noticeably bursty
burst-hi       → D ≥ 5         → highly bursty / clumps
```

#### I/O ratio — token-volume proxy

```text
io_ratio = total input tokens / total output tokens
```

Example:

```text
Request 1: 472 input, 18 output
Request 2: 1087 input, 230 output

total input  = 1,559
total output = 248
I/O ratio    = 1,559 / 248 = 6.29
```

That workload is input-heavy in *volume*. It is **not** measured prefill vs decode *latency* (server, model, GPU, batching, and cache decide that).

Bucketed to powers of two for the label:

```text
io0       → ratio ≤ 0
io1-8     → 0.125 ≤ ratio < 0.25
io1-4     → 0.25 ≤ ratio < 0.5
io1-2     → 0.5 ≤ ratio < 1
io1       → 1 ≤ ratio < 2
io2       → 2 ≤ ratio < 4
io4       → 4 ≤ ratio < 8
io8       → 8 ≤ ratio < 16
io16      → 16 ≤ ratio < 32
io32      → 32 ≤ ratio < 64
```

Informal mappings:

```text
high io ratio, e.g. io16 / io32 / io64
  → input-heavy / prefill-heavy / “long-prefill relative to decode”

low io ratio, e.g. io1-2 / io1 / io2
  → decode-heavy relative to prefill
```

`io32` does **not** necessarily mean each prompt is long. It means total input work is large relative to total output work. Many short prompts with very short outputs can also yield a high I/O ratio.

#### Arrival rate

```text
rate_rps = kept requests / span
```

Example: 400 valid requests over 200 s → `rate = 2` req/s. Another: 600 over 120 s → `5` req/s.

#### Input / output length buckets (`in…` / `out…`)

Input p50 floored to a power of two:

```text
in256     → 256 ≤ p50 < 512
in512     → 512 ≤ p50 < 1,024
in1k      → 1,024 ≤ p50 < 2,048
in2k      → 2,048 ≤ p50 < 4,096
in4k      → 4,096 ≤ p50 < 8,192
in8k      → 8,192 ≤ p50 < 16,384
in16k     → 16,384 ≤ p50 < 32,768
```

```text
in256 / in512  → short-input
in1k / in2k    → medium-input
in4k+          → long-input
in8k+          → very-long-input
```

“Long prefill” is usually `in…` together with a high `io…`. Label omits input p95; D4 distance includes it — a workload can be `in256` median with a very long p95 tail.

Output p50 likewise:

```text
out64      → 64 ≤ p50 < 128
out128     → 128 ≤ p50 < 256
out256     → 256 ≤ p50 < 512
out512     → 512 ≤ p50 < 1,024
out1k      → 1,024 ≤ p50 < 2,048
```

```text
out64 / out128 → short-decode
out256         → moderate decode
out512+        → long-decode
```

#### Concurrency and envelope

```text
copen   → concurrency=None; open-loop replay (arrival-driven)
c64     → explicit cap of 64 in-flight
c128    → explicit cap of 128
```

Caller/run config supplies this; not inferred from BurstGPT/Mooncake columns.

```text
no /xenv suffix → inside the fitted source envelope
/xenv           → intentionally sampled beyond the source envelope
```

Example:

```text
syn/io32/in8k/out512/burst-hi/copen/xenv
```

synthetic, input-heavy, long-input, moderately long-output, highly bursty, open-loop, deliberately beyond what the source observed.

#### Source kind

Provenance, not a statistical lookalike:

| Value          | Label     | Meaning                                                     |
| -------------- | --------- | ----------------------------------------------------------- |
| `production` | `prod`  | from a production trace (BurstGPT, Mooncake)                |
| `synthetic`  | `syn`   | sampled / extrapolated from a fit                           |
| `scoreboard` | `board` | fixed public benchmark condition (e.g. Artificial Analysis) |

`prod` means “originated from a production trace,” not “looks production-like.” Scoreboard evidence must not be treated as production behavior. Does **not** prove either public trace matches customer traffic (risk R4).

### Label vocabulary

```text
<source>/<io…>/<in…>/<out…>/<burst-…>/<c…>[/xenv]
```

Example Mooncake: `prod/io32/in8k/out256/burst-hi/copen`

| Piece        | From                 | Meaning         |
| ------------ | -------------------- | --------------- |
| `prod`     | `source_kind`      | production      |
| `io32`     | `io_ratio` ≈ 39   | bucket near 32  |
| `in8k`     | `input_p50` = 9075 | floor → 8192   |
| `out256`   | `output_p50`       | bucket near 256 |
| `burst-hi` | D = 6.74             | high burstiness |
| `copen`    | concurrency=None     | open-loop       |

Label is a **lossy summary for logs and exact-gate matching**. D4’s numeric distance still uses raw `input_p50`, `input_p95`, etc. — buckets alone would treat 1023 vs 1025 as different and 1025 vs 2047 as the same.

---

# Step 4 — Parameterized mode (envelope + margin)

Code: `runtime/gitm/traffic/parameterize.py` (`fit`, `sample_trace`, `grid`); CLI `python -m gitm.traffic --sweep ADAPTER PATH`.

### What it is

Two D1 workload modes:

| Mode                     | Question                                                                            | Output                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Replay (as-is)** | What happens under*this exact* traffic?                                           | Same arrivals/lengths as the kept trace →`timed_trace`                |
| **Parameterized**  | What happens across the*region* this traffic lives in — **and beyond it**? | Synthetic traces drawn from a fitted envelope, including`/xenv` points |

Parameterized: fit each trace’s distributions along the regime axes (input/output length distributions, arrival rate/burstiness, and the I/O volume implied by those lengths), then **sample a grid**, including points beyond any single trace. The traces define the realistic envelope; the sampler covers the envelope **plus margin**.

Concurrency remains an offered-load *setting* on `Regime` (`copen` / `c64`, …). It is not currently a fitted or swept axis in `parameterize.py` — callers still set it when tagging a run.

### Why it is needed

Replay alone answers only “this hour of BurstGPT / Mooncake.” A playbook row (D4) must survive **the customer’s next hour**, which is never the pinned fixture’s next hour. Without envelope+margin:

- you overfit one timeline;
- you never measure the load/burst/length corner where a knob is actually applied;
- an extrapolation cannot be told apart from a measurement (hence `/xenv`).

So: traces = realistic envelope; sampler = envelope + labeled margin.

### How it works

**1. Fit** (`fit(trace) → RegimeFit`)

- Need ≥2 requests and at least some output lengths.
- Store source `sha256`, request count, span, rate, burstiness.
- Fit **empirical** 101-point quantile grids for input and output lengths (`np.percentile` over 0…100). No lognormal / parametric family: production lengths are multi-modal (short chats + long docs); a parametric fit would smear modes and quietly move the prefill/decode ratio.

**2. Sample** (`sample_trace(fit, …) → Trace, Regime`)

Axes you can push:

| Knob                               | Effect                                    |
| ---------------------------------- | ----------------------------------------- |
| `rate_mult`                      | scale mean arrival rate                   |
| `burstiness`                     | target index of dispersion D              |
| `input_scale` / `output_scale` | scale drawn lengths                       |
| `duration_s`                     | how long to generate (default = fit span) |
| `seed`                           | reproducibility                           |

Arrivals:

- `D ≤ 1` → Poisson per bin
- `D > 1` → negative binomial (mean/variance set so dispersion = D)
- `D < 1` requested → **clamp to Poisson** and note (underdispersion not fully modeled yet)

Lengths: inverse-CDF draw from the fitted quantile grid, then scale and clamp to ≥1.

**Identity:** a synthetic trace has no file bytes; `TraceMeta.sha256` is the digest of the generating parameters. Same digest ⇒ same trace.

**Envelope flag:**

```text
in_envelope = True  only if rate_mult, burstiness, input_scale, output_scale
                 are all within the fitted source (≤ source on those axes)
else            in_envelope = False  → regime label gets /xenv
                source_kind = synthetic → syn/…
```

**3. Grid** (`grid(fit)`)

Default sweep is deliberately **asymmetric upward** — the interesting margin is *above* the observed trace (more load, burstier, longer outputs), where a playbook row gets applied and nobody measured:

```text
rate_mults:     0.5, 1.0, 2.0, 4.0
burst_targets:  1.0, 4.0, 16.0
output_scales:  1.0, 2.0
```

One derived seed per cell so the whole sweep is reproducible from a single root seed. CLI `--sweep` prints each cell’s regime label, request count, rps, and D.

### Validation standard (not the same as exact replay)

A parameterized sample is a *draw*, not a copy. `validate.compare` uses `SAMPLED_THRESHOLDS` (wider KS/rate/burst/count; **no** arrival-timeline KS). Exact replay uses `REPLAY_THRESHOLDS` (near-identical statistics + timeline). Comparing timelines on a sample would only prove “a sample is not a copy,” which is the point of sampling.

### After sampling

A sampled `Trace` can still go through Step 4 → emit `timed_trace` → fire → reconcile, same as a production fixture. Prefix blocks are empty unless you add them; do not invent Mooncake-style sharing on synthetic traffic.

### Relation to `/xenv` (Step 3 label)

```text
syn/io32/in8k/out512/burst-hi/copen/xenv
```

means: synthetic, beyond the fitted source on at least one swept axis — margin, labeled as margin.

---

# Step 5 — Emit timed_trace (replay file)

Write the vLLM-facing artifact from the canonical trace (`write_timed_trace` → `ReplayPlan`).

## CanonicalRequest → timed_trace

D1 does not send `CanonicalRequest` objects to vLLM. Adapters build them; `write_timed_trace` maps each request into one JSONL line and returns a `ReplayPlan`:

| Canonical                                 | timed_trace field                                       |
| ----------------------------------------- | ------------------------------------------------------- |
| `arrival_s`                             | `timestamp`                                           |
| `input_tokens`                          | `input_length`                                        |
| `output_tokens`                         | `output_length`                                       |
| `prefix_blocks`                         | `hash_ids`                                            |
| `session_id` / `source_e2e_latency_s` | **dropped** — format has no session or e2e field |

**`ReplayPlan` carries:** path, `requests`, `span_s`, `input_tokens_total`, `output_tokens_total`, `chunk_hash_size`, `sec_multiplier` (always 1 — we emit seconds), `self_timed`, `prefix_synthesized`, embedded source `TraceMeta`, and `notes`.

BurstGPT has no real hashes. Empty `hash_ids` would yield a zero-length prompt, so ids are synthesized at `SYNTHETIC_BLOCK_TOKENS = 512` — unique per request — so lengths are honored and **no fake prefix sharing** is invented (`prefix_synthesized=True`). Prefix-cache reuse on such a plan is understated, never overstated. Session identity stops at the emitter for the same reason.

`write_timed_trace` **refuses the whole file** if any request lacks output length, or if `len(blocks) * block_tokens < input_tokens` (would silently truncate).

`ReplayPlan.bench_serve_argv(...)` builds the exact command. Important contract details:

- Completions backend only (`openai` / `vllm`) — chat endpoints reject pre-tokenized prompts.
- Pass `--tokenizer` when the served model id is not HF-resolvable (stub / `--served-model-name`).
- Forces `--percentile-metrics ttft,tpot,itl,e2el` and `--metric-percentiles 50,95,99` (D2 needs p50/p95 beside p99; vLLM defaults omit them).
- Requires vLLM `>= VLLM_MIN_VERSION` (`0.23.0` — first release with `timed_trace`; 0.22.1 does not have it).

---

## timed_trace format vLLM accepts

One JSON object per line. Default labels (exactly what `write_timed_trace` emits):

| Field             | Type   | Meaning                                                             |
| ----------------- | ------ | ------------------------------------------------------------------- |
| `timestamp`     | number | arrival time (seconds when`--timed-trace-sec-multiplier 1`)       |
| `input_length`  | int    | prompt token count                                                  |
| `output_length` | int    | generation token count                                              |
| `hash_ids`      | int[]  | chained prefix-block IDs; each expands to`chunk_hash_size` tokens |

No prompt text. vLLM builds pre-tokenized prompts from `hash_ids ×` block size. Coverage rule D1 enforces: `len(hash_ids) * block_tokens >= input_length` per request, or the whole file is refused.

### Mooncake-style (real prefix identity)

```jsonl
{"timestamp": 0.0, "input_length": 6955, "output_length": 52, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]}
{"timestamp": 0.15, "input_length": 2048, "output_length": 64, "hash_ids": [46, 47, 48, 49]}
{"timestamp": 1.2, "input_length": 512, "output_length": 32, "hash_ids": [99]}
```

With `--timed-trace-chunk-hash-size 512`, the first row covers `14 × 512 = 7168 ≥ 6955` tokens. Shared leading IDs imply shared prefix KV.

### BurstGPT-style (synthesized blocks, no real sharing)

```jsonl
{"timestamp": 0.0, "input_length": 472, "output_length": 18, "hash_ids": [0]}
{"timestamp": 40.0, "input_length": 1087, "output_length": 230, "hash_ids": [1, 2, 3]}
{"timestamp": 113.0, "input_length": 417, "output_length": 276, "hash_ids": [4]}
```

### Pointing bench-serve at the file

```bash
vllm bench serve \
  --dataset-name timed_trace \
  --dataset-path /path/to/replay.jsonl \
  --timed-trace-sec-multiplier 1 \
  --timed-trace-chunk-hash-size 512 \
  --self-timed \
  ...
```

`--self-timed` fires each request at its own `timestamp`.

## Session-aware replay vs independent replay

Independent:

```text
request A arrives at 0s
request B arrives at 2s
request C arrives at 5s
```

Session-aware also knows conversation membership:

```text
session "abc": request A → request C
session "xyz": request B
```

Later turns may grow context and exercise session-oriented caching. Deriving `hash_ids` from Session ID would invent prefix hits the source never proved. Safe rule: understate reuse, never overstate. Session identity stops at the emitter for firing; available for analysis.

---

## Tokenization limitation and cross-tokenizer policy

Traces publish counts, not prompts or a tokenizer contract. D1 can reproduce **published counts and timing**; it cannot recreate original text or prove source tokens equal target-model (e.g. Qwen) tokens.

Same original text can tokenize differently:

```text
Original text: “Please summarize the following legal agreement …”

Azure/OpenAI tokenizer: 472 tokens
Qwen tokenizer:         530 tokens
another tokenizer:      410 tokens
```

If the source only reports `472`, D1 has no text to re-tokenize. Current behavior:

```text
Source reported input length: 472
Replay target input length:   472 synthetic pre-tokenized IDs
```

“BurstGPT’s reported `472` came from an unknown source tokenizer; D1 uses `472` as a target-model token-work budget, but cannot prove the original prompt would have been 472 Qwen tokens.”

Token count drives:

- prefill compute and KV-cache memory
- batching and queueing thresholds
- TTFT
- prefix-cache block boundaries
- the input/output ratio used by D1 and D4

So mismatched tokenizer coordinates corrupt D4 distance:

```text
Source regime A input p95: 8,000 Azure tokens
Source regime B input p95: 16,000 Azure tokens
D4 distance: log2(16000 / 8000) = 1   → looks like a clean 2× prefill difference

After target mapping A→9,500 and B→14,000:
target-side ratio ≈ 1.47×, not 2×
```

Cross-tokenizer reconciliation is required to make D1/D2/D4 claims defensible across tokenizer families — **not** required for D1 to run. Post-fire reconcile (Step 8) only proves synthetic work volume matched the plan.

### Evidence levels (policy still open)

#### 1. Same-tokenizer semantic replay — strongest

Requirements: original prompt text or token IDs; known matching tokenizers (or re-tokenize with target); record both counts.

```text
source tokenizer: Qwen revision abc
target tokenizer: Qwen revision abc
source input tokens: 472
target input tokens: 472
status: exact-tokenizer
```

Supports workload claims and, if real prefix ids exist, cache claims. BurstGPT cannot meet this today.

#### 2. Calibrated cross-tokenizer replay — approximate

Measure a mapping (with uncertainty) on a representative corpus, conditioned on language/script, code vs prose, length band, tokenizer family, domain — not a single global `× 1.11`.

```text
Azure-source count band: 256–512
Qwen/source ratio: p50 = 1.11, p95 = 1.23

Azure reported 472 tokens
target replay budget:
  p50 target = 524 tokens
  cautious p95 target = 581 tokens
```

Does not recreate specific BurstGPT prompts; approximates target token-work distribution. A single global rule such as `Qwen = Azure × 1.11` is often too weak — tokenizers diverge much more on CJK text, source code, whitespace-heavy prompts, structured JSON, and unusual Unicode.

#### 3. Count-shape-only — current BurstGPT / Mooncake position

```text
tokenization_status: source-tokenizer-unknown
replay_input_length_basis: source-reported count
cross-tokenizer_mapping: none
allowed interpretation:
  scheduling/load-shape experiment only
disallowed interpretation:
  semantic equivalence, exact Qwen prompt cost, cache-reuse claim
```

### Operational fields per run

```text
source_tokenizer: name/version or "unknown"
target_tokenizer: exact model/tokenizer revision used by vLLM
input_length_basis: source_reported | target_retokenized | calibrated_mapping
mapping_id: null or immutable calibration artifact ID/SHA
mapping_error: p50 / p95 estimation error, if mapped
prompt_content_available: true | false
cache_identity_status: source-provided | synthesized | unavailable
```

Gate:

```text
same tokenizer + original prompt/token IDs → semantic and workload claims may be considered
calibrated mapping → workload-shape claims within documented uncertainty
unknown source tokenizer / count-only → scheduling-load only; no semantic or cache-reuse claims
```

---

# Step 6 — Round-trip check (seam 1)

## D1 round-trip stage

After D1 reads and keeps requests, it writes a vLLM `timed_trace` JSONL and reads that file back. The canvas / e2e stage is `d1-replay` (“D1 round-trip”). Columns: requests, **Span (s)**, **Input tokens**, **Blocks**, **Round-trip**.

This stage does **not** start a vLLM process. It is a D1 self-check that the emitter’s artifact still matches the plan.

### Two levels of “round-trip”

| Level                         | Where                                           | What it checks                                                                                                                                |
| ----------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Thin (e2e / canvas)** | `e2e_d1_d4.py`                                | request count, total input tokens, span exact                                                                                                 |
| **Full (library)**      | `validate.compare` after `read_timed_trace` | KS on arrivals + input/output lengths, relative rate & burstiness, count; ASCII sparkline of arrival histograms (`ValidationReport.render`) |

Full compare uses `REPLAY_THRESHOLDS` (near-identical) for exact replay and `SAMPLED_THRESHOLDS` for parameterized draws (wider; no arrival-timeline KS). Arrivals are quantized at `ARRIVAL_RESOLUTION_S = 1e-6` so JSONL’s 6-decimal rounding does not fake a KS failure.

CLI `--replay` runs the full `compare` path and prints the report before optionally `--fire`.

---

## Span (s)

Wall-clock length of the kept workload: last arrival − first arrival, in seconds (`plan.span_s`).

| Trace      | Span     | Meaning                   |
| ---------- | -------- | ------------------------- |
| BurstGPT_1 | 37,269 s | ~10.4 hours of arrivals   |
| BurstGPT_3 | 41,453 s | ~11.5 hours               |
| Mooncake   | 141 s    | ~2.4 minutes, dense burst |

Same request count can imply very different rate and burstiness once span differs.

---

## Input tokens, Blocks, and Round-trip

These three are the silent-failure checks. Request count and span alone are not enough: the classic bug breaks total prefill work via block size while timeline and row count still look fine.

### Input tokens

Sum of `input_length` over every kept request (`plan.input_tokens_total`). Answers: *did we still plan the same total prefill work?*

| Trace      | Input tokens |
| ---------- | ------------ |
| BurstGPT_1 | 208,189      |
| BurstGPT_3 | 246,941      |
| Mooncake   | 5,710,530    |

Mooncake is large because prompts are long (p50 ~9k), not because it has more requests.

If the emitter or block expansion is wrong, request count can still match while total prefill is wrong. Observed case: wrong block size made Mooncake look like ~15k input tokens instead of ~506k — **32× short** — with row counts still matching.

### Blocks

`chunk_hash_size` on the replay plan — **512** on every plan in this run. That is the token size of each prefix-cache hash block (Mooncake’s native size). Answers: *under what block size is this plan valid?*

vLLM builds each prompt as `len(hash_ids) × chunk_hash_size` and stops when ids run out. Fire with 16 instead of 512 and every count-looking check can still pass while every prompt is truncated. Recording 512 on the plan is provenance for that contract.

### Round-trip

Pass/fail after write → read → compare. The runner requires:

1. row count == planned requests
2. summed `input_length` == planned input tokens
3. `last_timestamp − first_timestamp` == planned span

`all exact (0.0)` means all three matched. This run: **6/6**. That closes the format seam: D1’s JSONL is what `bench serve --dataset-name timed_trace` would consume.

Together:

- Input tokens = *what* was planned
- Blocks = *how* lengths are reconstructed
- Round-trip = *whether the file still matches*

---

## Our side vs vLLM side

**Our side** for what the canvas shows. D1:

1. writes the JSONL (`write_timed_trace`)
2. reads it back
3. checks requests / input tokens / span
4. records `blocks` (`chunk_hash_size`) on our `ReplayPlan`

**vLLM’s side** is the *contract* those checks target — not participation in this stage:

- field names (`timestamp`, `input_length`, `output_length`, `hash_ids`)
- prompt expansion: `hash_ids × chunk_hash_size`
- actual firing via `bench serve --dataset-name timed_trace`

Until `--fire`, vLLM is not in the loop. Round-trip only proves we speak its format correctly.

---

# Step 7 — Fire with vLLM (seam 2)

## Why vLLM; who replays; why replay

The intended load generator is `vllm bench serve --dataset-name timed_trace --self-timed`. That path already schedules each request at its own timestamp. D1 writes that JSONL and builds the argv; it does **not** invent a custom client or fire traffic itself.

Two different meanings of “replay”:

| Step                         | Who                 | What                                                 |
| ---------------------------- | ------------------- | ---------------------------------------------------- |
| Seam 1 (canvas`d1-replay`) | D1 only             | write`timed_trace` → read back → compare to plan |
| Real fire                    | vLLM`bench serve` | POST the workload at a live endpoint                 |

vLLM consumes the **replay file**, not in-memory Python objects.

**Why replay at all:** public traces supply workload shape (arrivals, lengths, optional prefix identity). Replay measures a real engine under that shape instead of fixed prompt/rate synthetics — so later stages can say a knob helped under *this* pattern, not “somehow.”

### How fire is launched (`runner.run_replay`)

Order matters — this is not a bare `subprocess.run`:

1. **`check_vllm`** — refuse before launch if missing or `< 0.23.0` (so the failure is a sentence about `timed_trace`, not an argparse “unknown dataset”).
2. **`vllm_executable()`** — binary from `sys.executable`’s directory so the version check and the process that runs cannot diverge.
3. Optionally resolve the served model id from `/v1/models`.
4. Launch `ReplayPlan.bench_serve_argv(...)`; support `--dry-run` (build/check, launch nothing).
5. Return `RunResult`: raw `result` JSON untouched, plus optional `joined` `BenchRun` when a `regime` was supplied.

CLI: `python -m gitm.traffic --replay … --fire --result-dir …` (needs vLLM + a server). Everything else is CPU-only.

# Step 8 — Reconcile after fire (seam 3)

## What D1 currently reconciles (post-fire)

Implemented by `results.join_result` → `BenchRun` (schema `gitm.traffic.benchrun/v1`). This is one **arm**; a playbook row is a delta between two arms.

D1 performs a transport / workload-shape reconciliation against the live `bench serve` result:

```text
Did vLLM receive the expected number of requests?     → requests_completed
Did it report zero failures?                          → no_failed_requests
Did it process the planned total input-token count?   → input_tokens_match_trace
Did self-timed replay respect the trace duration?     → paced_to_trace_span
```

For a live Mooncake run, it verified:

```text
planned input tokens:  506,280
vLLM input tokens:     506,280
```

That catches the 512-versus-16 block-size failure (at default 16 the total would be ~15,821 — **32× short** — while row counts can still look fine).

**Pacing drift** (`DRIFT_TOLERANCE=0.05`, `DRIFT_FLOOR_S=1.0`), only when `self_timed`:

- finished **faster** than span×(1−5%) → timestamps not honoured (not a replay)
- **overran** span×(1+5%)+1s → schedule drifted (often server saturation)

If `prefix_synthesized`, a non-failing `prefix_identity` check labels any cache-reuse reading as a **floor**, never an estimate.

### Misleading fields under `--self-timed`

vLLM still records CLI defaults `request_rate` (often `"inf"`) and `burstiness` (`1.0`) — not the trace. `join_result` **drops** them into `BenchRun.dropped` / `dropped_values` with a reason; true values live on `regime.rate_rps` / `regime.burstiness`. Metrics kept are an allowlist (`KEPT_METRICS`); new vLLM fields must be an explicit decision (`unjoined_keys`).

### Promotable bar

```text
BenchRun.promotable ⇔ reconciled ∧ failed == 0
```

Necessary for D2 to look at the run; never sufficient (D2 owns the promotion rule). `config_capture` stays `pending-config-capture` (R1). Raw JSON retained on `BenchRun.raw`.

This proves only:

> “vLLM processed the amount of synthetic token work D1 instructed it to process.”

It does **not** prove:

> “The synthetic tokens correspond to the same text, tokenization, semantics, or cache sharing as BurstGPT’s (or Mooncake’s) original requests.”

---

## Prefill vs decode *time*

I/O token ratio is only a volume proxy. Accurate phase times must come from the **target serving run**, with server instrumentation. Capture at least:

```text
arrival time
scheduled / start-processing time
prefill completion / first-token time
final-token time
```

Then distinguish:

```text
queue delay          = processing start − arrival
TTFT                 = first token − arrival   # includes queue/batch/schedule
prefill service time = prefill done − processing start
decode duration      = final token − first token
```

TTFT alone is not pure prefill time. Prefer engine phase metrics or GPU/kernel traces that attribute time to prefill and decode.

---

---

# Step 9 — Hand off to D2–D4

## How D1 connects to D2–D4

```text
raw trace
  → CanonicalRequest[] + TraceMeta + Regime     (D1 read/tag)
  → timed_trace JSONL + ReplayPlan               (D1 emit; seam 1)
  → vLLM bench serve → result JSON               (real fire)
  → reconcile vs plan + join regime              (seams 2–3)
  → D2 evaluate(base, treat)                     (promotion)
  → D4 row_from_runs + lookup / regime_distance
```

What later stages take from D1:

| Stage        | Uses from D1                                                                                                                                                                  |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D3** | Regime input p50/p95 (prompt lengths). Does not need the JSONL for ratio math.                                                                                                |
| **D2** | `ReplayPlan` totals (`requests`, `input_tokens_total`, `span_s`, …) plus regime, joined onto bench-serve metrics so a result must reconcile to the planned workload. |
| **D4** | Regime (and plan provenance) so a playbook row is tied to traffic shape; then distance/lookup against other regimes.                                                          |

In the session-19 e2e canvas, D2/D4 did **not** fire a GPU: they reused a committed bench-serve JSON and rewrote counts to match each plan, exercising arithmetic and refusals, not a live effect size.

---

## What D1 and D4 provide (roles)

**D4** — store and safely reuse evidence: model + GPU + regime + knobs → measured delta + provenance. Does not replay; uses D1’s regime/provenance to decide whether a past row applies:

```text
specific model + GPU + workload regime + knob setting
    → measured performance outcome + provenance
```

Example outcome:

```text
For long, bursty, prefix-sharing traffic on H100:
enable_prefix_caching = true improved TTFT p99 without harming ITL.
```

Does **not** provide: customer prompts, proof of customer representativeness, results before the target server is measured, or guarantees outside the measured regime.

### Conceptual playbook intent (not the built schema)

BurstGPT-shaped intent — knob win under bursty short-chat production proxy; must not auto-apply to Mooncake long-prefill / prefix-sharing traffic:

```json
{
  "identity": {
    "model_revision": "Qwen/Qwen3.6-35B-A3B-FP8@95a723d0",
    "gpu_sku": "NVIDIA H100 80GB",
    "regime": "production|…|bursty|…",
    "knob_set": { "max_num_seqs": 64 }
  },
  "measured_delta": {
    "throughput_pct": 8.1,
    "ttft_p99_ms": -4.2,
    "itl_p99_ms": 1.0
  },
  "provenance": {
    "trace_source": "burstgpt",
    "trace_sha256": "…",
    "trace_drops": { "zero_input_tokens": 16, "zero_output_tokens": 16 },
    "config_capture": "pending-config-capture"
  }
}
```

Mooncake-shaped intent — gated on long input + prefix identity:

```json
{
  "identity": {
    "model_revision": "Qwen/Qwen3.6-35B-A3B-FP8@95a723d0",
    "gpu_sku": "NVIDIA H100 80GB",
    "regime": "production|long-prefill|short-decode|bursty|prefix-sharing",
    "knob_set": { "enable_prefix_caching": true }
  },
  "measured_delta": {
    "throughput_pct": 14.5,
    "ttft_p99_ms": -18.0,
    "itl_p99_ms": 0.3
  },
  "provenance": {
    "trace_source": "mooncake",
    "trace_sha256": "…",
    "raw_time_unit": "ms",
    "prefix_block_tokens": 512,
    "has_prefix_identity": true,
    "config_capture": "pending-config-capture"
  }
}
```

(Built schema differs: separate `model` / `model_revision`, key `delta`, whole `Regime` object, `evidence` field — see Notes carried forward.)

---

# Step 10 — CLI, GUI, and selftests

### CLI (`python -m gitm.traffic`)

| Flag                                                    | Role                                              |
| ------------------------------------------------------- | ------------------------------------------------- |
| `--selftest`                                          | run every pinned check in`_selftest.py`         |
| `--describe ADAPTER PATH`                             | load + print meta / regime                        |
| `--replay ADAPTER PATH`                               | `write_timed_trace` + `compare` + print argv  |
| `--fire`                                              | actually run argv (needs vLLM ≥ 0.23.0 + server) |
| `--sweep ADAPTER PATH`                                | `fit` + `grid` printout                       |
| `--gui` / `--gui-port` / `--gui-root`             | localhost viewer                                  |
| `--out`, `--model`, `--tokenizer`, `--base-url` | replay / fire options                             |
| `--max-rows`, `--kind`                              | slice /`SourceKind`                             |
| `--result-dir`, `--dry-run`                         | fire options                                      |

Everything except `--fire` is CPU-only.

### GUI (`gui.serve`)

Read-only viewer bound to `127.0.0.1`. Routes: `/api/describe`, `/api/replay`, `/api/sweep`. Paths sandboxed under a configured root (default: committed fixtures); refuses anything outside that root.

### Selftests (`_selftest.run_all`)

Pin the behaviors this doc describes, including: fixture labels; every `DropReason` fires; `FILTERED_OUT` ≠ defect; provenance reconcile; replay round-trip; refuse truncate at `block_tokens=16`; session replay understates reuse; regime axes separate traces; parameterized envelope; argv; version guard; join drops misleading fields / catches 32× short / both pacing failures; `unjoined_keys` empty for the recorded real-run keys.

---

## Open and closed items (as of source update)

| Item                                                           | Status                                                                                                          |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| D4 playbook schema / selftests                                 | **Built** — calibrated regime-distance threshold still open (R5 / D4-9); shipped policy exact-match only |
| Shared config-capture schema                                   | **Open** (R1)                                                                                             |
| Cross-tokenizer reconciliation rule                            | **Open**                                                                                                  |
| Faithful arrival offsets via`timed_trace` + `--self-timed` | **Closed** (read from vLLM source; no custom load gen)                                                    |
| Live endpoint fire on authoring box                            | Was open when written; seams later closed against vLLM 0.28.0 under WSL — see Seams Expl / earlier V2 section  |
| Customer traffic representativeness                            | **Open** (R4)                                                                                             |

External refs: [BurstGPT](https://github.com/HPMLL/BurstGPT), [NVIDIA BurstGPT format docs](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/profile-with-burst-gpt-traces), [Mooncake paper/trace](https://arxiv.org/html/2407.00079v3).
