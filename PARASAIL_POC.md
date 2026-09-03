# Parasail POC — traffic replay and the playbook schema

Two pieces of the validation infrastructure, built to answer one question: **when
we say a knob is worth applying, what makes that statement checkable by someone
who was not in the room?**

- **`gitm/traffic/`** — replay real production traces through a real serving
  endpoint, and tie the result back to the workload that produced it.
- **`gitm/playbook/`** — the row that says *this knob, on this model and GPU,
  under this traffic, moved these numbers by this much, and here is everything
  needed to re-check it.*

CPU-only except for firing a replay. No new dependencies.

---

## Five-minute tour

```bash
# 1. Every claim below, as one runnable check
python -m gitm.traffic  --selftest        # 24 checks, 3 real traces
python -m gitm.playbook --selftest        # 19 checks, 2 real regimes

# 2. What a real production trace actually contains
python -m gitm.traffic --describe mooncake \
    benchmarks/traffic_replay/fixtures/mooncake_slice.jsonl

# 3. Replay it, and prove the replay preserved it
python -m gitm.traffic --replay mooncake \
    benchmarks/traffic_replay/fixtures/mooncake_slice.jsonl --out /tmp/mc.jsonl

# 4. What a playbook row looks like, and why none of these is selectable
python -m gitm.playbook --show   benchmarks/playbook/examples.json
python -m gitm.playbook --lookup benchmarks/playbook/examples.json ex1-prefix-cache-mooncake
```

Step 3 prints a validation table in which **every statistic is exactly 0.0**, and
the arrival-rate profiles of source and replay drawn one above the other. Step 4
returns **nothing** and says why, per row.

---

## What is measured, and what is not

The distinction is load-bearing, so it is stated before anything else.

| | state |
|---|---|
| The adapters, on real published bytes | ✅ measured |
| Replay fidelity, source vs the file vLLM reads | ✅ measured — every statistic exactly 0.0 |
| Firing at a live vLLM endpoint | ✅ **done**, against real `vllm bench serve` 0.28.0 |
| Joining a result back to its trace | ✅ measured, end to end, exit 0 |
| **Any playbook row being true** | ❌ **no.** Every shipped row is `illustrative` and its delta is invented |
| The regime-distance threshold | ❌ **uncalibrated, and the code says so at runtime** |
| Pacing under a *saturating* server | ❌ needs a GPU |

**Nothing here claims a performance win.** It is the machinery that would make
such a claim checkable.

---

## `gitm/traffic/` — traffic replay

### Real traces, not synthetic load

Two published production traces, pinned by sha256 through
`gitm.bench.manifest`. Measured by the adapters:

| trace | rows | input p50 / p95 | dispersion `D` | regime label |
|---|---|---|---|---|
| BurstGPT_1 (Azure OpenAI) | 383/400 | 353 / 1,638 | 1.01 | `prod/io1/in256/out128/burst-poisson/copen` |
| BurstGPT_3 (8-column layout) | 399/400 | 309 / 1,497 | 1.01 | `prod/io2/in256/out64/burst-poisson/copen` |
| Mooncake (Kimi serving) | 400/400 | 9,075 / 49,904 | 6.74 | `prod/io32/in8k/out256/burst-hi/copen` |

The two axes that matter separate cleanly — **D 6.74 vs 1.01, input p50 9,075 vs
353.** If the regime axes could not tell two real production traces apart they
would be decoration; `check_regime_axes_separate_the_traces` asserts they do.

### Real data is the work, not the parsing

**7 named drop reasons**, each firing exactly once on both a CSV and a JSONL
dirty fixture. The counts land in `TraceMeta`, and **a trace whose counts do not
reconcile cannot be constructed** — `rows_read == rows_emitted + dropped` is
enforced in `__post_init__`, not by convention.

The zero-token rows in BurstGPT are real and are the majority defect: **744 of
9,382 `Conversation log` rows carry 0/0**, while all 19 `API log` rows are
well-formed. A caller's own filter counts separately from bad data and still
reconciles.

### No custom load generator

Replay goes through vLLM's native `bench serve --dataset-name timed_trace
--self-timed`, which schedules every request at its own timestamp. Fidelity is
checked against **the file vLLM will actually read**, not an in-memory copy:

```
trace validation [replay]  mooncake(8090d6a38401) -> timed_trace(dfd989ad6d50)
check                 value     threshold  result
request_count             0             0  pass
arrival_ks                0         0.001  pass
input_len_ks              0         0.001  pass
output_len_ks             0         0.001  pass
rate_rps                  0         1e-06  pass
burstiness                0         1e-06  pass
PASS — the pipeline preserves the trace
```

**Fired end to end against real vLLM 0.28.0**: 40/40 completed, paced to
**12.008 s against a 12.000 s trace span** — confirmed by two independent clocks,
vLLM's own and the receiving server's.

### The silent failure this exists to catch

vLLM's `--timed-trace-chunk-hash-size` defaults to **16**. Mooncake's cache
blocks are **512**. At the default, every prompt is **32× short** — while
`completed`, `duration`, throughput and every latency percentile still read
perfectly.

The emitter refuses to write such a file, and the joiner catches one fired from a
plan built elsewhere. Confirmed on the real run: `total_input_tokens` came back
**506,280 against the trace's 506,280** — exact; at 16-token blocks it would have
been ~15,821.

### The result JSON is not merely incomplete — two fields are wrong

`bench serve --save-result` writes 34 keys. Under `--self-timed` vLLM still
records the CLI's `request_rate` (`"inf"` — a *string*) and `burstiness` (`1.0`),
against the trace's real 2.837 rps and D 6.74. **They are exactly the two axes a
playbook row keys on.**

They are **dropped with a stated reason** rather than merged or renamed, with the
values kept visible. A new vLLM field cannot fall out silently either —
`unjoined_keys()` fails on any key neither kept nor deliberately dropped, and it
caught `rtfx` on its first run.

---

## `gitm/playbook/` — the row, and when it may be used

### Row identity is a split, not a lookup

```
(model+revision, GPU SKU, environment, workload regime, knob set)
    -> measured delta + provenance
```

**Exact equality** on model+revision, GPU SKU, engine+version, `source_kind` and
concurrency. **Distance** on the numeric regime axes, because live traffic never
lands on a measured point. Each gate is a named field on `MatchPolicy`, so
loosening one is an edit visible in a diff rather than an accident inside a
comparison.

`source_kind` is asserted the hard way — **with every numeric axis identical**, so
only the gate can be doing the work. A scoreboard result never satisfies a
production query.

### Distance is a log2 ratio, combined with L-infinity

`|log2(a/b)|` — scale-free, symmetric, zero at equality, and `inf` when one side
is zero, because "no output tokens" is not a small version of "some".

**L-infinity, not a mean.** A row is as far away as its worst axis. Asserted on
the case it exists for: a candidate identical on five axes and 8× off on
`input_p95` reads **0.5 as a mean** — close enough to apply — and **3.0 as
L-inf**. For scale, the two real traces are **4.929 apart, limited by
`input_p95`** (49,904 vs 1,638).

### The threshold is uncalibrated, and the type refuses to pretend otherwise

```python
AxisTolerance(max_distance=1.0)                                        # ValidationError
AxisTolerance(max_distance=1.0, calibration="prereg E4: sign flip at 1.4")   # ok
```

No tolerance ships. Consequences, all asserted:

- an **exact** regime match still returns a row, so the schema is usable today;
- any nonzero distance returns `UNCALIBRATED` **naming the limiting axis**, and
  routes to discovery;
- a placeholder cannot quietly become a production constant.

Calibration needs the same knob measured across nearby regimes, which needs a
GPU. **The cost of the open state is a discovery run; the cost of an invented
threshold is a knob applied to traffic nobody measured it on.**

### A row cannot exist without its evidence

Refused at construction: no knobs, `repeats < 2`, a delta missing its latency
percentiles, an empty `trace_sha256`. `MeasuredDelta` carries throughput **and**
TTFT/ITL percentiles — the schema cannot express a throughput-only row.

Invalidation is **a field with a reason, never a deletion**. A deleted row leaves
no record that the claim was made, which is the first thing a reviewer asks for.

### `delta_is_floor` — the guard that falls out of the traces

BurstGPT carries no prefix identity, so a replay of it synthesizes unique blocks
per request. A prefix-cache knob measured there saw the **least** reuse the real
traffic could have had, so its delta is a **lower bound** — usable as "at least
this much", never quotable as the gain. `summary()` prints
`[FLOOR: prefixes synthesized]` so it cannot be lost in a paste.

`ex6` is that case. `ex2` is the control: same synthesized trace, a knob that does
not depend on reuse, so **not** a floor.

### The shipped examples contain nothing selectable

```
$ python -m gitm.playbook --show benchmarks/playbook/examples.json
benchmarks/playbook/examples.json: 6 rows, 0 selectable
```

The **regimes are real**, measured off the pinned fixtures. The **deltas are
invented** and every row says so. A perfectly matching query returns nothing and
names the reason per row. The **largest claimed delta in the file (+22%) is the
scoreboard row** — the one most likely to be copied, and the one gated out by
equality rather than distance.

---

## Verification

```
python -m gitm.traffic  --selftest    24 checks, 3 real traces, 7 drop reasons
python -m gitm.playbook --selftest    19 checks, 2 real regimes, 0 calibrated axes
pytest tests/test_traffic.py tests/test_playbook.py -q          45 passed
ruff check gitm/traffic gitm/playbook                  All checks passed!
```

Assertions are written **once** and run from both entry points, so the check a
reader is told about and the check CI runs are the same check.
`test_every_check_is_registered` fails if a `check_*` function is defined and left
out of the list.

Full suite: **11 failures, all pre-existing** — module-for-module identical to the
list on clean `b2da5b6`, all missing optional deps or importer goldens. **None in
`gitm/traffic` or `gitm/playbook`.**

---

## Two bugs found by running it rather than reading it

- **`runner.py` executed a bare `vllm` from `PATH`**, which fails under an
  absolute interpreter (a conda env used without activation — normal in CI and
  WSL) and, worse, **could have validated one install and run another**:
  `check_vllm()` reads metadata for *this* interpreter. The binary is now derived
  from `sys.executable`, so the two cannot differ.
- **`--tokenizer` was missing from the argv builder.** `bench serve` constructs a
  tokenizer from `--model` even though `timed_trace` sends pre-tokenized prompts,
  so any served name HuggingFace cannot resolve dies in
  `AutoTokenizer.from_pretrained` — long after the endpoint answered.

---

## Deliberately not built

- **A custom load generator.** vLLM replays timestamps natively.
- **Adapters beyond BurstGPT and Mooncake.** Two prove the canonical schema.
- **Discovery mode.** The handoff is defined; the mode is not designed here.
- **A calibrated distance threshold.** The type refuses a number without the run
  that produced it.
- **Session-aware *firing*.** BurstGPT_3 carries session identity and the adapter
  reads it, but `timed_trace` has no session field. Deriving prefix blocks from
  sessions would invent cache hits.

## Known open

- **Knob and environment fields are `D2`** until the shared
  config-capture schema exists. `EnvCapture` is the named subset this needs; it
  is **deleted** and those types imported verbatim the moment they land. No
  translation layer.
- **Pacing under saturation.** The smoke-test server answers instantly, so pacing
  was verified against a server that never queues.
- **The viewer's chart has not been looked at in a browser** — the HTTP layer and
  page bytes are checked; the rendering is not.
