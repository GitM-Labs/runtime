# Playbook schema

The contract between detection and apply. A **row** is one promoted tuning
result; a **lookup** decides whether a past row applies to the workload in front
of you. Types in `gitm/playbook/schema.py`, matching in `gitm/playbook/match.py`,
worked rows in `benchmarks/playbook/examples.json`.

> **Read this first.** The distance *metric* below is designed and implemented.
> The distance *threshold* is **not calibrated**, no measurement exists that
> could calibrate it yet, and the shipped policy carries no number at all. Under
> `UNCALIBRATED_POLICY` the only automatic match is an **exact** regime match;
> everything else routes to conservative discovery. That is the honest state, and
> §5 says exactly what would change it.

```bash
python -m gitm.playbook --selftest                          # 17 checks
python -m gitm.playbook --show     benchmarks/playbook/examples.json
python -m gitm.playbook --distance benchmarks/playbook/examples.json ex1-… ex2-…
python -m gitm.playbook --lookup   benchmarks/playbook/examples.json ex1-…
```

---

## 1. What a row is

```
(model + revision, GPU SKU, environment, workload regime, knob set)
    -> measured delta + provenance
```

| part | type | matched how |
|---|---|---|
| `model`, `model_revision` | `str` | **exact** |
| `gpu_sku` | `str` | **exact** |
| `env` (`EnvCapture`) | engine + engine version | **exact** (§3) |
| `regime.source_kind` | `production \| synthetic \| scoreboard` | **exact** |
| `regime.concurrency` | `int \| None` | **exact**, by policy |
| `knobs` | `dict[str, bool\|int\|float\|str]` | **exact**, key set |
| `regime` numeric axes | `Regime` | **distance** (§2) |

`Regime` is deliverable 1's type, **imported**. There is no second copy — a
second one would drift inside a week and the distance would be measured in two
different coordinate systems. `check_regime_is_imported_not_redeclared` asserts
the field's annotation is literally `gitm.traffic.regime.Regime`.

### What the types refuse to hold

A type cannot see how a number was produced, so it cannot enforce "came through
the promotion rule". What it does instead is refuse rows that **could not** have:

| refused | why |
|---|---|
| `knobs={}` | a row with no knob set says nothing |
| `repeats < 2` | a single run has no variance and cannot clear D2-1 |
| a delta with throughput but no TTFT/ITL | D2 criterion 3 would be unenforceable |
| `trace_sha256=""` | a claim that cannot be traced back to bytes |
| any unknown field | `extra="forbid"` everywhere except `EnvCapture` (§3) |

### Two fields that are states, not absences

- **`evidence`** — `measured` or `illustrative`. The worked examples ship in the
  same format as real rows, so a field has to separate them. Every row in
  `examples.json` is `illustrative` and **none of them is selectable**; a lookup
  against the example file returns `no_match` with that as the stated reason.
- **`invalidated`** — an `Invalidation(reason, at, by)`, never a deletion. A
  deleted row leaves no record that the claim was ever made, which is the first
  thing a reviewer asks for.

### `delta_is_floor` — the D1-11 guard

BurstGPT has no prefix identity. When D1 replays it, `write_timed_trace`
**synthesizes** prefix blocks, unique per request, so lengths hold and *no prefix
sharing is invented that the source never had*. A prefix-cache knob measured on
such a trace therefore saw the **least** reuse the real traffic could have had.

`PlaybookRow.delta_is_floor` is `True` when `provenance.prefix_synthesized` is
set **and** the knob set touches prefix caching. Such a row's delta is a lower
bound — usable as "at least this much", never quotable as the gain — and
`summary()` prints `[FLOOR: prefixes synthesized]`. `ex6` in the examples is that
case; `ex2` is the control (same synthesized trace, a knob that does not depend
on reuse, so not a floor).

### Replay conditions in provenance

`replay_chunk_hash_size` is a field because it is deliverable 1's finding with
the worst failure mode: at vLLM's default of 16 against Mooncake's 512-token
blocks, every prompt is 32× short **while every count in the result still reads
correctly**. A row that does not record the block size cannot be checked for it.

### Where a row comes from — `row_from_runs`

A `BenchRun` (deliverable 1, seam 3) is **one arm**. A row is the **difference
between two**, so the two are joined here and nowhere else:

```python
from gitm.playbook import row_from_runs

row = row_from_runs(
    "prefix-cache-mooncake-h100", baseline_runs, treatment_runs,
    model="Qwen/Qwen3.6-35B-A3B-FP8", model_revision="95a723d0",
    gpu_sku="NVIDIA H100 80GB", env=EnvCapture(engine="vllm", engine_version="0.28.0"),
    knobs={"enable_prefix_caching": True},
)
```

Provenance is lifted off the runs, never retyped by the caller: the trace
checksum, drop counts, regime label and replay conditions come from the arms that
actually ran, which is the only way the row's checksum and the run's checksum
cannot drift apart.

**It refuses, rather than producing a row, when:**

| refusal | why |
|---|---|
| the arms did not run the same trace bytes or label to the same regime | a delta across two workloads measures the workloads |
| any run is not `promotable` (did not reconcile, or had failures) | below that bar there is nothing to take a difference of |
| the arms have unequal repeat counts | D2 interleaves A/B/A/B; unequal means the interleave broke |
| `repeats < 2` | delegated to `MeasuredDelta` — a single run has no variance |
| either arm is missing `output_throughput`, `p99_ttft_ms` or `p99_itl_ms` | D2's criterion 3 is unenforceable without both sides |

**What it deliberately does not do.** `throughput_pct` is a percentage of
`output_throughput` (`THROUGHPUT_METRIC`, stated once so two readers cannot mean
different numbers), computed on **medians, never means**. `throughput_ci95_pct`
and `latency_blowout` stay `None`: D2 owns the variance rule and the blowout
predicate, and inventing either here is the mistake `AxisTolerance` refuses to
make with a distance threshold. And "same config minus exactly one knob" is D2's,
enforced by diffing two config-capture records that do not exist yet (**R1**) —
until they do, the caller asserts it and the row carries a note saying so, which
disappears on its own the moment `config_capture` is real.

---

## 2. Regime distance

### The metric

Numeric axes are compared as **log2 ratios**, because what matters for token
counts and rates is the *factor*, not the difference:

```
|log2(a / b)|

1,024 vs 2,048 tokens  ->  1.00      (a 2x change)
1,024 vs 1,536 tokens  ->  0.58
1,024 vs 1,024 tokens  ->  0.00
```

Scale-free by construction: 100 vs 200 and 10,000 vs 20,000 are the same
distance, which is the property the axis needs. Symmetric. Two exact zeros are
equal; **one** zero is `inf` — "no output tokens at all" is not a small version
of "some output tokens", and infinity is what makes the combination say so
without a special case.

**Burstiness** uses a shifted ratio, `|log2((1+a)/(1+b))|`. A perfectly paced
trace has `D = 0` and a bare ratio would make it incomparable to everything
including another paced trace. The shift anchors the axis on the Poisson
reference:

```
flat (0.00)     vs poisson (1.00)   ->  1.00
burstgpt (1.01) vs mooncake (6.74)  ->  1.94
moderate (5.00) vs mooncake (6.74)  ->  0.37
```

### Combination: L-infinity

```
regime_distance = max(input_p50, input_p95, output_p50, output_p95,
                      io_ratio, burstiness)
```

Not a mean, not a Euclidean norm. **A row is as far away as its worst axis.** The
case this exists for, asserted in `check_linf_is_the_worst_axis`: a candidate
identical on five axes and 8× off on `input_p95`. The mean calls that a 0.5
mismatch and would apply the row; L-inf calls it 3.0 and does not. That is a
long-context workload against a short-context row.

For reference, the two real traces deliverable 1 measured are **L-inf 4.929
apart, limited by `input_p95`** (49,904 vs 1,638). If those two collapsed to a
small distance, the axes would be decoration.

### `rate_rps` is deliberately not an axis

It exists on `Regime` and is **not** in `DEFAULT_AXES`. Adding it because it is
there would be exactly the mistake this module is written to avoid.

| | |
|---|---|
| **include it if** | knob outcomes are shown to depend materially on offered load *after* concurrency and burstiness are accounted for |
| **omit it if** | the selected knobs are insensitive to rate once those two are fixed |
| **either way** | the decision is recorded with the experiment that settled it |

The decision is material, not cosmetic: two regimes identical except for an 8×
difference in offered rate are distance **0.0** by default and **3.0** with the
axis on (`check_rate_is_not_in_the_default_axes`). Turning it on is one field:

```python
MatchPolicy(name="with-rate", axes=(*DEFAULT_AXES, "rate_rps"), tolerances=…)
```

---

## 3. The exact-match gates

Checked **before** any distance is computed, and each returns a stated reason
rather than a large number.

| gate | rule | why not a distance |
|---|---|---|
| `model`, `model_revision` | exact | "nearly the same weights" is not a thing |
| `gpu_sku` | exact | ditto for silicon |
| `env.engine`, `env.engine_version` | exact | scheduler rewrites ship in point releases; a version bump is the most common way a knob's effect changes with nothing in the workload changing |
| `source_kind` | exact | a `scoreboard` row can never satisfy a production query, even when every numeric axis is identical (asserted) |
| `concurrency` | exact, `match_concurrency=True` | open-loop and a capped-concurrency run are different experiments |
| knob key set | exact | a lookup asks about a *specific* knob |

Loosening any of them is an edit to a named field on `MatchPolicy`, not an
accident inside a comparison. `EnvCapture` is the one model with `extra="allow"`,
so a capture record from a newer engine round-trips without being truncated —
but `compatible_with` reads the **named** fields only, so an unknown extra key
can never change a match decision.

When the shared config-capture schema lands (**R1**), `EnvCapture` is deleted and
Adit's types are **imported verbatim**. No translation layer: two schemas that
translate into each other are two schemas that drift. Everything waiting on this
is marked `pending-adit` and is grep-able.

---

## 4. Lookup, precedence, and the handoff

```python
result = lookup(playbook, query_identity, UNCALIBRATED_POLICY)
if result.route_to_discovery:
    ...                              # conservative discovery
else:
    apply(result.row.identity.knobs)
```

| status | meaning | returns a row? |
|---|---|---|
| `exact_regime` | distance 0.0 on every axis | ✅ |
| `near_regime` | inside every calibrated tolerance | ✅ |
| `uncalibrated` | candidates exist; some nonzero axis has no calibrated tolerance | ❌ → discovery |
| `no_match` | nothing passed the gates, or the nearest exceeds its tolerance | ❌ → discovery |

`route_to_discovery` is `True` for everything that is not a returned row. A
status that is sometimes a row and sometimes a suggestion is how a wrong row gets
applied inside a 72-hour window.

A miss carries what it rejected and why — `rejected: {row_id -> reason}` plus the
ranked `candidates` with their per-axis distances. Discovery starts warm, and a
human reading a miss learns which axis was the problem, not just that there was
one. **Discovery mode itself is not designed here**; only the handoff.

### Precedence

1. **Distance** — nearest first.
2. **In envelope** — among equally near rows, a row measured inside the observed
   envelope beats one sampled beyond it (`/xenv`). *This sits below distance,
   where `todo.md` had it above*: a nearby extrapolated point was still genuinely
   run, and preferring a 4×-away in-envelope row over it answers the wrong
   question.
3. **Recency** — most recently verified.
4. **Smallest claim** — conservative. A wrong row applied in the live window
   costs more than a missed opportunity.

Each of the four is asserted separately.

---

## 5. Calibration — what would remove `uncalibrated`

`AxisTolerance` **refuses a number without the experiment that produced it**:

```python
AxisTolerance(max_distance=1.0)                       # ValidationError
AxisTolerance(max_distance=1.0, calibration="prereg E4, 2026-09-14: "
                                            "sign flip at 1.4 on input_p95")   # ok
```

That is the enforcement behind "the threshold is currently unknown". A
placeholder cannot quietly become a production constant.

**Why `PLAYBOOK_MATCH_MAX_DISTANCE = 1.0` was not shipped.** Under log2 it means
"accept up to a 2× mismatch on every axis at once". That may well be safe for
`output_p50`. It is not obviously safe for long-context `input_p95`, for
prefix-cache reuse, or for a queue-sensitive scheduling policy — and nothing
measured says which. A number that reads as derived when it was picked is the
kind of thing that gets defended in front of a customer and then collapses.

### The procedure

1. Run the same knob across nearby regimes, varying **one** axis at a time.
2. Find where the effect changes sign, or where D2's latency-percentile criterion
   flips from pass to fail.
3. Set that axis's tolerance strictly inside the distance at which it flipped,
   and cite the run in `calibration`.
4. The L-inf limit is then **the strictest relevant per-axis tolerance, by
   construction** — there is no separate global number to choose.

### What blocks it today

- No run against a live endpoint **with a real knob** has happened. Seam 3 is
  closed and `row_from_runs` (§ 6) builds a row from two arms, but calibration
  needs the *same knob measured across nearby regimes*, which needs a GPU.
- D2 does not exist, so "the effect flipped" has no agreed predicate.

Until then the cost of `uncalibrated` is a discovery run, which is the cheap
failure. `MatchPolicy.uncalibrated_axes` reports the current state; today it is
all six.

---

## 6. Worked examples

`benchmarks/playbook/examples.json` — 6 rows, **0 selectable**, regenerated by
`python benchmarks/playbook/make_examples.py` from the repo root, with `gitm`
importable (`pip install -e .`, or `PYTHONPATH=.` — a bare script path does not
put the root on `sys.path` the way `python -m` does). The **regimes are real**, measured
off deliverable 1's sha256-pinned fixtures. The **deltas are invented** and every
row says so.

| row | what it demonstrates |
|---|---|
| `ex1-prefix-cache-mooncake` | the ordinary case: long-input, prefix-sharing production traffic |
| `ex2-max-num-seqs-burstgpt` | the counter-example — same model and GPU, no prefix identity in the source. Synthesized prefixes but **not** a floor, because the knob does not depend on reuse |
| `ex3-chunked-prefill-qwen` | the row D3 would produce; stays illustrative until Phase B runs (needs one 80 GB card) |
| `ex4-retired-engine-bump` | invalidated by an engine bump — **kept, with a reason** |
| `ex5-scoreboard-not-production` | the biggest claimed delta in the file, gated out of every production lookup **by equality, not by distance** |
| `ex6-prefix-cache-on-a-synthesized-trace` | `delta_is_floor` — a prefix-cache knob on a source with no prefix identity |

---

## 7. Not in scope

- **Discovery mode.** The handoff is defined; the mode is not designed here.
- **The apply runtime** (Seojun's) and **env capture** (Adit's).
- **The promotion rule** (D2). D4 says what a row *is*; D2 says when one may be
  created. `MeasuredDelta.latency_blowout` is the stored field where D2's
  criterion-3 verdict lands, so the promotion gate and the live revert trigger
  read the same value instead of each re-deriving it.
- **A calibrated threshold.** See §5.
