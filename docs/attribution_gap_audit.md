# Attribution & deviation gap audit

What the attribution and deviation code actually computes and supports,
measured against `docs/invariants.md` and the modules' own docstrings. The
audit answers three questions:

1. **Q1.** Which residuals are computed?
2. **Q2.** Which mechanisms can the observations tell apart?
3. **Q3.** Where does the code assert a cause that the observations cannot
   establish?

Every gap is filed in the [gap register](#gap-register) with a code path and a
concrete failure case.

**Pinned to:** `21348b5` (2026-09-23), first audited at `7450b93`
(2026-09-22). Environment: `.venv`, Python 3.14, statsmodels 0.15.0.

**Revalidation at `21348b5`.** #119 and #120 touched three in-scope files:
`deviation.py` gained `stream_observed_by_phase`, `loop.py` gained history-based
ranking (a `use_history` flag, with history passed into candidate selection), and `monitor.py` changed a comment only. None of the code behind a gap
changed, so every verdict stands; line references were remapped to the new
commit. G-01 was re-run and still reproduces. The new `deviation_table.py` has
no production consumer yet (tests only). It prices per op and layer, so it
avoids the G-06 and G-07 shapes, but it inherits G-05 (`_verdict` uses the same
±0.4 band around the peak point, so 60% of peak reads `over_floor`) and G-21,
G-22 and G-23 through `observed_op` (an NVTX container label becomes its own
row, and a router in the MoE range is keyed `moe_routed`), all VERIFIED.

**Blind re-verification at `21348b5`.** Each gap was re-checked by an
independent verifier that saw only its claim, code path and failure case, not
the evidence or verdict. None was refuted. Seven were narrowed (G-01, G-10,
G-11, G-15, G-21, G-35, G-38), three citations were corrected (G-16, G-28,
G-32), and caveats were added to G-08, G-14, G-17, G-18, G-19, G-24 and G-29.
The entries below include those changes.

**Scope:** `gitm/optimizer/deviation.py`, `attribution.py`, `dr.py`,
`invariants.py`, `monitor.py`, `scheduler_attribution.py`,
`collective_signal.py`, `measure.py`; `gitm/importers/node_rollup.py`;
`gitm/runtime_driver.py`; and the consumer, `gitm/scheduler/loop.py`.

**Evidence standard.** Each gap carries one of two verdicts:

| verdict | meaning |
|---|---|
| **VERIFIED** | A check ran against the code at the pinned commit and reproduced the behaviour. Measured values are quoted. |
| **VERIFIED-BY-READING** | No executable check is possible (a domain claim or a constant). Reading the code confirms it. Partial reading-only support inside a VERIFIED gap is marked *by reading*. |

Checks ran from the repo root with `.venv/bin/python` and fixed random seeds.
Where a check builds its own input, it uses real kernel names from
`tests/test_deviation_alignment.py:55-68` wherever they exist; names with no
real source are marked *illustrative*. Because Granger does not run as
installed (G-01), its behaviour when it does run was checked with the removed
`verbose=` argument dropped; those checks are marked *(G-01 bypassed)*.

**Bottom line.**
- Of the three invariants, only kernel time is computed. It is measured against a single point at peak speed, and its severity is always 1.0. Memory traffic is never computed. Stream concurrency computes a different quantity that fires on ordinary single-stream runs.
- The observations separate "an op ran outside its floor" only partly. They cannot separate memory-traffic excess, lost overlap, contention, straggler waits or low offered load from their look-alikes.
- Wherever statsmodels is not pinned by `constraints.txt` (CI, local dev), Granger attribution returns nothing, so every claim without a live A/B reads "no strong causal signal" whatever the data (G-01).
- When Granger does run, its series are misaligned in time, untested against multiple comparisons and open to shared drivers. The doubly-robust estimator shares the misalignment, and its "agreement" requirement is never checked.
- Scheduler and collective "causes" are observed conditions with an unmeasured mechanism and knob advice attached. The knob match to candidates is by name, not direction.

---

## Q1: Which residuals are computed

| quantity | `invariants.md` says | what the code computes | status | gaps |
|---|---|---|---|---|
| `r_kt` (kernel time) | `t_pred_lo ≤ t_obs ≤ t_pred_hi`, band from the vendor efficiency window (`invariants.md:12-16`) | `(t_obs − t_pred)/t_pred` against a single `t_pred` at 100% of peak (`roofline.py:728-732`), with a flat ±0.4 band (`invariants.py:22`). `eff_lo`/`eff_hi` (`roofline.py:85-86`) are never read. | Computed, against the wrong yardstick | G-05 |
| `r_kt` pairing unit | "for every kernel `k` in the predicted execution graph" (`invariants.md:8-9`) | Each observed kernel is compared with its whole op's prediction (`deviation.py:284-289`, `monitor.py:142-157`), so cache-insert and activation launches carry the full attention or GEMM prediction. | Computed, wrong unit | G-07 |
| `r_kt` reference node in `gitm deviate` / `deviations.json` | same as above | `deviation.py:278-280` takes the first node of each op as the reference for every layer. `monitor.py` pairs per structural class; `deviation.py` does not. | Computed, wrong reference on heterogeneous stacks | G-06 |
| `r_mt` (memory traffic) | Tier 1, "always check" (`invariants.md:20-26`, `invariants.md:51`) | Needs `bytes_read`/`bytes_written`, which nothing sets (`_common.py:282-283`). Every `r_mt` is `None`; `check_invariants` never emits `memory_traffic` (`monitor.py:254-258`). | **Never computed** | G-04 |
| `r_sc` (stream concurrency) | `serialized_fraction(C)` over the planner's concurrent set `C` (`invariants.md:30-34`) | No `C` exists; `expected_stream_id` (`graph.py:101`) has no reader. `_serialized_fraction` (`monitor.py:187-205`) counts adjacent same-stream pairs across all kernels, and fires above 0.5 rather than on any pair (`monitor.py:271`). | Computes a different quantity | G-03, G-25 |
| Severity | `clamp(|r|/band, 0, 1)`, comparable across invariants (`invariants.md:38-47`) | Kernel-time and memory violations are emitted only when `|r| > band`, so severity is always 1.0 (`monitor.py:243-251`). Stream concurrency ranges over (0.5, 1.0]. | Saturates at 1.0 | G-09 |
| Tiers | Tier 2 "only for multi-stream configs" (`invariants.md:52`) | `Invariant.tier` (`invariants.py:18`) is never read. | Never read | G-26 |
| Unmodeled and unobserved ops | Every predicted kernel is checked | Kernels with no op, or an op not in the graph, are dropped (`monitor.py:135-140`). Predicted ops with no observed kernel produce no residual. Nothing counts either. | Dropped silently | G-08 |
| Per-rank scoping | Implicit: one trace vs one graph | `monitor.py:123` reads every kernel with no pid/device filter; the loop calls it on merged captures (`loop.py:651`). | Absent | G-24 |
| Second "kernel_time" | Invariant 1 is against the roofline prediction | `measure.py:80-89` and `runtime_driver.py:410-423` compute each kernel's duration against its own name's median, then emit `kernel_time` violations under the same name. | A different quantity under the same label | G-10 |
| Graphs checked by `gitm deviate` | Any predicted graph | Dense models exit with code 2 (`deviation.py:872-874`). | Partial coverage | G-34 |
| Violation record | — | `Violation.detail` (`invariants.py:35`) is never populated. | Unused field | G-38 |

---

## Q2: Which mechanisms the code can tell apart

"Can the observations separate it?" asks whether anything the code receives
distinguishes this mechanism from its look-alikes, not whether a report names it.

| mechanism | separable? | why | gaps |
|---|---|---|---|
| Op slower than its floor | **Partly** | Only when a kernel classifies to the right modeled op, the layer is known, and one kernel is one op. Multi-kernel ops (G-07), the peak-point yardstick (G-05), the first-node reference (G-06) and misrouted kernels (G-22, G-23) all blur it. | G-05, G-06, G-07, G-22, G-23 |
| Memory-traffic excess | **No** | No byte counts are captured, so extra KV bytes show up only as kernel time and can't be told apart from low bandwidth efficiency. | G-04 |
| Lost stream overlap | **No** | There is no planned concurrent set to compare against, and same-stream serialization is normal CUDA ordering. | G-03, G-25 |
| Unmodeled work | **Partly** | The `gitm deviate` table keeps an `<unmodeled>` row (`deviation.py:746-748`), but opaque NVTX labels leak out of it as "observed but not predicted" (G-21). `monitor.residuals` drops unmodeled kernels without a count (G-08), so attribution never sees them. | G-08, G-21 |
| Preemption | **Condition only** | Preemption counts are observed directly. What they cost in throughput is not measured, and no kernel residual is joined to them. | G-32, G-35 |
| Scheduler limit vs low offered load | **No** | Occupancy is `num_running / max_num_seqs` (`vllm_stats.py:399-407`). Low occupancy fires whether or not requests were waiting; peaks from different moments are compared. | G-16, G-31 |
| Comm cost vs straggler wait vs merged ranks | **No** | `device_comm_stats` reads only kernel intervals, so a transfer and a wait on a late peer are identical. Ranks sharing a local `device_id` are merged, custom all-reduce is counted as compute, PP point-to-point is priced as TP all-reduce, and idle time dilutes the share. | G-18, G-19, G-20, G-22, G-36 |
| Cache/bandwidth contention | **No** | `attribute()` and `attribute_dr()` receive only `r_kt`. No L2, HBM, SM-occupancy or clock counter reaches them. | G-11, G-13 |
| Inter-op causation | **No** | Series are paired by launch index, not step (G-02); a shared driver makes every pair significant (G-11, G-13); the graph's edges are ignored (G-28). | G-02, G-11, G-13, G-28 |

---

## Q3: Where a cause is asserted beyond the evidence

One entry per place a cause is emitted. Each gives the exact text or field, where
it surfaces, and why the observations cannot support it.

### Q3.1 Granger `cause→effect` in claims
- **Emitted:** `causal_evidence = "{cause_op}→{effect_op} (p={p_value:.2g})"` for `hypotheses.top(2)`, or `"no strong causal signal"` when the list is empty (`loop.py:850-853`; the autoresearch path repeats it at `loop.py:916-918`). The module docstring (`attribution.py:3-5`) promises that "the MLP that contended for cache shows up as a Granger-cause of attention's residual".
- **Surfaces in:** `Claim.causal_evidence` in the report; `residuals.json` `top_hypotheses_granger`.
- **Why unsupported:**
  - With statsmodels 0.15 (any install not pinned by `constraints.txt`), every test raises and is swallowed, so the empty-list fallback is printed on every claim without a live A/B. It reads as a negative result (G-01).
  - Row `i` of two series is the `i`-th launch of each op, not the same step, so "lag" is not time (G-02).
  - Only `r_kt` is input. A shared slowdown makes every ordered pair significant, and no counter can name contention (G-11).
  - 380 uncorrected tests at 20 ops, each the minimum over lags, make the top pair significant on noise (G-12).
  - `direction` is the cause's mean sign, not the effect's (G-27). The `graph` argument is ignored (G-28). Trends and interval zeros enter unfiltered (G-29).
  - Every catalog claim without a live A/B carries the same pair and the same run-level residual, whatever its knobs touch (G-15).
- **Gaps:** G-01, G-02, G-11, G-12, G-15, G-27, G-28, G-29.

### Q3.2 The doubly-robust "agreeing is the bar"
- **Emitted:** docstring `dr.py:8`: "Running both and agreeing is the bar before we act on a cause." The ATE is presented as the causal effect (`dr.py:3-8`).
- **Surfaces in:** `residuals.json` `top_hypotheses_doubly_robust` only (`loop.py:681-685`). Claims never cite it.
- **Why unsupported:**
  - No agreement check exists; claims cite Granger alone (G-14).
  - DR pairs the same misaligned rows as Granger (G-02).
  - The only covariate is sequence position (`dr.py:123`), so double robustness does not cover a shared driver (G-13).
  - Treatment is `|r| > band`, which lumps fast and slow anomalies (G-30). Groups of 3 get a normal-approximation p-value (G-37, *by reading*).
- **Gaps:** G-02, G-13, G-14, G-30, G-37.

### Q3.3 Scheduler causes and their knob attachment
- **Emitted:** `SchedulerCause.effect` strings such as `"decode is launch-bound (small batches)"` (`scheduler_attribution.py:70-89`), `"decode throughput (recompute after preemption)"` and `"throughput ceiling (requests waiting, not running)"` (`scheduler_attribution.py:95-113`), each with a note that recommends knobs ("Raise max_num_seqs…").
- **Surfaces in:** `residuals.json` `scheduler_causes`; appended to a candidate's `causal_evidence` as `"; scheduler[{signal}]: {note}"` (`loop.py:856-859`) when any of its knobs appears in `motivates_knobs` (`loop.py:785-793`).
- **Why unsupported:**
  - "Launch-bound" is inferred from occupancy alone; no kernel-timeline data is consulted, and low offered load produces the same number. That raising the cap changes nothing there is *by reading* (G-16).
  - The knob match ignores direction: a preemption cause arguing to *lower* `max_num_seqs` attaches to a candidate that raises it (G-17).
  - Backlog compares peaks from different moments and still says "Raise max_num_seqs" when KV exhaustion is the block (G-31; the different-moments part is *by reading*).
  - No cause is checked against a kernel-time symptom (G-32). Severities are unnormalized for window length, and a matching scheduler cause always wins over a collective one, whatever either's severity (G-35).
- **Gaps:** G-16, G-17, G-31, G-32, G-35.

### Q3.4 Collective causes
- **Emitted:** `exposed_collective` ("step time inflated by non-overlapped communication … Consider a smaller tensor_parallel_size … or an executor backend that overlaps comm with compute", `collective_signal.py:108-125`) and `collective_dominant` ("The parallelism topology may be over-split", `collective_signal.py:129-144`), with knobs from `collective_signal.py:36-40`.
- **Surfaces in:** `residuals.json` collective causes; appended to `causal_evidence` as `"; collective[{signal}]: {note}"` by the same knob-name match.
- **Why unsupported:**
  - Exposed collective time cannot be separated from waiting on a slow peer (G-20; the mechanism difference is *by reading*).
  - `distributed_executor_backend` selects the worker launcher, not NCCL overlap (G-33, *by reading*).
  - Merged ranks hide each other's comm (G-18). Custom all-reduce reads as compute (G-19). Wall-time dilution suppresses the cause on long captures (G-36).
- **Gaps:** G-18, G-19, G-20, G-33, G-36.

### Q3.5 Deviation "departures"
- **Emitted:** `deviations.json` (`loop.py:712-714`) lists the ops whose kernels were kept as departures. The `gitm deviate` table prints a verdict per op (`deviation.py:755-757`) and "observed but not predicted" for ops with no floor (`deviation.py:751-753`).
- **Why unsupported:**
  - Healthy compressed-layer kernels are kept because they are measured against layer 0 (G-06).
  - Cache-insert and activation launches are "departures" from the whole op's prediction (G-07).
  - NVTX container labels are reported as work the graph failed to predict (G-21).
- **Gaps:** G-06, G-07, G-21.

### Q3.6 Stream-concurrency violations
- **Emitted:** `Violation(invariant="stream_concurrency", node_op="<stream-set>", …)` (`monitor.py:269-281`).
- **Surfaces in:** `violations.json` (`loop.py:656`) and `residuals.json` `serialized_concurrency_fraction`.
- **Why unsupported:** it asserts that planned concurrency was lost, when no concurrency was planned and same-stream order is normal. On merged traces, the value depends on stream-ID collisions (G-03).
- **Gaps:** G-03.

---

## Gap register

Severity. **Critical:** the reported number or cause is wrong
on ordinary inputs. **High:** wrong on a common production configuration.
**Medium:** wrong on a realistic edge case, or a documented claim that isn't
implemented. **Low:** cosmetic, or dead code.

### Critical

#### G-01 · Critical · Q3
**Granger attribution returns nothing, silently, with statsmodels 0.15 (any install not pinned by `constraints.txt`).**
- **Code path:** `attribution.py:79` passes `verbose=False`, which statsmodels 0.15 removed; the `TypeError` is swallowed by `except Exception: continue` at `attribution.py:82-83`. `pyproject.toml` allows `statsmodels>=0.14`. `constraints.txt:18` pins 0.14.6, which still accepts `verbose=`, and both Dockerfiles install with it (`Dockerfile:56`, `Dockerfile.rocm:65`), so the shipped images are unaffected. CI (`tests.yml`) and a plain `pip install -e .` skip the constraints file and resolve 0.15. No test calls `attribute()`. A missing statsmodels takes the same silent path (`attribution.py:53-58`).
- **Failure case:** on CI or any install without the constraints file, `attribute()` returns an empty list, and `loop.py:850-853` writes "no strong causal signal" into every claim without a live A/B. That can't be told apart from a real negative result.
- **Evidence:** VERIFIED. On data with a planted link, 0 hypotheses as installed and 6 with `verbose=` dropped. The end-to-end run printed "no strong causal signal" 5 times. With the import blocked, 0 hypotheses and no warning.
- **What would close it:** a record of how many pair tests completed versus raised, reported with the hypotheses.

#### G-02 · Critical · Q2, Q3
**Series for different ops are paired by launch index, not by time.**
- **Code path:** `attribution.py:61-63` builds one series per op in trace order; `attribution.py:69-75` truncates to the shortest and stacks columns. `dr.py:115-123` pairs treatment `t[i]` and outcome `y[i]` the same way (`dr.py:127`, `dr.py:134`).
- **Failure case:** 200 steps of 8 layers with a planted step-level link from attention at step `s` to lm_head at `s+1`. After truncation, row `i` of attention is step `i // 8` and row `i` of lm_head is step `i`.
- **Evidence:** VERIFIED *(G-01 bypassed)*. Granger ranked the true pair 5th of 6 at p = 0.95; the same data aligned per step gave p = 2e-142. DR on the same data: ATE +0.036, p = 0.46, missing the link.
- **What would close it:** an engine-step (and layer) index on every kernel.

#### G-03 · Critical · Q1, Q2, Q3
**`serialized_concurrency_fraction` measures adjacent same-stream pairs, not a planned concurrent set, and flags single-stream runs.**
- **Code path:** `monitor.py:187-205`. `PredictedNode.expected_stream_id` (`graph.py:101`) has no reader in the optimizer; no set `C` exists.
- **Failure case:** 10 back-to-back kernels on one stream (normal CUDA ordering) produce a `stream_concurrency` violation. On a merged two-GPU trace the value follows stream-ID collisions, not behaviour.
- **Evidence:** VERIFIED. One stream: fraction 1.0, violation severity 1.0. Two GPUs both on stream 7: 1.0; streams 7 and 8: 0.0. The violation also appeared, unprompted, in the checks for G-06 and G-10.
- **What would close it:** a planner-declared concurrent set `C` with expected streams.

#### G-04 · Critical · Q1, Q2
**The memory-traffic residual is never computed.**
- **Code path:** `monitor.py:143-147` and `monitor.py:158-162` need `bytes_read`/`bytes_written`. The only writer, `_common.py:282-283`, sets `None`; the CUPTI decoder (`_cupti_decode.py:96-110`) never passes them. The dependent branch in `deviation.py:240-248` never runs.
- **Failure case:** every `r_mt` is `None`, so `memory_traffic` is never emitted (`monitor.py:254-258`) on a Tier 1 "always check" invariant. Extra KV bytes surface only as kernel time.
- **Evidence:** VERIFIED. 0 of 18,112 fixture kernels carry bytes; nothing in `gitm/` writes a non-`None` value. The only way in is a hand-written JSONL through `Trace.model_validate` (`replay.py:81`).
- **What would close it:** per-kernel bytes read and written from a hardware counter source.

### High

#### G-05 · High · Q1, Q2
**The kernel-time band is ±40% around a peak-rate point, not the efficiency interval.**
- **Code path:** `invariants.py:22` (`band_width=0.4`); `roofline()` returns one `t_pred` at 100% of peak (`roofline.py:728-732`); `eff_lo`/`eff_hi` (`roofline.py:85-86`) are never read.
- **Failure case:** a kernel at 60% of peak is inside the doc's (0.55, 0.95) band but is flagged. A kernel faster than the physical floor is reported at the same severity as a slow one.
- **Evidence:** VERIFIED. 60% of peak: residual +0.667, severity 1.0. Twice as fast as the floor: residual −0.5, severity 1.0.
- **What would close it:** a predicted interval from the efficiency window instead of a point.

#### G-06 · High · Q1, Q2, Q3
**`deviation.py` measures every layer against the first node of each op.**
- **Code path:** `deviation.py:278-280` (`by_op.setdefault(pn.op, pn)`), used at `deviation.py:285-289`. The docstring (`deviation.py:14-17`) says it "mirrors `check_invariants`"; `monitor.py:97-107` explains why one node per op is wrong for heterogeneous stacks.
- **Failure case:** DeepSeek-V4-class graph (43 layers, layers 0-1 sliding-window, the rest compressed). Every compressed-layer `attn_score_value` kernel is compared with layer 0's `t_pred`, and `deviations.json` blames attention on a healthy model.
- **Evidence:** VERIFIED. All 483 nodes fed a kernel lasting exactly its own `t_pred`; 104 were still kept (`attn_score_value` 42/44, `attn_kv_a` 42/44, `attn_kv_compress` 20/42), at `r_kt` +0.00 to +4.00. `monitor.residuals` + `check_invariants` on the same trace gave 0 kernel-time violations.
- **What would close it:** the layer (structural class) of each observed kernel, used to pick its reference node.

#### G-07 · High · Q1, Q2, Q3
**Each observed kernel is compared with the prediction for its whole op.**
- **Code path:** `deviation.py:284-289` → `_departs` (`deviation.py:235-237`); the same pattern in `monitor.py:142-157`.
- **Failure case:** on the llama-2-7b graph at batch 1, kv 4096 (attention `t_pred` 32.9 µs), a 3 µs `reshape_and_cache` launch and the `silu_and_mul` activation are each compared with a whole-op prediction. A bare cuBLAS GEMM classifies to `None`.
- **Evidence:** VERIFIED. `reshape_and_cache` gave `r_kt = -0.909`, `silu_and_mul` against `mlp_gate_up` gave `-0.955`, the bare GEMM gave no row (3 rows from 4 kernels), and all 4 were kept. At the default kv 128 the same launch reads slow (+1.9): the sign depends on the operating point.
- **What would close it:** predictions and observations in the same unit — per kernel, or per op per step.

#### G-08 · High · Q1, Q2
**Unmatched kernels and unobserved predicted ops are dropped without a count.**
- **Code path:** `monitor.py:135-140` (`continue` when the op is `None` or not in the graph). `_agg_kt_residual` (`loop.py:383-399`) then summarises the remainder.
- **Failure case:** bare-GEMM launches vanish, and the run-level residual reads as if it described the whole step. (`deviations.json` does count unmodeled kernels; the residual and claim path does not.)
- **Evidence:** VERIFIED. 3 kernels (2 bare GEMMs, 1 FlashAttention) gave 1 residual row, with no record of the drop. Five predicted ops with no observation gave no signal: `attn_out_proj`, `lm_head`, `mlp_down`, `mlp_gate_up`, `qkv_proj`.
- **What would close it:** the count and time of dropped kernels, and the list of predicted ops never observed.

#### G-09 · High · Q1
**Kernel-time and memory-traffic severity is always 1.0.**
- **Code path:** `monitor.py:243-251` emits only when `|r| > band`, then takes `min(|r|/band, 1.0)`; the same at `monitor.py:254-266`.
- **Failure case:** `r_kt = 0.41` and `r_kt = 12.0` rank equally, while stream concurrency ranges over (0.5, 1.0] (`monitor.py:271`, `monitor.py:279`), so severities are not comparable across invariants.
- **Evidence:** VERIFIED. Residuals 0.41 and 12.0 both gave severity 1.0.
- **What would close it:** the distance outside the band, unsaturated.

#### G-10 · High · Q1
**A second `kernel_time` residual, against each kernel's own median, is reported under the same name.**
- **Code path:** `measure.py:80-89` and `runtime_driver.py:410-423` compute `(dur − median(dur of same name)) / median`, then run `check_invariants` and Granger on families (`measure.py:91-104`, `runtime_driver.py:433-451`).
- **Failure case:** a consistently slow kernel never produces a violation; jitter does, and reaches the report as "kernel_time deviation on …" claims (`measure.py:129-134`, `runtime_driver.py:484-495`). `*_measure.json` stores only the violation count.
- **Evidence:** VERIFIED. 50 identical 5 ms launches gave 0 `kernel_time` violations; a jittered kernel produced them. The `measure_trace` docstring (`measure.py:67-68`) discloses the median baseline; the defect is the output label.
- **What would close it:** a predicted duration to compare against, or a distinct invariant name when there is none.

#### G-11 · High · Q2, Q3
**Granger reports a cause and a mechanism (cache contention) that nothing observed can support.**
- **Code path:** `attribution.py:3-5` (docstring); output as `cause_op → effect_op` at `loop.py:852` and `loop.py:917`.
- **Failure case:** a shared driver, such as clock throttling or a batch-size change, slows every op at once and makes most or all ordered pairs significant.
- **Evidence:** VERIFIED *(G-01 bypassed)*. 5 ops with no links and one shared AR(1) slowdown: 20 of 20 ordered pairs at p < 0.05, top p = 8e-9. An independent re-run gave 18 of 20 with a step driver and 7 of 20 with AR(0.9), so the share depends on the driver's strength. That only `r_kt` is input was confirmed by reading `attribute()`'s inputs.
- **What would close it:** cache, bandwidth, SM-occupancy and clock counters aligned with the residuals.

#### G-12 · High · Q3
**Hundreds of uncorrected tests, each the minimum p over lags.**
- **Code path:** `attribution.py:71-89` (all ordered pairs); `attribution.py:80-81` (`min(pvals)`).
- **Failure case:** 20 ops give 380 ordered pairs. On pure noise the top pair is significant, and `loop.py:852` prints it as `p=…`.
- **Evidence:** VERIFIED *(G-01 bypassed)*. 20 i.i.d. ops, 60 samples each, 60 seeds: median top p 0.001; top p < 0.01 in 60 of 60 runs. An empirical rate, not a bound.
- **What would close it:** the number of tests behind the reported pair and the top-p null distribution for that count.

#### G-13 · High · Q2, Q3
**DR still needs no unmeasured confounding, and its only covariate is sequence position.**
- **Code path:** `dr.py:12-15`; `dr.py:123` (`X = position`). The docstring at `dr.py:3-8` presents the ATE as the causal effect.
- **Failure case:** a shared slowdown puts every op out of band together, making every ATE large and significant with no causal link.
- **Evidence:** VERIFIED. On the no-link shared-throttle data from G-11, 20 of 20 DR pairs had p < 0.05.
- **What would close it:** per-step covariates for shared drivers (clock, batch size, kv_len).

#### G-14 · High · Q3
**The "both must agree" requirement is never checked.**
- **Code path:** `dr.py:8`; `loop.py:654` computes `dr_hypotheses`, which are only serialized (`loop.py:681-685`). Claims cite Granger alone (`loop.py:852`, `loop.py:917`).
- **Failure case:** Granger's and DR's top pairs disagree, and the claim still quotes Granger as the evidence. (Claims with a live A/B quote the A/B instead.)
- **Evidence:** VERIFIED. `dr_hypotheses` is used at `loop.py:654` (computed) and `loop.py:684` (serialized) only. With G-01, claims currently cite neither.
- **What would close it:** the DR result for the same pair, carried into the claim beside Granger's.

#### G-15 · High · Q3
**Every catalog claim without a live A/B cites the same run-level residual and the same Granger evidence.**
- **Code path:** `loop.py:807` (`kt_residual = _agg_kt_residual(res)`, "Same for every claim"); `loop.py:850-853`, `loop.py:916-918`; `residual_invariant="kernel_time"` hard-coded at `loop.py:863`.
- **Failure case:** a `kv_cache_dtype` candidate and a `max_num_batched_tokens` candidate carry the same `residual_value` and evidence string (illustrative candidates), though neither involves the ops the other touches. A live A/B replaces the Granger text, and autoresearch claims use the target op's residual instead (`loop.py:919`).
- **Evidence:** VERIFIED. `_agg_kt_residual` returns one scalar per run. In the end-to-end run, all 5 claims showed `kernel_time: -90.9%` and "no strong causal signal". The shared-pair half applies once G-01 is fixed.
- **What would close it:** residuals for the ops each candidate's knobs act on.

#### G-16 · High · Q2, Q3
**`under_filled_batch` asserts "launch-bound" from occupancy alone.**
- **Code path:** `scheduler_attribution.py:70-89`; occupancy is `num_running / max_num_seqs` (`vllm_stats.py:399-407`).
- **Failure case:** a benchmark offering 8 concurrent requests to a server with `max_num_seqs=256` gets "decode is launch-bound" and "Raise max_num_seqs". The cap isn't binding; load is low.
- **Evidence:** VERIFIED. Occupancy 3% with queue depth 0 produced the cause, with `max_num_seqs` first among its knobs. `scheduler_causes()` takes no kernel data. That raising the cap changes nothing is *by reading*.
- **What would close it:** queue depth over the window and kernel-timeline launch gaps.

#### G-17 · High · Q3
**Causes attach to candidates by knob name, without direction.**
- **Code path:** `loop.py:785-793` (`_find_motivating_cause`). Knob lists: `scheduler_attribution.py:64-66` (preemption: lower `max_num_seqs`), `scheduler_attribution.py:84-87` and `scheduler_attribution.py:111` (raise it). Appended at `loop.py:856-859`.
- **Failure case:** `max_num_seqs_dynamic` ("…to raise decode concurrency") picks up the preemption cause, whose note argues for lowering `max_num_seqs`.
- **Evidence:** VERIFIED. The nested closure, replicated verbatim on the real library entry, matched `kv_cache_preemption`. In the end-to-end run with `top_n_interventions=5` and no live engine, that candidate was not ranked, so the mismatch appears only when it is. The sweep's 0.5× candidate does lower `max_num_seqs`, so only its 2× and 4× candidates get the wrong cause.
- **What would close it:** the direction each cause argues for each knob.

#### G-18 · High · Q2, Q3
**`worst_device_comm` splits by `device_id` alone, merging ranks that share a local ID.**
- **Code path:** `collective_signal.py:61-63`. Its docstring (`collective_signal.py:46-53`) says overlap math must stay within one device; `deviation.observed_scopes` (`deviation.py:395-413`) keys on `(pid, device_id)`.
- **Failure case:** in one-process-per-GPU TP where every worker reports `device_id=0`, one rank's GEMMs hide another's all-reduce and exposed comm reads as 0. The run reads clean unless merged comm clears the 20% `collective_dominant` floor.
- **Evidence:** VERIFIED. Two pids sharing `device_id=0`: 0 ns exposed comm; distinct IDs: 2,000,000 ns. `tests/test_collective_signal.py:96` covers only distinct IDs.
- **What would close it:** process (rank) identity alongside `device_id` as the device key.

#### G-19 · High · Q2, Q3
**The comm classifier misses vLLM's custom all-reduce.**
- **Code path:** `_COMM_PATTERNS` (`node_rollup.py:21-34`) via `is_comm_kernel`, used by `device_comm_stats` (`node_rollup.py:153-154`).
- **Failure case:** on the TP custom all-reduce fast path (conditional on topology), custom all-reduce time reads as 0 (all comm, when no NCCL kernel runs), `collective_causes` can return nothing, and the all-reduce cost becomes "busy".
- **Evidence:** VERIFIED. `cross_device_reduce_1stage` and `cross_device_reduce_2stage` give `is_comm_kernel = False`, while `kernel_taxonomy.classify_kernel` says `collective` and `deviation.classify_op` says `tp_all_reduce`. `kernel_taxonomy.py:52-53` already has the complete list.
- **What would close it:** one collective-kernel list shared by all three classifiers.

#### G-20 · High · Q2, Q3
**Exposed collective time is read as comm-bound when it may be waiting on a slow peer.**
- **Code path:** `collective_signal.py:108-125`, `collective_signal.py:129-144`.
- **Failure case:** with MoE expert imbalance, early ranks spin inside `ncclDevKernel_AllReduce` waiting for the slow rank, and the report says "Consider a smaller tensor_parallel_size".
- **Evidence:** VERIFIED. `device_comm_stats` reads only kernel intervals; a 2 ms transfer and a 2 ms wait give identical stats and the same causes (`exposed_collective`, `collective_dominant`). The mechanism difference is *by reading*.
- **What would close it:** per-rank arrival times at each collective.

### Medium

#### G-21 · Medium · Q2, Q3
**`observed_op` returns opaque NVTX container labels as op names.**
- **Code path:** `deviation.py:202-203` (`return guessed or range_op`). The docstring (`deviation.py:196`) says container labels "are not identities".
- **Failure case:** a bare GEMM in `model.layers.3.mlp` becomes op `"model.layers.3.mlp"`, printed as "observed but not predicted" (`deviation.py:751-753`) instead of counted in `<unmodeled>` (`deviation.py:746-748`), one row per layer. vLLM's JSON layerwise NVTX ranges are first parsed to `mlp` with a layer number, so there the leak is a single merged `mlp` row.
- **Evidence:** VERIFIED. A two-kernel JSONL through `stream_observed` → `render_deviation` produced that row; `<unmodeled>` was absent.
- **What would close it:** a known-op check on the NVTX label before it is used as an identity.

#### G-22 · Medium · Q2
**Several `_OP_RULES` patterns route kernels to the wrong op.**
- **Code path:** `deviation.py:51-52` (`"nccl"` → `tp_all_reduce`), `deviation.py:176` (`"logits"` → `lm_head`), `deviation.py:158` (`"index_select"` → `embed_tokens`).
- **Failure case:** `ncclDevKernel_SendRecv` (PP point-to-point) and `ncclDevKernel_Broadcast_*` are priced against the TP all-reduce prediction.
- **Evidence:** VERIFIED with real NCCL names (`ncclDevKernel_SendRecv` also appears at `tests/test_importers_hypothesis.py:156`). The `logits` and `index_select` rules misroute *illustrative* names (`apply_penalties_to_logits`, `index_select_kernel`); no captured trace in the repo contains such a kernel, and PyTorch's real `indexSelectLargeIndex` is not matched.
- **What would close it:** the NCCL collective type and a name→op table built from captured traces.

#### G-23 · Medium · Q2
**The NVTX range overrides the router's own identity.**
- **Code path:** `deviation.py:204-207`; the exemption list omits `moe_router`, `attn_qnorm_rope_insert` and `embed_tokens`.
- **Failure case:** a router inside the MoE NVTX range is compared with the expert-GEMM prediction, though `moe_router` has its own node.
- **Evidence:** VERIFIED. `observed_op("topk_softmax_kernel", "moe_routed")` returns `"moe_routed"`; `classify_op` alone returns `"moe_router"` (real name, `tests/test_kernel_taxonomy.py:47`). The kimi-k2.6 graph has a `moe_router` node.
- **What would close it:** kernel identity taking precedence for every op that has its own graph node.

#### G-24 · Medium · Q1
**Residuals aren't scoped per worker or device.**
- **Code path:** `monitor.py:123` (no pid/device filter), called from `loop.py:651`; `_serialized_fraction` sorts all devices together (`monitor.py:198`).
- **Failure case:** a merged multi-GPU capture is compared with the loop's single whole-model graph (`loop.py:360-364`), so every rank's kernels pair with the same nodes and the prediction is counted once per rank. `gitm deviate` refuses this (`deviation.py:841-846`); the loop does not.
- **Evidence:** VERIFIED. A two-device trace gave 2 residual rows, with no scoping or refusal.
- **What would close it:** a `(pid, device_id)` scope on every residual.

#### G-25 · Medium · Q1, Q2
**The stream-concurrency threshold is 0.5, not "any pair".**
- **Code path:** `monitor.py:271` (`> band_width * 0.5`).
- **Failure case:** 40% of a planned-concurrent set serializing would not be reported (once G-03 is fixed).
- **Evidence:** VERIFIED. A serialized fraction of 0.4 gave 0 violations.
- **What would close it:** per-pair overlap results for the set `C`.

#### G-26 · Medium · Q1
**`tier` is never read, so stream concurrency runs on every configuration.**
- **Code path:** `invariants.py:18`.
- **Failure case:** a single-stream decode run is evaluated for stream concurrency, feeding G-03.
- **Evidence:** VERIFIED. Nothing reads `Invariant.tier`; a single-stream run emitted `stream_concurrency`.
- **What would close it:** the configuration's stream count.

#### G-27 · Medium · Q3
**`direction` is the sign of the cause's mean residual, not of its effect.**
- **Code path:** `attribution.py:84`.
- **Failure case:** a cause that runs slow on average with a negative lagged effect is labelled `"+ slower"`.
- **Evidence:** VERIFIED *(G-01 bypassed)*. Cause mean +0.3, lag-1 coefficient −0.8: p = 1e-103, direction `'+ slower'`.
- **What would close it:** the sign of the fitted lagged coefficient.

#### G-28 · Medium · Q2, Q3
**The `graph` argument is ignored, so there is no "residual subgraph".**
- **Code path:** `attribution.py:40`; `PredictedNode.depends_on` (`graph.py:102-115`) is never consulted.
- **Failure case:** pairs with no edge or shared resource rank the same as connected ones.
- **Evidence:** VERIFIED *(G-01 bypassed)*. `attribute(res, llama_graph)` and `attribute(res, None)` were identical.
- **What would close it:** graph edges and shared resources between op pairs.

#### G-29 · Medium · Q3
**Non-stationary series and interval residuals enter the test unfiltered.**
- **Code path:** `attribution.py:62-63` uses raw `kr.r_kt`, including `interval_based` rows, which are 0 by construction inside the predicted span (`monitor.py:71-84`).
- **Failure case:** during decode, kv_len grows every step, residuals drift together, and the shared trend produces spurious links.
- **Evidence:** VERIFIED *(G-01 bypassed)*. Two independent random walks, 60 seeds: p < 0.05 in 25% of runs against a nominal rate of at most about 10%. The interval-zero part is *by reading*. A series of only interval zeros makes the test raise, and that pair is dropped silently.
- **What would close it:** kv_len (or step index) per residual, and a flag on interval-based rows.

#### G-30 · Medium · Q3
**DR treatment lumps fast and slow anomalies together.**
- **Code path:** `dr.py:127` (`np.abs(...) > band`).
- **Failure case:** a cause sometimes −60% and sometimes +60% counts as treated both ways, and the effects partly cancel.
- **Evidence:** VERIFIED. Cause ±0.6 at 60 of 300 steps, effect = 0.8 × cause (correlation 0.98): `|x|` treatment gave ATE +0.131; slow-only treatment gave +0.523.
- **What would close it:** treatment split by the sign of the anomaly.

#### G-31 · Medium · Q2, Q3
**`admission_backlog` compares peaks from different moments and ignores memory-blocked admission.**
- **Code path:** `scheduler_attribution.py:95-113`; the summary keeps only maxima (`vllm_stats.py:521-523`).
- **Failure case:** a 64-request burst at t=0 that drains to 32 running reports a backlog for the whole window. When KV exhaustion blocks admission, the note still says "Raise max_num_seqs".
- **Evidence:** VERIFIED. Queue 64 / running 32 gave severity 1.0. With cache usage 99%, `admission_backlog` and `kv_cache_pressure` both fired and the backlog note was unchanged. The different-moments part is *by reading*.
- **What would close it:** time-aligned queue, running and KV-usage series.

#### G-32 · Medium · Q2, Q3
**Scheduler causes are never joined to the kernel residuals they claim to explain.**
- **Code path:** the docstring (`scheduler_attribution.py:8-10`) says they enter attribution "alongside the kernel-level Granger hypotheses"; `loop.py:688-692` only lists them side by side.
- **Failure case:** `effect="decode throughput (recompute after preemption)"` is asserted with no kernel-time violations at all.
- **Evidence:** VERIFIED. `scheduler_causes()` takes only a `SchedulerStatsSummary`; with no kernel data it still emitted that effect.
- **What would close it:** the named symptom measured in the same run.

#### G-33 · Medium · Q3
**`distributed_executor_backend` is suggested for comm/compute overlap.**
- **Code path:** `collective_signal.py:36-40`, `collective_signal.py:119-121`.
- **Failure case:** switching `mp` and `ray` changes how workers launch, not whether NCCL kernels overlap GEMMs.
- **Evidence:** VERIFIED-BY-READING (a domain claim about vLLM; not executable here).
- **What would close it:** a measured overlap change from switching the backend.

### Low

#### G-34 · Low · Q1
**`gitm deviate` refuses dense models.**
- **Code path:** `deviation.py:872-874`.
- **Failure case:** `gitm deviate trace.jsonl --model <dense config.json>` exits 2, though the invariants apply to any predicted graph.
- **Evidence:** VERIFIED. A Llama `config.json` gave exit 2, "…resolves to the dense family."
- **What would close it:** a dense-family graph on the `deviate` path.

#### G-35 · Low · Q3
**Scheduler severities ignore window length, and scheduler causes win over collective ones regardless of severity.**
- **Code path:** `scheduler_attribution.py:58` (`preemptions / 10`); `scheduler_attribution.py:100` (`peak_running == 0` makes the ratio the queue depth). `_find_motivating_cause` (`loop.py:785-793`) takes any matching scheduler cause before any collective cause, without comparing severities.
- **Failure case:** 10 preemptions in 10 s and in 10 h both score 1.0.
- **Evidence:** VERIFIED. Both windows gave 1.0; queue 1 / running 0 gave backlog severity 1.0.
- **What would close it:** window length and request count with each count.

#### G-36 · Low · Q2
**`exposed_comm_share_of_wall` divides by the whole trace, including idle time.**
- **Code path:** `node_rollup.py:152`, `node_rollup.py:164`; the per-device copy keeps the full `duration_ns` (`collective_signal.py:67`).
- **Failure case:** a 60 s capture with 5 s of decode: exposed comm at 20% of active time shows as 1.7% of wall, under `exposed_floor=0.05`.
- **Evidence:** VERIFIED. Share 1.67%; no `exposed_collective` fired.
- **What would close it:** the active-window boundaries of the capture.

#### G-37 · Low · Q3
**DR accepts groups of 3 with a normal-approximation p-value.**
- **Code path:** `dr.py:42`, `dr.py:129`; p-value at `dr.py:148`.
- **Failure case:** 3 treated samples yield a p-value from `erfc(|z|/√2)` with no small-sample correction.
- **Evidence:** VERIFIED-BY-READING.
- **What would close it:** treated and control counts large enough for the approximation, reported with the p-value.

#### G-38 · Low · Q1
**`Violation.detail` is never populated; the module ends in a stray string literal.**
- **Code path:** `invariants.py:35`, `invariants.py:37-44`.
- **Failure case:** every violation record has an empty `detail`. Violations carry `node_op`, `layer`, `residual` and `severity`, but none names the kernels behind it.
- **Evidence:** VERIFIED. No `detail=` is written in `gitm/optimizer`.
- **What would close it:** the kernels behind each violation.
