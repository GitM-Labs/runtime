# Mechanism model for deviation attribution

When a kernel or step runs slower than predicted, five mechanisms could be the
cause. This document states which of them the monitor's observations can tell
apart, under what assumptions, and what would separate the pairs it cannot.

**Pinned to** `21348b5` (main, 2026-09-24); code citations are lines at that commit.
Every lemma and tested claim has a test in `tests/test_mechanism_fixtures.py`,
built with the CPU-only generator `gitm/optimizer/mechanism_fixtures.py` and run
through the real `monitor.residuals`, `check_invariants` and
`node_rollup.device_comm_stats`. P5 and P6 are derived only. Nothing has run on
a GPU. A confound is shown by construction (two parameter settings, one
observation); a separation by an observation that differs between them.

## Summary

| pair | residuals | documented residuals | raw trace | untraced serving | separated by | kind of fix |
|---|---|---|---|---|---|---|
| **P1** slowdown vs additive cost, one operating point | same | same | same | same | an operating-point sweep | intervention |
| **P2** serialization vs idle gap | same | differs | differs | same | exposed side-stream time | code |
| **P3** regime gate vs slowdown (z constant or rising) | same | same | same | same | varying z out of time order, z recorded per step | intervention + telemetry |
| **P4** additive cost vs tracer overhead | same | same | same | differs | the `off` arm's TPOT | intervention (cheap) |
| **P5** efficiency loss vs extra traffic | same | differs | same | same | measured bytes (offline replay) | telemetry |
| **P6** slowdown on A vs B mislabeled as A | same | same | differs with NVTX | same | NVTX identity, per-step counts, a sweep | code or telemetry |

**Take P1 to hardware.** It is confounded at every passive layer, and its sweep
runs on the existing Kimi loop pod. The experiment is specified in
[`experiments/p1_fp8_kv_separation.md`](experiments/p1_fp8_kv_separation.md).

## 1. Setup

**Prediction.** The planner's roofline time for op `o` at operating point `x`
(`gitm/planner/roofline.py:729-732`):

```
t_o(x) = max( F_o(x) / P_peak,  B_o(x) / BW_peak,  n_launch · t_launch )
```

with FLOPs `F` and bytes `B` from shape arithmetic (decode attention has
`B ∝ kv_len`, `graph.py:228-229`). A `Graph` is priced at **one** operating point
`x_G` (`graph.py:190-203`), and every kernel is compared against `t_o(x_G)`.

**Truth.** Fixtures take true durations from a separate cost model, so truth
and prediction do not agree by construction:

```
d_o(x) = c0_o + max( F_o(x) / (P · η_c,o(x)),  B_o(x) · (1 + τ_o) / (BW · η_m,o(x)),  n_launch · t_launch )
```

It shares `F` and `B` with the planner (A0) but has its own fixed cost `c0`,
efficiencies `η`, and traffic excess `τ`. Under **matched truth** (`c0 = 0`,
`τ = 0`, constant `η`), `d = t / η`; statements below assume it unless noted.

**Lemma 3.** Saturating efficiency `η(B) = η_max · B / (B + B½)` gives
`d = B/(BW·η_max) + B½/(BW·η_max)`: constant efficiency plus a fixed cost. "Small
kernels are inefficient" and "every kernel pays a fixed cost" are one model.
(`test_lemma3_saturating_efficiency_is_additive`: 0 ns difference over 644
kernels.)

**Observation layers.** A confound is relative to what is observed:

| layer | contents |
|---|---|
| **residuals** | `monitor.residuals(trace, graph)` and its violations (§2) |
| **documented residuals** | the definitions in `docs/invariants.md` (§2, last table) |
| **raw trace** | every timestamp, stream, name, grid and NVTX range (`schema.py:26-44`) |
| **untraced serving** | TPOT of the `off` arm (`serving_summary.json`) |

Residuals are a function of the raw trace and the graph; so are documented
residuals except `r_mt`, which needs bytes no trace carries. Only untraced
serving bypasses the tracer. A confound is **fixable in code** if the raw trace
already separates it, **needs telemetry** if an uncaptured measurement would,
and **needs intervention** if only changing what is run would.

## 2. The residuals as implemented

**Kernel time, `r_kt = (t_obs − t_pred) / t_pred`**, for each kernel whose op
resolves (`deviation.py:195-207`) to a graph op (`monitor.py:134-140`), with
`t_obs` from the kernel's timestamps (`monitor.py:142`). It pairs with the exact
`(op, layer)` node when NVTX gives the layer, with the op's single class when
all layers agree, and otherwise uses an interval residual that is zero inside
the op's per-layer range (`monitor.py:149-176`, `71-84`). Consequences:

- `t_pred` is the peak-rate time, so a healthy kernel reads `r_kt = 1/η − 1`:
  0.25 at η = 0.8, but 0.43 at η = 0.7, already outside ±0.4
  (`test_g05_healthy_kernel_violates_at_eta_07`: 644 of 644 flagged).
- Kernels whose op does not resolve are dropped uncounted (`monitor.py:136-140`).
- Events carry no step, batch or kv_len (`schema.py:26-44`), so an op's series is
  in trace order only.

**Memory traffic, `r_mt`**, needs `bytes_read` and `bytes_written`
(`monitor.py:143-147`). No importer sets them (`_common.py:282`), so `r_mt` is
always `None` and `memory_traffic` never fires.

**Stream concurrency, `r_sc`** (`monitor.py:187-205`): sort all kernels, all
devices together, by start; count consecutive pairs that share a stream and do
not overlap; divide by the number of pairs.

**Lemma 1.** If kernels on one stream never overlap (A3), `r_sc` is the fraction
of start-order neighbours on the same stream. It depends only on the sequence of
stream IDs, not on durations or gaps, and is 1 for a single stream.
*Proof:* two start-order neighbours on the same stream are consecutive on that
stream, so by A3 they cannot overlap; the pair counts exactly when the streams
match. ∎ Hence any change that keeps start order keeps `r_sc`, and so does moving
an isolated kernel within a run of another stream (it adds two switches
wherever it lands). Same-nanosecond ties break by input order.
(`test_lemma1_r_sc_is_stream_sequence_only`: 0.667964 before and after halving
every duration; 1.0 on one stream.)

**Violations** (`monitor.py:208-282`) read only the residuals, so equal residuals
give equal violations (**Lemma 2**, `test_lemma2_equal_residuals_equal_violations`).
Kernel time is reported when `|r_kt| > 0.4` (`invariants.py:22`) and the op is
systematic (median above 0.4, `monitor.py:233-236`) or confirmed at robust
`z > 3` in two bases (`multibasis.py:49-75`); its severity is then always 1.0.
Stream concurrency fires at `r_sc > 0.5` (`monitor.py:271`), so a healthy
single-stream run always fires it.

| | implemented | documented |
|---|---|---|
| `r_kt` | against the peak point, ±0.4 | against `[t_pred_lo, t_pred_hi]` from the vendor efficiency window |
| `r_mt` | `None` | from measured bytes |
| `r_sc` | same-stream neighbours (Lemma 1) | share of the planned concurrent set `C` that failed to overlap |
| severity | 1.0 whenever reported | `clamp(|r|/band, 0, 1)` |

## 3. The five mechanisms

| id | mechanism | effect |
|---|---|---|
| **M1** | region slowdown `(R, α)` | `d ← (1+α)·d` on kernels in `R` (an op, layers, a device, or steps from `s*` on) |
| **M2** | serialization | a planned-concurrent kernel on another stream waits for its partner to end |
| **M3k** | additive cost, in-kernel `(R, δ)` | `d ← d + δ` on `R` |
| **M3g** | additive cost, gap `(R, δ)` | idle `δ` before each kernel in `R` |
| **M4** | regime gate `(M, z*)` | mechanism `M` only on steps with `z_s > z*` |
| **M5ε** | tracer overhead `(ε)` | the traced run pays `ε` per kernel; the untraced system does not |
| **M5m** | misroute `(B → A)` | kernels of `B` are recorded as `A` |

Mechanisms apply in the listed order (M1 then M3k is `(1+α)d + δ`; the reverse
is `(1+α)(d + δ)`). Tracer overhead is a distortion because the system being
diagnosed runs untraced.

Effect on each observation (matched truth, A3, no contention):

| | `r_kt` | `r_sc` | step time | exposed side-stream time | untraced TPOT |
|---|---|---|---|---|---|
| M1 | `1+r` × `(1+α)` on `R` | unchanged if start order kept | up | may change | up |
| M2 | 0 | unchanged if stream-switch count kept | up (lost overlap) | up | up |
| M3k | `+δ/t_pred` on `R` | as M1 | up | may change | up |
| M3g | 0 | unchanged | up | 0 | up |
| M4 | inner mechanism, gated steps only | same | same | same | same |
| M5ε | `+ε/t_pred` everywhere | as M3k | traced run only | as M3k | 0 |
| M5m | `B`'s kernels read `d_B/t_A − 1` in `A`'s series | 0 | 0 | 0 | 0 |

So the residuals see only durations and stream order: M2 and M3g are invisible
to them. And multiplicative (M1) and additive (M3k, M5ε) changes differ only
through `t_pred`, which is where P1's separation comes from.

## 4. Propositions

### P1. Region slowdown vs in-kernel additive cost

At one operating point, `M1(R, α)` and `M3k(R, δ_k = α·d_k)` give the same
execution, since `(1+α)d = d + αd`. They are confounded at every passive layer.
(A1; A5 allows a per-kernel `δ_k`; one `δ` suffices if every kernel in `R` has
the same shape.)

**Separation: an operating-point sweep** with the mechanism held fixed, each
point predicted at its own `x` (not the monitor's single `x_G`). Fit
`d = a·t(x) + b`:

| | `a` | `b` |
|---|---|---|
| M1 | `(1+α)/η` | 0 |
| M3k | `1/η` | `δ` |
| M1 then M3k | `(1+α)/η` | `δ` |
| M3k then M1 | `(1+α)/η` | `(1+α)δ` |

Two distinct `t(x)` identify `(a, b)`; three are needed to test the line. This
separates the multiplicative and additive parts of the deviation, not `α` from
`η`. To attribute a *change*, compare with a reference sweep `(a0, b0)`: M1
scales both (`a/a0 = b/b0 = 1+α`), M3k moves only `b` (`b − b0 = δ`).

Caveats:

- **Baseline fixed cost `c0`.** M1 alone then gives `b = (1+α)c0 > 0`, so `b > 0`
  means "the deviation has an additive part", not "the mechanism is additive".
  Only the reference comparison attributes a change.
- **Efficiency that changes with `x`** (a kernel variant switch at long context)
  breaks A4: a straight line gets a spurious intercept of either sign.
  Undetectable with two points; with three or more, a lack-of-fit test or a
  change of kernel name/grid flags it. Report it as a broken assumption.
- **Proportional traffic excess `τ`** moves only `a` (P5's confound inside M1);
  a constant byte excess is an additive cost.
- **Weaker passive separations**, each needing an extra assumption: a shared `δ`
  across ops makes `r' − r = δ/t_o` vary as `1/t_o`, while a shared `α` keeps
  `(1+r')/(1+r)` constant; and duration spread separates them only given a noise
  model (multiplicative noise: M1 keeps the CV, M3k lowers it; additive noise:
  M1 scales the SD, M3k keeps it).

Tests (attention, batch 8, η = 0.8):

- `test_p1_confound_single_operating_point`: α = 0.5 and δ = 82,282 ns at kv 2048
  give identical trace events (644), `r_kt` 0.8750, TPOT 16,180.932 µs, and the
  same flagged op.
- `test_p1_sweep_separates`: over kv 1024 to 16384, duration ÷ baseline is 1.5
  throughout for M1 and 2.0 → 1.0625 for M3k; fits give `a` 1.875, `b` 0 (M1) and
  `a` 1.25, `b` 82.282 µs (M3k).
- `test_p1_reference_ratio_attributes_change`: with `c0` = 5 µs, M1 gives
  `a/a0 = b/b0 = 1.5`; M3k (20 µs) gives `a/a0 = 1`, `b − b0 = 20 µs`.
- `test_p1_mismatched_c0_enters_intercept`: `b` = 25 µs (c0 + M3k) and 7.5 µs
  (c0 + M1 alone).
- `test_p1_efficiency_step_flagged_by_lack_of_fit`: η 0.8 → 0.6 above kv 4096
  gives curvature 0.9448 (vs 7.65e-6 for a true line) and a spurious
  `b` = −68.971 µs.
- `test_p1_traffic_excess_moves_slope_only`: τ = 0.1 gives `a` = 1.375, `b` = 0.

### P2. Serialization vs idle gap

Setup: side-stream kernel `c` (a collective) is planned to overlap compute
kernels `u, v, w`; the next compute kernel `q` depends on `c`; every launch pays a
gap `g`. In the baseline `c` launches with `v` and is hidden inside it.

M2 (`c` waits for `w`) and M3g (idle `δ = d_c + g` before `q`) give identical
residuals and step time (A1, A3, no contention). *Why:* no duration changes.
The start order goes from `u,v,c,w,q` to `u,v,w,c,q`, still two stream switches,
so `r_sc` is equal (Lemma 1). In both cases `q` starts `d_c + 2g` after `w` ends.

**Separation: the raw trace.** Exposed side-stream time
(`node_rollup.py:147-165`) is `d_c` per occurrence under M2 and 0 under M3g. The
documented `r_sc` over a planned set containing `(c, v)` would also separate
them. One trace cannot say the overlap was *lost* rather than never planned;
that needs a reference run or the planner's set `C`, and `expected_stream_id`
(`graph.py:101`) exists but nothing in `gitm/optimizer` reads it. Under
contention, serializing also shortens `v` and `w`, but reading that needs a
contention model.

Tests: `test_p2_confound_serialization_vs_gap` (identical residuals, `r_sc`
0.667964, step time 15,353,060 ns, TPOT 15,353.060 µs);
`test_p2_exposed_comm_separates` (exposed comm 7,092,608 ns = 128 × 55,411 ns vs
0).

### P3. Regime gate vs region slowdown

A gate whose condition always holds is a plain slowdown. A gate on a condition
that only rises over time (kv_len during decode at fixed batch) is a slowdown
with an onset at the first step past the threshold. Both are confounded at every
passive layer (A1).

**Separation:** vary `z` out of time order (interleaved or shuffled arms), so a
gate follows `z` while an onset follows time; and record `z` per step. Events
have no step index or operating point (`schema.py:26-44`), so today `z` is known
only from the design (one point per arm) or from a kernel's position in the
series, which needs A6. Production batch variation cannot be used until step
metadata is captured.

Tests: `test_p3a_gate_always_on_equals_slowdown` and
`test_p3b_monotone_gate_equals_onset` (identical traces);
`test_p3_shuffled_z_separates` (z = 3,1,4,2 slows steps 0 and 2; no onset
reproduces that).

### P4. In-kernel additive cost vs tracer overhead

A real `δ` in every kernel and a tracer overhead `ε = δ` give the same traced
run, so every trace-derived layer is confounded. The untraced system differs
(A7: the `off` arm reflects it).

**Separation:** the `off` arm's TPOT rises under real cost and not under
overhead (the harness runs `off` arms, `runtime_harness/data/overhead.yaml`).
Varying the collector (`cupti` vs `nvtx`) also bounds `ε` if `δ` does not depend
on it. And `(TPOT_traced − TPOT_off) / critical kernels per step` bounds a
per-kernel `ε`, to subtract from P1's fitted `b`.

Tests: `test_p4_confound_overhead_vs_additive` (identical events, step time
13,869,908 ns); `test_p4_off_arm_separates` (off-arm TPOT 13,547.908 µs
baseline, 13,869.908 µs real cost, 13,547.908 µs overhead).

### P5. Efficiency loss vs extra traffic (derived)

A memory-bound kernel with efficiency `η/(1+α)` and one moving `(1+α)×` the bytes
have equal durations. With `r_mt` always `None` and no bytes in the trace, they
are confounded everywhere but the documented `r_mt` (0 vs `τ`). Byte counters
serialize kernels, so they must come from offline single-kernel replay, joined
by kernel name and grid (`monitor.py:143-147`, `_common.py:282`).

### P6. Slowdown on A vs B mislabeled as A (derived)

The monitor pairs by op name, so B's kernels recorded as A contribute
`d_B/t_A − 1` to A's series; if steady, the systematic rule flags A exactly as
M1 on A would. Example: vLLM's `reshape_and_cache_flash_kernel` is classified as
`attn_score_value` (`deviation.py:131`), so a ~3 µs launch is scored against all
of attention. **Separated by** NVTX identity, by A's per-step kernel count
exceeding its expected count (A6), or by a sweep (`d_B/t_A` drifts unless B and
A scale alike). Identity is decided by the name rules (`deviation.py:51-52`
sends every `nccl` kernel to `tp_all_reduce`) and the NVTX override
(`deviation.py:204-207`).

## 5. Assumptions

| id | assumption | used by | fails when |
|---|---|---|---|
| A0 | truth shares the planner's `F`, `B` (not its timing) | fixtures | a byte or FLOP formula is wrong in shape |
| A1 | no noise, or noise independent of the mechanism | P1–P4 | noise scales with duration |
| A2 | NVTX identity; one kernel per op per layer | fixtures, P1 | multi-kernel ops (split-KV, cache insert); unnamed GEMMs |
| A3 | same-stream kernels never overlap | Lemma 1, P2 | timestamp resolution; merged devices reusing stream IDs |
| A4 | cost is affine in `t_o(x)` across the sweep | P1 separation | variant or split count changes with kv_len |
| A5 | mechanism parameters fixed across operating points | P1 separation | a fixed cost that grows with batch |
| A6 | known step structure, no dropped kernels | P3, P6 | unmodeled kernels; CUDA graphs hiding launches |
| A7 | the `off` arm reflects the untraced execution | P4 | the `off` arm inherits injection variables (`verify_arm` checks this) |

## 6. Not covered

Multi-rank mechanisms (a rank waiting on a slow peer looks like a transfer);
scheduler-level mechanisms (admission limits vs low load act on the serving
plane); and estimation under noise, which the separating-experiment spec
([`experiments/p1_fp8_kv_separation.md`](experiments/p1_fp8_kv_separation.md))
covers with its decision rule and inconclusive outcome.

See also [`attribution_gap_audit.md`](attribution_gap_audit.md) (PR #126), which lists the
same code paths as defects with reproductions.
