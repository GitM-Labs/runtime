# P1 separating experiment: fp8 KV on Kimi K2.5 MLA decode attention

**Question.** Switching the KV cache to fp8 (`--kv-cache-dtype fp8`) changes
decode attention time. Is the change **multiplicative** (a region slowdown or
speedup, M1: `d → (1+α)·d`), **additive** (a fixed per-launch cost, M3k:
`d → d + δ`), both, or neither?

At one operating point these are indistinguishable: they produce the same trace,
residuals and TPOT (`docs/mechanism_model.md`, P1). A known knob change is a
known action, not a known mechanism, so this experiment establishes which it is.

**Status:** specified, not run. Written to be run as-is on the Kimi K2.5 /
8×MI355X loop pod (`deploy/k8s/mi355x-kimi-loop.yaml`). The decision rule is
fixed in code (`gitm/optimizer/separation.py`) and tested on CPU before any
hardware run.

## 1. Why the intervention separates them

Run the same op at several operating points `x` under both configurations, and
regress the candidate's per-launch time on the baseline's:

```
d_cand(x) = k · d_base(x) + m
```

| hypothesis | `d_cand(x)` | `k` | `m` |
|---|---|---|---|
| M1 (multiplicative) | `(1+α) · d_base(x)` | `1+α` | 0 |
| M3k (additive) | `d_base(x) + δ` | 1 | `δ` |
| both | `(1+α) · d_base(x) + δ` | `1+α` | `δ` |
| neither | `d_base(x)` | 1 | 0 |

With two distinct values of `d_base` the line, and so `(k, m)`, is identified;
with three or more its linearity can be tested. Varying context length moves
`d_base` over a 16× range here, because Kimi's MLA decode attention reads the
whole latent cache: bytes ∝ batch × kv_len (`gitm/planner/glm_graph.py:279-297`,
`index_topk` never binds for Kimi).

The baseline is the regressor, not the roofline `t(x)`. So the rule does not
need the planner to be right about how attention scales (assumption A4 in the
mechanism model). It needs only:

- **A5:** the mechanism's parameters (`α`, `δ`) do not change with `x`. The
  lack-of-fit gate tests this.
- **Same launches:** both arms measure the same unit of work: one attention call
  per layer per decode step, at the same decode batch. The grid filter and the
  load gates enforce this.

## 2. Design

| | |
|---|---|
| **Held fixed** | model and checkpoint, image, TP=8, node, vLLM flags other than the lever, concurrency c = 16, output length 256, GuideLLM synthetic data, tracer build, capture window length |
| **Varied (the intervention)** | KV cache dtype: baseline default (bf16) vs candidate `--kv-cache-dtype fp8` |
| **Varied (the separating condition)** | prompt length L ∈ {2048, 4096, 8192, 16384, 32768}, so decode kv ∈ [L, L+256] |
| **Observable** | per-launch time of MLA decode attention: the anchor kernel plus the target kernels that follow it on its stream (split-KV reduce), median over launches at the anchor's modal grid, per window |
| **Required telemetry** | GITM kernel trace (arm B/I: names, device timestamps, stream, grid); vLLM `/metrics` at 1 Hz (running, waiting, preemptions, KV usage); GuideLLM JSON (ITL p50/p95); untraced ITL from the off arms |

**Why fp8 KV:** it is a real lever the loop already runs (E8, `intervene` pod),
and both mechanisms are plausible. Halving the bytes read predicts a
multiplicative change; a dequantize step or a different kernel variant with a
fixed setup cost predicts an additive one.

**Why c = 16 and these lengths:** the KV cache fits comfortably at every point
(about 37 GB per GPU at L = 32768), so there is no preemption and the decode batch
is the same in both arms. The lengths span a 16× range of attention time.

### Arms and configurations

| node | phase | arm (`arm.sh`) | config | traced | purpose |
|---|---|---|---|---|---|
| 1 | `base1` | B | baseline | yes | reference sweep |
| 1 | `cand` | I `--kv-cache-dtype fp8` | candidate | yes | candidate sweep |
| 1 | `base2` | B | baseline | yes | drift check (ABA) |
| 2 | `base1` | A | baseline | no | untraced ITL, corroboration |
| 2 | `cand` | L `--kv-cache-dtype fp8` | candidate | no | untraced ITL, corroboration |

The decision uses node 1 only, so baseline and candidate share a node and
clocks. Node 2 checks that the multiplicative part also appears with tracing off
(P4), and costs nothing extra in wall time because it runs in parallel.

### Per point

GuideLLM runs 150 s at c = 16. On node 1, a 30 s capture window opens 40 s in,
once every stream is past its first prefill; that gives tens of thousands of
attention launches per GPU per window. Metrics are scraped for the whole point.

**Repetitions:** 3 per point per phase. Points run in a fixed shuffled order per
rep, the same on both nodes and in every phase:

| rep | order |
|---|---|
| 1 | 8192, 2048, 32768, 4096, 16384 |
| 2 | 16384, 32768, 4096, 2048, 8192 |
| 3 | 4096, 16384, 2048, 8192, 32768 |

Shuffling keeps time trends (thermal, clock drift) from lining up with context
length. That would be P3's confound.

### Budget

| | node 1 | node 2 |
|---|---|---|
| phases | 3 | 2 |
| per phase | sanity + warm-up ≈ 5 min, 15 points × 150 s ≈ 38 min | same |
| arm switches | 3 × model reload (≤ 60 min each, per `arm.sh`) | 2 × reload |
| total wall | ≤ 5.2 h | ≤ 3.5 h |

About 9 node-hours worst case (72 GPU-hours), with both nodes running in parallel.
Traces are 30 s windows, 45 per run. Pull them selectively (see §5).

## 3. Gates

Two kinds of gate. A **window** gate drops that one window, and the rest of the
run continues; the verdict still needs the reps and points gates below. A
**run** gate makes the outcome **inconclusive**. Every failure is recorded with
what would resolve it. No gate is waived after the fact.

| gate | kind | rule | why |
|---|---|---|---|
| correctness | run | after every arm switch, 32 fixed greedy prompts each return ≥ 16 tokens, not collapsed onto ≤ 2 distinct tokens (`separation sanity`) | fp8 KV must not have broken the model; the runner also stops the phase |
| capture | window | a trace exists, with ≥ 1000 anchor launches at the modal grid | enough launches for a stable median |
| load | window | median running requests ≥ 0.9 × 16 **inside the capture window** (`cap/metrics_samples.jsonl`); zero preemptions over the point | same decode batch in both arms |
| latency | window | `guidellm.json` present with a readable ITL; ITL p95 / p50 ≤ 3 | a stalling server is not a steady operating point. GuideLLM reports ITL per request, so this catches gross stalls only |
| points, reps | run | ≥ 3 operating points, each with ≥ 2 surviving reps in both arms | a line needs two points; testing it needs three; noise needs reps |
| repeatability | run | rep-to-rep CV ≤ 5% at every point, per arm | noise small enough for the margins |
| anchor | run | the same anchor kernel in every window of an arm, and in `base1` and `base2` | a per-window anchor flip would change what "one launch" means |
| kernel set | run | the same resolved target names at every point within an arm | a variant switch inside one arm |
| drift | run | `base2 ÷ base1`, over the points where both were measured (≥ 3): pooled change within ±2%, and no point beyond ±2% **and** 3 standard errors (from each phase's own noise) | the node did not change under the experiment |
| control | run | `rms_norm` per-launch time, candidate ÷ baseline, same rule as drift, with standard errors from the control's own rep-to-rep noise | KV dtype must not move an op that does not read KV |
| fit | run | fails only if the curvature exceeds 0.05 **and** the lack-of-fit F test gives p < 0.01 | one `(k, m)` describes all points (A5); a bend that noise explains is not a failure |

Reported but not gating: the modal grid per point and arm (a mismatch is a
warning to check the batch), and the per-kernel time difference between arms
(`kernel_diff`, below).

## 4. Decision rule (pre-registered)

Implemented as `decide()` in `gitm/optimizer/separation.py`. Its constants are
fixed there and written into every report.

1. **Per window:** resolve the op's kernels by rule, per arm: names containing
   `mla`, `attn`, `attention`, `flash`, `paged` or `decode`, and none of
   `cache`, `rope`, `rotary`, `norm`, `slot_mapping`, `concat`, `quant`,
   `prefill`, `varlen` or `proj`. The anchor is the target kernel with the most
   **total time** (split-KV stage 1, not its reduce, whose launch count ties
   it). A launch is the anchor plus the target kernels that follow it on its
   stream. Take the median per-launch time at the anchor's modal grid.
2. **Per point:** the mean over the surviving reps, for each arm.
3. **Noise:** per arm, `sd² = a² + (c·d)²` fitted over all points by
   non-negative least squares: an absolute floor `a` (timer resolution, launch
   jitter) plus noise proportional to time `c`. Degrees of freedom are the
   pooled rep dof less the two fitted terms (8 for 5 points × 3 reps).
4. **Fit** `d_cand = k · d_base + m` by weighted least squares, each point
   weighted by the inverse of its variance in both arms. 95% intervals for `k`
   and `m` come from 4000 redraws of every point mean from a t distribution with
   the pooled noise and degrees of freedom.
5. **Margins:** `κ = 0.03` on `k`. On `m`,
   `μ = max(0.25 µs, 3% of d_base at the shortest point, 2 × tracer leak)`,
   where the tracer leak is `|n_c − k·n_b| × ε_max`, with `n_b`, `n_c` the
   kernels per launch in each arm and `ε_max = 0.25 µs` (see §7). The baseline
   reads `d + n_b·ε` and the candidate `k·d + n_c·ε`, so that is what a pure
   slowdown leaks into `m`.
6. **Classify:**

| outcome | `k` interval | `m` interval |
|---|---|---|
| **multiplicative** | entirely outside `[1−κ, 1+κ]` | entirely inside `[−μ, μ]` |
| **additive** | entirely inside `[1−κ, 1+κ]` | entirely outside `[−μ, μ]` |
| **mixed** | outside | outside |
| **no effect** | inside | inside |
| **inconclusive** | a run gate failed, or either interval straddles its margin | |

An inconclusive result lists `remedies`: for example, more reps when an interval
straddles its margin, a denser sweep when the fit fails, or a rerun on a quiet
node when drift or the control fails.

**Outside the op.** Cache insert and quantize kernels are excluded from the op
on purpose: fp8 KV changes the insert, and folding it in would be P6's
misattribution. An additive cost can still live there, so the report's
`kernel_diff` lists, per point, the ten kernels whose time per attention launch
changed most between arms, flagging any present in only one arm. The verdict
covers the attention core; `kernel_diff` covers the rest.

**Corroboration (reported, not gating).** On node 2, regress the untraced ITL
difference (candidate − baseline) on `d_base` over at least 3 points. The slope
(with its standard error) should be close to `61 · (k − 1)`, since 61 layers each
run attention once per token. The slope is immune to tracer overhead. The
intercept is not a check on `m`, because fp8 KV also changes the cache insert.

## 5. How to run

Two single-node Deployments that already exist and idle until told to run:
`deploy/k8s/parallel/kimi-traced.yaml` (born in arm B, node 1) and
`deploy/k8s/parallel/kimi-sweep-long.yaml` (born in arm A, node 2). Both mount
the same `/mnt/shared`, so both halves land under one run id.

```bash
kubectl apply -f deploy/k8s/parallel/kimi-traced.yaml -f deploy/k8s/parallel/kimi-sweep-long.yaml
TRACED=$(kubectl get pods -l app=kimi-traced -o name | cut -d/ -f2)
OFF=$(kubectl get pods -l app=kimi-sweep-long -o name | cut -d/ -f2)

# 1. the sidecars run gitm from the shared wheel: ship one built from this branch
python -m build --wheel
kubectl cp dist/gitm_labs-*.whl $TRACED:/mnt/shared/gitm/wheel/ -c gitm
kubectl exec $TRACED -c gitm -- bash -c 'pip install -q --force-reinstall --no-deps /mnt/shared/gitm/wheel/gitm_labs-*.whl'
kubectl exec $OFF    -c gitm -- bash -c 'pip install -q --force-reinstall --no-deps /mnt/shared/gitm/wheel/gitm_labs-*.whl'

# 2. ship the two scripts, flat, as for run_parallel.sh
kubectl cp scripts/kimi_loop/p1_separation.sh $TRACED:/mnt/shared/gitm/scripts/p1_separation.sh -c gitm
kubectl cp scripts/kimi_loop/arm.sh $TRACED:/mnt/shared/gitm/scripts/arm.sh -c gitm

# 3. start both halves with one run id, detached (a dropped session must not kill a 5 h run)
RUN=$(date -u +%Y%m%d)-p1
R=/mnt/shared/gitm/results/$RUN/p1
kubectl exec $TRACED -c gitm -- bash -c "mkdir -p $R && GITM_RUN=$RUN nohup bash /mnt/shared/gitm/scripts/p1_separation.sh traced > $R/traced.log 2>&1 &"
kubectl exec $OFF    -c gitm -- bash -c "mkdir -p $R && GITM_RUN=$RUN nohup bash /mnt/shared/gitm/scripts/p1_separation.sh off > $R/off.log 2>&1 &"

# 4. once base1's first window exists, check what the rule resolves
kubectl exec $TRACED -c gitm -- python -m gitm.optimizer.separation inspect \
    /mnt/shared/gitm/results/$RUN/p1/traced/base1/L8192_r1/cap/trace.jsonl
```

Step 4 must show an MLA **decode** kernel as `anchor` (not prefill, not a GEMM,
not a reduce), a launch count in the tens of thousands, `kernels_per_launch` of
1 or 2, and a non-null `control_us` (the rms_norm control; without it every run
is inconclusive). If any of these is wrong, stop the run
(`pkill -f p1_separation`) and fix the needles in `separation.py` before going
on. That is a change to the rule, so commit it before looking at any candidate
data. Progress is in `p1/<node>.log`; failed windows are listed in
`p1/traced/<phase>/FAILED`.

When both halves have finished (the last line of each log is `p1 … complete`),
analyse **in the pod**: the traces are several GB per window, too many to pull,
and the analysis needs a few GB of memory per window.

```bash
kubectl exec $TRACED -c gitm -- python -m gitm.optimizer.separation analyze \
    /mnt/shared/gitm/results/$RUN --n-layers 61 --concurrency 16
# pull everything except the traces
kubectl exec $TRACED -c gitm -- tar -C /mnt/shared/gitm/results --exclude='trace.jsonl*' \
    -cf - $RUN/p1 | tar -C evidence/kimi-mi355x/runs -xf -
```

Output: `p1/decision.json` has the outcome, `k` and `m` with intervals, the
margin and the tracer leak, every gate, reasons and remedies, dropped windows,
the anchors, target names and grids per arm, per-point means, `kernel_diff`,
the off-arm corroboration, and the constants used.

## 6. Prediction, recorded before the run

The planner gives an fp8 KV entry exactly half the bytes of a bf16 one (576 vs
1152: latent and RoPE key both fp8), and prices decode attention as memory-bound
at every point:

| L | kv (mid) | planner `t_pred`, bf16 | fp8 | ratio |
|---|---|---|---|---|
| 2048 | 2176 | 5.01 µs | 2.51 µs | 0.500 |
| 4096 | 4224 | 9.73 µs | 4.87 µs | 0.500 |
| 8192 | 8320 | 19.17 µs | 9.59 µs | 0.500 |
| 16384 | 16512 | 38.04 µs | 19.03 µs | 0.500 |
| 32768 | 32896 | 75.79 µs | 37.91 µs | 0.500 |

**Prediction:** multiplicative, with `k` between 0.50 and 0.65. The upper end
allows for the kernel not reaching the same bandwidth efficiency on fp8. A
**mixed** result with `m > 0` would mean a per-launch dequantize or setup cost
the planner does not model. **Additive** or **no effect** would mean the fp8
path is not reading half the bytes. That would be a finding about the backend,
checkable by kernel name in the report.

**What the run can detect** (simulated at this scale: baseline = roofline ÷ 0.7
+ 1 µs, 3 reps, 1% noise on the control op, 300 runs per row;
`tests/test_separation.py` pins the key rows):

| rep-to-rep noise | pure `k` = 0.55 | `k` = 0.55 + `m` = 1.5 µs | pure `m` = 1.5 µs | nothing |
|---|---|---|---|---|
| 0.5% | multiplicative 97% | mixed 98% | additive 98% | no effect 96% |
| 1% | multiplicative 97% | mixed 93% | additive 97% | no effect 32%, else abstains |
| 2% | multiplicative 89% | mixed 92% | additive 25%, else abstains | abstains |

With noise proportional to time, no row ever returns a wrong verdict; the failure
mode is abstention. The 95% intervals cover the true `k` and `m` in 95–98% of
runs. With an absolute noise floor as well (0.2 µs plus 0.2%, 400 runs per
case), `m`'s intervals cover 92–95%, and a true 0.5 µs fixed cost (twice the
margin) on top of `k` = 0.55 was called multiplicative in 4 of 400 runs: the
error rate the 95% intervals allow, not a bias.

"No effect" is the hardest verdict to reach, because it needs `m` pinned inside
±0.25 µs; that is by design, since absence has to be shown, not assumed. If the
first windows show noise above 1%, add reps before trusting a "no effect".

## 7. What could invalidate the inference

| effect | what it does | guard |
|---|---|---|
| tracer inflates device-side durations by `ε` per kernel | under a pure `k`, the intercept becomes `(n_c − k·n_b)·ε` | `μ ≥ 2 × tracer leak`, reported with every `m`; off-arm slope is tracer-free. `ε_max = 0.25 µs` is an assumption: timestamps are device-side, and the tracer's per-dispatch work is on the host |
| noise with an absolute floor (timer resolution, jitter) | short points over-trusted if noise were assumed proportional | two-part noise model; §6 gives its calibration |
| decode batch differs between arms (fp8 frees KV, changing scheduling) | changes work per launch | in-window load gate; modal-grid filter; grid mismatch warning |
| a separate candidate-only kernel (dequantize, cast) | an additive cost outside the op | `kernel_diff` reports it; the verdict states it covers the attention core |
| kv spread within a point (kv grows by 256 during decode) | per-point time is a mean over kv | exact if cost is linear in kv (it is, for memory-bound MLA); fit gate catches otherwise |
| the candidate runs a different kernel variant | `k`, `m` then describe variant vs variant, not one kernel's mechanism | anchors and target names reported per arm |
| kernel variant or split count switches with context length | no single `(k, m)` | fit gate; kernel-set gate within an arm |
| node, clock or thermal drift | a multiplicative shift unrelated to the lever | same node, ABA drift gate, rms_norm control, shuffled order |
| prefill contention inside the window | inflates some launches | ramp before capture; modal grid; ITL tail gate |
| the rule resolves the wrong kernels | measures the wrong op | `inspect` step before analysis; anchor gate |
| fp8 changes the tokens generated | different sequences, same lengths | synthetic data with fixed output length; content is irrelevant to attention cost |

## 8. Validated on CPU

`tests/test_separation.py` (22 tests) runs the rule on synthetic traces from
`gitm/optimizer/mechanism_fixtures.py`, through the same per-window reduction as
the hardware analysis. The baseline carries a 5 µs fixed cost, so the
roofline-intercept test (`b > 0`) would be fooled; this rule is not. These
fixtures are at batch 8, so `μ` is 5.08 µs here; §6 gives the hardware-scale
calibration.

| injected into the candidate | outcome | `k` [95% CI] | `m` [95% CI] |
|---|---|---|---|
| speedup α = −0.4 | multiplicative | 0.600 [0.599, 0.601] | 0.00 [−0.43, 0.43] µs |
| fixed cost δ = 10 µs | additive | 1.000 [0.998, 1.002] | 10.00 [9.28, 10.72] µs |
| both | mixed | 0.600 [0.599, 0.601] | 10.00 [9.56, 10.44] µs |
| nothing | no effect | 1.001 [0.999, 1.003] | 0.00 [−0.53, 0.54] µs |
| α = −κ with 3% noise | inconclusive (straddles) | 0.968 [0.966, 0.971] | |
| efficiency drops above kv 8192 | inconclusive (lack of fit, curvature 0.11) | | |
| α = −0.5, both arms traced with ε = 0.25 µs | multiplicative (`m` inside `μ`) | 0.500 | 0.13 [−0.24, 0.49] µs |

The tests also cover: one operating point, drift, the control op moving, noisy
reps, windows dropped by the load, latency and capture gates while the verdict
stands, a failed correctness check, the anchor choice when stage 1 and its
reduce tie on count, a candidate-only kernel surfacing in `kernel_diff`, the
mixed case at hardware scale, the fit and drift gates' false-failure rate, and
interval coverage.
