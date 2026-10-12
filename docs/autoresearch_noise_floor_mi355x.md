# Noise floor: Kimi K2.5 on 8x MI355X, rag 4096/512, c=64

First hardware noise-floor entry for the autoresearch harness (task 3). Run
`20261012-182236`, 2026-10-12. Agent-written, human-read.

**Scope.** One operating point, one server session. The 8 reps ran back to back
on one warm server with no reload between them, so this is **within-session**
noise. Reload-to-reload variance (what an A/B across an arm switch also pays)
was not sampled; the true floor for a switched comparison is at least this wide.

## Result

| Metric | Mean | SD | CV | Min / max | 95% CI on mean |
|---|---|---|---|---|---|
| Output throughput (tok/s) | 949.80 | 2.33 | 0.25% | 947.12 / 954.38 | 947.85 to 951.75 |
| ITL p50 (ms) | 66.80 | 0.16 | 0.24% | 66.58 / 67.03 | 66.67 to 66.93 |
| ITL p95 (ms) | 67.54 | 0.49 | 0.73% | 66.90 / 68.61 | 67.12 to 67.95 |
| TTFT p50 (ms) | 359.57 | 42.72 | 11.88% | 282.97 / 407.86 | 323.85 to 395.28 |

Every rep completed 320 requests with 0 errors.

**Two sample sizes.** The same statistics from the first 4 reps and from all 8:

| Metric | CV, n=4 (reps 1–4) | CV, n=8 |
|---|---|---|
| Output throughput | 0.35% | 0.25% |
| ITL p50 | 0.34% | 0.24% |
| ITL p95 | 0.49% | 0.73% |
| TTFT p50 | 15.98% | 11.88% |

The n=4 and n=8 estimates agree to within the precision n=4 allows. There is
no sign of drift across the session.

## Minimum detectable effect

Relative effect a two-arm comparison detects at α = 0.05 (two-sided) and
power = 0.80, with n reps per arm, unpaired, using the n=8 SD
(t-quantiles, df = 7): `MDE = (t₀.₉₇₅ + t₀.₈₀) · CV · √(2/n)`.

| Reps per arm | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Output throughput | 1.13% | 0.80% | 0.57% | 0.40% | 0.28% |
| ITL p50 | 1.10% | 0.78% | 0.55% | 0.39% | 0.28% |
| ITL p95 | 3.37% | 2.38% | 1.68% | 1.19% | 0.84% |
| TTFT p50 | 54.8% | 38.7% | 27.4% | 19.4% | 13.7% |

**Reading it:**
- Throughput and ITL are tight. With 4 reps per arm, a 0.6% change is
  detectable, before adding reload-to-reload variance.
- TTFT p50 is too noisy to gate on. A latency gate on TTFT needs either many
  reps or a different statistic (e.g. a per-request distribution test rather
  than the per-rep median). This feeds contract open question 4.
- These numbers apply only to this operating point and protocol. A hypothesis
  at another point is `not_established` until it has its own entry.

## Protocol

| | |
|---|---|
| Hardware | 8x AMD MI355X (gfx950), one node, National Compute `us-mi355x-gitmachine`, Kueue queue `market` |
| Model | `moonshotai/Kimi-K2.5`, HF snapshot `4d01dfe0332d63057c186e0b262165819efb6611` |
| Engine | vLLM 0.29.0, ROCm 7.2.3, image `vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1` |
| Launch | `deploy/k8s/mi355x-kimi-loop.yaml`: TP=8, `--gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 256`, `VLLM_ROCM_USE_AITER=1` |
| Arm | **A**: no tracer (`ROCP_TOOL_LIBRARIES=""`) |
| Load | GuideLLM 0.8.1, concurrent profile, 64 streams, synthetic text, 4096 prompt / 512 output tokens, `/v1/chat/completions` |
| Rep duration | 180 s; GuideLLM drops the first and last 10%, so 144 s measured per rep |
| Warmup | one 300 s run at the same point, discarded, after the arm-A restart |
| Reps / order | 8, back to back, same arm, no reload; started 19:16:08Z, every ~189 s |
| Estimator | per-rep GuideLLM mean output tok/s and per-rep ITL/TTFT percentiles; mean, sample SD, t-interval across reps |
| Raw data | `/mnt/shared/gitm/results/20261012-182236/aa/guidellm_aa_r{1..8}.json` on the cluster's shared FS (~11 MB each); local copy under `evidence/kimi-mi355x/runs/20261012-182236/` (gitignored) |
| Analysis | `scripts/kimi_loop/analyze_aa.py` → `analysis/aa_noise.md` |

Kimi K2.5 is a reasoning model, and with 512 output tokens every request spent
them all on reasoning. Throughput and ITL are unaffected; GuideLLM's
time-to-first-*output*-token (~35 s) is not meaningful for this workload and is
not reported.

## Findings from the same run

1. **The tracer hangs vLLM under load.** With the gitm ROCm tracer injected and
   a capture window active (arms B and I), the engine stalled on the first
   batch (3 running, 61 waiting, GPUs 0.3% active, 52% of kernel time in
   collectives) and died after ~5 min: `RPC call to sample_tokens timed out`.
   Both e8 arms produced 0 completed requests. Arm A served normally at the
   same point.
2. **Cold start costs ~24% throughput.** The warmup run right after the arm-A
   restart averaged 726 output tok/s against 950 warm, while kernels JIT
   compiled. `run_loop.sh e8` measures immediately after each switch with no
   warmup.
3. **`arm.sh` restarts the whole container.** `pkill -f 'vllm serve'` also
   matches the supervisor loop's command line, so the container exits (143)
   and restarts, which discards the JIT caches on every switch.
4. **`arm.sh` reloads needlessly.** Its "already active" check compares
   `/scratch/arm.env` text, and the init container writes
   `GITM_EXTRA_VLLM_ARGS=` where `arm.sh` expects `GITM_EXTRA_VLLM_ARGS=""`.
5. **The A/A reps have no MANIFEST entry**, because they were run by hand
   rather than through `run_loop.sh`. Provenance above is reconstructed from
   the GuideLLM JSON, the server log and the e8 MANIFEST of the same pod.

## Timings

| Step | Duration |
|---|---|
| Submit → GPUs granted (Kueue admission) | 9.5 min |
| Init (wheel install, tracer build) | 2.5 min |
| First model load to ready | ~12 min |
| Arm switch (reload), measured 3 times | ~10–12 min each (arm A: 9 min 51 s) |
| `run_loop.sh e8` end to end | 30 min 35 s |
| GPUs held in total | ~2 h 4 min (≈ 16.5 GPU-hours) |

## Next

- Reload-to-reload variance: repeat a few reps across 2–3 server restarts.
- A second operating point (e.g. chat 1024/256 or c=16).
- Fix or bypass the tracer hang before any traced arm is used for verdicts.
- Warmup after every switch is required; the contract already specifies it.
