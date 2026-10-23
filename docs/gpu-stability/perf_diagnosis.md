# Performance diagnosis plan — Kimi-K2.6 on alpha01 (MI300X, vLLM 0.30.0)

**Goal:** determine *whether* throughput/latency is actually low and *what* causes it, by isolating one variable at a time. Do **not** assume the container version — it's the last thing to test, after config levers are ruled out.

**Stack under test** (see `mns512_report.md` §0 for full versions): 8× MI300X (gfx942), ROCm 7.2.3, vLLM 0.30.0, AITER 0.1.21.post2, Triton 3.7.1, Kimi-K2.6, TP8.
**Guardrail (customer):** all runs at **`max_num_seqs ≤ 256`**. No 512. Keep the graceful-teardown + idle-gate from `mns512_report.md` §5 in place.

---

## 0. Why we can't conclude anything yet

There is **no performance data in the workspace** — only wedge/crash artifacts. Before any test, we need numbers and a reference. "Low" is meaningless without "low vs what."

**First question to answer:** did the slowdown start when we capped to `mns256`? If yes, the guardrail (P1) is the likely cause, not the container.

---

## 1. Controls — hold these fixed on every run

Any of these drifting invalidates the comparison:

- **Fixed workload:** same input/output lengths (8k/1k), same dataset/prompts, `--ignore-eos` so every request emits exactly 1k tokens (deterministic token count).
- **Clean start:** all 8 GPUs `gpu_busy_percent == 0`, VRAM at baseline, dmesg quiet (reuse the idle-gate).
- **Warm, then steady-state:** discard the first ~30 s (compile/warmup/cache fill); measure a fixed steady-state window only.
- **Repeat ×3** per point; report median + spread. One run is noise.
- **Rule out the client first:** confirm the load generator (InferenceX) and the network aren't the bottleneck — if the server GPUs aren't saturated, the problem isn't the server (see P8).
- **One variable per arm.** Everything else identical to the baseline.

### Metrics to capture every run
| Metric | Why |
|---|---|
| Output throughput (tok/s), total throughput | Headline number |
| Goodput (req/s meeting an SLO) | What the customer actually feels |
| TTFT (time to first token) | Prefill / queueing health |
| TPOT / ITL (per-output-token / inter-token latency) | Decode-step health |
| Per-concurrency: 4, 8, 16, 32, 64 | Where it scales vs saturates |
| GPU telemetry @ 5 s: `gfx_util_pct`, `mem_bw_pct`, power, VRAM | Saturation / bottleneck signal |
| `gitm capture` kernel-time breakdown | *Where* GPU time goes — the single most useful artifact |
| vLLM log: PIECEWISE/eager downgrades, scheduler stalls, recompiles | Hidden perf killers |

---

## 2. Test matrix (one variable at a time, all mns ≤ 256)

| # | Test | Vary | Hold | Rules in/out |
|---|------|------|------|--------------|
| **P0** | **Baseline + "is it slow?"** | nothing | best-known config | Establishes the reference curve vs a target/prior run. Without this, nothing below is interpretable. |
| **P1** | **mns cap cost** | `max_num_seqs` ∈ {128, 192, 256} | backend, rest | If throughput still climbing at 256 → the cap is throttling you (the guardrail is the cause). If it plateaus below 256 → the cap is *not* the bottleneck; look elsewhere. |
| **P2** | **Attention backend** | `TRITON_MLA` vs AITER MLA | mns256, rest | AITER materially faster → the slow runs were just on the slower backend, not a version bug. |
| **P3** | **Cudagraph vs eager** | full/piecewise capture vs `--enforce-eager` | mns256, rest | Big gap + PIECEWISE-downgrade warnings in the log → capture/compile path is hurting you. |
| **P4** | **Spec decode (EAGLE3)** | EAGLE3 on vs off; log **draft-token acceptance rate** | mns256, rest | Low acceptance → spec decode *adds* overhead; turning it off is faster. |
| **P5** | **DCP** | DCP=8 vs off | mns256, rest | DCP slower at this context length → it's costing latency you don't need. |
| **P6** | **quick-reduce quant** | INT4 vs FP (off) | mns256, rest | Rarely the cause; confirm the INT4 allreduce isn't a regression on this ROCm. |
| **P7** | **Container/stack version** | current image vs a known-good older image | **identical config** | *Only after P1–P6.* Same config slower on this image → genuine vLLM/ROCm/AITER regression. This is the test that actually answers "is it the container version." |
| **P8** | **Saturation / bottleneck** | (analysis of P0 telemetry) | — | GPUs at ~100% `gfx_util` + high `mem_bw` → compute/bandwidth-bound (expected ceiling). GPUs idle at peak concurrency → bottleneck is client, tokenization, or scheduling, **not** the GPUs or the container. |

---

## 3. Order of execution (fastest path to the answer)

1. **P0** — get a baseline curve and a target. Decide if it's even slow.
2. **P8** (reads P0's telemetry) — are the GPUs saturated? If idle at peak → stop blaming the server; it's the client/pipeline.
3. **P1** — did the mns256 cap cause it? (Most likely culprit given the recent guardrail.)
4. **P2** — backend A/B (biggest config lever on MI300X).
5. **P3–P6** — remaining config levers, as the breakdown points.
6. **P7** — container version A/B, **only if** P1–P6 came back clean.

---

## 4. How each result reads

- **Throughput still rising at mns256 (P1)** → the guardrail is throttling you. Perf is a *cost of the mitigation*, not a bug. Document the tradeoff; revisit when 512 is re-allowed.
- **AITER ≫ Triton (P2)** → you were benchmarking the slow backend. Switch, re-baseline.
- **Eager ≈ capture, or capture worse + downgrade warnings (P3)** → capture/compile is not helping (or hurting); run eager or fix capture.
- **Low EAGLE3 acceptance (P4)** → disable spec decode for this workload.
- **GPUs idle at peak (P8)** → the ceiling is upstream of the server; profile the client/tokenizer/scheduler, not vLLM.
- **P1–P6 clean but P7 shows a gap** → *then* it's the container/stack version — file it with the kernel-time breakdown as evidence.

---

## 5. What I need from you to run / interpret this
- A **target or prior "fast" number** (so P0 has a reference).
- Which **config arm** the current slow numbers came from (backend, mns, DCP, spec).
- Access to the **bench output** and, if possible, a **known-good older image** for P7.
- Confirmation the **client/load-gen** can push past the server's rate (for P8).
