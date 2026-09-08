# MI355X / Kimi K2.5 experiment plan — GITM tracer validation

Objectives, the run matrix, and the exact measurement inventory for the GITM
tracer side: validating the ROCm port, measuring its observation cost, and
feeding taxonomy/correlation/deviation. Deployment is a vLLM ROCm serving pod
with the GITM tracer injected on the head node (see `deploy/k8s/`). Companion:
`docs/rocm.md` for the tracer mechanics and smoke checks.

Scope note: the bulk serving matrix (multi-model TP=8 sweeps, rocprofv3 +
amd-smi telemetry, GuideLLM load) runs through the rex experiment harness, not
this plan. This document owns what rex deliberately does not: the injected
GITM tracer arms (B/C below) and the pipeline they feed. E2/E5/E6-style
serving measurements are rex's job; they appear here only as the A-arm context
the tracer arms compare against.

## Objectives — what "done" means

1. **Port validated**: an injected capture on MI355X round-trips — non-zero
   events, `vendor: amd`, real kernel names, clock-domain window verified, no
   `dropped_records` at merge.
2. **Observation cost known**: tracer overhead measured (not assumed) for the
   no-trace / GITM-trace / GITM+correlation arms, so the 24h pilot can decide
   traced vs sampled. (H200 reference: CUPTI 1.72x, +NVTX 3.46x.)
3. **Taxonomy coverage on ROCm**: % of decode kernel-time classified; the
   unclassified names become intern vocabulary task cards (AITER / hipBLASLt /
   CK symbols, not `nvjet_*`/CUTLASS).
4. **Headline baseline**: throughput/latency envelope for Kimi K2.5 on this
   SKU, captured cleanly enough to sit next to the Qwen3.6/H200 baseline.
5. **Residual decomposed**: predicted-vs-actual through the deviation pipeline —
   does the MI355X gap decompose like the H200's (1.62x on-device, rest eager
   host dispatch), or differently? This directs per-vendor optimization effort.
6. **Comm + energy characterized**: RCCL algorithm/protocol choices, xGMI/NIC
   utilization, tokens/watt and J/token measured at two power caps.

## Measurement inventory — collected on every experiment unless marked

### Serving plane (client harness + vLLM `/metrics`)
- tokens/s (prefill and decode split), requests/s
- TTFT p50/p95/p99, ITL p50/p95/p99, TPOT, end-to-end latency
- offered concurrency (swept), active sequences (`vllm:num_requests_running`,
  `vllm:num_requests_waiting`)
- KV occupancy: vLLM exposes it — `vllm:gpu_cache_usage_perc` (plus preemption
  counters); scrape `/metrics` at 1 Hz alongside the load generator.

### Kernel/trace plane
- **GITM injected tracer** (our schema, feeds taxonomy/correlation/deviation):
  kernel dispatch timeline with names, queue ids, grid/block, memcpys; HIP
  runtime calls + roctx ranges only on the correlation arm.
- **rocprofv3** (vendor-native artifact, one arm per config): kernel dispatch
  timeline, HIP runtime calls, stream/queue IDs, barriers/waits, memory copies
  and allocations, **RCCL ops (type, bytes, duration, rank)** — the one plane
  our tool does not decode yet
  (`rocprofv3 --kernel-trace --hip-trace --memory-copy-trace --rccl-trace`).
- GITM and rocprofv3 both register through rocprofiler-sdk; do NOT assume they
  coexist in one process. They are separate arms of the same config until a
  one-off coexistence check says otherwise.
- vLLM layerwise-tracing flag: verify once whether it emits ROCTx on ROCm
  (grep a correlation-arm trace for its range names). If it does not, skip it —
  kernel names + queue IDs are enough for the taxonomy.
- **Explicitly excluded from serving runs: hardware perf counters (`--pmc`).**
  Counter collection serializes kernels and poisons every latency number.
  Counters are offline Tier-2 work on replayed single kernels, later.

### Node/GPU telemetry plane
- amd-smi sampler at **100 ms or better**, both nodes, for the full run:
  GPU util, VRAM used, power draw, power cap, accumulated energy, GFX and
  memory clocks, junction and memory temperature, throttle/violation counters,
  PCIe bandwidth and replay errors, xGMI per-link bytes and utilization
  (`amd-smi metric --xgmi`).
- Energy numbers come from the accumulated-energy counter, never integrated
  from sampled power.

### Comm plane
- `RCCL_DEBUG=INFO` on **one run per model** (init-time log of chosen
  algorithm and protocol; near-free, log-only).
- RCCL Inspector on **one K2.5 config only** for per-communicator detail —
  not the whole matrix.
- Multi-node runs (TP16 is multi-node, so: yes): `ethtool -S` and the RDMA
  `hw_counters` on the NICs, snapshotted before/after each experiment.
  Do not assume InfiniBand — identify the fabric first (RoCE likely).

## Run matrix

Arms per config: **A** clean (no tracing env), **B** GITM-traced,
**C** GITM + correlation (`GITM_TRACE_NVTX=1`, needs redeploy),
**D** rocprofv3. Telemetry sampler runs on all arms; A is the arm headline
numbers are quoted from.

| # | Experiment | Config | Arms | Primary readout | Pass / deliverable |
|---|---|---|---|---|---|
| E0 | Smoke + clock domain | 1 request | B, C | events > 0, vendor=amd, names real, window non-empty, `range_op` populated in C | port validated (Obj 1) |
| E1 | Tracer overhead | fixed load, mid concurrency | A vs B vs C vs D | tokens/s, TPOT, ITL deltas | overhead table (Obj 2); decides pilot tracing mode |
| E2 | Concurrency sweep | c = 1, 4, 16, 64, 128... until saturation | A (B at one mid point) | full serving plane; KV occupancy vs preemptions | latency-throughput envelope (Obj 4) |
| E3 | Decode taxonomy | saturated decode window | B | kernel-time by class; % unclassified | taxonomy report + vocabulary task cards (Obj 3) |
| E4 | Correlation / layer attribution | mid concurrency | C | % kernel time attributed to layer/op; ROCTx-flag check | attribution working or documented-skip |
| E5 | Comm characterization | mid + saturated | D, +RCCL_DEBUG once, +Inspector once | RCCL op mix/bytes/durations, xGMI per-link, NIC counters | comm profile; is TP16-over-fabric the ceiling? (Obj 6) |
| E6 | Power/energy | repeat one E2 point at default cap and one lowered cap | A | tokens/watt, J/token from energy accumulator; clocks/throttle counters | measured (not inferred) efficiency curve (Obj 6) |
| E7 | Predicted vs actual | best E2 point's trace | B (+C) | deviation pipeline residual, decomposed host vs device | residual decomposition vs H200 (Obj 5) |
| E8 | Cross-vendor baseline | collate E1–E7 | — | tokens/s per GPU, per $, per W; GPU-busy; kernel-time breakdown vs H200 | the comparison chart this port exists for |

Sequencing: E0–E1 the first afternoon GPU nodes attach; E2–E4 next; E5–E6 can
interleave (E6 needs cap-setting rights on the nodes); E7–E8 close the loop.

## Standing rules

- Every run records: image digests, vLLM flags, `/opt/rocm/.info/version`,
  RCCL env, power cap, arm label — no anonymous numbers.
- One variable per run. A tracing arm is never also a power-cap arm.
- Arm files gate collection: confirm shards are NOT growing outside capture
  windows before trusting any A-arm number.
- Load generator runs from the gitm sidecar (`localhost:8000`) so network
  variance stays out of TTFT/ITL.
