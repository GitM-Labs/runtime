# H-002: Kimi K2.5 stores its MLA cache in bf16 where the engine could store fp8

| | |
|---|---|
| Status | Registered 2026-09-24. Not run. No measured verdict. |
| Owner | Tarun (derivation, interpretation). Isaiah (noise floor, evaluation). Abhiram (budget). |
| Code | `gitm/agents/hypotheses.py` (`H002`), proposed through `HypothesisProposer` |
| Tests | `tests/test_hypotheses.py` (derivation, gating, proposal path; no performance claim) |
| Engine | vLLM v0.19.1 (`b1388b1f`). Paths below are relative to `vllm/`. |

## Claim

On the team's 8xMI355X Kimi K2.5 deployment at TP8, adding `--kv-cache-dtype fp8` reduces decode inter-token latency (ITL) by 11.1% at the loop's headline point (`rag`, 64 concurrent), with a derived range of 9.2% to 14.0%. The saving in ms is proportional to the cached tokens read per step, and it does not shrink with tensor parallelism.

This is the lever the loop runbook already plans for its `e8` arm (`INTERVENTION='--kv-cache-dtype fp8'`). This spec registers the prediction before that run.

## Mechanism

vLLM resolves `--kv-cache-dtype auto` from the checkpoint (`engine/arg_utils.py:1567-1570`, calling `utils/torch_utils.py:324-342`):

- A modelopt `kv_cache_scheme` of static 8-bit float resolves to fp8 (`:262-296`).
- Anything else stays `auto`, which becomes the model dtype (`:345-351`).

`moonshotai/Kimi-K2.5` is compressed-tensors with `kv_cache_scheme: null`, so its cache is bf16. `nvidia/Kimi-K2.6-NVFP4` declares a static fp8 scheme, so its default is already fp8. The applicability check refuses it.

| | Bytes per cached entry (576 elements) | Bytes per token, 61 layers |
|---|---:|---:|
| bf16, what K2.5 stores by default | 1,152 | 70,272 |
| Generic fp8, one scale per layer | 576 | 35,136 |

The generic fp8 layout stores the whole entry (512 latent plus 64 RoPE) at one byte. `attention/mla_attention.py:316` sets `head_size = kv_lora_rank + qk_rope_head_dim`, and `:1130-1137` shapes the cache `(blocks, block_size, head_size)` in the cache dtype. Scales are per tensor only (`quantization/kv_cache.py:88-91`).

MLA keeps one latent per token shared by all 64 heads, so every TP rank holds and reads the whole cache. At decode the attention core is memory-bound and its bytes are the cache read, so halving the element width halves that node's floor. Nothing else in the step moves.

## Total overhead versus the recoverable part

At the headline point, the attention core's cache read above an fp8 cache is 1.224 ms of floor per step, or 1.631 ms on the measured basis (floor divided by the planner's mid-band efficiency, 0.75). That is the total overhead.

The intervention can recover all of those bytes. How much time comes back depends on whether the kernel changes:

- **MI355X keeps the kernel.** With AITER enabled, `ROCM_AITER_MLA` is first for MLA (`platforms/rocm.py:317-322`) and accepts both bf16 and fp8 (`v1/attention/backends/mla/rocm_aiter_mla.py:32-38`). One efficiency prices both sides, so the kernel's time halves anywhere in the band. The band only moves the attention core's share of the step.
- **Blackwell keeps the kernel.** `FLASHINFER_MLA` serves both (`flashinfer_mla.py:40-45`, `:65-66`).
- **Hopper switches the kernel.** `FLASH_ATTN_MLA`, first on sm90 (`platforms/cuda.py:92-97`), has no fp8 path (`flashattn_mla.py:45-49`, `:322-323`), so fp8 moves the core to `FLASHMLA` (`flashmla.py:48-53`, `:73-74`). The old and new kernels may then land at different efficiencies. In the worst case in the band, the kernel's time falls by only 13.6% instead of 50%.

What the intervention does not recover: the core's launch floor, and the unchanged bf16 query and output traffic. A second benefit, twice the cache capacity per rank, is real but is not a per-step latency effect and is not part of this prediction.

## Intervention

`--kv-cache-dtype fp8` (EngineArgs `kv_cache_dtype = "fp8"`). No weight change. Rollback is a server restart without the flag. On the loop pod that is `arm.sh`, which restarts in-pod.

## Applicability conditions

Checked in code before a candidate is emitted (`_h002_applies`), each with a reason when it fails:

1. The model uses MLA (`kv_lora_rank > 0`). A GQA cache splits across TP ranks, and the effect would divide by TP.
2. The executed cache is not already fp8. An explicit `serving["kv_cache_dtype"]` wins. Otherwise it is the catalogue's `kv_dtype`, which records what `auto` resolves to.
3. The architecture has an fp8 MLA decode backend pinned: Hopper, Blackwell, or CDNA4 with `VLLM_ROCM_USE_AITER=1` in the serving config (the loop deployment sets it). Without AITER, ROCm selects `TRITON_MLA` (`platforms/rocm.py:324-326`), which this prediction does not price.
4. The step is decode.
5. The mean predicted step reduction clears the measured noise floor (`HypothesisProposer(noise_floor=...)`).

The emitted `InterventionSpec` carries `requires_hardware` (H100, H200, B200, B300, GB200, GB300, MI355X), `requires_dtype` (bf16) and `workloads` (vllm-decode). The existing precondition gate in `select_interventions` checks those.

## Expected effect, derived before any run

Metric: decode ITL p50, ms per step, over the steady-state window. Baseline: the same server without the flag. Cached tokens follow `scripts/kimi_loop/predict_sweep.py`: mid-generation, prompt plus half the output. From `H002.predict`, TP8, positive numbers are reductions:

| SKU | Sequences | Cached tokens | Floor bf16 (ms) | Floor fp8 (ms) | Total overhead, measured (ms) | Recoverable lo / mean / hi (ms) | ITL reduction lo / mean / hi |
|---|---:|---:|---:|---:|---:|---:|---:|
| MI355X | 64 | 1,152 (`chat`) | 9.22 | 8.89 | 0.43 | 0.34 / 0.43 / 0.59 | 2.8% / 3.5% / 4.7% |
| **MI355X** | **64** | **4,352 (`rag`, headline)** | **11.02** | **9.79** | **1.63** | **1.29 / 1.63 / 2.22** | **9.2% / 11.1% / 14.0%** |
| MI355X | 128 | 4,352 | 15.36 | 12.91 | 3.26 | 2.58 / 3.26 / 4.45 | 13.5% / 15.9% / 19.5% |
| H200 | 32 | 8,192 | 13.32 | 11.40 | 2.56 | 0.55 / 2.56 / 4.96 | 3.3% / 14.4% / 25.3% |
| B200 | 32 | 8,192 | 8.61 | 7.46 | 1.54 | 1.21 / 1.54 / 2.09 | 11.2% / 13.4% / 16.6% |

Arithmetic for the headline row. The core reads 64 x 4,352 entries x 61 layers per rank per step. At 1,152 B that is 19.57 GB, or 2.447 ms at 8 TB/s; at 576 B it is 1.223 ms. The step floor falls from 11.017 to 9.794 ms.

- **Mean.** The unchanged rest of the step is (11.017 - 2.447) / 0.75 = 11.43 ms on the measured basis. The core is 2.447 / 0.75 = 3.26 ms, and it halves, saving 1.63 ms of 14.69, which is 11.1%.
- **lo.** Both kernels run at 0.95: 1.29 ms saved out of 11.43 + 2.58.
- **hi.** Both kernels run at 0.55: 2.22 ms saved out of 11.43 + 4.45.

The kernel's own time falls by exactly 50% at every end. That covered-op figure is what `expected_delta_*` carries on the emitted spec, because `replay.predict_delta` multiplies it by the trace's attention coverage.

The `rag` point fits comfortably. The fit ledger (`memory_fit`, the loop's 0.92 utilisation, 4.5 GB workspace) leaves 184 GB per MI355X rank for KV, and the bf16 cache needs 19.6 GB.

## Experiment

On the loop deployment (`deploy/k8s/mi355x-kimi-loop.yaml`: TP8, `VLLM_ROCM_USE_AITER=1`, `--gpu-memory-utilization 0.92`, `--max-num-batched-tokens 8192`), two arms, changed by `arm.sh` in-pod:

| Arm | Extra flags | MLA backend |
|---|---|---|
| A | none (the loop's existing headline run) | ROCM_AITER_MLA, bf16 |
| C | `--kv-cache-dtype fp8` | ROCM_AITER_MLA, fp8 |

1. **Load.** Run `INTERVENTION='--kv-cache-dtype fp8' bash run_loop.sh e8`. It re-runs the headline (`rag`, c=64) under the lever. Add the `chat` config at c=64 for the scaling check, with 3 repetitions each.
2. **Kernel plane.** Take one GITM-traced window per arm (the loop's arm B tracer, not during the timed runs) and read the per-layer MLA decode kernel duration.
3. **Pre-run check.** Confirm that the AITER MLA decode kernel's name contains one of the spec's `kernel_scope` substrings (`flash_mla`, `flashmla`, `cutlass_mla`, `flashinfer_mla`, `mla_decode`, `aiter_mla`, `mla_fwd`). The replay credits only those, deliberately: `classify_op` also files `reshape_and_cache` and `slot_mapping` under `attn_score_value`, and an fp8 cache does not halve them. If the AITER name matches none, add it to the scope; otherwise `predict_delta` sees zero coverage and ranks this candidate at nothing.

On H200, run a third arm to separate the backend switch from the byte halving. Arm B is `--attention-backend FLASHMLA` with a bf16 cache (`engine/arg_utils.py:597`). Its fp8-vs-bf16 comparison on the same kernel then isolates the mechanism.

## Gates

The loop's keep decision (`optimizer/apply.py`) measures throughput only, so a faster but wrong cache could be kept. `HypothesisProposer` therefore refuses to emit this candidate unless it is built with a `correctness_gate`, and the emitted `InterventionSpec` carries that gate in its `correctness_gate` field. `apply_intervention` runs it after measuring and before keeping, with whatever applicator the caller passed, so there is no wrapper to forget. A failing gate restores the change and puts the reason in `ApplyResult.error`. `tests/test_hypotheses.py::test_a_faster_but_wrong_candidate_is_rolled_back` and `::test_the_gate_is_on_the_spec_not_the_proposer` pin that. The gate itself is the sentinel below; wiring it to lm-eval is the loop's job, not this spec's.

**Correctness.** Kimi K2.5 ships no `k_scale`/`v_scale`, so vLLM stores with scale 1.0 (`quantization/kv_cache.py:71-75`). Run on arms A and C:
- GSM8K (lm-eval-harness `gsm8k`, 5-shot, all 1,319 items, `local-completions` against the server). Arm C may not score more than 1.0 point below arm A.
- Needle retrieval at the loop's `long` context (8,192 prompt), 100 prompts. Arm C may not score more than 2 points below arm A.

**Latency.** Arm C's TTFT p50 may not rise more than 5%. Arm C's ITL p99 may not exceed arm A's by more than the noise floor.

## Rejection conditions

1. **Effect.** If the ITL reduction at the headline is below 9.2% minus the noise floor, reject at this operating point.
2. **Mechanism.** The kernel is unchanged on MI355X, so the MLA decode kernel's own time should halve. If it falls by less than 25% (half the predicted reduction), the kernel is not HBM-bound at head size 576 on gfx950. Reject the mechanism, and record the measured time as a fixture.
3. **Scaling.** The predicted saving at `rag` is 3.8x the saving at `chat` (4,352 against 1,152 cached tokens at the same concurrency). If the measured ratio is below 2, the saving is not cache traffic.
4. **Correctness.** A gate failure rejects the lever for this family with uncalibrated scales. Reconsider it with `--calculate-kv-scales` (`engine/arg_utils.py:1007`) or a checkpoint that ships scales.

## What each outcome changes

- **Supported.** The catalogue entry `kv_cache_dtype_fp8` in `gitm/kernels/library.yaml` gets a hardware gate that includes MI355X, H200 and B200. Today it lists only A100, H100 and L40S, and the gate matches by substring, so it never fires on this team's hardware. Its flat 6% becomes this derivation. The measured AITER MLA efficiency replaces the band for that node on MI355X.
- **Rejected on mechanism.** Record a scoped exclusion for AITER MLA fp8 on gfx950 at this head size, pinned by the measured kernel time.
- **Rejected on correctness.** An applicability rule: fp8 cache on this family only with calibrated scales.

## Budget and dependencies

| Item | Estimate | Owner |
|---|---|---|
| Hardware | the existing MI355X loop pod; weights already on `/mnt/shared/hf-cache` | team |
| Server restart for arm C | in-pod through `arm.sh`, up to 1 h cold | |
| Load runs, 2 configs x 3 reps x 3 min, plus one traced window per arm | about 1 h | |
| GSM8K and needle, 2 arms | about 1.5 h | |
| Total | about 3.5 node-hours (28 GPU-hours); arm A reuses the loop's headline run if the config matches | Abhiram approves |
| Noise floor for ITL p50 and p99 at the headline point | not yet measured; rejection 1 and the proposer's `noise_floor` need it | Isaiah |
| AITER MLA kernel name in `classify_op` | not checked | Tarun, before the run |

A local test is not a measured win. The tests in `tests/test_hypotheses.py` show only that the candidate is generated, gated and rolled back correctly.
