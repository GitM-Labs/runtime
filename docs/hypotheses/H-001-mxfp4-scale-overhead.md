# H-001: MXFP4's 32-element scales recover half of the 4-bit scale traffic

| | |
|---|---|
| Status | Registered 2026-09-24. Not run. No measured verdict. |
| Owner | Tarun (derivation, interpretation). Isaiah (noise floor, evaluation). Abhiram (budget). |
| Code | `gitm/agents/hypotheses.py` (`H001`, `mxfp4_variant`). Registered but not proposable: it swaps weights. |
| Tests | `tests/test_hypotheses.py` (size reconciliation, derivation, the CUDA exclusion) |
| Engine | vLLM v0.19.1 (`b1388b1f`). Paths below are relative to `vllm/`. |

## Claim

On 8xMI355X at TP8, serving `amd/Kimi-K2.5-MXFP4` instead of `moonshotai/Kimi-K2.5` reduces the routed-expert HBM read per decode step by 5.55%. That is half of the checkpoint's scale traffic. The other half stays, because MXFP4 still carries one scale per 32 weights. The decode-latency reduction is predicted at 3.3% mean. The latency band is far wider than that mean, so the verdict is taken on the MoE kernels' bytes and time, not on ITL alone.

## Mechanism

Both 4-bit formats Kimi ships spend the same bytes on scales. NVIDIA's NVFP4 stores one e4m3 byte per 16 weights. Moonshot's INT4 (compressed-tensors, group 32) stores one bf16 per 32 weights. Both come to 0.0625 B per weight on top of a 0.5 B payload. MXFP4 stores one e8m0 byte per 32 weights, so it pays half.

| Format | Payload (B/weight) | Scales (B/weight) | Total | Scales' share of the stream |
|---|---:|---:|---:|---:|
| NVFP4 (e4m3 / 16) | 0.5 | 0.0625 | 0.5625 | 11.1% |
| INT4 g32 (bf16 / 32) | 0.5 | 0.0625 | 0.5625 | 11.1% |
| MXFP4 (e8m0 / 32) | 0.5 | 0.03125 | 0.53125 | 5.9% |

Decode on Kimi is memory-bound on the routed experts. At the registered 128 sequences, 358 of 384 experts wake per layer with 2.86 rows each, against a compute knee near 67 rows on H200 and higher on MI355X. The expert kernels' time should therefore follow their bytes.

The AMD checkpoint bundles a second change: it also quantises the shared expert and the dense layer-0 MLP, which are bf16 in both NVIDIA's and Moonshot's releases. Its `quantization_config.exclude` list names exactly the attention projections, `mlp.gate` and `lm_head` (61 x 5 + 60 + 1 entries for the text model). That bundled precision change is not scale recovery, so it is priced separately below.

### Size reconciliation, before any run

The format table and the exclude list together must reproduce the published checkpoint sizes, and they do. `amd/Kimi-K2.6-MXFP4` (rev `18ecc30a`) is 558,995,180,568 B against `nvidia/Kimi-K2.6-NVFP4` at 595,148,192,736 B, a 36.153 GB gap. The planner predicts it to 0.1%:

| Term | Weights | B/weight change | GB |
|---|---:|---:|---:|
| Routed experts, 60 x 384 x 3 x 7168 x 2048 | 1.0147e12 | -0.03125 | 31.71 |
| Shared expert, 60 x 3 x 7168 x 2048 | 2.642e9 | -1.46875 | 3.88 |
| Dense layer 0, 3 x 7168 x 18432 | 3.964e8 | -1.46875 | 0.58 |
| Predicted gap | | | 36.17 |

This check found a planner bug on the way. `model_weight_bytes` counted the dense FFN at bytes-per-weight squared, and it ignored precision overrides on the dense MLP (fixed in `a6570bc`).

## Total overhead versus the recoverable part

At the registered operating point (128 sequences, 1,024 cached tokens, TP8, MI355X):

- **Total overhead.** The routed experts' scale stream is 0.926 ms of floor per step (1.234 ms on a measured-time basis at mid-band efficiency 0.75).
- **Recoverable by this intervention.** At most half of it: 0.462 ms of floor. The bundled shared-expert and dense-layer requant adds 0.008 ms, because at TP8 those nodes sit at their launch floor.
- **Cost the intervention adds.** The AMD checkpoint quantises activations too (`input_tensors`: fp4, dynamic). A W4A4 MLP needs two quantised inputs: the MLP input and the SiLU output that feeds the down projection. If the backend runs those as separate kernels, that is 2 x 61 = 122 launches, 0.244 ms. The low end of the band charges all of it, the mean half, and the high end assumes both fuse into the GEMMs.

## Applicability conditions

Checked in code (`_h001_applies`):

1. The experts are a 4-bit format with a 1/16 B/weight scale (NVFP4 or INT4 g32).
2. The platform executes Quark OCP-MX MXFP4 natively. That is ROCm gfx95x only. The rule is spelled out in the next section.

### The exclusion, derived without a run

On every CUDA part, an MXFP4 Quark checkpoint emulates. `platforms/interface.py:577-581` returns `supports_mx() = False`, and only ROCm overrides it, for gfx95x (`platforms/rocm.py:744-745`). With `supports_mx()` false and the W4A4 scheme `w_mxfp4_a_mxfp4` mapped to `Mxfp4MoeBackend.NONE` (`model_executor/layers/quantization/quark/quark_moe.py:703-707`), `self.emulate` is true (`:737-744`). The fused MoE path then dequantises the whole local weight bank to bf16 on every forward (`model_executor/layers/fused_moe/fused_moe.py:1758-1762`).

At TP8 on H200 that is every local expert, 384 x 44.04 M / 8 = 2.11 G weights per layer. Each one is read at 0.53 B, written at 2 B and read back at 2 B, which comes to about 9.6 GB per layer, or 575 GB and about 120 ms per step, against a 7.3 ms expert floor. That estimate is a floor on the regression, since `dequant_mxfp4`'s own intermediates are not counted. The test `test_h001_is_excluded_where_mxfp4_emulates` pins this as a scoped rejection rule: never swap to a Quark MXFP4 checkpoint on a platform where `supports_mx()` is false. It should be reconsidered when vLLM routes `w_mxfp4_a_mxfp4` to a native CUDA backend.

## Expected effect, derived before any run

Metric: primary, the summed duration and HBM read bytes of the MoE kernels per decode step. Secondary, decode ITL p50. Baseline: `moonshotai/Kimi-K2.5` (INT4 W4A16), the team's existing MI355X loop deployment. From `H001.predict`. Positive numbers are reductions; a negative number is a regression:

| Sequences | Cached tokens | Step floor INT4 (ms) | Step floor MXFP4 (ms) | Routed bytes per layer per rank | Recoverable lo / mean / hi (ms, measured basis) | ITL reduction lo / mean / hi |
|---:|---:|---:|---:|---:|---:|---:|
| **128** | **1,024** | **11.62** | **11.14** | **1,110.9 MB -> 1,049.3 MB (-5.55%)** | **-5.87 / 0.51 / 6.98** | **-44.8% / 3.3% / 35.6%** |
| 64 | 4,352 | 11.02 | 10.64 | | -4.72 / 0.38 / 5.56 | -36.9% / 2.6% / 30.9% |
| 32 | 8,192 | 8.62 | 8.37 | | -3.24 / 0.21 / 3.72 | -31.7% / 1.9% / 27.2% |

The row at 128 sequences is registered because the experts are the largest share of the step there: the cache is short and the effect is largest. The 64 x 4,352 row is the team loop's `rag` headline, for comparison.

Why the band dwarfs the mean. The two arms run different MoE kernels: INT4 W4A16 through whatever ROCm vLLM selects, and MXFP4 W4A4 through AITER. Letting each land anywhere in the 0.55 to 0.95 efficiency band spans about 6 ms either way on an 8.5 ms MLP floor, and a 0.46 ms byte effect cannot be seen through that. The bytes are the mechanism, and they are what this hypothesis can decide. Latency is recorded and reported, but a latency result alone cannot confirm or reject the scale-overhead claim.

## Experiment

Two arms on the MI355X loop deployment (`deploy/k8s/mi355x-kimi-loop.yaml`: TP8, `VLLM_ROCM_USE_AITER=1`, `--max-num-batched-tokens 8192`), with only the checkpoint changed:

| Arm | Model | Quantisation as served |
|---|---|---|
| A | `moonshotai/Kimi-K2.5` | compressed-tensors INT4 W4A16 |
| B | `amd/Kimi-K2.5-MXFP4` (rev `42c6da36`) | Quark MXFP4 W4A4, native on gfx950 |

1. Load test. Use the team's `guide` helper (`scripts/kimi_loop/run_loop.sh`) at 128 streams with prompt 1,024 and output 256, for 180 s, with 3 repetitions per arm.
2. Kernel plane. Take one `rocprofv3 --kernel-trace` window per arm and sum the MoE kernel durations per decode step. Then replay one layer's MoE kernel offline with memory counters, as the Tier-2 work in `docs/mi355x_experiment_plan.md` describes, to read its HBM bytes. Counters stay out of the serving runs.
3. Check the log line that selects the backend: arm B must print "The current mode supports native MoE MXFP4 computation" (`quark_moe.py:758`). If it prints the emulation warning instead, stop. Applicability condition 2 did not hold.

## Gates

**Correctness.** GSM8K (lm-eval-harness, 5-shot, 1,319 items) on both arms. Arm B may not score more than 1.0 point below arm A. W4A4 quantises activations dynamically, which is the larger accuracy risk here.

**Latency.** Arm B's TTFT p50 may not rise more than 5%.

## Rejection conditions

1. **Accounting.** If the replayed MoE kernel's HBM read per layer is not within 2% of the predicted ratio (1,049.3 / 1,110.9 = 0.9445), reject the format accounting on AITER. The backend repacks or pads the MXFP4 scales, and `expert_pad_factor` or the execution rule for (mxfp4, cdna4) needs the measured layout.
2. **Mechanism, on bytes not time.** The two arms run different kernels at different efficiencies, so a missing 5.55% in kernel time cannot show that the kernels are not HBM-bound; it can only show that the byte saving did not survive the kernel swap. The mechanism test is therefore the replayed HBM read (rejection 1). If bytes drop as predicted and the MXFP4 kernel is still no faster, record both kernels' measured efficiencies in place of the band and file the gap as a kernel-efficiency question for the AITER MXFP4 path, not as a verdict on scale recovery.
3. **Correctness.** A gate failure rejects the swap for this model family at W4A4. Reconsider it for a W4A16 MXFP4 checkpoint if one is published.

## What each outcome changes

- **Supported.** The (mxfp4, cdna4) execution rule loses its `estimated` flag, and the measured AITER efficiency becomes the band for MI355X MoE nodes.
- **Rejected on accounting.** The measured scale layout goes into `resolve_execution` for CDNA4, pinned by a test.
- **Rejected on mechanism or correctness.** A scoped exclusion is recorded for MI355X, with the measured kernel times as a regression fixture.

## Budget and dependencies

| Item | Estimate | Owner |
|---|---|---|
| Download `amd/Kimi-K2.5-MXFP4` (559 GB) to `/mnt/shared/hf-cache` | 1 to 2 h, no GPUs | cluster owner |
| Server start, arm B | up to 1 h cold (the loop runbook reports 3,600 s for 595 GB) | |
| Load runs and one trace window per arm | about 1 h | |
| GSM8K on both arms | about 1 h | |
| Total GPU time | about 3 node-hours (24 GPU-hours); arm A can reuse the loop's existing arm-A run if the config matches | Abhiram approves |
| Noise floor for MoE kernel time and ITL on MI355X | not yet measured | Isaiah |
| Which ROCm kernel runs INT4 W4A16 MoE, and AITER's MXFP4 scale layout | not read (ROCm sources are outside the pinned checkout) | Tarun, before the run |
