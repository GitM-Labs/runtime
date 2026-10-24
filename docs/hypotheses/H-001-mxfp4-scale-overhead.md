# H-001: MXFP4's 32-element scales recover half of the 4-bit scale traffic

| | |
|---|---|
| Status | Registered 2026-09-24. Not run. No measured verdict. |
| Owner | Tarun (derivation, interpretation). Isaiah (noise floor, evaluation). Abhiram (budget). |
| Code | `gitm/agents/hypotheses.py` (`H001`, `mxfp4_variant`); `gitm/planner/glm_graph.py` (`aiter_mxfp4_routed_quants`, the MXFP4 activation-quant nodes). Registered but not proposable: it swaps weights. |
| Tests | `tests/test_hypotheses.py` (size reconciliation, derivation, the CUDA exclusion, the serving-env refusals (AITER flags, shared-expert fusion, forced backends, disabled kernels, `None` as unset), the NVFP4 refusal on MI355X, the activation-quant count and widths, the mean as the floor ratio); `tests/test_glm_graph.py` (AITER's routed-quant rule at every padded token count) |
| Engine | vLLM v0.29.0 (`98dff2a`) with AITER v0.1.19 (`3135022`): the MI355X loop's `vllm` container (`deploy/k8s/mi355x-kimi-loop.yaml:126`) runs `vllm/vllm-openai-rocm` at the digest Docker Hub lists for `v0.29.0`, which pins AITER v0.1.19 (`docker/Dockerfile.rocm_base:12`). vLLM paths are relative to `vllm/`; AITER paths start `aiter/`. First derived on vLLM v0.19.1 (`b1388b1f`) with AITER v0.1.10.post2; every citation was re-read at v0.29.0 and v0.1.19. |

## Claim

On 8xMI355X at TP8, serving `amd/Kimi-K2.5-MXFP4` instead of `moonshotai/Kimi-K2.5` reduces the routed-expert HBM read per decode step by 5.55%. That is half of the checkpoint's scale traffic. The other half stays, because MXFP4 still carries one scale per 32 weights. The decode-latency reduction is predicted at 0.9% mean. The latency band is far wider than that mean, so the verdict is taken on the MoE kernels' bytes and time, not on ITL alone.

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
- **Recoverable by this intervention.** At most half of it: 0.462 ms of floor. The bundled shared-expert and dense-layer requant saves 0.008 ms of GEMM floor, all of it on dense layer 0 (at TP8 the shared expert sits at its launch floor in both arms) and adds 122 quant launches (two per shared expert x 60, plus two on dense layer 0), 0.244 ms: a net 0.236 ms of floor lost at every row below. The routed mechanism alone stays ahead at all three rows (+0.342, +0.127 and +0.003 ms of floor).
- **Cost the intervention adds.** The AMD checkpoint quantises activations too (`input_tensors`: fp4, dynamic), and each quant is its own kernel launch.
  - Routed experts: AITER runs a fused quant-and-sort before stage 1 (`aiter/fused_moe.py:2612`) and a second quant before stage 2 (`:2761-2773`), skipped when the tuned stage-1 kernel's name has `_fp4` after its last `_t`, since that kernel writes stage 2's fp4 input and scales itself (`:2056`, `:2107`, `:2713-2721`). Kimi's TP8 rows are `aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv:2-17`, keyed on the token count rounded up to a power of two (`aiter/fused_moe.py:1102-1117`, `:2530-2531`). At 128 tokens stage 1 is `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4` (csv `:9`), so one quant; at 64 and 32 it is `..._t32x32x256_w3` and `..._t32x64x256_w4` (csv `:8`, `:7`), so two. The TP4 rows (per-rank intermediate 512, csv `:18-33`) carry `_fp4` at padded counts 2, 4, 8, 16 and 128 to 2,048 (csv `:19-22`, `:25-29`). The planner reads both shapes (`_AITER_FP4_FUSED_M`), keyed on the per-rank intermediate after ROCm's round-up to 256 (`model_executor/layers/fused_moe/oracle/mxfp4.py:759-761`), the `w2` shape AITER reads (`aiter/fused_moe.py:432-438`), so TP16 and TP32 take the TP8 rows; any other shape gets two quants. Stage 2's quant reads that padded width too (`aiter/fused_moe.py:2761-2773`), so at TP16 and TP32 it moves as many bytes as at TP8. Above 256 tokens (stage 1) and 1,024 (stage 2) each quant is two kernels, a per-token quant then a sort (`aiter/ops/quant.py:1090-1135`); the planner counts the quant and not the second launch, which only prefill-sized steps reach.
  - Shared expert and dense layer 0: two MXFP4 linears each (gate-up, then down), each quantising its own input (`model_executor/kernels/linear/mxfp4/aiter.py:28-104`, `:191-203`). No ROCm compile pass fuses an fp4 quant (`compilation/passes/fusion/act_quant_fusion.py:36-40` is NVFP4 on CUDA).
  - Per decode-only step (experiment step 2 selects them): 60 x 1 + 60 x 2 + 2 = 182 launches at 128 sequences, 60 x 2 + 60 x 2 + 2 = 242 at 64 or 32; at the planner's 2 µs per launch, 0.364 ms and 0.484 ms. The graph emits exactly these nodes, and the band charges them at every end, divided by that end's efficiency like every other node: each quant feeds a GEMM on the same stream, so it cannot hide behind one.
- **The A16W4 `cktile` path no longer applies.** At AITER v0.1.10.post2, MXFP4 with `token x top_k <= 384` and shuffled weights took a `cktile` path with bf16 activations. At v0.1.19 that path needs `ksplit > 1` (`aiter/fused_moe.py:2301-2308`); every tuned Kimi row sets `ksplit` 0 (csv `:2-17`), and the FlyDSL branch those rows select returns before the `cktile` check (`:2048-2118`). Without the tuned table, `get_ksplit` (`:1042-1065`) splits K only while M x 8 experts per token x 2 N-tiles is under the GPU's 256 CUs, a padded M of 8 or less. No registered row comes near it.

## Applicability conditions

Checked in code (`_h001_applies`):

1. The experts are a 4-bit format with a 1/16 B/weight scale; on CDNA4 that means INT4 g32. NVFP4 has the same scale cost but is refused there: vLLM v0.29.0's NVFP4 MoE priority list ends at `EMULATION` (`model_executor/layers/fused_moe/oracle/nvfp4.py:196-205`), every backend ahead of it fails its device check on ROCm, and the first that passes wins (`:293-307`). That emulation is a Triton MoE kernel that dequantises each packed weight tile in registers inside the GEMM (`model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py:51-254`, called at `:498` and `:529`): 0.5 B of fp4 plus 1/16 B of fp8 scale per weight, the same bytes as INT4 g32's Triton W4A16 kernel, plus a quant-dequant of each GEMM input (`:489-496`, `:520-527`). The scale traffic is real; what is missing is a planner rule for that kernel (`_FORCED[("nvfp4", "emulation")]` prices v0.19.1's unfused chain), so `predict` would raise and the refusal says so.
2. The platform executes Quark W4A4 MXFP4 MoE natively: ROCm gfx950 with AITER's fused MoE, checked on the hardware and the serving config:
   - The hardware is CDNA4 (`w.hw.arch == "cdna4"`). Everywhere else the experts emulate, as the next section derives.
   - The serving env enables AITER's fused MoE. `AiterExperts._supports_current_device()` is `rocm_aiter_ops.is_fused_moe_enabled()` (`model_executor/layers/fused_moe/experts/rocm_aiter_moe.py:451-453`), which is `VLLM_ROCM_USE_AITER and VLLM_ROCM_USE_AITER_MOE` (`_aiter_ops.py:1745`, `:1748`, `:1892-1893`). vLLM reads each as `os.getenv(name, default).lower() in ("true", "1")`, `VLLM_ROCM_USE_AITER` defaulting to `"False"` (`envs.py:1235-1237`) and `VLLM_ROCM_USE_AITER_MOE` to `"True"` (`envs.py:1255-1257`); the code parses `Workload.serving` the same way, so `serving={}` is refused, and an explicit `None` reads as unset and takes the default. With either flag off the oracle skips `AITER_MXFP4_MXFP4` and loads `EMULATION`. The loop deployment sets `VLLM_ROCM_USE_AITER=1` and leaves `VLLM_ROCM_USE_AITER_MOE` unset (`deploy/k8s/mi355x-kimi-loop.yaml:179-180`), so it passes.
   - The shared-expert fusion is off. `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` defaults to `"False"` (`envs.py:1326-1329`) and counts only with AITER's fused MoE on (`_aiter_ops.py:1897-1898`). On, with the routed and shared experts under the same Quark config (`model_executor/layers/quantization/utils/config_utils.py:140-181`, which the AMD checkpoint's exclude list allows), the shared expert joins the routed set (`model_executor/models/deepseek_v2.py:341-345`): 385 experts and top-9 (`model_executor/layers/fused_moe/routed_experts.py:987-993`; `model_executor/layers/fused_moe/experts/rocm_aiter_moe.py:60-86`). AITER then reads the 48 tuned 385-expert rows (csv `:34-81`), another kernel family, `flydsl_mxmoe_g1_a4w4_*` (`aiter/fused_moe.py:2005-2022`; `aiter/ops/flydsl/mxfp4_kname.py:7-9`, `:84-87`), with its own quant fusion (`f16in` is an inline activation quant, used up to 128 tokens in the csv) and no separate shared-expert quants. The planner models neither, so the 182 and 242 counts and the table below do not describe that run; the code refuses it. The loop deployment does not set it.
   - The MoE backend is `auto`. A `moe_backend` in `Workload.serving` (the key `gitm/optimizer/vllm_knobs.py:147` uses for `--moe-backend`) or in its `kernel_config`, other than `auto`, after vLLM's own normalisation (lower case, `-` to `_`; `engine/arg_utils.py:1680-1681`, `config/kernel.py:298-309`), is refused. vLLM keeps `kernel_config`'s backend unless the top-level flag is set (`engine/arg_utils.py:2534-2548`), and a forced backend skips the auto list (`model_executor/layers/fused_moe/oracle/mxfp4.py:534-560`). `aiter` would reach the same backend (`:302-308`, then the activation filter at `:443-456`) and is refused too, because the prediction is for what `auto` picks. (`aiter_mxfp4_mxfp4` is only the map key at `:308`; `--moe-backend` takes the `MoEBackend` literal's values, `config/kernel.py:122-142`, and rejects anything else at argument parsing.) The loop deployment leaves it unset.
   - The MXFP4 linears are AITER's. `--linear-backend emulation` filters the ROCm MXFP4 linear list (`model_executor/kernels/linear/__init__.py:358-386`, `:567-570`) down to `EmulationMxfp4LinearKernel`, and `VLLM_DISABLED_KERNELS` naming `AiterMxfp4LinearKernel` as one of its comma-separated entries skips AITER's (`:899-902`; `envs.py:1194-1198`). Either runs `EmulationMxfp4LinearKernel`, which dequantises the weight to bf16 on every call (`model_executor/kernels/linear/mxfp4/emulation.py:102-110`). `aiter` keeps AITER's kernel, and a backend with no MXFP4 kernel falls back to the whole list (`:341-355`). The code refuses `VLLM_DISABLED_KERNELS` and, as for the MoE backend, every linear backend but `auto`, top-level or in `kernel_config`: the shared-expert and dense-layer quants above are priced for the AITER linear. The loop deployment sets neither.

Unchecked assumptions, which the planner does not read; the arm B log lines and kernel names (experiment step 3) catch both:

- **Batch invariance and LoRA.** Either one makes `AiterExperts.is_supported_config` fail (`model_executor/layers/fused_moe/modular_kernel.py:592-595`), and the oracle then falls to `EMULATION`.
- **FlyDSL installed.** The tuned `_fp4` stage 1 runs only when `is_flydsl_available()` (`aiter/fused_moe.py:2052`). Without it AITER skips the branch that sets `fuse_quant` from the kernel name (`:2107-2115`), so the count of one routed quant at 128 tokens would be wrong; what runs instead was not read. The image should have it: AITER v0.1.19 requires `flydsl==0.2.4` (AITER's `setup.py:466-474`), and vLLM's base image installs AITER's wheel with its dependencies (`docker/Dockerfile.rocm_base:338-340`, `:427-429`).

### The exclusion, derived without a run

On every CUDA part, the experts of an MXFP4 Quark W4A4 checkpoint emulate. The Quark MoE method asks the MXFP4 oracle for a backend with MXFP4 activations (`model_executor/layers/quantization/quark/quark_moe.py:1102-1106`); only `AITER_MXFP4_MXFP4` and `EMULATION` qualify (`model_executor/layers/fused_moe/oracle/mxfp4.py:382-383`, `:443-456`, `:584-602`), and AITER's experts need gfx950 (`model_executor/layers/fused_moe/experts/rocm_aiter_moe.py:475-479`) with AITER's fused MoE on (`:451-453`, condition 2), so on CUDA the oracle picks emulation, which dequantises both expert weight banks to bf16 on every forward (`model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py:202-208`). At v0.19.1 the same exclusion ran through `supports_mx()`; the conclusion is unchanged.

At TP8 on H200 that is every local expert, 384 x 44.04 M / 8 = 2.11 G weights per layer. Each one is read at 0.53 B, written at 2 B and read back at 2 B, which comes to about 9.6 GB per layer, or 575 GB and about 120 ms per step, against a 13.9 ms expert floor at 128 sequences and 1,024 cached tokens. That estimate is a floor on the regression, since `dequant_mxfp4`'s own intermediates are not counted. The test `test_h001_is_excluded_where_mxfp4_emulates` pins this as a scoped rejection rule: never swap to a Quark MXFP4 checkpoint on a platform where the MoE oracle can only emulate. It should be reconsidered when vLLM routes `w_mxfp4_a_mxfp4` to a native CUDA backend.

## Expected effect, derived before any run

Metric: primary, the summed duration and HBM read bytes of the MoE kernels per decode step. Secondary, decode ITL p50. Baseline: `moonshotai/Kimi-K2.5` (INT4 W4A16), the team's existing MI355X loop deployment. From `H001.predict`. Positive numbers are reductions; a negative number is a regression:

| Sequences | Cached tokens | Step floor INT4 (ms) | Step floor MXFP4 (ms) | Routed bytes per layer per rank | Recoverable lo / mean / hi (ms, measured basis) | ITL reduction lo / mean / hi |
|---:|---:|---:|---:|---:|---:|---:|
| **128** | **1,024** | **11.62** | **11.51** | **1,110.9 MB -> 1,049.3 MB (-5.55%)** | **-6.29 / 0.14 / 6.59** | **-47.9% / 0.9% / 33.6%** |
| 64 | 4,352 | 11.02 | 11.13 | | -5.36 / -0.15 / 5.05 | -41.9% / -1.0% / 28.1% |
| 32 | 8,192 | 8.62 | 8.85 | | -3.87 / -0.31 / 3.21 | -37.9% / -2.7% / 23.4% |

The MXFP4 floor includes every activation-quant launch counted above (182 at 128 sequences, 242 at 64 or 32); the INT4 arm has none. At the mean both arms sit at the same efficiency, so the mean ITL column is the floors' ratio. At 64 and 32 sequences the MXFP4 step floor is above INT4's, and the bundled change is the whole reason: its 122 quant launches (0.244 ms) outweigh its 0.008 ms GEMM saving at every row, while the routed mechanism alone (0.462 / 0.367 / 0.243 ms of scale bytes against 60 or 120 routed quants) stays ahead by 0.342 / 0.127 / 0.003 ms.

The row at 128 sequences is registered because the experts are the largest share of the step there: the cache is short and the effect is largest. The 64 x 4,352 row is the team loop's `rag` headline, for comparison.

Why the band dwarfs the mean. The two arms run different MoE kernels: INT4 W4A16 through Triton `fused_moe_kernel_gptq_awq`, and MXFP4 W4A4 through AITER's tuned FlyDSL kernels (arm table below). Letting each land anywhere in the 0.55 to 0.95 efficiency band spans about 6 ms either way on an 8.5 ms MLP floor, and a 0.46 ms byte effect cannot be seen through that. The bytes are the mechanism, and they are what this hypothesis can decide. Latency is recorded and reported, but a latency result alone cannot confirm or reject the scale-overhead claim.

## Experiment

Two arms on the MI355X loop deployment (`deploy/k8s/mi355x-kimi-loop.yaml`: TP8, `VLLM_ROCM_USE_AITER=1`, `--max-num-batched-tokens 8192`), with only the checkpoint changed:

| Arm | Model | Quantisation as served (read from vLLM v0.29.0 + AITER v0.1.19) |
|---|---|---|
| A | `moonshotai/Kimi-K2.5` | compressed-tensors INT4 W4A16: Triton `fused_moe_kernel_gptq_awq`, int4 dequantised per tile in-kernel, bf16 MACs, no activation quant. The WNA16 oracle lists only CUDA backends ahead of `TRITON` (`model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:109`; `model_executor/layers/fused_moe/oracle/int_wna16.py:84-85`, `:117-124`; `model_executor/layers/fused_moe/experts/triton_moe.py:583`, `:713`; `model_executor/layers/fused_moe/fused_moe.py:238`, `:273-276`) |
| B | `amd/Kimi-K2.5-MXFP4` (rev `42c6da36`) | Quark MXFP4 W4A4 on gfx950, backend `AITER_MXFP4_MXFP4`: AITER's 2-stage MoE on the tuned FlyDSL kernels for Kimi's TP8 shape (`aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv:2-17`). The e8m0 weight scales stay 1 B per 32 through `e8m0_shuffle` (`model_executor/layers/fused_moe/oracle/mxfp4.py:1076-1087`; `aiter/ops/shuffle.py:338-364`, a permute whose padding to 256 rows and 8 columns is a no-op at Kimi's shape). Activation quant as counted above. |

1. Load test. Use the team's `guide` helper (`scripts/kimi_loop/run_loop.sh`) at 128 streams with prompt 1,024 and output 256, for 180 s, with 3 repetitions per arm.
2. Kernel plane. Take one `rocprofv3 --kernel-trace` window per arm and sum the MoE kernel durations per decode step. Then replay one layer's MoE kernel offline with memory counters, as the Tier-2 work in `docs/mi355x_experiment_plan.md` describes, to read its HBM bytes. Counters stay out of the serving runs.
   - Use decode-only steps. The 182-launch count and the routed-byte prediction hold for a step that schedules one token per running sequence, 128 tokens at 128 streams. A step that also carries a new request's prefill chunk schedules about 1,024 + 127 tokens, padded to 2,048, where the tuned stage 1 is `flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w4` (csv `:13`), which does not write fp4: two routed quants per layer, each as two kernels at that size (`aiter/ops/quant.py:1090-1135`), and different MoE bytes.
   - Select them by the no-arrival window: start all 128 requests together and keep the steps after the last prefill chunk finishes and before the first request completes. Every step in that window schedules one token per running sequence, 128 tokens, with no prefill chunk. Use the same window for both arms.
   - The stage-1 kernel name cannot select them on its own. AITER keys the tuned table on the token count rounded up to a power of two (`aiter/fused_moe.py:1102-1117`, `:2530-2531`), and the rows for 128, 256 and 512 tokens compile to the same stage-1 symbol (csv `:9-11`; step 3 gives the names a trace shows), so a filter on that name keeps every step with 65 to 512 tokens: 128 decodes plus a prefill remainder of 1 to 128 tokens (`--max-num-batched-tokens 8192` splits long prompts into chunks), or a decode-only step with fewer than 128 running sequences. Those steps still run 182 quants, but distinct experts woken grow with the row count, so their routed bytes differ from the prediction and a byte check over them is biased.
   - Use the kernel name as a consistency check inside the window: each step's 60 stage-1 launches should carry `_t32x128x256` and `_fp4q`. One with `_t128x128x256` and no `_fp4q` is the 2,048-token kernel (csv `:13`), a new request arrived: drop that step, and shorten the window if it happens often.
   - If the engine logs scheduled tokens per step, a step is decode-only when they equal its running sequences, a stricter check that can replace the window.
3. Check the log lines that select the backends. Arm B must print `Using 'AITER_MXFP4_MXFP4' Mxfp4 MoE backend.` (`model_executor/layers/fused_moe/oracle/mxfp4.py:412-413`, `:601`), `Using AITER_MXFP4_MXFP4 backend for OCP_MX_Scheme.w_mxfp4_a_mxfp4` (`model_executor/layers/quantization/quark/quark_moe.py:1136-1138`; the f-string renders the `OCP_MX_Scheme` member with its class name on the image's Python 3.12) and `Using AiterMxfp4LinearKernel for MXFP4 GEMM` (`model_executor/kernels/linear/__init__.py:916`). `Using OCP_MXQuantizationEmulationTritonExperts MOE backend` instead (`model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py:98-102`) means condition 2 did not hold: stop. In the arm B trace the MoE kernels must be FlyDSL's. The CSV's `flydsl_moe1_...` strings are lookup keys, not symbols (`aiter/ops/flydsl/moe_kernels.py:141-164`); the compiled kernels are `mfma_moe1_silu_mul_afp4_wfp4_*` and `mfma_moe2_afp4_wfp4_*` (`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:311-314`, `:467`, `:3253-3257`, `:3274`; FlyDSL 0.2.4 shows that name in the profiler), and the stage-1 symbol carries the tile (`_t32x128x256` for csv `:9-11`) and `_fp4q` when it writes fp4, not the CSV's `_w3`. AITER leaves the FlyDSL branch when `flydsl` is not importable (`aiter/fused_moe.py:2052`). The routed quants show as `fused_mx_quant_moe_sort_kernel` (AITER's `csrc/kernels/quant_kernels.cu:1734`; above 256 tokens `dynamic_per_group_scaled_quant_kernel`, `:30`, then `mxfp4_moe_sort_kernel`, `:2005`), the shared-expert and dense-layer quants as `_dynamic_mxfp4_quant_kernel` (`aiter/ops/triton/quant/quant.py`; the HIP `per_1x32_f4_quant_hip` when the asm fp4 GEMM is on, `model_executor/kernels/linear/mxfp4/aiter.py:28-104`, `:132`). `gitm.optimizer.deviation.classify_op` files the fused quant-and-sort and the sort kernels under `moe_permute`, the split path's `dynamic_per_group_scaled_quant_kernel` under `act_quant` and the Triton quant as unmodeled, so sum the quants by these names, not by op class.

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
| Which ROCm kernel runs INT4 W4A16 MoE, and AITER's MXFP4 scale layout | read at v0.29.0 / v0.1.19 (see the arm table): Triton W4A16, and e8m0 scales stay 1 B per 32 through `e8m0_shuffle` | done |
| Is FlyDSL importable in the loop image | the tuned MXFP4 kernels, and the 182-launch count, depend on it (`aiter/fused_moe.py:2052`); check the arm B trace for `mfma_moe1_silu_mul_afp4_wfp4_*` at 128 sequences (step 3) | Tarun, before the run |
