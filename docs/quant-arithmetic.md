# Quantisation arithmetic in the planner

This note is the derivation behind `gitm/planner/roofline.py`'s format and execution tables. It covers bytes per parameter, how a stored format executes on each SKU, ridge points and knees, and temporary traffic. The last three sections reproduce two existing design notes against the code: the Kimi K2.6 NVFP4 case and `docs/glm-5.2/DESIGN-NOTE.md`. Where a note was wrong, the discrepancy is resolved with evidence and pinned by a test in `tests/test_quant_arithmetic.py`.

Every engine fact is read from vLLM v0.19.1 (commit `b1388b1f`), the version the K2.6 case pinned. Engine paths are relative to `vllm/model_executor/layers/` unless they start with `vllm/`. CDNA4 facts are read from the AITER tag vLLM's ROCm image pins (`docker/Dockerfile.rocm_base:12`, v0.1.10.post2, commit `c3708fb`) and its composable_kernel submodule (`7b18f5f`); those paths start with `aiter/` or `ck/`.

## One weight, four numbers

A quantised checkpoint answers one question by itself: how a weight is stored. Three more depend on the engine and the SKU, and the planner now keeps them apart.

| Quantity | Decides | Owner in code |
|---|---|---|
| Stored bytes per weight (payload + block scales) | checkpoint size | `QuantFormat.bytes_per_elem` |
| Resident bytes per weight after load (storage x backend padding) | fit | `WeightExecution.resident_bytes`, `expert_pad_factor` |
| Streamed bytes per weight per use, plus scratch the backend writes and reads back | the memory term | `WeightExecution.bytes_per_use` |
| MAC dtype | the compute term and the ridge | `WeightExecution.compute_dtype`, via `resolve_peak` |

## Stored formats

bytes per weight = payload bits / 8 + scale bytes / block size

| Format | Payload | Block | Scale | Bytes/weight | Scale share | Read from |
|---|---:|---:|---|---:|---:|---|
| `int4_g32` | 4 | 32 | bf16 (2 B) | 0.5625 | 11.1% | Kimi K2.5 `quantization_config`: group 32, symmetric |
| `nvfp4` | 4 (e2m1) | 16 | e4m3 (1 B) | 0.5625 | 11.1% | K2.6 headers `U8 [2048,3584]` + `F8_E4M3 [2048,448]` |
| `mxfp4` | 4 (e2m1) | 32 | e8m0 (1 B) | 0.53125 | 5.9% | OCP MX v1.0 |
| `mxfp8` | 8 (e4m3) | 32 | e8m0 (1 B) | 1.03125 | 3.0% | OCP MX v1.0 |
| `fp8_block128` (label `fp8`) | 8 (e4m3) | 128 x 128 | fp32 (4 B) | 1.000244 | 0.02% | GLM-5.2-FP8 `weight_block_size [128,128]` |
| `fp8_group128` (activations) | 8 | 1 x 128 | fp32 | 1.03125 | 3.0% | `quantization/fp8.py:317`, `utils/fp8_utils.py:931-932` |
| `fp8_tensor` (KV cache) | 8 | none | per-tensor scalar | 1.0 | 0 | `quantization/kv_cache.py:88-91` |
| `fp8_ds_mla` (KV cache latent) | 8 | 128 | fp32 | 1.03125 | 3.0% | 656 B per 576-dim entry with a bf16 RoPE key |

NVFP4 and INT4 g32 cost exactly the same (a 1-byte scale per 16 weights equals a 2-byte scale per 32). That is why the NVFP4 and INT4 Kimi checkpoints are both 595 GB. Per-matrix fp32 scalars (NVFP4's `weight_scale_2` and `input_scale`) are 8 bytes against about 10^7 weights and are left out of the per-weight figure.

A cache is not a weight. The `fp8` label means a 128x128 block scale on the weight side and one scale per layer on the cache side, so `kv_elem_bytes` and `weight_bytes` answer differently.

## Execution per SKU, H200 first

| Stored | H200 (Hopper) | B200 (Blackwell) | MI355X (CDNA4) |
|---|---|---|---|
| `nvfp4` | Marlin W4A16, **bf16** MACs, streams 0.5625, no activation quant | TRT-LLM W4A4, fp4 MACs, one `scaled_fp4_quant` per MoE input | **no kernel**: every NVFP4 backend is CUDA-gated (`oracle/nvfp4.py:261`) |
| `mxfp4` | Triton (triton_kernels), bf16 MACs [estimated: triton_kernels not read] | TRT-LLM **MXFP4 x bf16**, bf16 MACs, per-rank dims padded to 256 | AITER CK 2-stage, scaled f8f6f4 MFMA (fp4 MACs), MXFP4 activations by a separate kernel fused with the MoE sort, per-rank dims padded to 256 [estimated: small-batch `cktile` dispatch unresolved] |
| `mxfp4`, forced Marlin | bf16 MACs, hidden padded to 256, per-rank intermediate to 128 | same | not available (Marlin is CUDA-only) |
| `mxfp8` | **no kernel**: raises `UnsupportedExecution` | TRT-LLM, fp8 MACs, activations MXFP8 | not pinned |
| `fp8_block128` | fp8 MACs, activations `fp8_group128` | same | AITER `fmoe_fp8_blockscale_g1u1` (or CK 2-stage at M <= 32), fp8 e4m3fn MACs, activations `fp8_group128` by a separate kernel, `shuffle_weight` is layout-only [estimated: assumes `VLLM_ROCM_USE_AITER=1`; the Triton block-fp8 path without it was not read] |
| `int4_g32` | Marlin W4A16, bf16 MACs | Marlin W4A16, bf16 MACs | Triton `fused_moe_kernel_gptq_awq` W4A16, int4 dequantised per tile inside the kernel, bf16 MACs, no repack |

Evidence for each cell is in `_EXECUTION_RULES` and `_FORCED`, with file and line. Four results were not obvious before reading the source:

1. **NVFP4 on H200 is W4A16.** The NVFP4 MoE oracle's priority list is TRT-LLM, CuteDSL, FlashInfer CUTLASS, vLLM CUTLASS, then Marlin (`fused_moe/oracle/nvfp4.py:140-146`), and only Marlin accepts sm90 (`fused_moe/fused_marlin_moe.py:567`). Marlin keeps the payload at K x N / 2 bytes (`quantization/utils/marlin_utils_fp4.py:335-344`) and the scales at one byte (`:84-110`). It refuses fp8 or int8 activations for NVFP4 (`:305-307`). The old planner fell down the ladder to the fp8 peak, which is 2x too fast for the MACs.
2. **MXFP4 on B200 also runs at bf16 by default.** `FLASHINFER_TRTLLM_MXFP4_BF16` is first in the MXFP4 priority list (`fused_moe/oracle/mxfp4.py:174-182`). The MXFP4 x MXFP8 variant is not.
3. **On MI355X none of the three formats streams anything but the checkpoint's bytes.** `shuffle_weight` (`aiter/ops/shuffle.py:7-26`) is a permute and a view with the same shape and dtype; `e8m0_shuffle` (`aiter/utility/fp4_utils.py:72-92`) pads scale rows to 256 and columns to 8, a no-op at Kimi's shape; the INT4 path only transposes and re-views (`compressed_tensors_moe.py:1824-1839`). What ROCm adds is on the activation side: block fp8 and MXFP4 both quantise activations in their own kernel before the GEMM, and INT4 does not. One open item: for `token x top_k <= n_experts` with shuffled MXFP4 weights, AITER routes to a `cktile` path that takes bf16 activations (`aiter/fused_moe.py:803-827, :944-954`), so small-batch decode may run W4A16 there; whether the `is_shuffled` attribute survives vLLM's custom-op boundary was not verifiable from source, and the rule remains marked estimated until dispatch is verified.
4. **Padding is a storage-versus-execution cost the format does not show, and it is applied per rank.** vLLM splits the intermediate dim across tensor-parallel ranks first (`fused_moe/layer.py:426`) and then rounds up what each rank holds (`:537-538` calls `maybe_roundup_sizes(hidden_size, intermediate_size_per_partition)`); hidden is not split for the expert GEMM. Marlin rounds hidden to 256 and the per-rank intermediate to 128 (`oracle/mxfp4.py:380-385`), TRT-LLM rounds both to 256 (`:386-388`), and the ROCm MXFP4 path rounds both to 256 (`:395-397`). Only the MXFP4 methods pad the intermediate at all; the base rule (`fused_moe_method_base.py:69-99`) pads hidden only for DeepEP/NIXL expert-parallel kernels and leaves the per-rank intermediate alone, so NVFP4, block fp8 and INT4 under plain TP pad nothing. The order matters: a 2,880-wide expert across eight ranks is 360 per rank, padded to 512 (TRT-LLM) or 384 (Marlin). Padding the whole 2,880 to 3,072 and then splitting gives 384, which the kernel would pad again; that order under-counts each rank's expert bytes by 25%. Kimi (2,048 / 8 = 256) and GLM-5.2 are exact multiples per rank, so their factor is 1 either way, which is how the wrong order survived the first tests.

## Temporary traffic

Emulation dequantises the whole matrix on every forward. For NVFP4 (`quantization/utils/nvfp4_emulation_utils.py:57-65,130-141`) that is an fp32 unpack (write 4 B), an fp32 scale multiply (read 4, write 4), a bf16 cast (read 4, write 2) and the matmul's read (2): 20 B of scratch per weight on top of the 0.5625 B read from storage. MXFP8's linear fallback follows the same chain (`quantization/utils/mxfp8_utils.py:71-82,157`, marked estimated).

The largest case in practice is an MXFP4 Quark checkpoint on any CUDA part. `supports_mx()` is false off ROCm gfx95x, so the Quark MoE method emulates and dequantises every local expert per forward (`fused_moe/fused_moe.py:1758-1762`). For Kimi at TP8 on H200 that is about 575 GB and about 120 ms per step, against a 7.3 ms expert floor. H-001 turns this into a scoped exclusion rule.

## Ridges and knees

The ridge uses the *execution* rate: `ridge(hw, dtype, tier)` resolves the stored label first.

| H200 | FLOP/byte (HBM 4.8 TB/s) | FLOP/byte (NVLink 900 GB/s) |
|---|---:|---:|
| bf16, and NVFP4/INT4/MXFP4 through Marlin or Triton | 206 | 1,099 |
| fp8 | 412 | 2,199 |
| fp32 (CUDA cores, the router's `.float()`) | 14 | 74 |

A weight-streaming GEMM of shape (r, k) x (k, n) has AI(r) = 2rkn / (wkn + ar(k + n)). The knee, where AI equals the ridge R, is

r* = R w k n / (2kn - R a (k + n)), which is about R w / 2 when k and n are much larger than R a.

`critical_rows` computes it exactly. For a Kimi expert (k = 7,168, n = 2,048, bf16 activations) on H200:

| Format and backend | Knee (rows per expert) | Rows per expert at decode, 32 sequences |
|---|---:|---:|
| NVFP4 or INT4 via Marlin (R 206, w 0.5625) | 67 (58 in the large-matrix limit) | 1.36 |
| block fp8 (R 412, w 1.0002) | 278 (206 in the limit) | 1.36 |

Decode is two orders of magnitude below the knee. A prefill chunk of 8,192 tokens puts 171 rows on each expert, above the knee, so the expert GEMM is compute-bound there. The MAC rate that was wrong in the old planner therefore matters for prefill.

`gitm plan` prints the expert line for every `glm_moe_dsa` entry:

```
experts   nvfp4: 0.5625 B/weight stored (11.1% scales) -> marlin, bf16 MACs, 0.5625 B/weight per use
          1.36 rows/expert at this batch; compute-bound above 67
```

## Reproduction 1: the Kimi K2.6 NVFP4 case

The case's figures are from `analysis/derive.py` and the design note, with B200 at TP4, 32 x 8,192 and a bf16 cache.

| Figure | Case | This code | Verdict |
|---|---:|---:|---|
| Bytes per expert (3 x 7,168 x 2,048 at 0.5625) | 24.77 MB | 24,772,608 B | reproduced |
| Distinct experts per layer, 32 x top-8 of 384 | 188.2 | 188.23 | reproduced |
| Routed bytes per layer per rank | 1,165.7 MB | 1,165.7 MB (+0.1% activations) | reproduced |
| Routed-expert time, 60 layers | 8.74 ms | 8.751 ms | reproduced |
| KV bytes per token, bf16 | 70,272 | 70,272 | reproduced |
| KV bytes per token, fp8 | 39,040 | 35,136 | **case wrong** |
| Cache dtype NVIDIA's command serves | bf16 (claim 1.5) | fp8 | **case wrong** |
| KV room at TP4 after weights | 10.9 GB | 7.0 GB with 4.5 GB workspace | **case inconsistent** |
| Footprint vs 595.15 GB | -0.03% (planner) | -0.16%, the gap being the 0.94 GB vision tower | **match was two errors cancelling** |
| Ridges bf16 / fp8 / fp4 | 281 / 562 / 1,125 | 281 / 562 / 1,125 | reproduced |

The four discrepancies:

- **fp8 KV.** The case priced an fp8 cache as fp8 latent plus bf16 RoPE, 640 B per layer. Dense MLA under `--kv-cache-dtype fp8` uses vLLM's generic layout, which stores all 576 dims at one byte: `head_size = kv_lora_rank + qk_rope_head_dim` (`attention/mla_attention.py:316`) and cache shape `(blocks, block_size, head_size)` (`:1130-1137`). The `fp8_ds_mla` layout that keeps RoPE wide is used only on sparse MLA backends (`:341-362`).
- **Served cache dtype.** The case said the published command, which passes no `--kv-cache-dtype`, stores bf16. vLLM v0.19.1 resolves `auto` from the checkpoint before it builds the cache config (`vllm/engine/arg_utils.py:1567-1570` calling `vllm/utils/torch_utils.py:324-342`). This checkpoint's modelopt `kv_cache_scheme` (static, 8-bit float) maps to fp8 (`:262-296`). The case cited the field's default string (`vllm/config/cache.py:49`) and the scheme parser, and it stopped there. So NVIDIA's default is an fp8 cache with scale 1.0, and the case's attention-core row (2.303 ms at TP4) is really 1.151 ms. The `kimi-k2.6` entry already had fp8; `docs/audits/kimi-k2.6.md` F1 records this.
- **Workspace.** The case's fit charged 4.5 GB of workspace, but its "left for KV" line did not. `MemoryFit` keeps one ledger: budget minus weights minus workspace, then KV. At TP4 that leaves 7.0 GB. At bf16 that holds 99 k tokens, 12 sequences at 8K rather than 18. At the fp8 default it holds 199 k tokens, 24 sequences, against a need of 9.2 GB for 32. The recipe's TP4 shape does not hold the baseline under either dtype.
- **Footprint.** `model_weight_bytes` multiplied the dense FFN by bytes-per-weight twice (fixed in `a6570bc`). That added 0.79 GB to Kimi, which roughly cancelled the 0.94 GB vision tower the planner does not model.

### The 13.56 ms planner run against the 13.38 ms hand derivation, term by term

The case note explained the gap as "the planner pricing collectives as bandwidth-only". That is wrong about the net. Pricing collectives without a latency floor makes the planner *lower* on that term, and other terms pushed it higher by more. This is the full breakdown for the submitted YAML on `main` (`21348b5`), B200 at TP4:

| Term | Hand (ms) | Planner (ms) | Difference | Cause |
|---|---:|---:|---:|---|
| All-reduces, 122 | 1.464 | 0.244 | -1.220 | The planner charges one 2 us launch plus wire time per collective. The case assumed a 12 us latency floor. |
| Router GEMM + gating | 0.041 | 0.662 | +0.621 | The submitted YAML marked the router fp32 (the reference implementation's `.float()`), and B200 had no fp32 peak, so it priced at A100's 19.5 TF/s: 0.542 ms of compute. vLLM runs it as a bf16 GEMM with fp32 output (`router/gate_linear.py:28-29,110-127`). |
| Norms, RoPE insert, permute, combine, embed | not added | 0.610 | +0.610 | The case listed the launch floor separately (2.2 ms) instead of adding one launch per node. |
| Attention linears | 0.559 | 0.767 | +0.208 | The planner has an unabsorbed `kv_b` GEMM, separate `q_a`/`kv_a`, and launch floors on three of them (audit F5, F6). |
| Shared expert | 0.165 | 0.120 | -0.045 | The submitted YAML's `op_dtype_overrides` replaced the parent's list and dropped `moe_shared: bf16`, so the shared expert priced at NVFP4. |
| Routed experts | 8.743 | 8.751 | +0.008 | activation bytes |
| `lm_head`, layer-0 MLP, logits all-gather, attention core | 2.409 | 2.411 | +0.002 | |
| **Total** | **13.381** | **13.564** | **+0.183** | The planner column's rounded terms sum to 13.565. |

With both input errors fixed (router bf16 per the engine, or fp32 at B200's real 75 TF/s; shared expert back to bf16), this branch prices the same configuration at 13.335 ms. That includes the 60 W4A4 activation-quant launches the case did not model.

## Reproduction 2: GLM-5.2 on 8xH200, TP8/EP8, FP8

| Figure | Note | This code | Verdict |
|---|---:|---:|---|
| Ridges fp8 / bf16 / fp32 | 412 / 206 / 14 | 412.3 / 206.0 / 14.0 | reproduced |
| Decode floor, 32 x 8,192 | 16.254 ms | 16.254 ms | reproduced |
| `moe_routed` | 12.052 ms | 12.052 ms | reproduced |
| KV per token | 52,618 B (the note flags it) | 52,608 B | **resolved** |
| bf16 checkpoint vs 1,506,659,919,872 B | +0.08% | +0.00% | **resolved** |
| `act_quant` bytes per layer at 32 rows | 0.590 MB | 0.596 MB | **corrected**; the node stays launch-bound, so the floor is unchanged |

- **KV.** The note used the weight constant 1.000244 for cache bytes and said so. With a per-tensor cache scale an fp8 element is exactly 1 B: 78 x (512 + 128) + 21 x 128 = 52,608.
- **bf16 footprint.** The +0.08% was the dense FFN double-count: three dense layers, 1.36 GB.
- **act_quant.** Block fp8 quantises activations with one fp32 scale per 128 channels, 48 scales per row at hidden 6,144 (`quantization/fp8.py:317`). The old node charged one scale per row.

On B200 the GLM entries now price 0.2% to 2.2% faster, because the fp32 router runs at B200's 75 TF/s CUDA-core rate (HGX B200: 600 TFLOPS FP32 across eight GPUs) instead of A100's.

## What moved across the catalogue

`gitm plan` totals before (`21348b5`) and after, for every entry at 32 x 8,192 with TP 1 and 8 on H200, B200 and MI355X. 20 of 36 predictions are unchanged. Every change has one of these causes:

| Change | Why |
|---|---|
| `glm-5.2`, `glm-5.2-fp8`, `mimo-v2.5` on B200: -0.2% to -3.4% | fp32 peak 75 TF/s replaces 19.5 |
| `kimi-k2.6` on B200: +0.3% (TP1), +1.6% (TP8) | 60 NVFP4 activation-quant nodes |
| `kimi-k2.6` on H200: same floor, no longer flagged as fallback | Marlin's bf16 rate is the right rate |
| `glm-5.2-fp8` on H200: -0.00% | fp8 cache at 1.0 B instead of 1.000244 |

## Reproduce

```bash
python -m pytest tests/test_quant_arithmetic.py tests/test_hypotheses.py
gitm plan kimi-k2.6 --gpu H200 --batch 32 --kv-len 8192 --tp 8 --workspace-gb 4.5
gitm plan kimi-k2.6 --gpu H200 --batch 32 --kv-len 8192 --tp 8 --kv-cache-dtype auto
gitm plan glm-5.2-fp8 --gpu H200 --batch 32 --kv-len 8192 --tp 8 --ep 8
```

## Validation status

Everything above is validated against source: checkpoint headers and configs, vLLM v0.19.1, AITER v0.1.10.post2 and its CK submodule. None of it is validated against hardware. The efficiency band (0.55 to 0.95), the 2 us launch floor, and the rules still marked estimated (MXFP4 via Triton on Hopper because `triton_kernels` was not read; block fp8 and MXFP4 on MI355X because both assume `VLLM_ROCM_USE_AITER=1`, which the planner cannot see, and MXFP4's small-batch dispatch remains unresolved) are the parts a measurement should replace first. MI355X peaks are not stated anywhere in the AITER or vLLM trees; the catalogue's figures come from AMD's published dense rates. H-001 and H-002 are the first two measurements registered for that.

`gitm plan --json` includes requested and resolved KV dtype, utilization, workspace bytes, and the same per-rank `memory_fit` ledger as text output. All ledger sizes are bytes. Families without a fit model return `memory_fit: null` with an explicit reason, and so does a SKU with no HBM capacity in the catalogue (the unknown-SKU fallback), rather than a ledger against zero capacity. Utilization must be finite and in (0, 1]; workspace must be finite and nonnegative.
