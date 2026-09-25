# Audit: the `kimi-k2.6` catalogue entry against checkpoint and engine evidence

| | |
|---|---|
| Entry | `gitm/planner/models/kimi-k2.6.yaml` (extends `kimi-k2.5`), family `glm_moe_dsa` |
| Checkpoint | `nvidia/Kimi-K2.6-NVFP4` @ `2fd3a800`; base `moonshotai/Kimi-K2.6` @ `7eb5002f` |
| Engine | vLLM v0.19.1 (`b1388b1f`). Engine paths are relative to `vllm/`. |
| Operating point | 8xH200, TP8, EP1, 32 sequences x 8,192 cached tokens, decode, no speculation |
| Date | 2026-09-24 |

The audit checks four things against config and code evidence: precision, execution backend, parallelism and traffic. Every finding lists its evidence, the code path it affects, the prediction error at the operating point (planner floor at vendor peak, from `gitm plan`) and a proposed correction.

Each finding is also classified. A **missing modelled cost** is time the hardware spends that the planner does not predict. An **over-modelled cost** is predicted time the engine does not spend. Neither one is an optimisation opportunity by itself: finding a cost does not show that it can be recovered.

## What was checked and agrees

| Dimension | Catalogue | Evidence | Verdict |
|---|---|---|---|
| Expert storage | `expert_dtype: nvfp4` | shard headers `U8 [2048,3584]` + `F8_E4M3 [2048,448]` per matrix; `hf_quant_config.json` group 16 | agrees |
| Non-expert precision | attention, shared expert, layer 0, `lm_head` bf16 (`moe_shared` restated) | `hf_quant_config.json` exclude list; `quantization/modelopt.py:186-199` gives excluded modules `UnquantizedLinearMethod` | agrees |
| Router precision | bf16 (listed as estimated) | `models/deepseek_v2.py:341-346` and `router/gate_linear.py:28-29,110-127`: bf16 weight, `router_gemm_bf16_fp32` on bf16 tensor cores with fp32 output | agrees; can move from estimated to verified |
| Parallelism | priced as TP with experts sharded on the intermediate dim, no EP | NVIDIA's card serves `--tensor-parallel-size 4` with no EP flag; the team's Kimi deployment (`deploy/k8s/mi355x-kimi-loop.yaml`) and its predictions (`scripts/kimi_loop/predict_sweep.py`) are both TP8 with no EP; `fused_moe/layer.py:425-426,873-946` shards experts on the intermediate dim under TP | agrees; no EP-pinned/TP-modelled contradiction for Kimi |
| KV sharding | cache replicated per rank, not divided by TP | MLA builds one latent (`num_kv_heads=1`); every rank reads the whole cache | agrees |
| Collectives | two all-reduces per layer plus one logits all-gather | `models/deepseek_v2.py:384-398`, `layers/logits_processor.py:75-86` | agrees on count (see F4 for the norm fused into them) |
| Footprint | 595.19 GB, entry claims -0.03% | now -0.16%, the gap being the 0.94 GB vision tower (F7) | agrees; the old -0.03% was two errors cancelling |
| Served cache dtype | `kv_dtype: fp8` | `auto` resolves from the checkpoint's `kv_cache_scheme` to fp8 (F1) | agrees |

## Findings

### F1. The served cache dtype: the entry is right, and the K2.6 case was wrong

- **Evidence.** The entry sets `kv_dtype: fp8`. NVIDIA's model card serves with `--tensor-parallel-size 4 --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code` and no `--kv-cache-dtype`. vLLM v0.19.1 resolves `auto` from the checkpoint before building the cache config (`engine/arg_utils.py:1567-1570`, calling `utils/torch_utils.py:324-342`). A modelopt config whose `kv_cache_scheme` is `{dynamic: false, num_bits: 8, type: float}`, as this checkpoint's is, resolves to fp8 (`:262-296`). So the published command stores a one-byte cache, the same as the entry.
- **What this corrects.** The K2.6 case (engine claim 1.5) said the default cache was bf16. It cited `config/cache.py:49` (the field's default string) and `quantization/modelopt.py:300-311` (the scheme parser), but it did not follow the `auto` resolution in `arg_utils.py`. Every conclusion built on a bf16 default moves:
  - The attention core at TP4 on B200 is 1.151 ms, not 2.303 ms.
  - The step is about 1.15 ms faster.
  - The recipe's TP4 shape still does not hold 32 x 8,192 with 4.5 GB of workspace: it needs 9.2 GB of KV against 7.0 GB available.
- **Residual caveat.** The checkpoint ships no `k_scale`/`v_scale` tensors, so the fp8 cache runs with scale 1.0 (`quantization/kv_cache.py:71-75`). That is an accuracy question, not a prediction error.
- **Change made.** `gitm plan --kv-cache-dtype {auto,bf16,fp16,fp8}`. `auto` keeps the catalogue's resolved value, and an explicit dtype prices that cache instead.
- **Class.** No discrepancy in the entry. The error was in the case note, and the audit records it so that nobody reintroduces it.

### F2. NVFP4 on Hopper was priced at the fp8 peak; vLLM runs Marlin at bf16

- **Evidence.** H200 has no FP4 tensor cores. The NVFP4 MoE oracle's only sm90-capable backend is Marlin (`fused_moe/oracle/nvfp4.py:140-146`; `fused_marlin_moe.py:567`), which asserts bf16/fp16 activations (`quantization/utils/marlin_utils_fp4.py:135`) and refuses W4A8 for NVFP4 (`:305-307`).
- **Affected path.** `roofline.resolve_peak` fell down the ladder from fp4 to fp8, and flagged every H200 prediction as a fallback.
- **Expected prediction error.** Decode is unchanged, because the expert GEMM is memory-bound at 1.36 rows per expert. Prefill is not. An 8,192-token chunk puts 171 rows on each expert, above the 67-row knee. The routed-expert node's compute time was 21.9 ms and should be 43.8 ms, and that node sets the prefill step.
- **Correction.** Shipped in `13e4d7f`: `resolve_execution` maps (nvfp4, hopper) to Marlin at bf16. `peak_is_fallback` is now false because the rate is the right one, not a guess.
- **Class.** A missing modelled cost, 2x on prefill expert compute.

### F3. NVFP4 activation quantisation on Blackwell was unmodelled

- **Evidence.** The entry listed it as unmodelled. The TRT-LLM NVFP4 MoE path launches one `scaled_fp4_quant` per MoE input (`fused_moe/oracle/nvfp4.py:370-381`; `_custom_ops.py:78,1633`).
- **Expected prediction error.** 60 launches per step: +0.120 ms at B200 TP8 (+1.6%). On H200 it is zero, because Marlin is weight-only.
- **Correction.** Shipped in `13e4d7f`: an `act_quant` node, gated on the execution's activation format. The entry's `unmodelled` line should be removed.
- **Class.** A missing modelled cost.

### F4. The all-reduce and the following RMSNorm run as one kernel under the default compile level

- **Evidence.** At `-O2`, the default, `fuse_allreduce_rms` is `enable_allreduce_rms_fusion` (`config/vllm.py:203-208`). That is on for TP > 1 on sm90 or the sm100 family with FlashInfer installed (`:116-130`). The fused kernel is used while the all-reduce payload fits the FlashInfer cap (`compilation/passes/fusion/allreduce_rms_fusion.py:54-68,759`): 0.5 MiB at TP8 on sm90. That cap is 36 tokens of hidden 7,168 at bf16. At 32 sequences the payload is 458,752 B and the fusion applies; at 64 it does not.
- **Affected path.** `glm_graph._emit_layer` emits `rms_norm` as its own node after each collective (`add_rms_norm(with_residual=True)`).
- **Expected prediction error.** 120 of the 123 norm nodes follow a collective. The planner over-predicts 120 launches, -0.240 ms at H200 TP8 with 32 sequences (2.1%). The error is zero above 36 tokens per step on H200 TP8, and above 73 on B200 TP8.
- **Proposed correction.** When `hw.arch` is hopper or blackwell, TP > 1, and `rows x hidden x 2` fits the cap for (arch, TP), fold the post-collective `rms_norm` into the collective node. The fused kernel's name then has to classify to the collective in `deviation.classify_op`. That touches the pairing contract, so it is filed for Rahul rather than changed here.
- **Class.** An over-modelled cost. The fusion is already on by default, so it is not an opportunity.

### F5. MLA is modelled unabsorbed; vLLM absorbs at decode

- **Evidence.** vLLM splits `kv_b_proj` into `W_UK_T` and `W_UV` (`layers/attention/mla_attention.py:734-797`). At decode it applies them as two bf16 bmm's, one on the query before attention (`:664`) and one on the output after it (`:847-872`). The attention kernel then reads the 576-wide latent with all heads.
- **Affected path.** The `attn_kv_b` node, which the entry already lists as estimated.
- **Expected prediction error.** Launches: +1 per layer, since two bmm's replace one GEMM. That is +0.122 ms, 1.1% under-predicted. Bytes are unchanged, because the same weights are read. Attention-core FLOPs are 3.4x under-predicted, but the core stays memory-bound, so the floor does not move. Pairing: `attn_kv_b` is predicted as one GEMM, while the capture shows two bmm kernels.
- **Proposed correction.** An `mla_absorbed` flag on the spec. At decode, it emits two bmm nodes and prices the core's FLOPs at `kv_lora_rank + qk_rope_head_dim` per head. Prefill stays unabsorbed, matching `forward_mha`.
- **Class.** A missing modelled cost (launches and FLOPs), plus a node-identity mismatch.

### F6. `q_a_proj` and `kv_a_proj_with_mqa` run as one fused GEMM

- **Evidence.** `models/deepseek_v2.py:870-874` builds `fused_qkv_a_proj` whenever `q_lora_rank` is set.
- **Expected prediction error.** -0.011 ms: two nodes' memory time plus one extra launch becomes one node. The larger effect is on pairing, because `attn_kv_a` is predicted and never observed.
- **Proposed correction.** Emit one replicated node of width `q_lora_rank + kv_entry_dim` when the family has a query LoRA.
- **Class.** An over-modelled launch, which matters for identity more than for time.

### F7. The resident vision tower is not in the fit

- **Evidence.** The checkpoint's 595.148 GB includes 0.942 GB of vision tower and projector (summed from the shard headers in the K2.6 case). With `--mm-encoder-tp-mode data` the tower is replicated on every rank.
- **Affected path.** `model_weight_bytes`, and so `memory_fit`.
- **Expected prediction error.** The fit overstates KV room by 0.94 GB per rank, 2.0% of the 46.2 GB available at TP8 with 4.5 GB of workspace. Before `a6570bc` this was hidden: the dense FFN was counted twice (0.79 GB), which is why the footprint matched to -0.03%.
- **Proposed correction.** A `resident_extra_bytes` field on the entry (0.942e9 for K2.x), added to `model_weight_bytes` unsharded.
- **Class.** A missing modelled cost, affecting fit only.

## The vendor configuration, re-priced

| | Decode floor, H200 TP8, 32 x 8,192 (ms) |
|---|---:|
| Entry as catalogued (fp8 cache, which is also vLLM's resolved default) | 11.404 |
| F4: fused all-reduce + norm | -0.240 |
| F5: absorbed MLA, one extra launch per layer | +0.122 |
| F6: fused `q_a`/`kv_a` | -0.011 |
| Corrected estimate | 11.275 |

F2, F3 and F7 do not change this row: F2 and F3 affect prefill and Blackwell, and F7 affects fit only. The corrected estimate is still a floor at vendor peak, and no hardware measurement backs it yet.

## Summary

The checkpoint's shape, precision, parallelism, traffic model and served cache dtype all agree with the evidence. The entry's largest errors at this operating point are engine-lowering details: the default all-reduce+norm fusion (F4) and MLA absorption (F5), each about 1 to 2% and in opposite directions. Two costs were missing and are fixed on this branch: NVFP4's execution rate on Hopper (F2, 2x on prefill expert compute) and its activation quantisation on Blackwell (F3). Three corrections are filed (F4, F5, F6), and one fit term is filed (F7). F1 records an error in the K2.6 case, not in the entry.
