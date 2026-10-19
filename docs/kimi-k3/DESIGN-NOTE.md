# Kimi K3 — Working Design Note

**Catalogue entry `gitm/planner/models/kimi-k3.yaml`: what the checkpoint is, what
the planner prices, the gap between them, and the code change that closes it.**

Built from the model repo's own files at revision
`f831ab66814297da540d832a5235f8e904f29d06` of `moonshotai/Kimi-K3`: `config.json`,
`README.md`, `configuration_kimi_k3.py`, `modeling_kimi_linear.py`,
`model.safetensors.index.json` (497,220 tensors, 96 shards) and the safetensors
headers of shards 00001 (layer 0), 00002 (layer 1), 00004 (layer 3), 00094
(embed / lm_head / final norms), 00095 (mm_projector) and 00096 (vision tower),
fetched by HTTP Range. **No traces, and no deployment recipe was read.** Every
planner number is a roofline at vendor peak.

Reproduce:

```bash
gitm plan kimi-k3 --gpu MI355X --batch 1  --kv-len 8192 --tp 8
gitm plan kimi-k3 --gpu MI355X --batch 32 --kv-len 8192 --tp 8
.venv/bin/python -m pytest tests/test_catalogue_kimi_k3.py
```

---

## 1. Architecture map

| | |
|---|---|
| Layers | 93: **69 KDA** (linear attention) + **24 gated MLA** |
| Hidden | 7168, vocab 163,840, untied `lm_head` |
| MLA (24 layers) | 96 heads, q_lora 1536, kv_lora 512, nope 128 / "rope" 64 / v 128, **NoPE** (`mla_use_nope`, `rotary_emb = None`), **output gate** `g_proj [12288, 7168]` |
| KDA (69 layers) | 96 heads x 128, q/k/v `[12288, 7168]`, **full-rank** output gate `g_proj [12288, 7168]`, low-rank forget gate `f_a [128, 7168]` / `f_b [12288, 128]`, `b_proj [96, 7168]`, three depthwise convs `[12288, 1, 4]` F32, `A_log`, `dt_bias`, gated RMSNorm `o_norm` |
| FFN | layer 0 dense (33,792, SiTU-GLU); layers 1-92 MoE |
| MoE | 896 routed, sigmoid `noaux_tc` top-16, renormalised, scaling 1.0; **latent**: experts in 3584 space; 2 shared experts as one 6144-wide MLP at full width |
| Residual | **Attention Residuals**, block size 12 (softmax over stored block states, twice per layer) |
| Precision | routed experts MXFP4 (e2m1 + e8m0 per 32, 0.53125 B/weight); everything else BF16 (F32 scalars/convs/bias) |
| MTP | none (`num_nextn_predict_layers 0`, no tensors) |
| Vision | MoonViT-V2, 27 blocks x 1024, 401M + 46M projector |
| Params | **2,779.93B** total (README 2.8T); **104.19B** active excl. input embedding (README 104B) |

### 1.1 The layer lists are 1-indexed

`configuration_kimi_k3.py:152` — `is_kda_layer(i)` is `(i + 1) in kda_layers`. The
weight map is the arbiter and agrees exactly:

* `self_attn.A_log` (KDA) on 0-indexed layers `{0,1,2, 4,5,6, ..., 88,89,90}` — 69 layers
  = `kda_layers - 1`.
* `self_attn.kv_a_proj_with_mqa` (MLA) on `{3, 7, 11, ..., 87, 91, 92}` — 24 layers
  = `full_attn_layers - 1`.
* Layer 0 is **KDA and dense-FFN**. It is missing from both config lists only because
  they count from 1. The final two layers, 91 and 92, are **both MLA**: the 3:1 pattern
  breaks at the end (`full_attn_layers` ends `..., 92, 93`).

GLM-5.3-Flash uses the same key names (`linear_attn_config.kda_layers`) 0-indexed.
Any reader that reads `kda_layers` directly must know which convention it holds; a
generic reader that guesses shifts every layer by one, and K3's tail (two MLA layers
back-to-back) is where it shows.

### 1.2 Latent MoE, read from the header and the code

Shard 00002 (layer 1):

| tensor | dtype | shape | role |
|---|---|---|---|
| `routed_expert_down_proj.weight` | BF16 | [3584, 7168] | 7168 -> 3584, **once per token**, before dispatch |
| `experts.N.w1/w3.weight_packed` | U8 | [3072, 1792] | gate/up, logical [3072, 3584] |
| `experts.N.w2.weight_packed` | U8 | [3584, 1536] | down, logical [3584, 3072] |
| `experts.N.w*.weight_scale` | U8 | [.., 112] / [.., 96] | e8m0 per 32 |
| `routed_expert_norm.weight` | BF16 | [3584] | RMSNorm on the combined routed output |
| `routed_expert_up_proj.weight` | BF16 | [7168, 3584] | 3584 -> 7168, **once per token**, after combine |
| `shared_experts.{gate,up}_proj` | BF16 | [6144, 7168] | full width — sees the un-projected input |
| `gate.weight` / `e_score_correction_bias` | BF16 / F32 | [896, 7168] / [896] | router on full width |

`KimiSparseMoeBlock.forward`: router(x) -> `down_proj` -> `moe_infer` (dispatch,
experts, weighted combine, all at 3584) -> `routed_expert_norm` -> `up_proj` ->
`+ shared_experts(x)`. Consequences: (a) a routed expert is 3 x 3584 x 3072 =
33.03M weights, half what a 7168-wide expert of the same intermediate would be;
(b) every token pays 2 x 3584 x 7168 dense bf16 weights (102.8 MB/layer) whatever
it routes to; (c) EP dispatch/combine moves 3584-wide activations, half the usual.

### 1.3 Attention Residuals

Every layer carries `self_attention_res_{proj,norm}` and `mlp_res_{proj,norm}`
(`proj` is `[1, 7168]` — a learned pseudo-query). `_apply_attn_res` RMS-normalises
the stack `[block_residual..., prefix_sum]`, scores each vector against the query,
softmaxes and returns the weighted sum, in fp32. A new block state is pushed every
12 layers (at 0, 12, ..., 84: 8 blocks), so a layer mixes 2-10 hidden vectors per
token, twice. Over the stack that is **906 hidden vectors per token** (13.0 MB bf16).
Weights: 5.3 MB in total.

### 1.4 KV cache and fixed state

| | per token | per sequence (fixed) |
|---|---:|---:|
| MLA latent, 24 layers x 576 x bf16 | **27,648 B** | — |
| KDA recurrent, 69 x 96 x 128 x 128 x fp32 | — | **434.1 MB** |
| KDA conv, 69 x 3 x 12288 x 3 x bf16 | — | 15.3 MB |

Crossover: the fixed state equals the KV cache at ~16.3k tokens. MLA is
single-latent, so the cache is replicated across TP ranks; KDA state shards on heads.

---

## 2. Tensor-byte ledger — closes on `total_size` to the byte

Shapes and dtypes from the six headers above, applied per layer kind (layer 0; KDA
+ MoE from layer 1; MLA + MoE from layer 3) to all 497,220 names in the weight map.
`tests/test_catalogue_kimi_k3.py` re-derives each row from the header shapes and
asserts the sum.

| class | bytes | % |
|---|---:|---:|
| routed experts, MXFP4 (92 x 896 x 3 x 3584 x 3072 x 0.53125) | 1,446,456,066,048 | 92.67 |
| KDA attention, 69 layers (887,800,832 each) | 61,258,257,408 | 3.92 |
| shared experts, 92 x 3 x 7168 x 6144 x 2 | 24,310,185,984 | 1.56 |
| MLA attention incl. `g_proj`, 24 layers (464,392,192 each) | 11,145,412,608 | 0.71 |
| latent down/up proj + norm, 92 layers | 9,454,620,672 | 0.61 |
| `lm_head` | 2,348,810,240 | 0.15 |
| `embed_tokens` | 2,348,810,240 | 0.15 |
| dense FFN, layer 0 | 1,453,326,336 | 0.09 |
| router (weight + F32 bias), 92 layers | 1,182,074,880 | 0.08 |
| vision tower | 802,428,928 | 0.05 |
| mm_projector | 92,289,024 | 0.01 |
| attn_res proj + norm, 93 layers | 5,332,992 | 0.00 |
| layer norms | 2,666,496 | 0.00 |
| final norm + output attn_res | 43,008 | 0.00 |
| **sum** | **1,560,860,324,864** | |
| `total_size` | 1,560,860,324,864 | residual **0 B** |

Per-shard tensor counts check the per-kind templates: shard 00002 holds 5,404
tensors (896 x 6 expert tensors + 28 for a KDA+MoE layer), shard 00004 holds 5,398
(+22 for MLA+MoE), shard 00001 holds 23 (KDA + dense).

---

## 3. What the planner gets wrong (entry as committed)

Fit: `glm_moe_dsa`, all 93 layers MLA, no indexer (the kimi-k2.5 trick), routed
experts `mxfp4`, and two **stand-in fields** for the latent MoE:
`moe_intermediate_size: 1536` (3 x 7168 x 1536 = 3 x 3584 x 3072, so routed weight
bytes and MACs are exact) and `n_shared_experts: 4` (4 x 1536 = 6144, exact for the
shared MLP). Why not `hybrid`: it has KDA-shaped linear layers but no expert
precision — 1.45 TB of MXFP4 would be priced at bf16 (3.76x), or the whole model at
4 bits — and no MLA.

### 3.1 Weight bytes: −3.585 %, decomposed exactly

`model_weight_bytes` = **1,504,904,085,504 B** against 1,560,860,324,864 B.

| cause | bytes |
|---|---:|
| KDA priced as ungated MLA, 69 x (887,800,832 − 288,227,328) | −41,370,571,776 |
| MLA output gate `g_proj` + 2 norms absent, 24 x 176,164,864 | −4,227,956,736 |
| latent down/up projections + norm absent | −9,454,620,672 |
| vision tower + mm_projector | −894,717,952 |
| attn_res vectors, layer norms, final norms | −8,042,496 |
| router F32 bias | −329,728 |
| routed experts, shared, dense, embed, lm_head | 0 |
| **total** | **−55,956,239,360** |

### 3.2 Decode step bytes (whole model, TP1, MI355X rates)

Planner = sum of node bytes. Hand = weights read + KV read + KDA state read/write
(activations excluded, so the hand figure is slightly low).

| shape | planner | hand | planner vs hand |
|---|---:|---:|---:|
| B=1, S=8k | 82.9 GB | 138.1 GB | **−40.0 %** |
| B=32, S=8k | 721.4 GB | 781.0 GB | −7.6 % |
| B=256, S=8k | 1,740.4 GB | 1,831.3 GB | −5.0 % |
| B=32, S=128k | 1,142.7 GB | 889.7 GB | **+28.4 %** |

At B=1 the step is all dense weights and the planner is missing 41.4 GB of KDA
projections, 9.5 GB of latent projections and 4.2 GB of MLA gates. At B >= 32 the
exact expert term dominates (634 GB at B=32, 1.43 TB at B=256) and the errors shrink
but do not cancel: the missing KDA state alone is 28.8 GB at B=32 and **230 GB at
B=256** (868 MB of fp32 read+write per sequence per step). At long context the
93-layer latent KV overtakes: 449 GB planner `attn_score_value` against 116 GB real
KV at B=32 / 128k.

### 3.3 Memory per sequence

Planner: 107,136 B/token, no fixed state. Real: 27,648 B/token + 449.4 MB. Equal at
~5.7k tokens; the planner is low below that and high above it (1.76 GB vs 0.90 GB at
16k, 112.3 GB vs 29.4 GB at 1M — 3.8x). Concurrency from `gitm plan`'s fit line is
therefore optimistic for short chats and pessimistic for long-context serving.

### 3.4 Smaller, unmodelled terms

* **Stand-in activation shape.** `moe_permute` / `moe_combine` / EP all-to-all are
  priced at 7168 wide (real 3584, 2x over); the expert intermediate at 1536 (real
  3072, 2x under). `expert_pad_factor` sees `(7168, 1536/n)` instead of
  `(3584, 3072/n)`: identical at TP 1, 2, 8 and under EP, **+33 % at TP4** (384 -> 512
  where the real 768 needs no pad), **2x at TP16**.
* **Attention Residuals**: no node. 0.42 GB at B=32 decode (0.05 %); 106 GB of
  activation traffic per 8,192-token prefill chunk in the reference layout (~13 ms
  at 8 TB/s against a ~0.7 s compute-bound chunk, ~2 %). A fused kernel may do far
  less.
* **SiTU-GLU** priced as SwiGLU — same FLOP class, no byte effect.
* **MXFP8 activations.** README: QAT for MXFP4 weights / MXFP8 activations. The
  planner's `resolve_execution` picks the MAC dtype per SKU; on MI355X it chooses
  `aiter_ck2stages` fp4 MACs. Settle on a capture.
* **NoPE MLA**: `attn_qnorm_rope_insert` charges RoPE FLOPs that do not run (tiny).
* **Vision** (0.89 GB): off the text path.

---

## 4. Deployment shape (unverified)

1.561 TB of weights. 8 x H200 (1.13 TB) does not fit. 8 x MI355X or 8 x B300 (288
GB each) fits: `gitm plan kimi-k3 --gpu MI355X --tp 8` prices 251.9 GB/rank
(routed experts carry the aiter 384 -> 512 pad at TP8, x1.33) leaving 7.3 GB for KV
at 0.9 utilisation. That KV figure is wrong for this model (§3.3): the real cache is
4x smaller per token but each sequence also needs 56 MB/rank of KDA state at TP8.

---

## 5. Code-change proposal

Target: `glm_moe_dsa` (`gitm/planner/glm_graph.py`), not a new family. MLA, per-op
precision, MXFP4 experts, the no-indexer trick and the MoE node set are all already
right for K3; what is missing is three per-layer/per-block features. Start by
landing `origin/fix/glm53-flash-kda-layers` (c51462d + 2291e13), which already adds
the KDA layer kind to this exact family.

### 5.1 What c51462d gives, and where it stops for K3

It adds `layer_types` / `linear_num_heads` / `linear_head_dim` /
`linear_conv_kernel_dim` / `linear_state_dtype`, `kda_weight_bytes_per_layer`,
`kv_fixed_bytes_per_sequence`, MLA-only `kv_bytes_per_token`, and `_emit_kda`
(`linattn_in_proj`, `linattn_conv`, `linattn_recurrent`, `attn_out_proj`); 2291e13
pairs vLLM's KDA kernels in `gitm/optimizer/deviation.py`. Byte-identical for
existing entries when `layer_types` is empty. Applied to K3 unchanged it prices the
output gate as **low-rank** (`g_a [128, 7168]` + `g_b [12288, 128]`, 2.5M weights)
where K3 has a **full-rank** `g_proj [12288, 7168]` (88.1M): 716.3 MB/layer against
887.8 MB, −11.8 GB over 69 layers. It also reads the schedule only from a config
`layer_types` key, which K3's config does not have.

### 5.2 The change, field by field

| # | change | where | effect on K3 |
|---|---|---|---|
| 1 | Merge c51462d + 2291e13; set `layer_types` (69 `linear_attention` / 24 MLA, 0-indexed) on the entry | glm_graph.py, deviation.py | KV rate 107,136 -> 27,648 B/token; +449 MB fixed state/seq; KDA nodes replace 69 x MLA |
| 2 | `linear_full_rank_gate: bool = False`. True: `h*qkv/tp` replaces `2*h*hd + ...g_b` in `kda_weight_bytes_per_layer`, and `_emit_kda`'s up-gate GEMM becomes `(rows, h) @ (h, qkv/tp)` | glm_graph.py | KDA lands at 887,455,744 B/layer vs 887,800,832 on disk (−0.039 %; the rest is F32 convs/`A_log`/`dt_bias`/`o_norm`) |
| 3 | `mla_output_gate: bool = False`. True: `+ h * n_heads * v_head_dim / tp` in `attn_per_layer`, and an `attn_gate_proj` node (sharded GEMM + sigmoid-mul) on MLA layers; add the name to `deviation._OP_RULES` | glm_graph.py, deviation.py | +4.23 GB weights, +4.23 GB/step at B=1 |
| 4 | `routed_expert_hidden_size: int = 0` (0 = `hidden`), `latent_moe_norm: bool = False`. Experts use `lh` in `model_weight_bytes`, `per_expert_weights = 3*lh*inter`, `per_position_flops = 6*lh*inter`, `expert_pad_factor(ex, lh, inter, ...)`; `moe_permute` / `moe_combine` / EP all-to-all at `lh`; new `moe_latent_down` (before `act_quant`/permute) and `moe_latent_up` (after combine, + norm) nodes at `weight_dtype`; weights `+ n_sparse * 2*h*lh / tp`. Shared stays at `hidden` | glm_graph.py, deviation.py | entry drops the stand-ins (3072 / 2); +9.45 GB weights; correct pad at TP4/16; 2x-smaller EP traffic |
| 5 | `attn_res_block_size: int = 0`. When set, emit an `attn_res` node twice per layer: bytes `rows * (nb+1) * h * aw` + output, FLOPs `~6 * rows * (nb+1) * h`, `nb = layer // block + {0, 1}`; mark `estimated` (engine layout unknown) | glm_graph.py | +0.05 % decode, ~+2 % prefill |
| 6 | `spec_from_hf_config`: build `layer_types` from `linear_attn_config.kda_layers` when `layer_types` is absent, **with the offset chosen by `model_type`** (`kimi_linear` 1-based; GLM-5.3 provides `layer_types` 0-based), and validate both lists partition `range(n_layers)`. Read `routed_expert_hidden_size`, `latent_moe_use_norm`, `mla_use_output_gate`, `use_full_rank_gate`, `attn_res_block_size` | glm_graph.py | the HF-config path stops pricing K3 as 93 x MLA |
| 7 | Tests: K3 weight bytes within 0.01 % of `total_size` minus vision; KV rate 27,648; fixed state 449.4 MB; node counts (69 `linattn_recurrent`, 24 `attn_score_value`, 92 `moe_latent_down`); byte-identical graphs for every existing `glm_moe_dsa` entry with the new fields at default | tests/test_glm_graph.py | regression guard |

After 1-4, `model_weight_bytes` for K3 = 1,560,860,324,864 − vision/projector
(894,717,952) − norms/attn_res/bias/F32 residue (~32 MB) ≈ **−0.06 %**, and batch-1
decode bytes move from −40 % to within a few percent of the hand ledger
(the remainder is activation traffic, which the planner prices and the hand ledger omits).

### 5.3 Effort

| item | effort |
|---|---|
| 1 rebase fix branch (merge-base 21348b5; it predates `stored_dtype_overrides`, `index_kv_dtype`, `mtp_separate_embedding_head` and the dense-FFN double-multiply fix now on HEAD, and edits the GLM-5.3 entry that has since landed) | 0.5 day |
| 2 full-rank KDA gate | 2 h |
| 3 MLA output gate + deviation rule | 2-3 h |
| 4 latent MoE (weights, 2 nodes, permute/combine/EP width, pad) | 1 day |
| 5 attn_res node | 0.5 day |
| 6 HF-config reader + index convention | 2-3 h |
| 7 tests, byte-identity diff across existing entries | 0.5 day |
| **total** | **~3.5 engineer-days**; 1-3 alone (~1 day) remove 81 % of the weight gap and the KV/state error |

Fields 2, 4 and 5 are also what a `hybrid`-family Kimi-Linear would need, so if the
shared-code owner prefers one place for linear attention, items 2 and 5 can land in
`hybrid_graph.py` too — but `hybrid` additionally lacks MLA and expert precision, so
`glm_moe_dsa` is the cheaper host.

---

## 6. Open questions

1. **Engine.** Which vLLM/SGLang build serves `KimiK3ForConditionalGeneration`, at
   what TP/EP, on what SKU. Every per-rank number above assumes TP8 without EP.
2. **KDA state dtype in the engine.** fp32 in the reference (434 MB/seq); bf16 halves
   it, and it is the second-largest per-sequence cost at B >= 32.
3. **Attention Residuals kernel.** Materialised `[T, nb, h]` stack or fused.
4. **Latent projection sharding.** Replicated or TP-split; moves 9.45 GB/TP per rank.
5. **Expert MAC dtype.** W4A8 (MXFP8 activations, per README QAT) vs W4A16.
