# GLM-5.2 — standup brief (execution-graph view)

**What this is:** GLM-5.2 (`zai-org/GLM-5.2`, `GlmMoeDsaForCausalLM`) modelled end-to-end
in the GitM runtime/execution-graph stack, as a new **`glm_moe_dsa`** family forked from
the DeepSeek-V4 sparse-MoE graph. Everything below is derived **from `config.json` and the
checkpoint's own `model.safetensors.index.json`** — no traces, no observed performance. The
numbers are a roofline **floor at vendor peak**, i.e. a lower bound on time, not a target.

Reproduce any figure here:
```bash
gitm plan glm-5.2 --gpu H200 --batch 1 --kv-len 4096        # op-share table
gitm plan glm-5.2 --gpu H200 --sweep 1,4,16,32,64,128,256   # batch crossover
gitm plan glm-5.2 --gpu H200 --batch 1 --kv-len 131072 --json > graph.json
```

---

## 1. What GLM-5.2 is, in one breath

A **754B-parameter** (1.507 TB bf16, validated to **+0.1%** against the published checkpoint)
sparse-MoE decoder, **~40B active per token** (config-derived, using the repo's `active_params`
convention — which folds in the ~1B untied embed + lm_head, so ~39B is the real per-token
multiply). Three things make it its own family rather than "another DeepSeek":

1. **MLA attention** — one compressed KV latent (`kv_lora_rank=512`) shared across all 64
   query heads. There is **no** per-layer compression schedule (no CSA/HCA, no sliding
   window) — every layer runs the same attention.
2. **DeepSeek Sparse Attention (DSA)** — a lightning indexer scores the whole history and
   keeps the top **`index_topk=2048`** positions for the attention core.
3. **IndexShare** — the mechanism this fork exists to price. Only **21 of 78 layers**
   compute the index; the other **57 reuse** a neighbour's selection and physically carry
   **no indexer weights**.

## 2. Layer stack / block structure

| | |
|---|---|
| Layers | **78** transformer + **1** MTP draft head |
| Hidden | 6144, vocab 154,880 (untied embed + lm_head) |
| MLP schedule | layers **0–2 dense** FFN (`intermediate=12288`); **3–77 MoE** (`first_k_dense_replace=3`) |
| MoE | 256 routed experts, **top-8**, **1 shared**, `moe_intermediate=2048`, sigmoid + `noaux_tc` routing, `routed_scaling=2.5` |
| Attention | MLA: `q_lora=2048`, `kv_lora=512`, per-head `qk=256` (nope 192 + rope 64), **`v_head=256`** |
| Indexer | 32 heads × 128 dim, top-2048, **IndexShare period 4** |
| MTP | 1 draft layer, `index_share_for_mtp_iteration=true` (reuses the main index) |
| Precision | **bf16 throughout** — no `quantization_config` in the release |

Each block = `attn_q_a → attn_q_b → attn_kv_a → [attn_index_proj → attn_index_score on full
layers only] → attn_score_value → attn_qnorm_rope_insert → attn_out_proj → {dense FFN | moe_router → moe_shared → moe_routed}`.

## 3. Attention pattern & IndexShare — the headline

`indexer_types` is read verbatim from the checkpoint: layers **0,1,2 `full`**, then a strict
**period-4** cycle — one `full` layer that recomputes the top-2048 selection, then three
`shared` layers that reuse it (`index_topk_freq=4`, `index_skip_topk_offset=3`).

**This is proven from the weight map, not inferred:** indexer tensors (`*.indexer.*`) exist on
exactly the 21 `full` layers and on **none** of the 57 `shared` layers. A shared layer that
recomputed the index would need those weights; it doesn't have them. So the fork emits indexer
nodes on 21 layers only. Pricing all 78 at full rate — the naive reading of `index_topk` —
would overstate the indexer's share ~4× and mis-rank it against the MoE term.

**Consequence in the graph:** attention is *flat in context*. The core reads at most
`index_topk=2048` selected positions however long the context grows; only the indexer *scan*
grows, and IndexShare caps that to 21 layers. Measured in the graph, 4K → 128K context:

| op | 4K ctx | 128K ctx | behaviour |
|---|---|---|---|
| `attn_score_value` (core) | 0.039 ms | 0.039 ms | **flat** — bounded by top-k |
| `attn_index_score` (scan) | 0.005 ms | 0.147 ms | grows with context, but only 21 layers, still <1% |

## 4. MoE / routing structure

75 sparse layers + the MTP head each run: a replicated **`moe_router`** (h→256), one always-on
**`moe_shared`** expert, and **`moe_routed`** — top-8 of 256. The routed term is the whole
decode story: **FLOPs scale with `positions × 8`**, but **weight traffic scales with the number
of _distinct_ experts the batch woke**, which saturates at 256 (`distinct_experts`). At batch 1
a step fetches ~8 experts/layer; by batch ~256 it fetches essentially all of them for the same
per-token FLOPs — which is why decode is memory-bound at low batch and only turns compute-bound
at high batch.

## 5. Expected kernel classes & roofline (the op-share table)

**H200 (989 TFLOP/s bf16, 4.80 TB/s), batch 1, kv 4096, whole-model floor** — ridge 206 FLOP/byte, so **every node is memory-bound**:

| op | share | bound | what it is |
|---|---:|---|---|
| `moe_routed` | **56.1%** | memory | routed-expert weight fetch (the union term) |
| `attn_out_proj` | **19.4%** | memory | output GEMM — wide because `v_head=256` → 16,384-wide input |
| `moe_shared` | 7.0% | memory | always-on shared expert |
| `attn_q_b` | 6.5% | memory | query up-projection (64×256 heads) |
| `attn_kv_b` | 2.8% | memory | KV up-projection from latent (unabsorbed MLA — see §7.3) |
| `attn_q_a` | 2.4% | memory | query down-projection (replicated) |
| `lm_head` | 2.3% | memory | vocab projection |
| everything else | <2% each | memory | kv_a, dense FFN, **indexer (0.4%)**, router, core, norms |

Floor: **17.0 ms/step** at batch 1 on this shape. **This is a per-GPU rate, not an achievable
config — the model is 1.5 TB and does not fit one H200 (see §7).** Kernel classes the graph
expects: **memory-bound GEMV/GEMM** (all projections + experts), **grouped-GEMM** (routed
experts), **paged-attention core** over selected KV, **indexer score/top-k** (21 layers), **fused
qnorm+RoPE+cache-insert**, and — under sharding — **all-reduce** (TP) / **all-to-all** (EP).

**Honest end-to-end throughput** (a shape that fits — 16×H200, TP16/EP16, batch 32, kv 8192):
**15.5 ms/step, ~2,060 tok/s**, collectives priced, still fully memory-bound.

**Batch crossover** (`--sweep`, TP=1 per-GPU reference, kv 4096): 59 tok/s @ b1 → 782 tok/s @
b256, where **341/830 nodes turn compute-bound** as the expert union saturates. (Per-GPU rates —
multiply by ranks, minus collective overhead, for a real deployment.)

## 6. Algebraic roofline structure & assumptions

Per op: `t = max(flops/peak_dtype, bytes/HBM_bw)`, peak resolved **per dtype** (bf16 here).
Load-bearing terms and the assumptions behind them:

- **Expert traffic = a set-union, not a multiply:** `distinct(B) = 256·(1−(1−8/256)^B)`. Assumes
  **uniform routing** — real skew touches *fewer* experts, so this over-predicts traffic (the
  safe direction).
- **KV traffic uses the shared latent** `kv_lora_rank + qk_rope = 576` elems/token/layer — **not**
  `num_key_value_heads × head_dim`. The config's `num_key_value_heads=64` is a red herring; a GQA
  reading inflates decode KV ~57×.
- **Attention core is not divided by TP** — a single shared latent can't be split, so the cache is
  replicated and every rank reads all of it. **TP buys no KV bandwidth on this architecture.**
- **`q_a` / `kv_a` are replicated** across TP ranks (they make the shared latent), so TP's speedup
  on attention is strictly less than `tp`.
- **Collectives are bandwidth-only** (no latency floor) → optimistic at decode message sizes,
  flagged `estimated`. **EP imbalance stays at 1.0** (perfect balance) — it is trace-calibrated by
  design and, with the no-traces constraint, is declared rather than fitted.
- **Decode only.** Prefill (chunked, different indexer asymptotics) is unmodelled, as in every
  family.

## 7. Where the headroom likely is — and what to validate first

Ranked by expected payoff, all **hypotheses from the graph** to be confirmed against a capture:

1. **The model doesn't fit one box — the deployment shape is the first-order lever.** 1.507 TB
   bf16 needs **≥11 H200 / ≥8 B200** for weights alone (realistically **16×H200 = 2 nodes** with
   KV+activation headroom). **FP8 halves it to ~0.75 TB → fits 8×H200.** Validate first: does an
   FP8 expert cast hold GLM-5.2's quality? Everything else is downstream of this.
2. **`moe_routed` (58%) is the memory-bound heart.** Levers: **EP with a good expert-placement /
   EPLB** to cut per-rank distinct-expert traffic and, at serving batch, the **grouped-GEMM
   backend** (DeepGEMM). Validate: measured per-rank expert traffic vs the `distinct_experts`
   prediction, and the **real EP imbalance** (the one number we must calibrate from a trace).
3. **`attn_out_proj` (19%) + `attn_kv_b` (3%) hinge on one question: does the engine run
   _absorbed_ MLA?** We model it **unabsorbed** — `kv_b` runs as its own GEMM and `o_proj` is
   narrow (16384→6144). If vLLM absorbs MLA (the usual decode path), `kv_b` disappears and
   `o_proj` **doubles** to 32768→6144. Same resident weights, but the #2 line moves by 2× in
   either direction. Validate first: which path the serving engine takes — it re-ranks the whole
   attention side.
4. **IndexShare is already a big win the graph shows is _cheap_ (0.4%).** The validate-first
   question is the opposite of a lever: **confirm the runtime actually skips the scan on `shared`
   layers** (and reuses across the group), because if it doesn't, a hidden 4× indexer cost is
   sitting off our books. Also confirm `index_topk_freq` is purely spatial (layer-group) vs also
   temporal (step-to-step) — extra temporal reuse only *reduces* cost.
5. **MTP acceptance** — we emit the draft as a shared (no-indexer) block per
   `index_share_for_mtp_iteration`. Validate the accepted-token rate to see whether MTP pays off,
   and that the draft indexer is genuinely shared.

**First capture to take:** a single decode step at a shape that fits (16×H200, TP16/EP16, batch
16–32, kv 8K — or 8×B200, TP8/EP8), attributed per-op, and diff it against
`gitm plan glm-5.2 --gpu H200 --batch 32 --kv-len 8192 --tp 16 --ep 16 --json`. The three numbers
that matter most: **EP imbalance**, **`moe_routed` per-rank traffic**, and whether **`shared`
layers emit any indexer kernel at all**.

## 8. How to run it on RunPod

The model is too large for a single GPU, so this is a **multi-GPU / multi-node** job. Two paths:

**A. Predict-only (no GPU needed — do this first, it's free and already works):**
```bash
pip install -e .              # this repo
gitm plan glm-5.2 --gpu H200 --batch 32 --kv-len 8192 --tp 16 --ep 16   # 16×H200, fits
gitm plan glm-5.2 --gpu B200 --batch 32 --kv-len 8192 --tp 8  --ep 8 --json  # 8×B200, fits
```

**B. Serve + capture on RunPod (to get the numbers §7 wants validated):**
1. Pick hardware by the footprint math above: **bf16 → 2× `8×H200` (or 8×B200)**; **fp8 →
   1× `8×H200`**. On RunPod choose an **8×H200 SXM** pod (or two, networked) with a **network
   volume ≥ 2 TB** for the 282-shard checkpoint.
2. Serve with a DSA-aware engine (vLLM/SGLang build with `glm_moe_dsa` + MLA + DSA support),
   e.g. `--tensor-parallel-size 8 --enable-expert-parallel`, `--kv-cache-dtype fp8` (serving
   choice, not a model fact), `--max-model-len` to taste.
3. Attach the GitM collector to the live server and capture a bounded decode window:
   ```bash
   gitm capture serve   # or: gitm capture attach   (see PR #79)
   ```
4. Import the trace and diff observed-vs-predicted per op; residuals outside the efficiency band
   are the leads §7 lists — **a residual here is a lead, not a defect.**

> ⚠️ The engine flags in step 2 are the shape to aim for, not verified command lines — confirm
> the serving engine actually supports `glm_moe_dsa` before booking a multi-node pod. Start with
> path A (free) and the FP8 fit question, which gates everything else.

---

### Artifacts in this directory
- `artifacts/glm-5.2_H200_b1_kv4096_tp1ep1.json` — canonical decode-step graph (830 nodes, per-GPU rate)
- `artifacts/glm-5.2_H200_b1_kv131072_tp1ep1.json` — long-context (indexer scan grows)
- `artifacts/glm-5.2_H200_b32_kv8192_tp16ep16.json` — a shape that fits: 16×H200, collectives priced
- Graph code: `gitm/planner/glm_graph.py` · catalogue: `gitm/planner/models/glm-5.2.yaml` · tests: `tests/test_glm_graph.py`
