# DeepSeek-V3.2 on SambaNova: a conditional decode ceiling

**Deliverable 2 of the SambaNova lane.** What 16 SN40L can deliver for DeepSeek-V3.2
decode, derived from public evidence only, with the SN50 customer projection kept
separate. Every input cites into `sn40l-source-base.md` **v2.1**, which ships with this note
(S29 and three derived rows were added; §10). No SambaNova trace, compiled plan or `legalizerInfo` output has been seen.

```bash
python v32_sn_ceiling.py check     # 66 reconciliations and self-tests; non-zero exit on failure
python v32_sn_ceiling.py report    # every table below, in order (≈15 s: the EP routing simulation)
python v32_sn_ceiling.py point --plan tp --context 131072 --attention dsa_masked   # any single point
```

Python 3.11, standard library only, no network. Every table in this note is pasted
unedited from `report`; §11 checks that they still match.

---

## 0. Headline

1. **HBM bandwidth is the binding resource in every supported scenario.** That holds for every placement, context, batch and chip count where the weights live in HBM. Compute is at most 12.5 % of the HBM time. SRAM is at most 3.2 % on SN40L.
2. **Hardware lower bound on 16 SN40L: 1.51 ms per step, so at most 660 tok/s per user** (B=1, 8K, fp8 weights in HBM). It is essentially flat in context under sparse DSA: 1.55 ms at 128K.
3. **Placement moves the bound more than anything else public evidence leaves open.**
   - **Tensor-parallel plan: 2.82 ms (≤ 354 tok/s).** Every chip reads the replicated latent producers, the replicated indexer and the whole cache.
   - **Expert-parallel plan: 5.23 ms (≤ 191 tok/s).** The step waits for the chip that drew the most of the token's 8 experts, which is 2.0 on average.
4. **The measured V3.1 point sits at 28 % of the hardware bound.** That is 184 tok/s at 10k input (D-V31-AA). It is consistent with the `tp` plan at 47 % of peak, or the `ep` plan at 92 %. **Public evidence cannot say which**, so both are carried as competing cases.
5. **Anchored to that measurement, V3.2 on SN40L-16 at B=1 lands at 154–190 tok/s per user at 8K, whichever attention case runs.** That assumes the same plan and attainment as V3.1. **At 128K the attention implementation (U7) decides it:**
   - **138–157 tok/s** if DSA gathers only the top-2048 latents;
   - **52–89 tok/s** if attention reads the whole cache, either as the reference code's mask or as dense MLA.
6. **A customer can distinguish the two attention cases with their own API key.** Time per output token at a ~30K prompt vs a ~1K prompt on SambaCloud's 32K V3.2 preview grows at most 7 % under sparse DSA, and 28–60 % otherwise (T5b).
7. **DDR binds as soon as weights leave HBM.**
   - Streaming fp8 weights from DDR: 24.1 ms per step.
   - A bf16 upcast: 18.7 % of the experts spill, 4.8 ms per step.
   - An 8-chip deployment on either generation spills too.
8. **SN50 (projection, `2HAND` bandwidth).**
   - **16 RDUs (two nodes) is the smallest HBM-resident deployment.** Its bound is 1.32 ms (≤ 760 tok/s).
   - **The tensor-parallel plan flattens at ~1.21 ms (≤ 825 tok/s) however many RDUs are added.** Its replicated bytes do not shrink with N.
9. **Inter-chip communication is unpriced on SN40L (U4).** At decode it can bind only through **latency**. It would overtake HBM if each of the ~123–181 dependent collectives per token costs more than **23 µs (`tp`) or 29 µs (`ep`)** on the critical path. The bandwidth needed is under 3 GB/s.

---

## 1. Operating point

**OP-1** is the primary case. Every row is either cited or labelled as an assumption.

| Field | OP-1 value | Class | Source |
|---|---|---|---|
| Model / revision | DeepSeek-V3.2, `deepseek-ai/DeepSeek-V3.2` @ `a7e62ac04ecb…` | MODEL, **provisional pin** | S25; Rahul owns the canonical revision |
| Architecture used | 61 layers (3 dense + 58 MoE) + 1 MTP layer; 256 experts, top-8, group-limited 4 of 8; MLA latent 512 + 64; indexer 64 × 128, top-2048 | MODEL | MD-LAYERS, MD-MOE, MD-MLA, MD-INDEXER |
| Weight precision | fp8 e4m3 **as stored**, 128 × 128 fp32 block scales; bf16 embed/head/router/`weights_proj`; fp32 norms | MODEL + DEPLOY | MD-QUANT; T1; bundles are fp8-named (D-BUNDLES) |
| Compute precision | bf16 (fp8 weights dequantised in the PCUs) | **UNKNOWN**, priced at bf16 | P-FP8-PEAK (none documented); U1 |
| Activation precision | bf16 | MODEL | MD-QUANT (`torch_dtype`) |
| Latent cache | bf16, **1,152 B/token/layer** | MODEL (reference code) | S29; SambaNova's RDU choice is unknown (U14b) |
| Indexer-key cache | fp8 + one fp32 scale, **132 B/token/layer** | MODEL (reference code) | S29 |
| Chips | **16 SN40L**, one SN40L-16 node, one bundle | DOC / DERIVED | P-NODE, D-NODE |
| Placement | `ideal` (HLB); `tp` and `ep` (plan-conditional bounds) | ASSUMED, bracketing | §2 |
| Decode batch B = concurrency per node | **1**; B = 4 at 8K as a variant | DEPLOY | D-V32-SEQ: 8K allows batch 1 or 4, longer contexts batch 1 |
| Context | **8K** primary; 32K and 128K | DEPLOY | D-V32-SEQ (up to 128K on SambaStack); D-V32-CLOUD (32K on SambaCloud) |
| Residency | weights and caches in HBM | DOC + DERIVED | X-PLACE; DV-FIT; DV-DDR-DECODE |
| Attention | `dsa_sparse` (V3.2 semantics); `dsa_masked` and `dense_mla` as implementation cases | MODEL / **UNKNOWN which runs** | MD-TOPK, MD-MQA, S29; U7 |
| MTP | off | **UNKNOWN** | U8. §6.5 prices it symbolically |

---

## 2. Three kinds of number, and what each needs

| Kind | Meaning | Needs, to be attained |
|---|---|---|
| **HLB**: hardware lower bound (`ideal` plan) | No compiled plan on this hardware can be faster | (a) Every byte spread evenly over 16 chips, **including** the latent producers and a context-parallel cache with a free top-k merge. (b) 100 % of 1.6 TB/s on every chip at once. (c) HBM, DDR and compute fully overlapped (X-OVERLAP). (d) Zero collective, kernel-boundary and host cost. (e) Routing as in A-ROUTE |
| **PCB**: plan-conditional bound (`tp`, `ep`) | The same peaks for one named placement, with the replication and imbalance that placement forces. A bound for that plan only | Same as HLB (b)–(e). The placement is stated below. **Neither plan is claimed to be SambaNova's** |
| **ANCHOR**: measurement-anchored projection | Measured V3.1 time × (V3.2 bound ÷ V3.1 bound), for the same plan | A-ANCHOR and A-NOSPEC below. **Not a bound** |

**The plans, precisely** (`PLANS` in the script):
- **`tp`** is GPU-convention tensor parallelism over 16 chips.
  - Attention is sharded by head.
  - `q_a`, `kv_a`, the router and the whole indexer are **replicated**, so every chip reads them and the entire latent and indexer cache.
  - Experts are sliced 1/16 per chip.
  - There are 2 all-reduces per layer.
- **`ep`** keeps attention as in `tp`.
  - Each chip holds 16 **whole** experts, laid out striped (expert e on chip e mod 16).
  - Each layer waits for the busiest chip.
  - There is 1 all-reduce plus dispatch and combine per MoE layer.

The striped layout is the better of the two simulated (T9). A contiguous layout aligns chips with routing groups, and DeepSeek's group-limited routing then piles a token's experts onto fewer chips.

**Assumptions in force.** Each is an input the script exposes.

| ID | Assumption | Direction if wrong |
|---|---|---|
| A-ROUTE | Routing is exchangeable: iid scores pushed through `noaux_tc` group-limited top-k. This gives the distinct-expert count and EP load | Real skew touches **fewer** experts at B > 1, so HLB bytes go down. At B = 1 it is exactly 8 either way (T8). Skew **raises** the EP busiest chip unless experts are rebalanced |
| A-OVERLAP | Step time = max of the tier terms, not the sum (X-OVERLAP) | Serialisation makes every bound larger |
| A-WSTREAM | Each weight byte crosses HBM once per pass, whatever the row count | — |
| A-ACT | Activations stay in SRAM; HBM carries weights and caches only | Adds bytes; B × 7168 × 2 B per layer is noise at decode |
| A-SRAM | SRAM traffic ≈ 2 × HBM traffic (in to PMU, out to PCU). Used only to show SRAM cannot bind | A 31× reuse factor would be needed to flip it |
| A-SPILL | A capacity deficit spills routed experts first, spread uniformly (X-ALLOC: smallest bandwidth need first) | A skew-aware spill of cold experts lowers DDR traffic |
| A-RESV | HBM reserved for program and buffers = 0 (**UNKNOWN**) | Concurrency limits in T2 are **upper** bounds |
| A-MTP | D > 1 reuses the single MTP layer per stage; a sequence's 1 + D verify rows share one cache read | Union of top-k selections > 2048 adds cache bytes |
| A-ANCHOR | SambaCloud serves V3.1 from the documented bundles. A 10k-token request needs a > 8K bundle, and the only one is `deepseek-4in1-fp8-128k` at **batch 1** (D-BUNDLES), so the anchor is B = 1. V3.2 would run the same plan at the same attainment | A different hosted configuration breaks the ratio |
| A-NOSPEC | The anchors are read as one token per step (U8) | With speculation, the implied attainment is lower. The ratio projection survives only if V3.2's acceptance matches V3.1's |

---

## 3. Model inputs

### 3.1 The checkpoint, rebuilt from `config.json` and reconciled

The script rebuilds every stored tensor from config fields. It reproduces the
checkpoint's own per-dtype parameter counts **exactly**, which checks the ledger's
byte basis three independent ways.

## T1. Parameter inventory rebuilt from config, reconciled

| stored dtype | rebuilt from config (params) | MD-PARAMS / MD-BYTES | difference |
|---|---|---|---|
| F8 | 681,408,069,632 | 681,408,069,632 | +0 |
| BF16 | 3,946,184,704 | 3,946,184,704 | +0 |
| F32 | 42,667,040 | 42,667,040 | +0 |
| **total** | **685,396,921,376** | **685,396,921,376** | **+0** |
| bytes | 689,471,107,200 | 689,471,107,200 | +0 |

Main model 674.03 GB + MTP block 15.44 GB = 689.47 GB. One routed expert = 44,050,944 B (44.05 MB, fp8 + fp32 block scales).

What the exact match pins (`DERIVED`, new row **DV-INV**):
- **The MTP layer is a full decoder layer.** It carries its own indexer, all 256 experts, and its own copies of the embedding and head. Together these make up the 15.44 GB.
- **Every norm is stored in fp32**, and so is the indexer's `k_norm`. The reference code declares both classes fp32 (S29), and only that assignment balances the F32 total.
- **The indexer's `weights_proj` is stored in bf16.** The reference *runs* it in fp32, but an fp32 copy would overshoot the F32 total by 28 M parameters.

### 3.2 Cache per token, and three attention cases

The reference inference code at the pinned commit (**S29**) fixes the model-side
cache layout. This closes U14 on the model side; SambaNova's own layout remains open
as U14b.
- **MLA cache:** `kv_cache` [.., 512] and `pe_cache` [.., 64] in the default dtype, which `generate.py` sets to bf16. That is 1,152 B per token per layer.
- **Indexer cache:** `k_cache` [.., 128] as `float8_e4m3fn`, plus `k_scale_cache` [.., 1] in fp32. That is 132 B per token per layer.

S29 also shows something S26 does not. **The reference decode path scores all cached
latents and applies the top-k as a mask**, reading `kv_cache[:bsz, :end_pos]` in full.
DSA's byte saving is therefore a property of a *gathering* kernel, not of the model.
Three implementation cases follow, and all three are priced:

| Case | Latent entries read / layer | Indexer keys read / layer |
|---|---|---|
| `dsa_sparse` (V3.2 semantics, gathering kernel) | min(S, 2048) | S |
| `dsa_masked` (the reference path, ported faithfully) | S | S |
| `dense_mla` (indexer not implemented; V3.1-style) | S | 0, and no indexer weights read |

---

## 4. Capacity ledger, per tier

| Tier | Capacity (SN40L-16) | What lives there in OP-1 | Verdict |
|---|---|---|---|
| **SRAM** | 520 MiB/chip, 8.1 GiB/node (P-SRAM-CAP) | Activations and streaming buffers. **No weights across steps**: even the replicated per-chip set (2.0 GB) is 3.7× SRAM | Not a capacity constraint on residency. Its role is kernel fusion, which is plan-dependent (§6.6) |
| **HBM** | 64 GiB/chip, 1,099.5 GB/node (P-HBM-CAP) | fp8 weights plus both caches | **Fits: 61.4 % (`ideal`), 65.0 % (`tp`) per chip at 8K** (T2) |
| **DDR** | 512 GB–1.5 TB/chip (C3) | The bundle's model library: four DeepSeek versions of about 689 GB, 172 GB/chip (D-DDR-RES) | Fits under either C3 value; carries no decode traffic in OP-1 |
| **P2P** | — | — | No capacity role |

## T2. HBM capacity per chip, OP-1 (SN40L-16, fp8, B=1, DSA)

| plan | context | weights / chip (GB) | cache / chip / sequence (GB) | HBM used at B=1 | max sequences before spill (≤) |
|---|---|---|---|---|---|
| ideal | 8K | 42.13 | 0.04 | 61.4 % | 663 |
| ideal | 32K | 42.13 | 0.16 | 61.5 % | 165 |
| ideal | 128K | 42.13 | 0.64 | 62.2 % | 41 |
| tp | 8K | 44.02 | 0.64 | 65.0 % | 38 |
| tp | 32K | 44.02 | 2.57 | 67.8 % | 9 |
| tp | 128K | 44.02 | 10.27 | 79.0 % | 2 |

**The cache spill case is plan-dependent, not weight-driven.**
- **Under `tp`, every chip holds a full replica of the latent and indexer cache.** 128K context therefore fits **at most 2 sequences per node**. That is at least consistent with SambaStack offering batch 1 only beyond 8K (D-V32-SEQ).
- **A sharded cache holds about 41.**
- **This is a customer-readable distinction.** At 128K and B=1, `legalizerInfo` memory utilization (X-LEGALIZER) would read about 79 % per chip under a replicated cache and about 62 % under a sharded one, plus whatever A-RESV turns out to be.

---

## 5. Traffic ledger, per step

## T3. Traffic per decode step, OP-1 at 8K (whole node)

| group | bytes read / step (GB, whole node) | share |
|---|---|---|
| expert (routed) | 20.440 | 52.7 % |
| attn | 10.493 | 27.1 % |
| shared | 2.555 | 6.6 % |
| head | 1.853 | 4.8 % |
| dense_mlp | 1.189 | 3.1 % |
| attn_repl | 0.924 | 2.4 % |
| indexer | 0.880 | 2.3 % |
| router | 0.213 | 0.5 % |
| cache read | 0.210 | 0.5 % |
| norm | 0.004 | 0.0 % |
| cache write | 0.000 | 0.0 % |
| embed | 0.000 | 0.0 % |
| **total** | **38.760** | 100 % |

At 128K (T3b in `report`) the only line that moves is the cache read: 0.21 GB becomes 1.20 GB, and the step grows to 39.75 GB.

The routed experts are the largest line. The next largest is not the MoE: the **attention
projections take 29.5 %** (`attn` + `attn_repl`). Their biggest part is `o_proj` at
117 M parameters per layer, where DeepSeek's `v_head` 128 × 128 heads makes it 16384 → 7168.

**Per tier, OP-1 at 8K:**
- **SRAM** carries at least the HBM stream plus activations. Its time is at most 3.2 % of the HBM time (T12).
- **HBM** carries all 38.76 GB.
- **DDR** carries 0.
- **P2P** carries 123 collectives (`tp`) or 181 (`ep`) per step, 1.76 MB or 14.24 MB in total (T11).

---

## 6. Time bounds and the binding resource

### 6.1 By placement plan

## T4. Bounds by placement plan, OP-1 at 8K

| case | kind | bytes / critical chip (GB) | HBM ms | DDR ms | compute | step bound ms | tok/s/user ≤ | binding |
|---|---|---|---|---|---|---|---|---|
| ideal | HLB | 2.42 | 1.514 | 0 | 11.5 µs | 1.514 | 660 | HBM |
| tp | PCB | 4.51 | 2.821 | 0 | 23.5 µs | 2.821 | 354 | HBM |
| ep | PCB | 8.36 | 5.228 | 0 | 23.5 µs | 5.228 | 191 | HBM |

At 128K (T4b): `ideal` 1.553 ms, `tp` 3.439 ms, `ep` 5.846 ms. All three are HBM-bound.

**The binding resource is HBM in all three plans, but different bytes dominate on the
critical chip.** These are the competing cases the evidence cannot separate:
- **`tp`: replication.** The replicated part is 2.23 of 4.51 GB per chip at 8K, and 3.22 of 5.50 GB at 128K. It is made of `q_a` and `kv_a` (0.92 GB), the indexer (0.88 GB), the router (0.21 GB) and the full cache.
- **`ep`: expert imbalance.** The busiest chip reads 58 × 2.01 experts = 5.13 GB, 61 % of its bytes.

Whole-expert placement cannot beat a busiest chip of 1 expert per layer at B = 1,
where slicing reaches 0.5 (T9). So **at B = 1, EP costs at least 2× TP on the expert
term, even with perfect expert rebalancing.**

### 6.2 By context and attention case

## T5. Attention case x context x plan (B=1)

| attention case | context | plan | step bound ms | tok/s/user ≤ | cache share of chip bytes | binding |
|---|---|---|---|---|---|---|
| dsa_sparse | 8K | ideal | 1.514 | 660 | 0.5 % | HBM |
| dsa_sparse | 8K | tp | 2.821 | 354 | 4.7 % | HBM |
| dsa_sparse | 32K | ideal | 1.522 | 657 | 1.0 % | HBM |
| dsa_sparse | 32K | tp | 2.945 | 340 | 8.7 % | HBM |
| dsa_sparse | 128K | ideal | 1.553 | 644 | 3.0 % | HBM |
| dsa_sparse | 128K | tp | 3.439 | 291 | 21.8 % | HBM |
| dsa_masked | 8K | ideal | 1.531 | 653 | 1.6 % | HBM |
| dsa_masked | 8K | tp | 3.091 | 324 | 13.0 % | HBM |
| dsa_masked | 32K | ideal | 1.606 | 623 | 6.2 % | HBM |
| dsa_masked | 32K | tp | 4.294 | 233 | 37.4 % | HBM |
| dsa_masked | 128K | ideal | 1.907 | 524 | 21.0 % | HBM |
| dsa_masked | 128K | tp | 9.106 | 110 | 70.5 % | HBM |
| dense_mla | 8K | ideal | 1.494 | 669 | 1.5 % | HBM |
| dense_mla | 8K | tp | 2.500 | 400 | 14.4 % | HBM |
| dense_mla | 32K | ideal | 1.561 | 640 | 5.8 % | HBM |
| dense_mla | 32K | tp | 3.579 | 279 | 40.2 % | HBM |
| dense_mla | 128K | ideal | 1.831 | 546 | 19.6 % | HBM |
| dense_mla | 128K | tp | 7.897 | 127 | 72.9 % | HBM |

**At the hardware bound, the attention case barely matters: 1.55–1.91 ms at 128K.**
**Under the replicated-cache plan it matters 2.6×** (3.44 vs 9.11 ms), because every chip
reads the whole cache at a single chip's bandwidth. At 8K the three cases span only
2.50–3.09 ms under `tp`.

A second finding: **`dense_mla` is faster than `dsa_sparse` at 8K** (2.50 vs 2.82 ms,
`tp`). Below about 17K, under every plan, the indexer's weights and keys cost more than the
latents DSA avoids reading.

**The observation that identifies the case:**

## T5b. The observation that identifies the attention case (B=1)

| attention case | ideal: t(30K) / t(1K) | tp | ep |
|---|---|---|---|
| dsa_sparse | 1.01× | 1.07× | 1.04× |
| dsa_masked | 1.06× | 1.53× | 1.28× |
| dense_mla | 1.06× | 1.60× | 1.28× |

Under any realistic plan, a time-per-output-token ratio of **≤ 1.07× means sparse DSA is running, and ≥ 1.28× means it is not**. The ratio cancels the unknown attainment, provided attainment is flat in context.

It cannot separate `dsa_masked` from `dense_mla`, but both carry the same long-context cost.

### 6.3 Residency cases

## T6. Residency cases (8K, ideal plan)

| case | kind | bytes / critical chip (GB) | HBM ms | DDR ms | compute | step bound ms | tok/s/user ≤ | binding |
|---|---|---|---|---|---|---|---|---|
| fp8 weights in HBM (OP-1) | HLB | 2.42 | 1.514 | 0 | 11.5 µs | 1.514 | 660 | HBM |
| fp8 weights streamed from DDR | HLB | 2.42 | 0.008 | 24.094 | 11.5 µs | 24.094 | 42 | DDR |
| bf16 upcast, experts spill to DDR | HLB | 4.70 | 2.638 | 4.770 | 11.5 µs | 4.770 | 210 | DDR |
| SN40L-8: 8 chips, experts spill | HLB | 4.85 | 2.725 | 4.855 | 23.0 µs | 4.855 | 206 | DDR |

bf16 upcast: 18.7 % of routed experts spilled; SN40L-8: 19.0 % of routed experts spilled.

- **DDR streaming (42 tok/s) is ruled out by every measured DeepSeek point** (D-R1-16, D-V31-AA), confirming DV-DDR-DECODE.
- **The bf16 upcast's hardware bound (4.77 ms) is already slower than the fp8 `tp` plan's bound (2.82 ms).** Together with the fp8 bundle names, the bf16 case survives only as a what-if.

### 6.4 Batch and routing

From T7 (`report`), at 8K:

| | B = 1 | B = 4 | B = 16 |
|---|---|---|---|
| distinct experts / layer | 8 | 30.5 | 102 |
| HLB tok/s/user, tok/s/node | 660, 660 | 264, 1,056 | 91, 1,453 |
| `tp` tok/s/user, tok/s/node | 354, 354 | 183, 732 | 71, 1,129 |

- **HBM binds at every batch up to 32.**
- **At B = 4 the uniform-routing HLB (3.79 ms) sits 2.5× above the routing-free floor**, where all 4 rows share 8 experts (1.54 ms, T8). A-ROUTE is the only assumption that moves the HLB at B > 1.

### 6.5 MTP, symbolic in the unknown acceptance

V3.2 ships one MTP layer (MD-LAYERS). Whether SambaNova runs it is unknown (U8), so
this table prices the step and leaves the accepted tokens per step, τ, as the variable.

## T10. MTP, symbolic in accepted tokens τ (8K, B=1)

| D | plan | MTP off ms | MTP step ms | step cost | break-even accepted tokens τ* | tok/s/user ≤ at τ = 1.5 / 2 / 2.5 / 3 |
|---|---|---|---|---|---|---|
| 1 | ideal | 1.514 | 2.392 | 1.58× | 1.58 | 627 / 836 |
| 1 | tp | 2.821 | 3.841 | 1.36× | 1.36 | 391 / 521 |
| 1 | ep | 5.228 | 6.936 | 1.33× | 1.33 | 216 / 288 |
| 2 | ideal | 1.514 | 3.245 | 2.14× | 2.14 | 462 / 616 / 770 / 925 |
| 2 | tp | 2.821 | 4.836 | 1.71× | 1.71 | 310 / 414 / 517 / 620 |
| 2 | ep | 5.228 | 8.546 | 1.63× | 1.63 | 176 / 234 / 293 / 351 |
| 3 | ideal | 1.514 | 4.075 | 2.69× | 2.69 | 368 / 491 / 614 / 736 |
| 3 | tp | 2.821 | 5.808 | 2.06× | 2.06 | 258 / 344 / 430 / 517 |
| 3 | ep | 5.228 | 9.981 | 1.91× | 1.91 | 150 / 200 / 250 / 301 |

**A draft stage is expensive here.** Each stage re-reads a full MoE layer, the
1.85 GB bf16 head and a 206 MB bf16 `eh_proj`, and the verify pass at 2 rows wakes
about 16 experts instead of 8.

**MTP pays off more easily under the slower plans** (τ\* 1.33–1.36 at D = 1), because their fixed costs dilute the draft.

The only DeepSeek acceptance figure published for SambaNova is SN50's *modeled* MTP3 with
an accept length of 3.1 (R50-R1-MOD, `VMOD`). By the source base's rules it is context,
not an input.

### 6.6 SRAM, compute, P2P and kernel boundaries

**SRAM cannot bind on SN40L.** P-SRAM-BW is a lower bound (> 100 TB/s), so the SRAM time
derived from it is an *upper* bound, at 3.2 % of the HBM time in every plan (T12).
SRAM would bind only if on-chip reuse moved more than 31× the HBM bytes through PMUs.

**Compute cannot bind.** It is at most **12.5 %** of the HBM time across every case in the
script's grid. The worst case is attention reading the whole cache at 128K (`dsa_masked`,
`ideal`): 128 query heads score 131K latents per layer, 0.24 ms against 1.91 ms. Compute is
priced at the bf16 rate, because fp8 compute is undocumented on SN40L (P-FP8-PEAK).

**P2P is unpriced (U4), and this is what it would take to bind:**

## T11. Inter-chip communication: what P2P would have to be to bind (8K)

| plan | B | dependent collectives / step | payload / step (MB) | PCB step ms | break-even latency per collective (µs) | break-even bandwidth (GB/s) |
|---|---|---|---|---|---|---|
| tp | 1 | 123 | 1.76 | 2.821 | 22.9 | 0.63 |
| tp | 4 | 123 | 7.05 | 5.463 | 44.4 | 1.29 |
| ep | 1 | 181 | 14.24 | 5.228 | 28.9 | 2.72 |
| ep | 4 | 181 | 56.94 | 9.512 | 52.6 | 5.99 |

- **Payloads are 14 KB per all-reduce at B = 1**, so P2P bandwidth is irrelevant.
- **Latency is what matters.** Under full overlap (A-OVERLAP), collectives bind once n × L exceeds the HBM time. Without overlap they add n × L to every step regardless.
- A dataflow machine can stream the next layer's weights during a dependent all-reduce, because weights do not depend on activations. So overlap is plausible; it is not documented for this case.

**Kernel boundaries are not numerically bounded.**
- A boundary is a forced synchronisation (X-KERNEL).
- Without kernel looping, boundaries took more than 30 % of the per-token time on Llama3.1-8B, and looping cut that about 8× (X-SYNC).
- The DeepSeek kernel count per token is unknown. The term is n_k × t_b and shares the same per-token latency budget as the collectives above.

### 6.7 Where the binding resource changes

| Scenario | Binding | Evidence status |
|---|---|---|
| fp8 weights in HBM, any plan, B 1–32, 1K–128K, SN40L-16 or SN50 ≥ 16 | **HBM** | Supported (DV-FIT, X-PLACE, D-BUNDLES) |
| ↳ critical-chip bytes dominated by | replication (`tp`) · expert imbalance (`ep`) · weights (`ideal`) | **Competing; not identifiable from public evidence** (§7) |
| Weights streamed from DDR | **DDR** (24.1 ms) | Ruled out by D-R1-16 and D-V31-AA |
| bf16 upcast, 18.7 % of experts spilled | **DDR** (4.8 ms) | Contradicted by the fp8 bundle names |
| 8 chips of either generation | **DDR** (4.9 ms for SN40L-8) | Not a SambaStack configuration (P-NODE) |
| Per-collective latency above ~23–53 µs on the critical path | **P2P** | **Not numerically bounded** (U4) |
| Kernel-boundary cost × count above the HBM time | **sync** | **Not numerically bounded** (X-KERNEL; count unknown) |
| Anything | compute, SRAM | Cannot bind (≤ 12.5 %, ≤ 3.2 %) |

---

## 7. Calibration, and the anchored projection

## T13. Calibration against measured DeepSeek decode on SN40L

| anchor | context | plan | V3.1 bound ms | measured tok/s | measured ms/token | implied attainment (bound ÷ measured) | verdict |
|---|---|---|---|---|---|---|---|
| D-V31-AA | 10,000 | ideal | 1.499 | 184 | 5.435 | 28 % | consistent |
| D-V31-AA | 10,000 | tp | 2.579 | 184 | 5.435 | 47 % | consistent |
| D-V31-AA | 10,000 | ep | 4.986 | 184 | 5.435 | 92 % | consistent |
| D-R1-16 | 1,024 | ideal | 1.474 | 198 | 5.051 | 29 % | consistent |
| D-R1-16 | 1,024 | tp | 2.185 | 198 | 5.051 | 43 % | consistent |
| D-R1-16 | 1,024 | ep | 4.592 | 198 | 5.051 | 91 % | consistent |
| D-R1-16 | 8,192 | ideal | 1.494 | 198 | 5.051 | 30 % | consistent |
| D-R1-16 | 8,192 | tp | 2.500 | 198 | 5.051 | 49 % | consistent |
| D-R1-16 | 8,192 | ep | 4.907 | 198 | 5.051 | 97 % | needs more than the best measured dense attainment (93 %) |

**Falsification test: passed.** No hardware bound exceeds a measured time; had one, the framework would be wrong.

The V3.1 ledger is V3.2's without the indexer. MD-LINEAGE makes that exact, and S12 extends it to R1.

**What the anchor says, and does not:**
- **SambaNova's served DeepSeek runs at 28–30 % of the hardware bound.** Two explanations fit equally well:
  - a `tp`-like plan at 43–49 % of peak;
  - an `ep`-like plan at 91–97 % of peak.
- **The dense-model range is 76–93 %** (M-KL-ROOF). The `ep` reading sits at its top, and it assumes no speculation. The `tp` reading needs about half the dense attainment.
- **Neither can be preferred on public evidence.** The discriminating observations are in §12 (Q2, Q3).

## T14. Anchored projection for V3.2 on SN40L-16 (B=1), from D-V31-AA

| attention case | context | anchored tok/s/user, tp plan (η = 47 %) | anchored tok/s/user, ep plan (η = 92 %) |
|---|---|---|---|
| dsa_sparse | 8K | 168 | 175 |
| dsa_sparse | 32K | 161 | 171 |
| dsa_sparse | 128K | 138 | 157 |
| dsa_masked | 8K | 154 | 167 |
| dsa_masked | 32K | 111 | 137 |
| dsa_masked | 128K | 52 | 80 |
| dense_mla | 8K | 190 | 187 |
| dense_mla | 32K | 133 | 153 |
| dense_mla | 128K | 60 | 89 |

This is the table closest to "what will the customer see". **It is a projection, not a
bound.** It inherits the anchor's caveats:
- Batch and speculation are unstated (U8, U9); the B = 1 reading follows from A-ANCHOR.
- The AA page drifts, and its dated screenshot is not saved yet (S15).
- V3.2 has no measured SambaNova speed at all: AA lists it with no metrics (D-V32-AA).

---

## 8. SN50: the customer projection, kept separate

The inputs change and nothing else does: HBM is 1.84 TB/s per RDU (`2HAND`), DDR is
100 GB/s over CXL (`2HAND`), fp8 compute is 3,200 TF (DOC), SRAM is 432 MB, and there
are 8 RDUs per node. **No SN50 DeepSeek measurement exists.** The R1 operating points
(R50-R1-MOD) are the vendor's model, so these rows carry **no anchor**.

## T15. SN50 projection by RDU count (8K, B=1, DSA)

| RDUs | plan | kind | weights / chip (GB) | experts spilled | HBM ms | DDR ms | step bound ms | tok/s/user ≤ | binding |
|---|---|---|---|---|---|---|---|---|---|
| 8 | ideal | HLB | 84.25 | 19 % | 2.369 | 4.855 | 4.855 | 206 | DDR |
| 8 | tp | PCB | 86.02 | 21 % | 3.400 | 5.407 | 5.407 | 185 | DDR |
| 8 | ep | PCB | 86.02 | 21 % | 5.081 | 13.712 | 13.712 | 73 | DDR |
| 16 | ideal | HLB | 42.13 | 0 % | 1.317 | 0 | 1.317 | 760 | HBM |
| 16 | tp | PCB | 44.02 | 0 % | 2.453 | 0 | 2.453 | 408 | HBM |
| 16 | ep | PCB | 44.02 | 0 % | 4.546 | 0 | 4.546 | 220 | HBM |
| 32 | ideal | HLB | 21.06 | 0 % | 0.658 | 0 | 0.658 | 1519 | HBM |
| 32 | tp | PCB | 23.02 | 0 % | 1.833 | 0 | 1.833 | 546 | HBM |
| 32 | ep | PCB | 23.02 | 0 % | 3.659 | 0 | 3.659 | 273 | HBM |
| 64 | ideal | HLB | 10.53 | 0 % | 0.329 | 0 | 0.329 | 3038 | HBM |
| 64 | tp | PCB | 12.52 | 0 % | 1.522 | 0 | 1.522 | 657 | HBM |
| 64 | ep | PCB | 12.52 | 0 % | 3.129 | 0 | 3.129 | 320 | HBM |
| 128 | ideal | HLB | 5.27 | 0 % | 0.165 | 0 | 0.165 | 6076 | HBM |
| 128 | tp | PCB | 7.27 | 0 % | 1.367 | 0 | 1.367 | 731 | HBM |
| 128 | ep | PCB | 7.27 | 0 % | 2.818 | 0 | 2.818 | 355 | HBM |

**One SN50 node (8 RDUs) cannot hold V3.2 in HBM.** It has 549.8 GB against 674 GB of
main-model weights, so about a fifth of the experts would stream from CXL DDR, and DDR
binds.

The smallest HBM-resident deployment is **16 RDUs across two nodes**. The `tp` plan's
all-reduces then cross the 2 × 200 GB/s inter-node links (P50-NET), which is priced below.

**Scaling out helps the hardware bound linearly and the `tp` plan hardly at all:**

| chips | `tp` step at 8K | `tp` step at 128K |
|---|---|---|
| 16 | 2.45 ms | 2.99 ms |
| 256 | 1.29 ms | 1.83 ms |
| floor as N → ∞ | **1.21 ms (≤ 825 tok/s)** | **1.75 ms (≤ 571 tok/s)** |

The replicated set does not shrink with N: latent producers, indexer, router and the
full cache, 2.23 GB per chip at 8K (T15b). That is the quantitative case for asking
SambaNova how the DeepSeek SN50 plan shards the indexer and the latent cache.

On the P2P side, SN50's link bandwidth is published, so its bandwidth term is priced:
18 µs (`tp`) or 142 µs (`ep`) per step at 16 RDUs (T15c). That is not binding. Latency
remains unpublished; the break-even is 12–25 µs per dependent collective.

**Each missing SN50 parameter: what it moves, and what resolves it.**

| Parameter | Treatment | Value / sensitivity | Resolving observation |
|---|---|---|---|
| HBM peak bandwidth | **Sourced range** | 1.80 (trace plateau, VMEAS, a floor on peak) – 1.84 TB/s (2HAND). HLB 1.346–1.317 ms (T17). Symbolic: every HBM-bound row ∝ 1/BW | S10 deck slide, or SambaNova confirmation |
| DDR bandwidth (CXL) | **Single 2HAND value, symbolic** | 100 GB/s. Matters only when weights spill (8 RDUs): DDR ms ∝ 1/BW_DDR | Same |
| SRAM bandwidth | **Not numerically bounded** | Binds only if SRAM bandwidth < ~2× HBM traffic ÷ step time (A-SRAM), about 3.7 TB/s per RDU; SN40L's PMUs exceed 100 TB/s | SambaNova figure for SN50 PMU bandwidth |
| P2P latency per collective | **Not numerically bounded** | Binds above 12–25 µs per collective if unoverlapped (T15c) | Measured all-reduce latency across 16 RDUs / 2 nodes |
| fp8 compute used in decode | **Immaterial** | Compute ≤ 12.5 % of HBM even at the bf16 rate | — |
| RDU count for the customer | **Customer input** | T15 row | Jalon: customer's planned rack count |
| Placement plan | **Competing cases** | `ideal` 1.32 / `tp` 2.45 / `ep` 4.55 ms at 16 RDUs | As for SN40L (§12 Q2) |
| Attention implementation (U7) | **Competing cases** | T16: 128K `tp` 2.99 (sparse) vs 6.87 ms (dense) | T5b test on SN50 when available |
| MTP acceptance τ | **Symbolic** | T10's form carries over | Engine acceptance metric |

---

## 9. Unknowns still visible in the outputs

| ID | Unknown | What it moves (largest effect) | In the script as |
|---|---|---|---|
| **Plan** | tp-like vs ep-like placement (new; feeds deliverable 3) | 1.9–3.5× over HLB at 8K; the anchor fits both | `--plan` |
| **U7** | Sparse DSA, masked reference path, or dense MLA | At 128K: 138–157 vs 52–89 tok/s anchored | `--attention` |
| **U4** | P2P latency (and bandwidth) on SN40L | Binds above 23–53 µs per collective | unpriced; break-even printed |
| **U8** | MTP / speculation, and acceptance τ | Break-even τ\* 1.33–2.69 (T10) | `--mtp`, symbolic τ |
| **U9** | Served concurrency | B = 4 roughly halves per-user speed and doubles node throughput (T7) | `--batch` |
| **U14b** | SambaNova's cache dtypes (model-side fixed by S29) | Cache share ≤ 22 % (sparse) and ≤ 73 % (dense, `tp`, 128K) scales with bytes/token | `--latent-bytes`, `--index-bytes` |
| **U1** | fp8 storage vs bf16 upcast; fp8 compute | bf16 upcast → DDR-bound 4.8 ms | `--weights` |
| **A-RESV** | HBM reserved for program/buffers | Concurrency limits in T2 are upper bounds | `--hbm-reserved-gb` |
| — | Kernel boundaries per token × cost | Not numerically bounded | not modelled; same budget as U4 |
| — | Reference `kv_b` dequantised to bf16 (S29) | +2.6 % HLB (1.554 vs 1.514 ms) | `--kv-b-bf16` |

---

## 10. Source-base additions (v2 → v2.1)

| ID | Addition | Class |
|---|---|---|
| **S29** | DeepSeek-V3.2 reference inference code, `inference/model.py` and `inference/generate.py` at commit `a7e62ac0…`. Fetched 2026-09-28 as **verbatim excerpts** through a fetch tool: register_buffer lines, Indexer `__init__`/`forward`, Linear, LayerNorm/RMSNorm, the MLA decode branch, and `set_default_dtype`. Establishes: latent cache bf16; indexer keys fp8 e4m3 with an fp32 scale per 128; norms fp32; `weights_proj` computed in fp32; the decode path scores the whole cache and masks with top-k; `kv_b` is dequantised once to bf16. Leaves open: SambaNova's port | MODEL |
| U14 | **Resolved model-side** by S29. **U14b**, SambaNova's cache dtypes on RDU, stays open | — |
| DV-INV | The parameter inventory rebuilt from config reproduces MD-PARAMS for all three dtypes and MD-BYTES exactly (T1). Implies a full MTP layer with its own indexer, experts, embed and head; fp32 norms; bf16-stored `weights_proj` | DERIVED |
| DV-ANCHOR-B1 | D-V31-AA's 10k input exceeds every 8K bundle; the only larger DeepSeek bundle is batch 1, so the anchor is B = 1 (A-ANCHOR) | DERIVED + ASSUMED |
| DV-EP-LOAD | Under `noaux_tc` with iid scores, 16-way whole-expert placement: busiest chip 2.01 (striped) / 2.30 (contiguous) experts at B = 1. The simulated distinct-expert count matches the closed form (T9; seeded) | DERIVED |

The updated `sn40l-source-base.md` ships alongside this note.

---

## 11. How this was checked

- **`check` runs 66 assertions** (all pass):
  - exact reconciliation of all three dtype totals and the byte total;
  - DV-FIT, DV-SWAP and DV-DDR-DECODE recomputed;
  - hand arithmetic for the expert bytes and the OP-1 routed and cache lines, written independently of the ledger code;
  - simulated vs closed-form distinct experts (the routing simulator's correctness);
  - **HLB ≤ every measured time** (falsification);
  - UNKNOWN inputs are never defaulted;
  - every input carries a source ID.
- **Plan byte sums hand-checked for OP-1 at 8K.**
  - `tp`: 16.09/16 + 20.44/16 + 2.02 + 0.21 = 4.51 GB ✓.
  - `ep`: 1.006 + 58 × 2.01 × 44.05 MB + 2.02 + 0.21 = 8.36 GB ✓.
- **Every table in this note was pasted from `report` and diff-checked** against a fresh run before delivery.
- **Not done:** no person has checked any row against the primary PDFs (source base §7). The rows this note leans on most are P-HBM-BW, P-HBM-CAP, D-BUNDLES, D-V31-AA and S29.

---

## 13. Next steps

- **Get Rahul's canonical pin**, and rerun `check`. The inventory reconciliation fails loudly if the pin changes the checkpoint.
- **Seed for deliverable 3.** `roofline.py` carries one `peak_mem_bw_bytes_per_s` and one `interconnect_bw_bytes_per_s`, so it cannot express what this note needed:
  - more than one memory tier with per-op residency and spill;
  - a critical chip whose bytes depend on placement (replicated vs sharded tensors);
  - expert imbalance derived from routing rather than calibrated (`ep_imbalance`);
  - a bandwidth that is known only as a lower bound (SRAM);
  - collectives whose cost is latency with no published figure.

  Each becomes a schema-gap entry. Plan, U7 and U4 are the platform-evidence gaps behind them.
