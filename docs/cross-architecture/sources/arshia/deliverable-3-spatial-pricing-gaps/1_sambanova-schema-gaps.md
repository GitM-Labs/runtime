# SambaNova: spatial pricing gaps in the GitM planner

**Deliverable 3 of the SambaNova lane.** These are entries for Ishaan's schema-gap
register. Each marks a place where `roofline.py` and the graph classes cannot express
what the DeepSeek-V3.2 ceiling (deliverable 2) needed. Every entry names its
platform evidence, the affected planner path, the missing representation, and the
expected modeling error. Where the error cannot be bounded, the entry gives a
symbolic explanation instead.

The planner was read at **`GitM-Labs/runtime` @ `21348b5`** (public `main`,
2026-09-22). Every `file:line` below is at that commit. Evidence IDs cite
`sn40l-source-base.md` v2.1; table IDs (T4, T9, …) cite `v32-sambanova-ceiling.md`.

```bash
python sambanova_gap_probe.py --runtime /path/to/runtime   # reproduces every measured error below (P1-P7)
```

It needs `v32_sn_ceiling.py` from deliverable 2 beside it. It is CPU-only and runs in under a second.

---

## 0. Summary

**Class** separates the two things the brief asks to keep apart:
- **Schema:** the planner cannot represent the thing even with perfect evidence.
- **Evidence:** the representation would exist, but no public SambaNova source supplies the value.
- **Both:** both of the above.
- **Coverage:** a model-family routing defect, not a spatial gap.

| ID | Gap | Class | Expected error (measured unless stated) | Priority |
|---|---|---|---|---|
| **SN-01** | V3.2 is silently priced as a DeepSeek-V4 checkpoint | Coverage | **17× at 8K, 128× at 128K** (bytes and time); no indexer emitted | **P0** |
| **SN-02** | One memory tier, no capacity: residency and spill are inexpressible | Schema | 0 when resident. When weights spill: **−45 %** (bf16 upcast), **−94 %** (weights in DDR); "does not fit" undetectable | P1 |
| **SN-03** | Placement is a compiled output; expert imbalance is a calibrated scalar; no sequence-sharded cache | Both | EP priced **= TP (1.00×)** where the derived value is **1.85×**. The needed scalar varies **4.0 → 1.9** across B = 1–8 | P1 |
| **SN-04** | Graph node ≠ execution region: a GPU launch floor is charged per node | Both | **+1.21 ms, 30 % of the prediction** (841 nodes × 2 µs); true RDU term not numerically bounded | P1 |
| **SN-05** | A fixed ±40 % residual band, where attainment is plan-conditional | Both | The one measured DeepSeek point reads as **3.6× headroom or "as expected"** depending on an unstated plan | P1 |
| **SN-06** | Collectives are bandwidth + one GPU launch: no link latency, no topology | Both | Not numerically bounded; binds above **23–53 µs** per collective (SN40L), 12–25 µs (SN50) | P2 |
| **SN-07** | Step time is a sum of nodes; no per-resource overlap | Schema | **×1.55** when two tiers are active (spill); symbolic otherwise | P2 |
| **SN-08** | The observation plane is GPU-kernel-shaped; RDU observables have no home | Both | No SambaNova residual can be computed at all | P2 |
| **SN-09** | The hardware catalogue is keyed by NVML names; RDUs fall back to A100 | Schema | HLB **21 % too fast** on HBM, compute peak 2× low; warning only | P2 |
| **SN-10** | One cache dtype for the latent and the indexer keys | Schema | **+14.9 % / −4.9 %** per-rank bytes at 128K | P3 |
| **SN-11** | Hardware inputs carry no bound direction or evidence class | Schema | ±2 % (SN50 HBM); presentation risk | P3 |
| **SN-12** | Storage dtype ≠ compute dtype is flagged as a fallback peak | Schema | 0 on time; a false warning on every SN40L plan | P3 |

**Owner routing.**
- **SN-01 and SN-10 are model-side.** They belong on Rahul's catalogue-integration list as well as in this register. SN-01 also affects **GPU** planning of V3.2, the customer's first target model, not just SambaNova.
- **Everything else is spatial.**

---

## 1. What the planner already gets right (not filed)

These were checked at `21348b5` so nobody files them by mistake.

- **The GLM family already has almost every model-side mechanism V3.2 needs.** It has an MLA latent that is **replicated, not divided, under TP**; replicated `q_a`, `kv_a` and indexer projections; per-op dtypes; DSA core and indexer asymptotics; and D-deep MTP chains. Routed through it with a synthesized schedule, V3.2 lands **within 2.4 %** of deliverable 2's hand-built `tp` ledger at 8K (probe P3). The spatial gaps below sit on top of a model side that is essentially right.
- **`Graph.has_unpriced_collectives`.** An unknown interconnect is reported as unpriced, not treated as free. That is the right behaviour for SN40L, whose P2P bandwidth is unknown (U4).
- **`PredictedNode.depends_on` and `Graph.serial_chain_s`** are now shipped (`graph.py:115`, `:137`). The GLM design note still lists them as unshipped (its G10), so **the exemplar note is stale on this point**.
- **`distinct_experts`** and the prefix-chain `tokens_per_step` are the same terms deliverable 2 uses.

---

## 2. Entries

### SN-01 · DeepSeek-V3.2 is routed to the DeepSeek-V4 family and priced 17–128× too slow
**Class:** Coverage (model family). **Priority: P0**, because it is silent.

- **Platform evidence:** the V3.2 config (S25 @ `a7e62ac0`) carries `index_topk` and `n_routed_experts`, like V4. Its semantics are V3.1-Terminus plus DSA (MD-LINEAGE): an MLA latent of 576 elements per token shared by all heads (MD-MQA, MD-KVHEADS), 3 dense layers (MD-LAYERS), no output LoRA, no compression schedule, and an indexer on every layer (MD-INDEXER).
- **Affected planner path:** `registry.detect_family` (`registry.py:12`) → `moe_graph.is_sparse_moe_config` (`moe_graph.py:617`) → `moe_graph.spec_from_hf_config` (`:632`). V3.2 fails the GLM `model_type` test and matches V4 structurally. The reader then:
  - takes `num_key_value_heads = 128` literally (`:655`), so the latent is read as 128 heads;
  - defaults `head_dim` to 512 and `o_lora_rank` to 1024 (`:656`, `:659`), fields V3.2 does not have;
  - with no `compress_ratios` and no `sliding_window`, treats every layer as global attention over the whole cache (`:120`), and **emits no indexer at all**;
  - has no dense-layer field, so `first_k_dense_replace` is dropped;
  - hard-codes an fp8 KV cache (`:682`).
- **Missing representation:** a DeepSeek-V3.2 reader. The GLM family already expresses the architecture. What's missing is a `deepseek_v32` detection plus a synthesized schedule: `indexer_types = ["full"] × 61` and `mlp_layer_types` built from `first_k_dense_replace`. The probe's `glm_adapter()` makes exactly that change, in five lines.
- **Expected modeling error (probe P2),** TP16 on an SN40L-shaped spec, B = 1:
  - at 8K the planner predicts 77.98 GB and 48.74 ms per rank, against 4.51 GB and 2.82 ms in the ledger: **×17.3**;
  - at 128K it predicts 702 GB and 439 ms, against 5.50 GB and 3.44 ms: **×127.6**.
  - Dominant false terms: `attn_score_value` (41.6 GB, the whole cache read as 128 KV heads) and `attn_kv_a` (32.8 GB).
  - Through the GLM adapter, the error falls to **−2.4 %** at 8K.
- **Resolves it:** schema-side, the reader (Rahul). No platform evidence is needed.

### SN-02 · One memory tier with no capacity: residency and spill cannot be expressed
**Class:** Schema. **Priority: P1.**

- **Platform evidence:** three tiers per SN40L socket (S04 Fig. 16.4.1):
  - SRAM: 520 MiB at > 100 TB/s;
  - HBM: 64 GiB at 1.6 TB/s;
  - DDR: 0.5–1.5 TB at 0.1 TB/s.

  They are separate, software-visible spaces (X-SPACES). The compiler places weights and caches in HBM and spills the lowest-bandwidth symbols to DDR first (X-ALLOC, X-PLACE). Deployments keep a model library in DDR and swap it into HBM (D-DDR-RES). SN50 attaches DDR over CXL (P50-DDR-BW).
- **Affected planner path:**
  - `HardwareSpec` (`roofline.py:56`) has one `peak_mem_bw_bytes_per_s` (`:74`) and **no capacity field**.
  - `roofline()` computes one `t_memory` (`:730`).
  - Each family's `model_weight_bytes` (`glm_graph.py:323`, `moe_graph.py:149`) is never compared against anything, so `gitm plan` cannot say "does not fit".
- **Missing representation:**
  - A tier list per chip: capacity, bandwidth, and the bound's direction (see SN-11).
  - A per-tensor-class residency, whether declared, compiled or unknown.
  - Nodes that charge bytes to the tier their tensors live in.
  - A fit check that emits "does not fit" and the spill set.
- **Expected modeling error:**
  - **0 while everything is HBM-resident**, which is the OP-1 case. The one-tier model is correct there.
  - When weights spill, the planner predicts the HBM-only time against a DDR-bound truth (T6):
    - bf16 upcast: 2.64 ms against 4.77 ms, **−45 %**;
    - SN40L-8 or SN50 × 8: 2.73 ms against 4.86 ms;
    - weights streamed from DDR: 1.51 ms against 24.1 ms, **−94 %**.
  - **On capacity it is not wrong but silent.** Under a replicated cache, 128K admits at most 2 sequences per node (T2), and the planner will price B = 4 at 128K without objection.
- **Resolves it:** schema-side. The tier values are documented. What a spill contains at 128K is compiled-plan evidence, and `legalizerInfo` would show it (X-LEGALIZER).

### SN-03 · Placement is a compiled output, but the planner takes it as flags
**Class:** Both. **Priority: P1.**

- **Platform evidence:**
  - The SambaNova compiler chooses the allocation and bandwidth plan with its own static bandwidth model (X-ALLOC, X-BWMODEL, X-STATIC). The customer supplies a bundle, not TP or EP degrees (X-BUNDLE).
  - Customers get no per-chip traffic or utilization metrics (X-METRICS), so there is **no customer trace to calibrate skew from** on this platform.
  - The routing is group-limited `noaux_tc` (MD-MOE). Under it, a whole-expert placement's busiest chip is derivable: 2.01 experts at B = 1 against a perfect spread of 0.5 (T9, DV-EP-LOAD).
- **Affected planner path:**
  - `ShardingConfig` (`roofline.py:472`) is `tp`, `ep`, `dp` plus a scalar `ep_imbalance` (`:502`), documented as "calibrated from a trace, not predicted". The family graphs use it at `glm_graph.py:734`, `moe_graph.py:483` and `hybrid_graph.py:771`.
  - There is no mode for a **sequence-sharded or context-parallel** cache, which is the placement the hardware lower bound needs.
  - There is no way to say "the plan is compiled and unknown".
- **Missing representation:**
  1. **Placement provenance:** declared, compiled-and-known, or compiled-and-unknown. An unknown placement produces a **set of bounded plans** (HLB / `tp` / `ep`), not one number.
  2. **A derived imbalance:** busiest-chip expert load as a function of batch, E / N, top-k, the group-limited routing and the expert layout, with a calibrated override kept for when a trace does exist.
  3. **A cache-sharding mode:** replicated per rank, or sequence-sharded with a merge collective.
- **Expected modeling error (probe P4, T4, T9):**
  - The planner prices EP16 and TP16 at the **same bytes (ratio 1.000)**. The derived ratio is **1.85×** (5.23 against 2.82 ms at B = 1).
  - The scalar that would reproduce it is 4.0 at B = 1, 2.96 at B = 2, 2.33 at B = 4 and 1.86 at B = 8. **A single calibrated value is off by up to 2.1× at other batch sizes.**
  - Without a context-parallel mode the HLB itself is unexpressible. The HLB-to-`tp` gap is **1.86×** at 8K and **2.21×** at 128K.
- **Resolves it:** schema, items 1–3 above. Evidence: SambaNova's expert placement and cache sharding (D2 §12 Q2, Q3). The measured V3.1 point fits both `tp` at η 47 % and `ep` at η 92 % (T13), so it cannot decide between them.

### SN-04 · A graph node is not an execution region: the GPU launch floor is charged per node
**Class:** Both. **Priority: P1.**

- **Platform evidence:**
  - An RDU kernel is a compiled region. Its boundary is a forced synchronisation plus a bitstream reload (X-KERNEL).
  - One kernel can cover a whole decoder layer (M-DEC-FUSED).
  - Kernel looping cut Llama3.1-8B to **4 kernel calls per token**, where the same model's decode issues 323 on DGX H100 (X-CALLS).
  - Without looping, boundary sync exceeded 30 % of the per-token time (X-SYNC).
  - The RDU orchestrates its kernel schedule without the host (X-ORCH).
  - Kernel boundaries are therefore explicit and few, not absent.
- **Affected planner path:**
  - `roofline()`'s launch bound (`roofline.py:708`, `serial_launches`) uses `HardwareSpec.kernel_launch_overhead_s = 2 µs` (`:84`), a CUDA-graph replay constant.
  - The families emit one node per GPU kernel, including norms, quantisation and gating. That was deliberate (GLM note G8), so that the node count *is* the launch count on a GPU.
- **Missing representation:**
  - **An execution region:** a named group of nodes compiled into one kernel.
  - **A region-boundary cost** as its own hardware parameter, separate from GPU launch overhead.
  - **Regions per token** as a plan-dependent quantity, which can be unknown.

  Node-level bytes and FLOPs stay as they are; only the launch or sync term moves to the region level.
- **Expected modeling error (probe P3):**
  - Priced on an SN40L-shaped spec, the GLM-path V3.2 graph adds **1.21 ms of launch floor to a 2.75 ms roofline**, from 841 launch-bound nodes of 1,273. That is **30 % of the predicted 3.96 ms**.
  - The true RDU term is N_regions × t_boundary. Neither value is public for DeepSeek, so it is **not numerically bounded**: Δ = 841 × 2 µs − N_regions × t_boundary.
  - It is over-predicted if SambaNova loops kernels as it does for Llama (≈ 4 regions per token), and under-predicted if not (X-SYNC).
- **Resolves it:** schema, the region concept. Evidence: kernel calls per DeepSeek token and the per-boundary cost (D2 §12 Q6). SambaNova holds both in its per-socket traces (S02 Fig. 13).

### SN-05 · A fixed residual band, where attainment depends on the plan
**Class:** Both. **Priority: P1.**

- **Platform evidence:**
  - SN40L's measured attainment against its own roofline is 76–93 % for dense Llama (M-KL-ROOF, M-HBM-85).
  - The one measured DeepSeek point (D-V31-AA) sits at **28 % of the HLB, 47 % of the `tp` bound and 92 % of the `ep` bound** (T13).
- **Affected planner path:**
  - Residuals are classified against `band_width = 0.4` for both kernel time and memory traffic (`optimizer/invariants.py:22–23`), fixed across hardware and plans.
  - `HardwareSpec.eff_lo` and `eff_hi` (`roofline.py:85`) are documented as the vendor efficiency band, but **nothing reads them**.
- **Missing representation:** the band is a property of the pair (hardware, plan), not a constant. When the plan is unknown (SN-03), a residual needs a band per candidate plan, and a verdict of "consistent with {plans}".
- **Expected modeling error (probe P7):**
  - The same measurement, 5.435 ms per token, reads as **3.6× deviation** against the HLB, **2.1×** against `tp`, and **1.09×, inside the band,** against `ep`.
  - GitM would report up to **3.6× of recoverable headroom that may be pure plan structure**. That is the error `roofline.py`'s docstring says the headroom report exists to avoid.
- **Resolves it:** schema, plan-conditional bands. Evidence: MoE attainment on RDU, which only SambaNova's traces or the customer's measurements can give.

### SN-06 · Collectives: bandwidth plus one GPU launch, with no link latency and no topology
**Class:** Both. **Priority: P2.**

- **Platform evidence:**
  - **SN40L:** 10 P2P ports per socket; bandwidth and latency unpublished (P-P2P-PORTS, U4). Collectives are **dataflow operators inside kernels, pipelined with compute** (X-COLL), not separate kernels.
  - **SN50:** 7 intra-node links and 2 inter-node links per RDU at 100 GB/s per direction (P50-NET). The smallest V3.2-resident SN50 deployment is 16 RDUs across two nodes, so its all-reduces cross the narrower links (D2 §8).
  - **Decode payloads are about 14 KB** (T11), so latency, not bandwidth, is what can bind.
- **Affected planner path:**
  - `glm_graph._emit_collective` (`glm_graph.py:763`) prices a collective by swapping `interconnect_bw_bytes_per_s` into the memory slot (`:797`) and adding `serial_launches = 1` when priced (`:807`), that is, one 2 µs CUDA launch. `moe_graph.py:512` does the same.
  - There is one scalar `interconnect_bw_bytes_per_s` (`roofline.py:78`) per SKU.
- **Missing representation:**
  - a per-collective latency parameter, distinct from kernel launch;
  - a topology split into intra-node and inter-node bandwidth, and N relative to node size;
  - a flag for whether the collective is fused into a region (SN-04) or stands as its own node.
- **Expected modeling error:**
  - **Not numerically bounded** on either generation.
  - Once an SN40L figure exists, the planner would add 123 × 2 µs = **0.25 ms (9 % of the `tp` step)**. That is a CUDA constant on a machine that pipelines collectives inside kernels. The real binding threshold is **23–53 µs** per dependent collective on SN40L (T11) and **12–25 µs** on SN50 (T15c).
  - For SN50 at N > 8, the bandwidth term would be **3.5× too optimistic** (7 links against 2), though it stays small at decode: 18 µs per step for `tp` at 16 RDUs.
- **Resolves it:** schema, the latency and topology fields. Evidence: SN40L link bandwidth and measured all-reduce latency across 16 RDUs; SN50 inter-node latency (D2 §12 Q5, Q8).

### SN-07 · Step time is a sum of nodes; the graph cannot express overlap across different resources
**Class:** Schema. **Priority: P2.**

- **Platform evidence:** AGCU tile loads, PMU traffic and PCU compute are "fully decoupled and overlapped", and compute overlaps communication **by default** (X-OVERLAP). On a dataflow machine the next layer's weights can stream during a dependent all-reduce, because weights do not depend on activations.
- **Affected planner path:**
  - `Graph.total_pred_s` (`graph.py:133`) sums node times.
  - `serial_chain_s` (`:137`) gives a floor for declared edges only.
  - Nodes carry `expected_stream_id` (`:101`), which nothing reads, but **no resource tag**.
- **Missing representation:** each node names the resource or resources it consumes: HBM, DDR, P2P, compute or SRAM. The step floor then becomes max(serial chain, max over resources of the per-resource sum). This is the form deliverable 2 uses under A-OVERLAP.
- **Expected modeling error (probe P6):**
  - With two active tiers (the bf16 spill), the planner's sum is 2.64 + 4.77 = 7.41 ms against a max of 4.77 ms: **×1.55**.
  - In the HBM-only case the sum and the max nearly agree for memory-bound nodes. The difference is exactly the collective and region terms of SN-04 and SN-06, whether serial or hidden.
- **Resolves it:** schema only. X-OVERLAP is documented.

### SN-08 · The observation plane is shaped around GPU kernels; RDU observables have no home
**Class:** Both. **Priority: P2.**

- **Platform evidence** (source base §2.2):
  - **What a SambaNova customer can observe:** router-level request latency, queue and completions (X-METRICS); deploy-time `legalizerInfo` memory utilization and warnings (X-LEGALIZER); and `modelprofiles` feature flags (X-SPEC).
  - **What is vendor-internal:** PMU and switch counters, per-socket HBM traces, and power telemetry (X-OBS-INT).
  - No kernel-level record reaches the customer.
- **Affected planner path:**
  - `tracer/schema.py`: `KernelEvent` (`:26`) has grid and block dimensions and a stream; `Trace.vendor` is nvidia or amd (`:75`); `source` is cupti, rocprof, nsys, torch or none (`:81`).
  - `optimizer/deviation.classify_op` (`:180`) pairs CUDA kernel names, and the invariants are per kernel.
- **Missing representation:**
  - **Step-level and request-level observations,** such as time per output token vs context and batch.
  - **Deploy-artifact observations,** such as memory utilization per chip and feature flags.
  - A residual computed at that granularity, against the step bound of each candidate plan (SN-05).
- **Expected modeling error:** today, **no SambaNova residual can be computed**. The two largest open forks in deliverable 2 are nevertheless decidable from step-level data alone:
  - attention case: the t(30K) / t(1K) ratio is ≤ 1.07 for sparse DSA and ≥ 1.28 otherwise (T5b);
  - cache sharding: `legalizerInfo` reads about 79 % against 62 % per chip at 128K (T2).
- **Resolves it:** schema, the step and deploy observation types. Evidence: kernel-level RDU data exists only inside SambaNova and has to be requested.

### SN-09 · The hardware catalogue is keyed by NVML GPU names; any RDU prices as an A100
**Class:** Schema (catalogue). **Priority: P2.**

- **Platform evidence:**
  - The SN40L numbers are documented: P-HBM-BW, P-BF16-PEAK, P-HBM-CAP.
  - The SN50 numbers are partly `2HAND`: P50-HBM-BW, P50-DDR-BW.
  - RDUs are not NVML devices.
- **Affected planner path:** `context._PEAKS`, `_QUANT_PEAKS` and `_INTERCONNECT` (`context.py:29`, `:64`, `:94`), via `peak_for_sku` (`:162`) and `hardware_spec_for` (`:172`). With no match the peak is `None`, which becomes `HardwareSpec()`, the **A100** defaults (`:187`). `gitm plan --gpu SN40L` only prints a warning (`registry.py:350`).
- **Missing representation:** RDU catalogue rows, keyed by something other than an NVML name. A row is only half the fix: without SN-02 and SN-11 it can carry neither a tier nor the evidence class.
- **Expected modeling error (probe P5):** A100 defaults are 2.039 TB/s and 312 TF/s. That makes SN40L's HBM-bound HLB **21 % too fast** and its compute peak 2.05× too low. Compute does not bind, so the time error is the 21 %.
- **Resolves it:** schema and catalogue. The evidence is in the source base.

### SN-10 · One cache dtype covers both the MLA latent and the indexer keys
**Class:** Schema (model-side). **Priority: P3.**

- **Platform evidence:** the V3.2 reference stores the latent in **bf16** (1,152 B per token per layer) and the indexer keys in **fp8 plus an fp32 scale** (132 B) (S29). SambaNova's own choice is open (U14b).
- **Affected planner path:** `GlmMoeDsaModelSpec.kv_dtype` (`glm_graph.py:156`) prices the latent (`:242`) and the indexer keys (`:598`, `:618`), plus a separate `kv_rope_dtype`.
- **Missing representation:** a dtype per cache class: latent, rope key and index key. Scale bytes are not yet counted either.
- **Expected modeling error (probe P3),** against the S29 layout at 128K under TP16:
  - an all-bf16 cache is **+14.9 %** per-rank bytes;
  - an all-fp8 cache is **−4.9 %**;
  - at 8K the error is −2.4 % or −5.3 %.
- **Resolves it:** schema, with Rahul, because this is a model-side field.

### SN-11 · Hardware inputs carry no bound direction and no evidence class
**Class:** Schema. **Priority: P3.**

- **Platform evidence:**
  - SN40L SRAM bandwidth is published only as a **lower bound**, > 100 TB/s (P-SRAM-BW).
  - SN50 HBM bandwidth, 1.84 TB/s, is **secondhand** (`2HAND`, from S27). Its only corroboration is a trace plateau at about 1.8 TB/s (DV-SN50-HBM).
- **Affected planner path:** `HardwareSpec` fields are bare floats (`roofline.py:56–86`). The model side has `provenance`; the hardware side has none.
- **Missing representation:** per-field `(value, kind ∈ {point, lower, upper}, class, source)`. A lower-bound bandwidth may be used only to show that a tier *cannot* bind, never as a ceiling.
- **Expected modeling error:**
  - ±2 % on SN50 (1.80–1.84 TB/s, T17).
  - Structurally, a > 100 TB/s SRAM entered as the point value 100 TB/s would turn a dominance argument into a false ceiling.
  - The main risk is **presentation**: a catalogue row makes a secondhand number look documented.
- **Resolves it:** schema.

### SN-12 · "Stored in fp8, computed in bf16" is flagged as a fallback peak
**Class:** Schema. **Priority: P3.**

- **Platform evidence:** SN40L documents no fp8 compute; its compute types are BF16, FP32, INT32 and INT8 (P-FP8-PEAK). fp8 storage with bf16 compute is the working assumption (U1, D-BUNDLES).
- **Affected planner path:**
  - `resolve_peak` (`roofline.py:684`) falls back from fp8 to fp16, and `Graph.has_fallback_peaks` (`graph.py:181`) then warns that "the ceiling is low".
  - `_canon_dtype` already special-cases int4 storage with 16-bit compute, but not fp8 on hardware without fp8 compute.
- **Missing representation:** a per-SKU declaration that fp8 storage runs on the bf16 path, so the bf16 peak is the **correct** ceiling rather than a fallback.
- **Expected modeling error:** **0 on time**, since compute is at most 12.5 % of HBM time (D2 §6.6). The cost is a false "fallback peaks" warning on every SN40L plan.
- **Resolves it:** schema, one catalogue attribute.

---

## 3. Compiled regions and observable execution boundaries on SambaNova

The brief asks which regions and boundaries the platform exposes, without assuming
kernel boundaries are absent. **They are not absent.** They are explicit, fewer than on a
GPU, and visible only to the vendor. The customer's boundary set is coarser:
deployment, residency and request.

| Boundary | On SambaNova | Evidence | Observable by | Planner today |
|---|---|---|---|---|
| **Deployment** | One bundle per 16-RDU node; switching is delete + redeploy | X-BUNDLE, D-NODE, S21 | **Customer** (deploy events) | Not represented: one model priced in isolation |
| **Model residency** | The active model is copied from DDR to HBM. A swap measures 650 ms on average; a "2 µs switch" is also claimed (C10) | X-PLACE, D-DDR-RES, DV-SWAP | **Customer** (swap latency) | Not represented (SN-02) |
| **Compiled region (kernel)** | Forced synchronisation plus bitstream reload. A decoder layer can be one kernel, and kernel looping removes per-layer boundaries: 4 calls per token for Llama3.1-8B | X-KERNEL, M-DEC-FUSED, X-CALLS, X-SYNC | **Vendor only** (per-socket traces, S02 Fig. 13) | A GPU launch per node (SN-04) |
| **Orchestration** | The RDU sequences its kernels without the host | X-ORCH | Vendor only | CUDA-graph replay constant (SN-04) |
| **Collective** | A dataflow operator over P2P, pipelined with compute inside a kernel | X-COLL | Vendor only | Its own node on stream 1 with a launch (SN-06) |
| **Inside a region** | AGCU, PMU and PCU decoupled and overlapped | X-OVERLAP | Vendor only (PMU/switch counters, PEU power) | A sum of nodes (SN-07) |
| **Deploy-time plan artifact** | Legalizer memory utilization and warnings; model-profile feature flags | X-LEGALIZER, X-SPEC | **Customer** | No observation type (SN-08) |
| **Request / token** | Router-level latency, queue and completions | X-METRICS | **Customer** | No step-level observation type (SN-08) |

**Consequence for the cross-architecture model.** On a GPU the planner's unit of
prediction and of observation is the same: a kernel. On SambaNova they come apart:
- **The unit of execution is a compiled region.** Its count and boundaries are a plan output, and only the vendor can observe them.
- **The unit a customer can observe is a step or a request.**

A SambaNova residual therefore has to be formed at the step level, against a plan-conditional
bound (SN-05, SN-08). The region is what "spatial execution model v1" (due Oct 19) must
define precisely.

---

## 4. Schema limitation vs missing platform evidence

| Only the schema is missing (evidence exists) | Only evidence is missing (schema exists) | Both |
|---|---|---|
| SN-02 tiers and capacity · SN-07 per-resource overlap · SN-09 catalogue · SN-10 cache dtypes · SN-11 bound direction and class · SN-12 storage vs compute dtype | **U4 P2P bandwidth** (the interconnect field exists and correctly reports "unpriced") · **U8 MTP acceptance** (`--acceptance-rate` exists) · **U9 served batch** (`--batch` exists) | SN-03 placement and imbalance · SN-04 regions and boundary cost · SN-05 plan-conditional band · SN-06 collective latency and topology · SN-08 RDU observations |

SN-01 is neither: it is a **coverage defect**. The representation exists in the GLM family,
but detection routes V3.2 away from it.

---

## 5. Next steps

1. **Get Ishaan's register template** and reshape these entries to it. Content should not change.
2. **Route SN-01 and SN-10 to Rahul now.** SN-01 misprices V3.2 on *every* hardware target, and V3.2 is the customer's first model. The probe's `glm_adapter()` is the minimal reader.
3. **Flag to whoever owns the GLM note** that its G10 ("`depends_on` unshipped") is stale at `21348b5`.
4. **Resolve the evidence halves through Jalon** (D2 §12): Q2/Q3 for SN-03, Q5 for SN-06, Q6 for SN-04, and Q1 plus the T5b test for SN-08.
5. **The Oct 19 spatial execution model** starts from §3: region, floor, observation and overlap, defined against these boundaries.
