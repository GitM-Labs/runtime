# SambaNova source base — v2.1

Deliverable 1 of the SambaNova lane. Every factual input and architectural claim in
the DeepSeek V3.2 ceiling note (deliverable 2) cites into this file by ID — `S04 Fig. 16.4.1`
for a source location, `P-HBM-BW` for a parameter row, `C1` for a recorded conflict,
`DV-FIT` for a derived check.


---

## 0. How to read this file

Every claim carries one **evidence class**. The classes do not rank sources; they say
what kind of statement it is, which determines what the ceiling note may do with it.

| class | meaning | may be used as |
|---|---|---|
| `DOC` | vendor-documented hardware capability or documented software behaviour | an input to the hardware bound |
| `MODEL` | read from the model publisher's checkpoint repo (config, index, file sizes) | a model-side input (Rahul owns the canonical pin) |
| `VMEAS` | measurement SambaNova ran and reported; not reproduced by anyone else | a calibration point, labelled vendor-reported |
| `VMOD` | vendor model or projection, stated as not measured | context only; never an input |
| `3PMEAS` | measurement by a party other than SambaNova | a calibration point, with its setup |
| `DEPLOY` | vendor statement about a specific served or shipped configuration | a deployment assumption, labelled |
| `2HAND` | a third party's transcription of a primary we cannot access (e.g. a paywalled deck) | a **provisional** input, always flagged, never sole support for a headline number; replace when the primary is obtained |
| `DERIVED` | our arithmetic on cited numbers; formula shown | an input or a check, carrying its parents' uncertainty |
| `UNKNOWN` | not established by any source found | must stay visible in outputs |

**Generation** is always stated: `SN40L`, `SN50`, or `both`. SN40L and SN50 numbers
are never mixed in one row. **SN40 = SN40L**: SambaNova's current marketing name
(resolved in C9).

**Access level** is stated per source: *full text* (read end to end), *abstract*,
*snippet*, or *not accessed*. Only full-text sources are cited for numbers.
Figures read off a plot are marked *read from figure* and carry reading error.

---

## 1. Sources

| ID | Source | Version / date | Gen | Access | Establishes | Leaves open |
|---|---|---|---|---|---|---|
| **S01** | Prabhakar et al., *SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts*, MICRO 2024. arXiv:2405.07518 **v2**; DOI 10.1109/MICRO61859.2024.00100 | v2, 2024-11-05 | SN40L | full text | Chip parameter table (Table II); PCU/PMU/AGCU architecture; compiler HBM/DDR allocation (§V-A); kernel orchestration (§IV-D); Llama 3.1 decode on 16 sockets (§VI-B) | No fp8; no P2P bandwidth; HBM/DDR bandwidths disagree with S03/S04 (C1, C2) |
| S01a | Same paper, arXiv **v1** | v1, 2024-05-13 | SN40L | full text | Differences only: v1 has no Table II and no Llama 3.1 section | Superseded; do not cite for numbers |
| **S02** | Koeplinger et al., *Kernel Looping: Eliminating Synchronization Boundaries for Peak Inference Performance*. arXiv:2410.23668 **v1** [cs.CL] | **v1, 2024-10-31**. The unversioned arXiv PDF served v1 on 2026-09-28, so v1 is current; no venue stated | SN40L | full text | Kernel-boundary cost and cause (§1–2); kernel calls per token (Fig. 3); decode efficiency vs a stated roofline (§4.4, Table 3); per-socket HBM trace exists (Fig. 13) | Boundary cost in µs; MoE only as figures (Mixtral) |
| **S03** | Prabhakar, *SambaNova SN40L RDU: Breaking the Barrier of Trillion+ Parameter Scale Gen AI Computing*, Hot Chips 36 slides | 2024-08 | SN40L | full text | 16-socket totals incl. HBM and DDR bandwidth (slide 3); PCU data types (slide 5); Llama 3.1 tok/s (slide 20) | P2P bandwidth; fp8 |
| **S04** | Prabhakar et al., *SambaNova SN40L: A 5nm 2.5D Dataflow Accelerator with Three Memory Tiers for Trillion Parameter AI*, ISSCC 2025, paper 16.4, pp. 288–289; DOI 10.1109/ISSCC49661.2025.10904578 | 2025-02-18 (conf.); downloaded 2026-09-28 via UC Berkeley IEEE Xplore | SN40L | **full text** (incl. figures) | **Per-socket tier bandwidths and capacities (Fig. 16.4.1)**; 10 P2P + 2 PCIe ports per socket (Fig. 16.4.1); PCU/PMU counts and PMU size; software placement policy (weights + KV in HBM); default compute–communication overlap; power-estimation unit and capping | P2P link bandwidth; fp8 |
| **S05** | SambaNova RDU product page, `sambanova.ai/products/rdu-ai-chips` | live page; accessed 2026-09-28 (fetch + user PDF print) | both | full text | SN40 vs SN50 spec table; SN50 BF16/FP8 peaks | No bandwidths; DDR capacity differs from S01/S04 (C3) |
| S06 | SambaNova SN40 RDU datasheet (PDF) | © 2026 | SN40L | full text | 10 kW average per SambaRack | No numeric chip specs |
| **S07** | SambaNova SN50 RDU datasheet (PDF) | © 2026 | SN50 | full text | 432 MB SRAM, 64 GB HBM2E, DDR5 up to 2 TB, 5 nm | No bandwidths, no compute; DDR capacity conflicts with S05 (C6) |
| **S08** | Prabhakar, *Hot Chips 2026: Dataflow at Scale* (SambaNova blog), **including six embedded slide images** | 2026-09-02; accessed 2026-09-28 (fetch + user PDF print) | SN50 | full text + slide images | MBU definition slide; SN50 node network slide (link bandwidths, PCIe); TP-GEMM slide (slide 28, exact %); **EP per-socket bandwidth trace**; DeepSeek-R1 MBU-at-scale slide; disaggregation slide | SN50 HBM/DDR peak bandwidth; which DeepSeek numbers are measured beyond what is stated |
| S09 | R. Smith, *SambaNova's SN50 RDU for AI at Hot Chips 2026*, ServeTheHome | 2026-08-25 (mod. 2026-09-07) | SN50 | full text | Third-party reading of the slides: 16 RDUs per rack in two nodes; two reticle dies, no I/O die; HBM2E; 512-RDU aggregate figure | Author's transcription of image slides |
| S10 | Hot Chips 2026 SN50 slide deck (primary, all slides) | 2026-08 | SN50 | **not accessed: paywalled** (noted 2026-09-28). Six slides are public inside S08; the rest reach us only via S27 (`2HAND`) | — | The primary for P50-HBM-BW, P50-DDR-BW and R50-TP32. **Obtain if access becomes available** |
| **S11** | SambaNova blog + Business Wire, *DeepSeek-R1 671B on SambaNova Cloud* | 2025-02-13 | SN40L | full text | DeepSeek-R1 on 16 SN40L, 198 tok/s/user | Precision, batch, context, residency, speculation |
| **S12** | J. Woo, *Start Building with the Fastest DeepSeek-V3.1-Terminus*, SambaNova blog | 2025-09-30 | SN40L | full text | Four DeepSeek versions on one rack with weights in DDR; 650 ms average measured hotswap | Whether the active model is copied to HBM |
| **S13** | SambaCloud models overview, `docs.sambanova.ai/docs/en/models/sambacloud-models` | live page; two captures 2026-09-28 (fetch; user copy adds MiniMax-M3, gemma-4) | SN40L (implied) | full text | DeepSeek-V3.2 on SambaCloud: **Preview, 32k**; V3.1 production, 128k | Hardware and precision behind each model |
| **S14** | Chitty-Venkata et al. (Argonne), *LLM-Inference-Bench*, arXiv:2411.00136v1 | 2024-10-31 | SN40L | full text | Independent measurement on 8 SN40L (bf16, SambaFlow), dense 7B–70B | Dense only; values in figures |
| **S15** | Artificial Analysis, provider pages: `/providers/sambanova`, `/models/deepseek-v3-2/providers` | fetched 2026-09-28; no page date shown | SN40L (implied) | full text (fetched); **screenshot not yet saved** | SambaNova DeepSeek V3.1: 184 tok/s non-reasoning, 116 reasoning; **DeepSeek V3.2 on SambaNova: listed, no metrics** ("deprecated"); default workload 10k input tokens | Values drift; save dated screenshots before citing |
| S16 | TechInsights, *SambaNova Releases Fourth-Gen Chip* | undated | SN40L | snippet (paywalled) | Claims 688 FP16 TFLOPS | Unverifiable; **excluded** (C5) |
| S17 | V. Sekar, *Inside SambaNova's Inference Architecture*, Vik's Newsletter | 2026-05-20 (paid post); user PDF print 2026-09-28 | both | **public portion only.** The spec table is an image that did not render in the print; the section after the paywall was not read | SambaNova-reviewed nomenclature; **both SN40L and SN50 are HBM2E on TSMC 5 nm** (post edited after SambaNova's correction); SN50 "roughly 2,080 PCUs" and "432 MB across ~2,080 PMUs"; rack = 2 chassis × 8 RDUs, "~7 GB SRAM and 1 TB HBM" | **Unit count conflicts with S27 (C13).** The author's claim that HBM is "not really used in the computation process" contradicts S02, S04 X-PLACE and S08's MBU framing, so it is not used |
| **S18** | SambaStack datasheet, 1 page, doc no. 081225-01 | 2025-08-12; user PDF print 2026-09-28 | SN40L | full text | SN40L = **fourth-generation** RDU; 10 kW average; model switching "as little as 2 microseconds" | Marketing sheet; conflicts with S12 on switching time (C10) |
| **S19** | SambaStack docs v2.1.2 — *Introduction* | accessed 2026-09-28 (user copy) | SN40L | full text | Hosted vs on-prem deployment split; prompt caching; bundles | No hardware or compiler detail |
| **S20** | SambaStack docs v2.1.2 — *Supported models and bundles* | accessed 2026-09-28 | SN40L | full text | **DeepSeek bundle names (all `fp8`), sequence lengths and batch sizes; V3.2 sequence-length/batch card** | No RDU count, precision or speculation fields per model |
| **S21** | SambaStack docs v2.1.2 — *Create a custom bundle* and *Deploy a bundle* | accessed 2026-09-28 | SN40L | full text | **Legalizer** checks memory fit and reports `.status.legalizerInfo` (memory utilization, warnings); **one bundle per node**; bundle switch = delete + redeploy | legalizerInfo field schema; which tiers it reports |
| **S22** | SambaStack docs v2.1.2 — *SambaRack components and cabling* (`sn40l-16-hardware-reference`) and *SambaRack overview* | accessed 2026-09-28 | SN40L | full text | **SN40L-16 node = 8 XRDU modules × 2 RDUs = 16 RDUs; one node per rack**; host module; data switch Juniper QFX5130, 400GbE split to 2×200GbE per endpoint | No HBM/DDR sizes, no P2P link specs |
| **S23** | SambaStack docs v2.1.2 — *Deploying with speculative decoding* | accessed 2026-09-28 | SN40L | full text | Speculative decoding exists; only documented pair is Llama 3.3 70B + 3.2 1B draft; capability flag `speculative_decoding` in `modelprofiles` | **DeepSeek not mentioned**; no MTP |
| **S24** | SambaStack docs v2.1.2 — *Metrics* and *RDU module administration* | accessed 2026-09-28 | SN40L | full text | Customer-visible metrics are router-level only (queue, latency, completions, worker state) | **No RDU utilization, HBM/DDR bandwidth, kernel timing or token-latency metrics documented** |
| **S25** | DeepSeek-V3.2, `huggingface.co/deepseek-ai/DeepSeek-V3.2`: `config.json`, `model.safetensors.index.json`, API metadata, file tree | **commit `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`**, lastModified 2025-12-01; fetched 2026-09-28 | model | full text (config verbatim; API fields) | Architecture fields; parameter count by dtype; stored bytes; indexer tensors present | **Provisional pin** (Rahul owns the canonical revision). The repo's reference inference code, which S26 says specifies DSA "unambiguously", has **not yet been read** (U14) |
| **S26** | DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:2512.02556 **v1** [cs.CL] | v1, 2025-12-02; user PDF 2026-09-28 | model | full text | V3.2 = V3.2-Exp architecture = **V3.1-Terminus + DSA only** (§2.1). Lightning indexer (Eq. 1). Top-k selection over MLA latents (Eq. 2). DSA built on the **MQA mode of MLA**. Complexity O(T·k) core vs O(T²) indexer (§2.3). 2048 selected tokens (§2.1.1). MQA mode for decode (App. A) | Indexer-cache dtype in deployment ("can be implemented in FP8", not "is"); cost figures are H800-only |
| **S27** | lukebest.github.io, *Hot Chips 2026: SambaNova SN50* (personal Quartz wiki, in Chinese), summarizing "raw/papers/HC2026_SambaNova_SN50.pdf (non-public local path)" | page dated 2026-08-26; user PDF print 2026-09-28 | SN50 | full text of the summary; **primary not seen** | **`2HAND`**: SN50 HBM **64 GB @ 1.84 TB/s**; **CXL DDR prompt cache 100 GB/s**; CoWoS-L; rack totals; power/capacity table with MBU and effective TB/s at 128/256/512 RDUs; modeled DeepSeek-R1 uses **MTP3, accept 3.1**; "DeepSeek TP-32 MoE layer > 70 % MBU" | Anonymous transcription. Its numbers **reproduce its own derived columns exactly** (DV-S27-CONSIST), which raises confidence but does not make it primary. "864 PCU + PMU" per die is ambiguous (C13) |
| **S29** | DeepSeek-V3.2 reference inference code, `inference/model.py` and `inference/generate.py` in the S25 repo | **commit `a7e62ac0…`** (same pin as S25); fetched 2026-09-28 | model | **verbatim excerpts** via fetch tool (register_buffer lines, Indexer `__init__`/`forward`, `Linear`, `LayerNorm`/`RMSNorm`, MLA decode branch, `set_default_dtype`); file not read end to end | **MLA latent cache bf16** (`kv_cache` [.., 512] + `pe_cache` [.., 64], default dtype set to bf16 in `generate.py`): **1,152 B/token/layer**. **Indexer key cache fp8 e4m3 + fp32 scale per 128**: **132 B/token/layer**. Norms (RMSNorm, LayerNorm) fp32; `weights_proj` run in fp32; fp8 `Linear` scales fp32. **Decode path scores the whole cache `kv_cache[:bsz, :end_pos]` and applies top-k as a mask**; `kv_b` dequantised once to bf16 | SambaNova's port (U14b, U7). Reference semantics, not an efficient kernel |
| **S28** | V. Mohan, *SemiAnalysis Benchmarks SambaRack SN50 with Fast Inference on MiniMax M2.7*, SambaNova blog, with embedded chart | 2026-07-30; chart "Updated: 07/22/2026"; user PDF 2026-09-28 | SN50 | full text + chart | 16-chip SambaRack SN50, vLLM, MTP: TP16 at C=1 ≈ **820 tok/s/user**; 2 × TP8 data-parallel at C=2 ≈ **420 tok/s/user** (read from figure) | Chart carries a disclaimer: "not hosted at inferencex.semianalysis.com and is not affiliated with or endorsed by InferenceX … may be unofficial". The per-chip normalization with 4 H200s in the loop is unclear. MiniMax, not DeepSeek |

---

## 2. Parameter register

The ceiling note and its script take inputs **only** from these tables. Where sources
disagree, every candidate is listed and the conflict is named.

### 2.1 SN40L, per socket

| Parameter ID | Value | Class | Source | Notes |
|---|---|---|---|---|
| P-BF16-PEAK | **638–640** TFLOPS | DOC | 638: S01 Table II, S03 slide 2, S02 Table 3 (5104 ÷ 8). 640: S04 p. 288, S05 | Gap immaterial (C5) |
| P-FP8-PEAK | **none documented** | UNKNOWN | S01§IV-A, S03 slide 5 | Compute types: BF16, FP32, INT32, INT8 (S03 slide 5). "Configurable storage data types" (S03) permits fp8 *storage* with bf16 compute; see U1 |
| P-INT8-PEAK | not published | UNKNOWN | S03 slide 5 | |
| P-PCU-PMU | 1040 PCUs + 1040 PMUs per socket (520 + 520 per die); ~3000 PCUs, PMUs and switches in total | DOC | S04 p. 288; S01§I | |
| P-SRAM-CAP | **520 MB** (= 1040 PMUs × 512 KB) | DOC | S04 p. 288 and Fig. 16.4.1; S01 | 1040 × 512 KiB = 520 MiB, so S01's "520 MiB" is the precise unit |
| P-SRAM-BW | **> 100 TB/s** aggregate per socket | DOC (lower bound) | S04 Fig. 16.4.1 | Replaces v0's "hundreds of TB/s". A lower bound only |
| P-HBM-CAP | **64 GB** HBM (HBM2E per S05), 4 stacks per socket | DOC | S04 Fig. 16.4.1; S01; S05 | |
| P-HBM-BW | **1.6 TB/s** per socket (preferred). S01 alternatives: 1.8 (Table II), ~2.0 (text) | DOC | 1.6: S04 Fig. 16.4.1, S03 slide 3 (25.6 ÷ 16), S02 Table 3 (12.8 ÷ 8). 1.8/2.0: S01 | **C1, resolved by weight of evidence.** Three sources, including the latest peer-reviewed circuit paper, agree on 1.6. Carry 1.8/2.0 only as an optimistic sensitivity |
| P-DDR-CAP | up to **1.5 TB** per socket (S04 Fig. 16.4.1, S01, S03) / up to **512 GB** (S05, current product page) | DOC, conflicting | S01, S03, S04, S05 | **C3.** DIMMs are pluggable (S01§I): a deployment fact |
| P-DDR-BW | **100 GB/s** per socket (preferred). S01 alternative: 200 | DOC | 100: S04 Fig. 16.4.1, S03 slide 3 (1600 ÷ 16). 200: S01 | **C2, resolved by weight of evidence.** Now 1/16 of HBM bandwidth |
| P-DDR-HBM-COPY | "over 1 TB/s" DDR→HBM per 8-socket node | DOC | S01§I, §VI-C | Exceeds 8 × P-DDR-BW = 0.8 TB/s, so S01 is internally optimistic on DDR. Use P-DDR-BW |
| P-P2P-PORTS | **10 P2P interfaces + 2 PCIe** per socket; token-based P2P protocol streams RDU-to-RDU | DOC | S04 Fig. 16.4.1, p. 288 | Port count only |
| P-P2P-BW | **not published** | UNKNOWN | — | U4 |
| P-DIES | 2 per socket, 600 mm² each, 102 B transistors per socket, TSMC 5 nm, 3× CoWoS-S | DOC | S04 p. 288 | |
| P-CLOCK | < 2 GHz; frequency **varies at run time** under power capping | DOC | S01 Table II; S04 p. 289 | Power estimation unit capping gives up to 39 % better token generation than static capping (S04). Clock is a run-time variable, not a constant |
| P-NODE | **SN40L-16 node:** 8 XRDU modules × 2 RDUs = 16 RDUs, 1 node per rack, x86 host module. Research systems: **SN40L-8** | DOC | S22; S04 p. 288; S01 | **C4, resolved.** Both exist; the SambaStack product is the 16-RDU node |
| P-DATA-NET | Rack data switch Juniper QFX5130, 400GbE split to 2 × 200GbE per endpoint (XRDUs and host) | DOC | S22 | Role (host data path vs RDU-to-RDU) not stated; do not use as P2P bandwidth |
| P-RACK-POWER | 10 kW average per SambaRack | DOC | S06, S18 | Not needed for the ceiling |

### 2.2 SN40L execution model — documented behaviour

| ID | Claim | Class | Source |
|---|---|---|---|
| X-ALLOC | Compiler: HBM by default while it fits; spill to DDR the symbols with the **smallest estimated bandwidth requirement first**; weights get highest priority to stay in HBM | DOC | S01§V-A |
| X-PLACE | "Software allocates Tensors requiring high bandwidth access – like weights and KV caches in generative inference – in HBM. DDR is used to store Tensors requiring high capacity and low bandwidth access." DDR can hold "parameters of hundreds of models along with persistent KV cache values"; "the most frequently used models are then moved to HBM dynamically by software" | DOC | S04 p. 288 |
| X-SPACES | DDR and HBM are two separate, software-visible memory spaces; no hardware caching between them | DOC | S04 p. 288; S01§IV |
| X-OVERLAP | Compute overlaps communication by default. AGCU tile loads, PMU reads/writes and PCU compute are "fully decoupled and overlapped" | DOC | S04 p. 288 |
| X-STATIC | No dynamic allocation or pointer aliasing; lifetimes analysed statically | DOC | S01§V-A |
| X-BWMODEL | Compiler contains a static bandwidth model used to allocate bandwidth and to tune statically | DOC | S01§VII |
| X-KERNEL | A kernel is a forced synchronization barrier; at a boundary the fabric is reprogrammed by loading a new configuration bitstream | DOC | S02§1–2 |
| X-CALLS | Llama3.1-8B decode: 35 kernel calls per token with layer fusion, **4** with kernel looping (323 on DGX H100) | VMEAS | S02 Fig. 3 |
| X-SYNC | Without kernel looping, end-of-decoder sync is > 30 % of time per token (Llama3.1-8B, SN40L-8); looping cuts it ~8× | VMEAS | S02§2 |
| X-ORCH | Graph control lets the RDU orchestrate a kernel schedule without host involvement; decode gains 1.4–8× from hardware orchestration | DOC / VMEAS | S01§IV-D, §VI-A2; S04 p. 288 |
| X-COLL | Collectives are dataflow operators over P2P, pipelined with compute | DOC | S01§VII; S02 Fig. 2 |
| X-OBS-INT | *Vendor-internal* observation surfaces: switch/PMU stall counters (S01§VII); per-socket HBM bandwidth trace (S02 Fig. 13); on-chip power estimation unit (S04) | DOC | S01, S02, S04 |
| X-LEGALIZER | *Customer-visible:* applying a bundle runs a **legalizer** that validates the configuration "fits in RDU memory"; `kubectl get modelbundle <name> -o jsonpath='{.status.legalizerInfo}'` returns **memory utilization and warnings** | DOC | S21 |
| X-BUNDLE | A bundle packages models + profiles, with batch sizes per sequence length. **One bundle per node**; switching bundles requires deleting and redeploying | DOC | S20, S21 |
| X-METRICS | *Customer-visible* metrics are router-level only (queue length, wait, completions, response latency, worker state). No RDU, memory-tier or kernel metrics are documented | DOC (negative) | S24 |
| X-SPEC | Speculative decoding is supported per profile (`spec.features` includes `speculative_decoding`, queried via `kubectl get modelprofiles`). Only documented pair: Llama 3.3 70B + Llama 3.2 1B draft. **No DeepSeek/MTP statement** | DOC | S23 |

**For deliverable 3.** SN40L has explicit, observable execution boundaries: kernel
calls (X-KERNEL, X-CALLS) and the bundle/node boundary (X-BUNDLE). What a *customer*
can observe is much narrower than what exists. They see legalizer memory
utilization and router latency. Counters, HBM traces and power telemetry are
vendor-internal. That split is itself a planner-relevant finding.

### 2.3 SN40L measured results (calibration points, not inputs)

| ID | Result | Class | Setup | Source |
|---|---|---|---|---|
| M-L31-16 | Llama 3.1: 405B **129**, 70B **457**, 8B **1042** tok/s/user | VMEAS | 16 sockets, bf16 weights, mixed bf16/fp32 activations, 8K; speculative decoding on 70B/405B | S01§VI-B Table IV |
| M-L31-HC | Same models: 114 / 380 / 1100 tok/s | VMEAS | 16 SN40L, bf16, Aug 2024 | S03 slide 20 |
| M-HBM-85 | Decode reaches over 85 % of HBM bandwidth | VMEAS | Llama 3.1, 16 sockets | S01§VI-B |
| M-DEC-FUSED | One kernel per decoder layer, ~90 % of PCUs/PMUs, ~85 % of HBM bandwidth | VMEAS | Llama2-7B, 8 sockets | S01§VI-C |
| M-KL-ROOF | 78 % of roofline (SN40L-8) and 76 % (SN40L-16), Llama3.1-8B BS=1; up to 93 % on Llama3.1-70B | VMEAS | Roofline uses **1.6 TB/s per socket** (S02 Table 3), which now matches P-HBM-BW, so these percentages stand as stated | S02§1, §4.4 |
| M-MIXTRAL | Mixtral 8x7B decode at BS 1/8/16, 4K, bf16 | VMEAS | Only MoE with SN40L decode data; figures only | S02 Table 2, Figs. 9–10 |
| M-ARGONNE | 8 SN40L, dense 7B–70B, bf16: lower inter-token latency than GPUs at BS 16, higher TTFT | 3PMEAS | Dense only | S14§VI |
| M-ENERGY | SN40L-8 vs DGX H100: energy per token 0.33× (Llama 3.1 8B) and 0.31× (70B) | VMEAS | Normalized bars | S04 Fig. 16.4.5 |

### 2.4 DeepSeek on SambaNova (deployment statements)

| ID | Claim | Class | Source |
|---|---|---|---|
| D-R1-16 | DeepSeek-R1 671B on **16 SN40L** ("one rack") at **198 tok/s per user** | DEPLOY / VMEAS | S11 (2025-02-13) |
| D-R1-AA | Artificial Analysis measured > 195 output tok/s | 3PMEAS, secondhand | Quoted in S11 coverage |
| D-V31-AA | DeepSeek V3.1 on SambaNova: **184** tok/s (non-reasoning, TTFT 5.66 s), **116** tok/s (reasoning, TTFT 4.03 s); default workload 10k input tokens | 3PMEAS | S15 (fetched 2026-09-28) |
| D-V32-AA | DeepSeek V3.2 on SambaNova: **listed, no metrics** ("deprecated" on the provider page) | 3PMEAS (absence) | S15 |
| D-DDR-RES | One rack holds four DeepSeek versions with **all weights in DDR device memory**; hotswap averages **650 ms** (measured) | DEPLOY / VMEAS | S12 |
| D-BUNDLES | DeepSeek bundles, all named **`fp8`**: `deepseek-r1-v31-fp8-8k`, `deepseek-r1-v3-fp8-8k`, `deepseek-v3-v31-fp8-8k`, `deepseek-v3-v3termi-fp8-8k` (8K; batch 1, 4); `deepseek-4in1-fp8-128k` (128K; batch 1) | DEPLOY | S20 |
| D-V32-SEQ | DeepSeek-V3.2 (SambaStack): sequence lengths **8K (batch 1, 4), 16K (1), 32K (1), 128K (1)**. **No V3.2 bundle row, so no precision stated** | DEPLOY | S20 |
| D-V32-CLOUD | DeepSeek-V3.2 on SambaCloud: **Preview, 32k context** | DEPLOY | S13 |
| D-NODE | One bundle per node, and a node is 16 SN40L, so a DeepSeek deployment occupies **16 RDUs** | DERIVED | X-BUNDLE + P-NODE |
| D-3TIER | SambaNova says DeepSeek-R1 suits the three-tier memory design | claim, no mechanism | S11 |

Still unstated anywhere:
- **Concurrency behind any tok/s figure** (only the bundle batch options).
- **Speculative decoding or MTP use** for DeepSeek.
- **Whether V3.2's sparse attention is implemented.**

### 2.5 DeepSeek-V3.2 checkpoint (model side — provisional pin, Rahul owns canonical)

All `MODEL`, from S25 at commit `a7e62ac0…`.

| ID | Field | Value |
|---|---|---|
| MD-ARCH | architecture / model_type | `DeepseekV32ForCausalLM` / `deepseek_v32` |
| MD-LAYERS | num_hidden_layers; first_k_dense_replace; num_nextn_predict_layers | **61**; **3** (so 58 MoE layers); **1** MTP layer |
| MD-DIMS | hidden_size; intermediate_size (dense); moe_intermediate_size | 7168; 18432; 2048 |
| MD-MOE | n_routed_experts; num_experts_per_tok; n_shared_experts; n_group / topk_group; scoring | 256; **8**; 1; 8 / 4; sigmoid, `noaux_tc`, routed_scaling_factor 2.5 |
| MD-MLA | q_lora_rank; kv_lora_rank; qk_nope / qk_rope / v_head_dim; num_attention_heads | 1536; **512**; 128 / **64** / 128; 128 |
| MD-KVHEADS | num_key_value_heads | **128**. The GLM note's trap: MLA caches one latent (kv_lora_rank + qk_rope = 576 elems/token/layer), not 128 heads |
| MD-INDEXER | index_n_heads; index_head_dim; index_topk | 64; 128; **2048**. Indexer tensors per layer: `wq_b`, `wk` (+ `weight_scale_inv`), `k_norm` (weight, bias), `weights_proj` |
| MD-QUANT | quantization_config | fp8 `e4m3`, block 128×128, scale_fmt `ue8m0`, dynamic activations; torch_dtype bfloat16 |
| MD-CTX | max_position_embeddings; rope | 163840; YaRN factor 40 over 4096 |
| MD-VOCAB | vocab_size; tie_word_embeddings | 129280; false |
| MD-PARAMS | parameters by stored dtype | BF16 3,946,184,704 · F8_E4M3 681,408,069,632 · F32 42,667,040 · **total 685,396,921,376** |
| MD-BYTES | weight bytes by dtype (DERIVED from MD-PARAMS) | **689,471,107,200 B = 689.47 GB = 642.12 GiB** |
| MD-STORED | repo usedStorage | 689,484,423,011 B. Differs from MD-BYTES by 13.3 MB (0.002 %): tokenizer, assets, headers. **Ledger check passes** |
| MD-SHARDS | shard files | 163 (e.g. 00002 = 4,302,383,956 B) |

**DSA semantics (from S26; model-side, `MODEL`).**

| ID | Claim | Source |
|---|---|---|
| MD-LINEAGE | "DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus … the only architectural modification … is the introduction of DeepSeek Sparse Attention (DSA) through continued training." The MoE and MLA dimensions are therefore V3.1-Terminus's | S26 §2.1 |
| MD-INDEXER-SEM | Index score I(t,s) = Σ over indexer heads of wᵢ · ReLU(qᵢ · k_s). There is **one indexer key k_s per preceding token**, shared across indexer heads (config: 64 heads × 128 dim, MD-INDEXER), so the indexer caches **128 elements per token per layer**. It has a small number of heads and "**can be implemented in FP8**" | S26 §2.1, Eq. 1 |
| MD-TOPK | For each query token, attention reads only the MLA latent entries of the **top-k index scores**; k = **2048** ("select 2048 key-value tokens for each query token") | S26 §2.1, Eq. 2; §2.1.1; config `index_topk` |
| MD-MQA | DSA is implemented on the **MQA mode of MLA**: each latent (576 elems/token/layer) is shared across all 128 query heads. MQA mode is the decode mode; short-sequence prefill uses a masked MHA mode | S26 §2.1, §2.3, App. A Fig. 7 |
| MD-COMPLEXITY | Core attention O(T²) → **O(T·k)**; the lightning indexer "still has a complexity of O(T²)" but with much less compute than V3.1-Terminus's MLA | S26 §2.3 |

**Consequence for the ledger.** At decode context S, attention bytes per layer are:
- *reads:* the indexer cache (S × 128 elements) plus min(S, 2048) latent entries × 576 elements;
- *writes:* one new latent and one new indexer key.

The byte counts depend on the cache dtypes, which S26 does not fix (U14).

### 2.6 Derived checks (arithmetic on cited rows — checks, not findings)

| ID | Check | Result | Depends on |
|---|---|---|---|
| **DV-FIT** | Weights vs HBM on one 16-RDU node (16 × 64 GiB = 1,099.5 GB) | **fp8 as stored: 689.5 GB = 62.7 % of HBM**, leaving ~410 GB for KV, indexer cache and activations. **All-bf16 upcast: 1,370.9 GB = 124.7 %: does not fit** | MD-BYTES, P-HBM-CAP, D-NODE. Excludes KV and activations (deliverable 2) |
| **DV-SWAP** | Can the 650 ms hotswap (D-DDR-RES) be a full DDR→HBM copy? Aggregate DDR = 16 × 100 GB/s = 1.6 TB/s | fp8 copy ≥ **0.43 s** (fits in 650 ms at ~66 % of DDR peak); bf16 copy ≥ **0.86 s** (does not fit) | P-DDR-BW (C2 resolution), MD-BYTES; assumes a full copy limited by DDR |
| **DV-DDR-DECODE** | Can decode stream weights from DDR at the served speed? At 198 tok/s, DDR supplies at most 1.6 TB/s ÷ 198 = **8.1 GB per token** | Routed + shared experts touched per token at batch 1 = 58 × 9 × 3 × 7168 × 2048 = **23.0 B params ≈ 23.0 GB at fp8, 2.8× the DDR budget**, before attention. **DDR-resident decode is ruled out for D-R1-16** unless > 2.8 tokens are accepted per step | D-R1-16, P-DDR-BW, MD-DIMS/MD-MOE. The same MoE dimensions across R1 → V3.1 → V3.2 are now sourced: S12 (V3.1 shares R1's architecture) and MD-LINEAGE (V3.2 = V3.1-Terminus + DSA) |
| **DV-SN50-HBM** | Does the SN50 EP trace bound HBM bandwidth? | The 64-socket EP trace (S08 slide "EP on 64-socket SN50") plateaus at **~1.8 TB/s** (read from figure). This is consistent with S27's 1.84 TB/s peak: the plateau is ~98 % of it | S08 slide (axis not labelled "HBM"; per socket assumed); S27 |
| **DV-S27-CONSIST** | Do S27's SN50 numbers reproduce each other? | N × 1.84 TB/s × MBU gives **102.9 / 212.4 / 376.8 TB/s** at 128/256/512 RDUs vs S27's stated 103 / 212 / 377. Capacity 8.2 / 16.4 / 32.8 TB = N × 64 GB. Rack totals = 16 × per-RDU (25.6 PF = 16 × 1600; 6.9 GB = 16 × 432 MB; 6.4 TB/s = 16 × 400 GB/s; 800 GB/s = 16 × 50). **All consistent** | S27. Internal consistency is evidence the transcription is faithful, not that 1.84 is right |
| **DV-D50-AGG** | What is S09's "> 350 TB/s at 512 RDUs"? | It is **effective** model bandwidth: 512 × 1.84 × 0.40 = 376.8 TB/s. Installed is 942 TB/s. The 0.68 TB/s-per-RDU reading is **ruled out** | S09, S27 |
| **DV-INV** | Does a tensor inventory rebuilt from `config.json` reproduce the checkpoint? | **Exactly**, for all three dtypes and the byte total: F8 681,408,069,632 · BF16 3,946,184,704 · F32 42,667,040 · 689,471,107,200 B. This pins that the MTP layer is a full decoder layer with its own indexer, 256 experts, embed and head (15.44 GB). Norms and `k_norm` are stored fp32. `weights_proj` is stored bf16 | MD-*, S29; `v32_sn_ceiling.py check` |
| **DV-ANCHOR-B1** | At what batch was D-V31-AA served? | Its 10k-token input exceeds every 8K bundle. The only larger DeepSeek bundle is `deepseek-4in1-fp8-128k`, **batch 1**, so the anchor is B = 1 | D-V31-AA, D-BUNDLES. **Assumes** SambaCloud serves from the documented bundle set |
| **DV-EP-LOAD** | Busiest chip when 16 chips hold whole experts, under `noaux_tc` group-limited routing with iid scores | **2.01** experts at B = 1 with a striped layout, **2.30** with a contiguous one (whole-expert floor 1; sliced equivalent 0.5). The simulated distinct-expert count matches the closed form | MD-MOE; seeded simulation in `v32_sn_ceiling.py` |
| **DV-MBU-INVERT** | Can the SN50 DeepSeek-R1 MBU points be inverted for bytes per step? | Now that MTP3 with accept length 3.1 is known (S27), bytes moved per verification step = N × 1.84 × MBU ÷ (tok/s ÷ 3.1) = **931 / 1,071 / 1,314 GB** at 64/128/256. Each exceeds the **entire** fp8 checkpoint (689 GB, MD-BYTES), and the figure grows with N | Consistent only if the points include concurrency > 1, or if MBU counts replicated per-chip reads. **Not invertible**; ask SambaNova how "parameter bytes" is counted for MoE across chips (§7 Q8) |

**What these change.**
- **Working assumption for deliverable 2:** fp8 weight storage, HBM-resident weights on 16 SN40L. It rests on DV-FIT, DV-SWAP, DV-DDR-DECODE and D-BUNDLES and is stated as an assumption, not a fact.
- **The competing case is narrower but still open:** weights fit, but KV plus indexer cache at long context, plus any bf16 dequantised staging, could spill.

### 2.7 SN50 (customer projection — kept separate)

| Parameter ID | Value | Class | Source | Notes |
|---|---|---|---|---|
| P50-BF16-PEAK | **1600** TFLOPS | DOC | S05 | |
| P50-FP8-PEAK | **3200** TFLOPS | DOC | S05 | "5× compute" (S09) = 3200 ÷ 640, i.e. SN50 fp8 vs SN40L bf16 (DERIVED) |
| P50-FP4 | not documented | UNKNOWN | — | |
| P50-SRAM | **432 MB** | DOC | S05, S07, S08 | Less than SN40L's 520 MB |
| P50-HBM-CAP | **64 GB HBM2E** | DOC | S05, S07, S09 | Same as SN40L |
| P50-HBM-BW | **1.84 TB/s** per RDU | **2HAND** (primary S10 paywalled); corroborated | S27; DV-SN50-HBM (S08 trace plateau ~1.8); DV-S27-CONSIST; DV-D50-AGG | Provisional input. 15 % above SN40L's 1.6 on the same HBM2E (S17). Replace with S10 when obtained |
| P50-DDR-CAP | **256 GB – 2 TB** per RDU (configurable) | DOC + 2HAND | 512 GB: S05. 2 TB: S07. 256 GB–2 TB: S27 | **C6 explained:** a DIMM-population range, not a contradiction. Customer's population still needed |
| P50-DDR-BW | **100 GB/s** ("CXL DDR prompt cache 100 GB/s") | **2HAND** | S27 | Provisional. Same as SN40L's per-socket DDR; attach path is **CXL** per S27. Not corroborated by any second source |
| P50-NET | **10 × 800G Ethernet per RDU = 2 TB/s**; each link **200 GB/s** (legend "200 GBps, 800G Enet links"); 7 links full-mesh the 8-socket node; 2 links = **400 GB/s per RDU inter-node scale-up**; **1 × 400G RoCEv2 NIC = 50 GB/s per RDU** scale-out; PCIe5 per XRDU: 2 RDUs × 2 × x8 to a PCIe switch, host x16 | DOC | S08 network slide | 200 GB/s per 800 Gb/s link is bidirectional (DERIVED: 800 Gb/s = 100 GB/s per direction) |
| P50-SCALE | 8 per node; 16 per air-cooled rack (two nodes); scale-up to 256 RDUs | DOC | S05, S08, S09 | |
| P50-DIES | Two reticle-size logic dies + HBM, no I/O die; TSMC 5 nm; **CoWoS-L** | DOC (via S09) + 2HAND | S05, S07, S09; S27 (CoWoS-L) | |
| P50-UNITS | "roughly 2,080" PCUs and PMUs per socket (S17) / "864 PCU + PMU" per die (S27) | conflicting | S17, S27 | **C13.** Not needed for a bandwidth bound |
| P50-RACK | Air-cooled rack = 16 RDUs / 2 nodes: 25.6 PF BF16, 51.2 PF FP8, 6.9 GB SRAM, 1 TB HBM, scale-up 6.4 TB/s, scale-out 800 GB/s; **typical 15–20 kW, max 34 kW** | 2HAND (totals DERIVED-consistent) | S27; DV-S27-CONSIST | |

| ID | SN50 result | Class | Source |
|---|---|---|---|
| R50-MBU | MBU = fraction of peak HBM bandwidth used for weights and KV; tok/s = HBM BW/chip × MBU × chips ÷ (parameter bytes + KV bytes/token) | DOC (definition) | S08 MBU slide |
| R50-R1-MOD | DeepSeek-R1 8K/1K: **200 / 300 / 500 tok/s/user** at 64 / 128 / 256 SN50, MBU **51 / 44 / 45 %**; plotted against 26 InferenceX B300-FP4 GPU points (mostly < 25 % MBU). Modeled with **MTP3, accept length 3.1** (S27) | **VMOD** ("modeled operating points, not measured") | S08 text + slide; S27 |
| R50-POWER | Power/capacity table (MTP3, accept 3.1): 128 RDU / 120 kW / MBU 43.7 % / 103 TB/s / 8.2 TB; 256 / 240 kW / 45.1 % / 212 TB/s / 16.4 TB; 512 / 480 kW / 40.0 % / 377 TB/s / 32.8 TB. Comparators: 4 × NVL72 at 5 % MBU (28.8 TB/s); 32 × DGX B300 at 25 % (16 TB/s) | VMOD, 2HAND | S27 |
| R50-TP32 | "DeepSeek TP-32 MoE layer > 70 % MBU" | VMEAS?, 2HAND (measured vs modeled not stated) | S27; slide exists per S09 but not seen |
| R50-GEMM | TP GEMM TFLOPs utilization **71.9 / 70.9 / 78.9 %** at 8/16/32 SN50; speedup 1.00 / 1.97 / 4.39; full overlap at 32 | VMEAS, kernel-scoped | S08 slide 28 |
| R50-EP | 64-socket EP: "**>70 %** Bandwidth To Load Experts"; per-socket trace plateau ~1.8 TB/s with periodic dips to ~0.75 TB/s | VMEAS (trace) | S08 slide |
| R50-MINIMAX | MiniMax M2.7: 4 × H200 prefill + **16 SN50 TP16** decode, engine **vLLM**, **763** tok/s at 10k input; SN40 public endpoint **482** (AA serverless, 2026-07-07). SemiAnalysis legend: "H200+SN50 (vLLM Custom, **MTP**)" | 3PMEAS (AA, SemiAnalysis) via vendor slide | S08 disaggregation slide |
| R50-SEMI | MiniMax M2.7 on one 16-RDU SambaRack SN50, vLLM + MTP: **TP16 at C=1 ≈ 820 tok/s/user**; **2 × TP8 data-parallel at C=2 ≈ 420 tok/s/user** (read from figure) | VMEAS (vendor-hosted chart in InferenceX format; not endorsed by InferenceX) | S28 |
| D50-AGG | 512 SN50: > 350 TB/s aggregate "model bandwidth capacity", MBU ~40 % | **Resolved: effective bandwidth** (DV-D50-AGG) | S09; S27 |

---

## 3. Conflicts register

| ID | What disagrees | Values | Status | Effect / action |
|---|---|---|---|---|
| **C1** | SN40L HBM bandwidth per socket | 1.6 (S02, S03, **S04**) · 1.8 (S01 Table II) · ~2.0 (S01 text) TB/s | **Resolved → 1.6** by three sources incl. ISSCC Fig. 16.4.1 | Use 1.6. Carry 1.8–2.0 only as an optimistic sensitivity. Ask SambaNova whether 1.8/2.0 was an earlier or other SKU |
| **C2** | SN40L DDR bandwidth per socket | 100 (S03, **S04**) · 200 (S01) GB/s | **Resolved → 100** | S01's > 1 TB/s DDR→HBM per 8 sockets exceeds 8 × 100; treat S01 as optimistic |
| **C3** | SN40L DDR capacity | 1.5 TB (S01, S03, S04) · 512 GB (S05) | Open (deployment) | Customer's DIMM population |
| **C4** | Node size | 8 (S01, S04) · 16 (S05, S22) | **Resolved:** SN40L-8 research system; SN40L-16 SambaStack product node | Use 16 for DeepSeek |
| C5 | BF16 peak | 638 · 640 · 688 (S16, excluded) | Immaterial | Use 638–640 |
| C6 | SN50 DDR capacity | 512 GB (S05) · 2 TB (S07) · 256 GB–2 TB (S27) | **Explained:** configurable range | Ask the customer's population |
| C7 | S02 70B efficiency | 93 % (text) · 95 % (Fig. 12 caption) | Immaterial | Cite 93 % |
| C8 | SN50 EP expert-load share | > 70 % (slide title) · ~80 % (S09) | **Resolved → "> 70 %"** (slide) | — |
| C9 | Chip name | "SN40L" (papers, S18) · "SN40" (S05, S06) | **Resolved:** S18 calls SN40L the 4th generation; S05 calls SN50 the 5th and SN40 the previous one | SN40 = SN40L |
| **C10** | Model switching time | "as little as 2 microseconds" (S18) · 650 ms average measured (S12) | Open | 5 orders of magnitude. Likely different events (switch among HBM-resident models vs DDR→HBM swap). Cite S12 for swaps |
| C11 | V3.2 context | 32k Preview (S13, SambaCloud) · up to 128K (S20, SambaStack) | Explained: different products | Customer is on-prem-like, so use S20 |
| **C12** | DeepSeek-V3.2 checkpoint size | index `total_size` = 1,370,793,842,752 B · stored = 689,484,423,011 B | **Resolved → 689.5 GB.** total_size = params × 2 exactly (bf16-assumed metadata) | Model-side trap: never size the model from `total_size` |
| C13 | SN50 compute/memory unit count | "roughly 2,080" PCUs and PMUs per socket (S17) · "864 PCU + PMU" per die (S27) | Open | Immaterial for a bandwidth bound; resolve from S10 |
| — | Sekar's claim that on RDUs HBM "is not really used in the computation process" (S17) | vs S04 X-PLACE (weights + KV in HBM), S02/S08 (decode bound by HBM bandwidth, MBU) | **Rejected** as author opinion contradicted by vendor primaries | Not used as evidence |

---

## 4. Unknowns that block or bound deliverable 2

Ranked by how much each still moves the SN40L DeepSeek V3.2 ceiling.

| ID | Unknown | Status after v1 | Resolving observation |
|---|---|---|---|
| **U1** | fp8 **storage** vs bf16 upcast for DeepSeek on SN40L | **Largely addressed.** Bundles are named `fp8` (D-BUNDLES), and fp8 fits while bf16 does not (DV-FIT). The 650 ms swap fits fp8, not bf16 (DV-SWAP). Still unknown: whether fp8 is dequantised to bf16 on the fly (no fp8 compute documented), and V3.2's own profile (no bundle row) | `legalizerInfo` for a V3.2 bundle; SambaNova statement |
| U2 | Served precision | Same as U1 | Same |
| **U3** | Which tier holds active weights during decode | **DDR-resident decode ruled out** for the served R1 speed (DV-DDR-DECODE); HBM residency fits (DV-FIT). **Open:** KV and indexer-cache placement at 32K–128K, and any spill | `legalizerInfo` memory utilization per tier |
| **U4** | SN40L P2P bandwidth and topology | **Still unknown.** 10 P2P ports per socket (P-P2P-PORTS) | Ask SambaNova. Collectives stay unpriced |
| **U7** | Does SambaNova implement V3.2 sparse attention (indexer + top-2048), or dense MLA? | Unknown. The Preview status, the 32K cloud cap and the "deprecated" AA status hint at a less mature V3.2 path. The model-side semantics are now pinned (MD-INDEXER-SEM, MD-TOPK, MD-MQA), so both cases can be priced | Ask; or decode time vs context length (DSA flattens beyond 2048 tokens; dense MLA grows linearly) |
| **U14** | Deployment dtypes of the MLA latent cache and the indexer-key cache | **Resolved model-side (v2.1, S29):** latent bf16 (1,152 B/token/layer), indexer keys fp8 + fp32 scale (132 B). **U14b, SambaNova's dtypes on RDU, stays open** | Ask SambaNova (deliverable 2 §12 Q7) |
| U8 | Speculative decoding / MTP for DeepSeek | Undocumented (S23). SN50 MiniMax uses MTP (R50-MINIMAX) | `kubectl get modelprofiles` feature flags; ask |
| U9 | Concurrency behind 198 tok/s (D-R1-16) | Bundles offer batch 1 or 4 at 8K (D-BUNDLES); served batch unknown | Ask |
| **U10** | SN50 HBM peak, DDR bandwidth, fp8 compute in decode | **Provisionally resolved via `2HAND`:** HBM 1.84 TB/s (corroborated by the S08 trace and S27's own consistency); DDR 100 GB/s over CXL (uncorroborated). Whether decode uses fp8 compute is unknown | Paywalled S10 deck, or SambaNova confirmation. Until then, SN50 projections carry the `2HAND` label |
| **U11** | Compiled-plan artifacts | **Partially addressed:** `legalizerInfo` is customer-visible (X-LEGALIZER). No kernel, tier-traffic or utilization metrics are exposed (X-METRICS) | Ask for `legalizerInfo` output and model-profile specs of the customer's V3.2 bundle |
| U12 | legalizerInfo schema: which tiers, what units | Unknown | Same request as U11 |
| U13 | SN40L run-time clock under power capping | Varies at run time (P-CLOCK); matters only if compute-bound | Low priority for decode |

---

## 5. What this source base supports today

**Can be stated with citations:**
- SN40L per-socket tiers: SRAM 520 MB at > 100 TB/s; HBM 64 GB at 1.6 TB/s; DDR up to 1.5 TB at 100 GB/s (S04 Fig. 16.4.1).
- 10 P2P ports per socket.
- A DeepSeek deployment is one 16-RDU node.
- DeepSeek bundles are fp8-named, with batch 1/4 at 8K and batch 1 to 128K.
- The V3.2 checkpoint is 689.47 GB and fits in the node's HBM at 62.7 %.
- DDR-resident decode cannot deliver the served R1 speed.
- The compiler's placement policy puts weights and KV in HBM.
- Customers can read legalizer memory utilization but no hardware metrics.

- V3.2's attention semantics: indexer over all S tokens, top-2048 latent reads, MQA mode in decode (S26).
- SN50, provisionally (`2HAND`): HBM 1.84 TB/s and DDR 100 GB/s per RDU.

**Cannot yet be stated:**
- SN40L P2P bandwidth.
- Whether V3.2 runs sparse attention on SambaNova.
- The latent and indexer cache dtypes (U14).
- Concurrency behind any vendor speed.
- SN50 numbers from a primary source.

**Consequence for deliverable 2.** The residency question narrows from three competing
cases to one primary case (fp8, HBM-resident weights, 16 SN40L) plus a spill case
driven by long-context KV and indexer cache. Collectives stay reported as unpriced (U4).

---

## 6. Coverage against the brief

| Brief asks for | Status |
|---|---|
| SN40L architecture papers | S01 (v2 canonical), S03, **S04 (full text)** |
| Kernel-looping paper | S02, **v1 confirmed** (2024-10-31) |
| Hot Chips 2026 SN50 deck | **Paywalled, not obtained** (S10). Public: six slides in S08. Secondhand: S27 |
| Model-side sources | S25 checkpoint (provisional pin), S26 technical report. Reference inference code not yet read (U14) |
| Compiler documentation | **Partial.** Compiler internals only in S01§V–VII, S02§3, S04. SambaStack v2.1.2 admin docs (S19–S24) cover bundles, profiles, legalizer, speculation and metrics. SambaFlow (compiler) docs do not appear in the public v2.1.2 docs index (keyword search of the index for compile/SambaFlow/profil/trace, 2026-09-28) |
| Compiled-plan artifacts | **None obtained.** Customer-visible candidates identified: `legalizerInfo`, `modelprofiles` specs (X-LEGALIZER, X-SPEC). Vendor-internal: counters, HBM traces, PEU telemetry (X-OBS-INT). Request list in §7 |
| Handover from last week | **Not located** as of 2026-09-28. Absent from the runtime repo (docs, planner models, all remote branches checked). Asking Jalon. Merge on receipt, keeping these IDs |

---

## 7. Remaining work


**Still missing (could not be found publicly)**
- **SN40L:** P2P link bandwidth and topology (U4).
- **Rahul: canonical V3.2 pin.** This file uses commit `a7e62ac0…` provisionally.
- **SN50:** a primary source for HBM and DDR bandwidth. The full HC2026 deck (S10) is **paywalled**; S27 stands in as `2HAND`.
- **DeepSeek on SambaNova:** V3.2 bundle profile and precision, sparse-attention implementation, speculative decoding/MTP, served concurrency (U1, U7–U9).
- **legalizerInfo schema** (U12).
- **Last week's handover.**
