# Cross family planner requirements version 1

Status update October 13, 2026: Ishaan authorized local commits. Earlier October 11 references to uncommitted work describe that snapshot. Maintainer review, pushing, publication and external contact remain separate; none is established by creating local commits.

## Evidence update on October 11

Planner statements were rechecked on October 11 against default-branch head `16d6d44c984d5446b6fc124550c995649cf96106`: none of the three intervening commits changed `gitm/planner/`. Measurements retain their original October 6 planner pin `05abd11b866bdabe2619b5f289db52d647eb55ad`. The current kernel-identity documentation and ROCm status were separately read; see the current `docs/kernel_identity.md`, `docs/rocm_correlation.md` and `sources/repository-comparison.json`. See [validation interpretation](validation-interpretation-update.md) and [implementation review](implementation-review.md).

**Status:** version 1 prepared for team review, 2026-10-06. This consolidates the K3 reconciliation, current planner audit, reviewed memory proposal, Arshia's SN40L evidence and a completed source-derived two-family exercise. Planner review pin: `05abd11b866bdabe2619b5f289db52d647eb55ad`. The recommendation is provisional pending team disposition; production support and measured deployment validation are separate outcomes.

## Purpose

GitM estimates model execution ceilings from workload shape, model structure, hardware rates, data movement, and scheduling assumptions. The planner must make those assumptions explicit. It must not produce a precise-looking ceiling when the model's execution plan, physical traffic, or concurrency is unknown.

Version 1 covers three execution families:

1. **Dynamic-kernel GPUs:** NVIDIA and AMD, where software launches kernels dynamically and tooling can often observe individual kernels.
2. **Compiler-managed accelerators:** AWS Trainium, where a compiler emits a NEFF and places work/data on device resources.
3. **Spatial dataflow accelerators:** SambaNova SN40L and Cerebras, where compiled mappings arrange compute, storage, and routes across processing elements. SN40L evidence below is architecture- and vendor-source-derived unless explicitly labeled otherwise; no DeepSeek V3.2 mapping or trace is available.

The goal is a comparable logical workload and auditable resource model, not one execution representation that erases differences between machines.

## Requirements

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| R1 | Preserve a shared logical workload core: requests, prefill/decode shape, token positions, logical operators/work, tensor identity, and true dependencies. | The same model/workload meaning can be referenced across families; family-specific lowering does not rewrite logical dimensions to make a loader accept them. |
| R2 | Distinguish logical tensor volume from physical traffic. | Bytes are attributed to a tier/resource and execution region only when measured or derived from an identified compiler/platform artifact. Unknown traffic remains unknown. |
| R3 | Describe memory tiers and transfers. | A hardware/plan description can state tier capacity, scope, resource rate, evidence class/bound direction, directed transfers, residency and live intervals. Capacity validation does not infer residency from logical bytes alone. |
| R4 | Distinguish execution regions from graph nodes and GPU launches. | A family plan can map logical operations to compiled regions or spatial stages and state boundary cost/provenance. One execution region is not assumed to equal one GPU kernel. |
| R5 | Represent dependencies, resource contention and overlap separately. | Demands sharing one physical resource aggregate; concurrency is allowed only when a plan or measurement supports it. Unknown concurrency does not silently become free overlap. |
| R6 | Represent collectives with route/topology, latency/startup, resource identity, and fusion/overlap semantics. | Unknown link cost is surfaced as unpriced/unknown; a separate GPU launch term is not applied to a compiler-fused collective without evidence. |
| R7 | Match prediction and observation granularity. | GPU kernel residuals, Trainium compiler/runtime observations, SN40L deploy/request observations, and vendor-internal telemetry are identified by source and access level. A residual is not reported at finer granularity than its evidence supports. |
| R8 | Carry evidence and uncertainty on model and hardware inputs. | Values are tagged measured, compiler-derived, source-derived, unknown, or another defined class; point/lower/upper/range direction is retained through derived calculations. Unknown catalogue fields continue to fail loudly. |
| R9 | Preserve compatibility until an explicit family plan opts in. | Existing GPU catalogue predictions retain their current meaning through a legacy adapter; Trainium/RDU entries do not inherit GPU bandwidth, launch, or tracing defaults. |
| R10 | Keep model architecture, hardware family and deployment plan distinct. | K3/V3.2 model semantics are not treated as chip-family features; compiled placement and deployment settings are not inferred from checkpoint dimensions. |
| R11 | Keep the gap register in steady use by FDE and Arshia's lane. | Prioritization and next decisions are recorded in the register; sources/status updates follow new planner behavior, source artifacts, or review decisions. Agree an update rhythm with the lane owners; do not leave the register as a one-time report. |

## Provisional schema position

Use **one shared logical core plus family-specific plan adapters**. For family 3, the current evidence points to a **sibling spatial-plan representation** joined to logical operations through stable IDs. That plan needs compiled-region/stage identity, placement, buffers, routes, boundary/fill-drain cost, steady-state initiation interval, and plan-specific observations. Family 2 may fit as a compiler-region adapter over the shared core and tier/transfer model, but this remains provisional until a versioned NEFF/profile and runtime evidence are reviewed.

The tiered-memory proposal is one shared resource-demand concept; it does not by itself define the spatial schedule. Its execution-region unit must be mapped by a family plan. The existing scalar GPU path remains the compatibility path until a family opts into evidence-backed tiered inputs.

## Cost of being wrong

**Force one kernel-shaped schema onto all families:** spatial streaming may be charged as repeated HBM reads; overlapped computation and communication may be summed as serial time; GPU launch overhead may be charged where the machine instead has compiled boundaries; graph-capture/stream interventions may be recommended on machines that do not expose them. This risks systematic ceiling error and false optimization advice.

**Split schemas too early or too broadly:** duplicate logical workload definitions, model geometry, provenance and validation; comparisons drift across families; every new model or measurement must be translated among sibling schemas. This raises maintenance and reconciliation cost and can make cross-family comparisons unauditable.

## Evidence that would change the position

Keep a single plan schema only if a reviewed SN40L compiler artifact plus trace maps placement, compiled regions, resource overlap and steady-state timing into the existing node/dependency model without family-specific semantic exceptions or systematic residuals. Keep Trainium on the shared plan adapter only if a versioned NEFF/profile plus runtime evidence can be represented without pretending compiler regions are dynamic GPU launches. Split more of the shared core only if the logical concepts themselves (workload shape, logical work, dependencies, provenance) fail to retain consistent meaning across families.

Evidence currently available for SN40L establishes SRAM/HBM/DDR tiers, static compiler placement behavior, documented overlap, explicit compiled-kernel boundaries, and customer/vendor observation boundaries (Arshia source base v2.1: P-SRAM-CAP/BW, P-HBM-CAP/BW, P-DDR-CAP/BW, X-ALLOC, X-PLACE, X-OVERLAP, X-KERNEL, X-COLL, X-LEGALIZER, X-OBS-INT, X-METRICS). It does not establish DeepSeek V3.2's compiled placement, region count, link latency, or measured ceiling. The source base records DDR configuration variability; use configuration-specific values, not a universal point.

## Two family exercise

The shared resource-qualified memory-traffic concept has been exercised for Trainium2 and SN40L through a local review adapter using the unmodified pinned planner's roofline and dependency functions. Trainium uses AWS's documented HBM-to-SBUF tile example. SN40L uses its documented DDR-to-HBM activation flow with explicitly selected small byte counts and Arshia's source-derived rate scenarios. No platform access or measured trace was supplied. The mission permits source-derived evidence in that situation.

The [exercise report](two-family-planner-exercise.md) records inputs, sources, component results, missing rates, 15 passing acceptance examples and reproduction instructions. Missing transfer-engine rates prevent a complete-region numerical ceiling. Aggregate fit does not prove distributed bank/layout legality. This is a source-derived prototype exercise of shared semantics; it requires team review and originally did not claim runtime integration or measured model performance. The new optional calculator and source-derived example are now integrated locally; measured model performance on those families remains unvalidated. The independent calculator rerun passes Arshia's 66 bundled checks.

For later measured validation, capture an identified GPU or Neuron workload with software versions, SKU, batch/context, physical traffic and region timing; pair it with an SN40L compiler/placement artifact and observations if access exists. Compare at observable granularity. Absence of access must remain recorded, not substituted with nominal peak arithmetic.

## Register and ownership

The [schema-gap register](schema-gap-register.md) is the local working prioritization record; it is not yet live in the GitHub repository or in steady use by FDE and Arshia's lane. Arshia's 12 SambaNova findings have been cross-referenced using stable CF topics, with CF-04a/04b distinguishing regions and observations; derived probe errors are not platform measurements. Keep the original SN IDs alongside those register IDs and agree the update rhythm with the lane owners.

- **Rahul:** review planner interface/compatibility for tiered traffic and model-catalogue handling of DeepSeek V3.2. He has said the proposal looks good and requested an “Execution region” table row; the row is in the proposal’s Execution region section. His requested edit has been addressed. October 6 accounting refinements are supplied for follow-up review; his positive review and the completed table edit are recorded.
- **Arshia:** validate that SN40L source interpretation stays within her evidence, especially the difference between family-level compiler behavior and a V3.2 compiled mapping.
- **Saiger:** decide which K3 mechanisms belong in the Kimi-family model representation versus shared planner fields.
- **Ishaan:** maintain the register and evidence labels, reconcile review feedback, and prepare this requirements draft. Maintain source-derived exercise evidence separately from implementation and measured validation.

## Review record

| Date | Item | Status |
| --- | --- | --- |
| 2026-09-25 | Friday drafts: register, assumptions audit, tiered-memory proposal | Drafts existed; not yet accepted as reviewed or implemented. |
| 2026-09-28 | Arshia's three SambaNova deliverable archives received | Source base v2.1, conditional V3.2 ceiling analysis, and 12-gap analysis reviewed and attributed to Arshia. |
| Date not recorded | Rahul response | “The proposal looks good”; asked for an execution-region row. |
| 2026-10-04 | Proposal | Execution-region row added to the DOCX; no implementation or Git publication. |
| 2026-10-04 | Register and audit | Updated to incorporate Arshia's evidence and label source-derived versus conditional results. |

## Decisions still open

1. Disposition of the October 6 accounting refinements and an owner for the V3.2 model-reader findings; Rahul's existing positive review is already recorded.
2. Saiger's Kimi-specific versus shared-field decisions.
3. Availability of platform artifacts for measured validation beyond the completed source-derived exercise.
4. Final review of the shared-core plus spatial-plan boundary, including explicit evidence that would reverse it.

## Operational use and completion criteria

The register is the proposed common intake record for FDE and Arshia's lane. Each new fact retains its source/class/date, each code change cites an implementation path, and each status change records the decision and owner. Proposed rhythm: update when evidence or planner behavior changes, and review unresolved priority changes weekly. This is a proposal for adoption, not a claim the lanes already use it.

Friday item 1 has a reconciled local register but still needs authorized repository publication. Item 2 has a current implementation audit and architecture evidence, with missing deployment evidence explicitly labeled. Item 3 has Rahul's positive proposal review, the requested execution-region addition, semantics, migration and distinguishing acceptance examples; it remains a reviewed-proposal outcome rather than a landed extension. October 19 v1 content and source-derived exercise are prepared. Team acceptance of the schema boundary and steady register use remain external decisions.

Prepared with AI assistance. The sources directory records the executed checks and source review; independent verification by Ishaan or a lane owner is still to be recorded. This update has not been committed, pushed, published or sent to teammates.

## Architecture evidence

- [Trainium2 overview v2.31.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html) and [NKI architecture guide v2.31.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/architecture/trainium2_arch.html): explicit on-chip memory and DMA; shapes/control flow require separate treatment from static allocation assumptions.
- [SN40L architecture paper v2](https://arxiv.org/html/2405.07518v2) and [Kernel Looping paper v1](https://arxiv.org/html/2410.23668v1): tiered residency and compiled boundary behavior. Arshia's source base provides the broader rate/conflict/access register.
- [Cerebras SDK conceptual view 1.3.0](https://cerebras-sdk-docs-130.netlify.app/computing-with-cerebras): data-arrival tasks and explicit dataflow. This establishes a family-level boundary; no Cerebras LLM mapping or performance validation is supplied.
- Current GPU implementation evidence is pinned in the [family-assumptions audit](family-assumptions-audit.md). These facts describe implementation choices, not universal execution laws.

## Requirements added after GPU validation

| ID | Requirement | Acceptance condition |
| --- | --- | --- |
| R12 | Distinguish a resource proxy, a conditional bound and an empirical latency estimate in every output surface. | Machine-readable and text outputs expose meaning and schedule/traffic/rate assumptions. Unknown or uncalibrated empirical latency is not fabricated. Existing numeric keys retain their meaning. The text CLI's current warning is credited. |
| R13 | Version physical lowering by backend and preserve missing coverage. | Canonical logical operations may map to multiple physical kernels or one fused region. Launch, scratch, GQA expansion and cache-write assumptions identify engine/version/artifact. A logical-node count alone does not establish physical work or boundary count. |
| R14 | Validate an empirical latency model separately from a peak model. | If a latency estimator is introduced, define its target interval, record runtime/hardware/precision, split fitting and held-out measurements, and retain uncertainty/coverage. Qwen residuals are diagnostic evidence, not a fitted correction. |

The October 11 GPU evidence exercises one measured execution family. Trainium2/SN40L exercises remain source-derived review adapters. Two physical A100 devices are not two execution families, and Qwen timings do not establish GLM/Kimi or spatial model accuracy. The choice between shared core and sibling spatial plan remains provisional.


## Local integration status

Ishaan authorized local integration on October 11. The optional `gitm/planner/tiered_memory.py` and `gitm/planner/backend_report.py` APIs and the registry reporting change are now present in this checkout with native tests. They are uncommitted and have no new maintainer approval. See [integration status](completion-status.md) for verification scope and remaining gates.

The [acceptance record](mission-acceptance.md) maps each mission requirement to its local artifact, verification and remaining external gate. It supersedes any informal percentage estimate as the completion record.
