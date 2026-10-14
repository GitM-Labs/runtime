# Proposal for CF-01: name the memory path

Status update October 13, 2026: Ishaan authorized local commits. Earlier October 11 references to uncommitted work describe that snapshot. Maintainer review, pushing, publication and external contact remain separate; none is established by creating local commits.

## Evidence update on October 11

Planner statements were rechecked on October 11 against default-branch head `16d6d44c984d5446b6fc124550c995649cf96106`: none of the three intervening commits changed `gitm/planner/`. Measurements retain their original October 6 planner pin `05abd11b866bdabe2619b5f289db52d647eb55ad`. The current kernel-identity documentation and ROCm status were separately read; see the current `docs/kernel_identity.md`, `docs/rocm_correlation.md` and `sources/repository-comparison.json`. See [validation interpretation](validation-interpretation-update.md) and [implementation review](implementation-review.md).

Status: **reviewed proposal with requested table edit addressed; implementation integrated locally, uncommitted and awaiting maintainer review**. Updated 2026-10-06 against main `05abd11b866bdabe2619b5f289db52d647eb55ad`. Rahul told Ishaan “the proposal looks good” and requested an execution-region row; that row is present in the proposal’s Execution region section. The receipt date was not supplied. This version adds current-planner reconciliation and explicit transfer-resource/bound semantics. Those additions are available for follow-up review. Arshia's source base supports the architecture case; no DeepSeek V3.2 compiled placement or platform trace has been supplied.

## The failure

`roofline.roofline(op, flops, bytes_moved, hw, dtype)` computes `max(flops/peak, bytes_moved/hw.peak_mem_bw_bytes_per_s, launch_time)`. `RooflinePrediction.bytes` is untyped. Three graph builders also price interconnect by making a copy of `HardwareSpec` with memory bandwidth replaced by link bandwidth. That trick works only while one node's traffic is known to use one link and one residence is assumed. It gives no place to say whether a read came from SRAM, HBM or DDR, whether a model first moved between them, and which tier must have spare capacity.

This is a confirmed **representation** gap with two independent architecture sources. [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html) has HBM and SBUF, and its [NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/architecture/trainium2_arch.html) describes PSUM and HBM-to-SBUF DMA. The [SN40L architecture paper](https://arxiv.org/html/2405.07518v2) documents SRAM/HBM/DDR, compiler-managed placement, and model activation from DDR into HBM. It does **not** prove how a particular customer model is placed. GPU HBM and caches provide a third relevant machine, but the proposal does not equate automatic GPU cache hits with explicit scratchpad placement. Current `HardwareSpec.memory_bytes` and `glm_graph.memory_fit` provide one GPU HBM fit ledger. `TierTraffic` already separates weight payload/scales, activation, temporary and interconnect bytes. These are useful existing mechanisms to reuse, but neither describes a named tier topology or directed transfer path. This is enough to propose the schema, not enough to predict an SN40L deployment.

## Semantics (smallest useful extension)

Add a resource-qualified demand to an execution **region** without committing to what a region is called on each machine:

```text
MemoryTier(name, capacity_bytes, scope, read_resource_id, write_resource_id)
MemoryResource(resource_id, bandwidth_value_or_range, bound_kind, rate_kind, scope, provenance)
TransferPath(source_tier, destination_tier, engine_resource_id, startup_s, provenance)
RegionTraffic(read_bytes_by_tier, write_bytes_by_tier, transfers)
Residency(tensor_or_bank_id, tier, live_region_start, live_region_end, bytes)
```

Names are scoped to one hardware/compiled plan, never global string conventions. `read_bytes_by_tier` and `write_bytes_by_tier` count *physical* traffic during that region, not logical tensor volume. A transfer is one directed source-to-destination movement with its own startup and engine resource. State in SRAM reused ten times is not charged as ten HBM reads. A model bank residing in DDR but activated into HBM has a transfer at activation and HBM reads during execution, not a DDR read on each op. Residency lists capacity occupancy over time; it is not inferred from per-op bytes. Provenance on each demand is `measured`, `compiler-derived`, `source-derived` or `unknown`, with an artifact identifier and, for estimates, a range. An unknown demand **fails closed** for ceiling claims instead of silently becoming zero.

### Resource accounting rule

Treat a tier's read and write resource IDs as the **same ID** by default, making its declared bandwidth an aggregate resource shared by reads and writes. Use separate IDs only when a platform artifact explicitly identifies separate ports or engines; separate counters alone do not prove they can overlap. Each transfer charges its source read resource, destination write resource, and transfer engine resource. These are three demands from one event, not three copies of logical data. A consuming read is a separate event. Region read/write maps exclude endpoint bytes already recorded by its transfer events. If source read and destination write use the same aggregate resource, both demands add there; an event never records its endpoint demands a second time. Sum demands that contend for the same resource before dividing by its rate. The maximum of required service lower bounds is an optimistic resource floor even without overlap evidence; it does not prove those services can run concurrently or that the floor is attainable. A duration or attainable ceiling needs scheduling/concurrency evidence. Missing resource demands/rates make the complete region claim unknown, while known component bounds can still be reported. `rate_kind` distinguishes peak from sustainable rates.

SN40L evidence now establishes more than the existence of its tiers: its compiler performs static placement, normally uses HBM while the working set fits, and can spill lower-priority symbols to DDR. The source base also records model-library activation from DDR into HBM. Treat those as family-level, source-derived behavior; the DeepSeek V3.2 placement, transferred bytes, and activation policy remain unknown without its compiled plan. The reported SN40L SRAM/HBM/DDR capacities and rates can seed a hardware description with provenance and bounds, but the DDR sources describe different configurations (current product 512GB versus paper up to 1.5TiB); select a configuration or keep capacity unknown. Do not turn distinct configurations into an observed statistical interval. The paper states HBM as 64GiB, while some later tables say 64GB. Keep source units: the local scenario explicitly uses paper-v2 64GiB; a deployment fit needs its actual capacity. See Arshia's `1_sn40l-source-base.md` entries P-SRAM-CAP, P-SRAM-BW, P-HBM-CAP, P-HBM-BW, P-DDR-CAP, P-DDR-BW, X-ALLOC, X-PLACE and D-DDR-RES.

This proposal defines only a per-region resource bound and residency validation. It does not define a whole-step scheduler. Dependencies must identify when a transfer completes before its consumer can run. Arshia's evidence says SN40L can pipeline communication with compute and overlap these with SRAM activity; this supports modeling concurrency when a compiled plan or measurement identifies it, not assuming every tier or transfer runs concurrently. Preserve the existing legacy serial aggregation until a family schedule supplies that behavior, and do not label it a whole-step lower bound.

The boundary cost also needs a family-specific expression. SN40L compiled-kernel boundaries include synchronization and bitstream reload; kernel looping can reduce repeated calls. Published Llama results do not establish the number or cost of boundaries for DeepSeek V3.2, so a GPU launch count cannot stand in for it. A future spatial schedule should carry compiled-region identity and boundary cost, and distinguish first-token fill/drain from steady-state initiation interval.

For one region with verified concurrent engines, resource lower bound is `max(compute demand / matching engine rate, tier read/write bounds, transfer-path demand, dependency/launch floor)`. For competing demands on the *same* physical resource, aggregate their bytes before division; do not `max` them independently. A transfer's `startup_s` belongs once per actual transfer, not to every consumer. Treat rate type (peak or sustainable) independently of bound direction (point/lower/upper/range). A peak acting as an upper rate produces a service-time lower bound. A documented lower bandwidth bound produces an upper bound on ideal bytes/rate service, not a latency lower bound or a measured duration. A point estimate is conditional. Propagate both interval endpoints and parent provenance; unknown is not zero. This region bound is **not** a whole-step schedule: `Graph.total_pred_s` remains the legacy serial sum until a schedule supplies overlap and contention. That sum should not be called a whole-step lower bound when regions overlap, even though the CLI currently calls it a “floor.” In a spatial plan, a sibling schedule will distinguish first-token fill/drain from steady-state initiation interval. No compiler mapping, no asserted overlap.

## Existing entries and migration

Keep `roofline(op, flops, bytes_moved, hw, dtype)` and `RooflinePrediction.bytes` unchanged for old callers. Default legacy traffic is `read_bytes_by_tier={"device_hbm": bytes_moved}` for the current GPU hardware adapter, with **no transfer** and the old scalar BW. The proposed extension must preserve every existing YAML and its `t_pred_s` bit-for-bit relative to the October 6 pin by using the old path unless an explicit tiered plan is supplied. Do not auto-map a Neuron or RDU catalogue entry through this default. Instead require named tiers, capacity and either per-region traffic evidence or `unknown`; a compiler artifact can populate them. Add an optional `traffic` field to a future region object or prediction input, not to existing model-spec dataclasses, because model shape does not determine physical placement. Keep `ShardingConfig` logical; transfer routing belongs to deployment evidence. Reuse `estimated` only for approximate costs, and add a distinct unknown-tier flag so it cannot be confused with an approximation that has been quantified.

The first implementation PR should contain the data types, validation, a standalone tier-bound calculator and legacy adapter. It should **not** rewrite `hybrid_graph`, assume SN40L bandwidth per PMU, or modify the catalogue's old YAMLs. The next PR can consume a real compiler/profile artifact, map region IDs to logical ops, and only then turn on tiered predictions for that machine. This split is deliberate: a new field with invented placement would make a false model load cleanly.

## Tests that today's scalar cannot pass

1. Legacy compatibility: the same existing `hybrid`, `sparse_moe` and `glm_moe_dsa` entries produce identical node count, bytes, bounds and `total_pred_s` with the adapter; no YAML migration is required.
2. Toy activation: a 64 MB transfer from DDR to HBM at 64 GB/s costs at least **1 ms** plus startup, even if HBM runs at 1 TB/s. A 0.2 ms compute region with 64 MB HBM traffic has a region floor of 0.2 ms; an explicitly dependent activation makes the combined floor at least 1.2 ms. The current scalar-BW model sees only HBM's 0.064 ms and cannot distinguish the two situations. These are synthetic numbers, not SN40L measurements.
3. Reuse: moving 64 MB once and reading it ten times from an on-chip tier charges one transfer plus ten on-chip reads. It does not charge 640 MB to DDR or HBM. Verify conservation of transfer bytes at both endpoints and no double charge on the destination read unless the consuming region actually reads it.
4. Capacity: two simultaneously live 40 MB buffers on a 64 MB tier fail validation; non-overlapping lifetimes pass. A model bank in DDR does not count as HBM-resident until activation; evicting it removes HBM occupancy without erasing DDR backing.
5. Resource collision: two 32 MB movements over the same 64 GB/s path require at least 1 ms aggregate transfer time; two independent paths may overlap only when their engine IDs and dependencies permit it.
6. Unknown placement: a Neuron or SN40L region without a tier assignment produces an explicit `unknown` result and no numerical whole-step ceiling. It must not inherit `device_hbm` by convenience.
7. Measured validation gate: compare measured GPU HBM/transfer bytes with the legacy adapter and, separately, a Neuron profile or SN40L compiler placement plus trace with the tiered path. This establishes measured deployment validation. The mission also permits a clearly labeled source-derived planner exercise where platform access is absent. The accompanying local exercise covers Trainium2 and SN40L through a review adapter and real planner primitives; it is not production family support or a deployed workload ceiling.

## Execution region

An execution region is the unit to which physical resource demands and optional boundary costs are attached. A family plan maps logical operations to that unit. A GPU kernel, a Neuron compiler region, and a spatial compiled kernel can each supply a region identity, but do not acquire identical timing or observation semantics. This is the concept row Rahul requested. Transfers can occur before a consuming region; their dependency must remain explicit.

| Concept | Meaning | Current expression and boundary |
| --- | --- | --- |
| Execution region | A family-defined unit to which resource demand is attributed; logical operations map into it through a family plan. | `RegionTraffic.region_id` identifies the accounting unit. It does not infer a GPU kernel, compiler region, spatial stage, scheduling order or boundary cost. |
| Memory resource | One physical service resource shared by all demands assigned its ID. | `MemoryResource` with scope, bandwidth semantics and provenance; demands aggregate before division. |
| Memory tier | A named storage domain with capacity and read/write resource references. | `MemoryTier`; aggregate capacity does not establish bank or layout legality. |
| Transfer | One directed movement charging source read, destination write and engine service. | `TransferPath` and `Transfer`; consuming reads are recorded separately. |
| Residency | An allocation occupying a tier over a half-open lifetime. | `Residency`; allocation identity is distinct from tensor identity, permitting explicit copies. |
| Evidence | The source and limitations of each input. | `Provenance`; measured, compiler-derived, source-derived, scenario or unknown. |

The execution-region row addresses Rahul’s requested table addition in this repository version. His earlier positive response is not a review of this implementation table.


## Review status and evidence boundaries

The proposal has Rahul's positive review and the requested row is included. The October 6 refinements reuse `TierTraffic`, `WeightExecution` and the existing HBM fit work rather than duplicating them. The local adapter exercises source-derived Trainium2 HBM-to-SBUF and SN40L DDR-to-HBM activation semantics and 15 acceptance examples. That earlier exercise did not install types in runtime or establish measured validation. The local integration now supplies explicit optional types; catalogue entries remain unchanged. See [exercise and results](two-family-planner-exercise.md).

The first production change remains a small opt-in extension with explicit legacy compatibility checks, reviewed interfaces and evidence-backed family lowering. The optional implementation is now integrated in this local checkout; no commit, push, publication or new maintainer approval has occurred. The register distinguishes the implemented GPU precedents from the remaining named-tier/transfer/residency gap. Arshia's evidence supports the family concepts, while V3.2 placement, boundaries and deployed cache dtypes stay unknown.

## October 11 validation and review boundary

New evidence narrows this proposal. The actual GLM catalogue path already supports weight amortization, distinct-expert reuse, compressed KV traffic and an aggregate HBM fit ledger. The two-device GPU recurrence/pipeline experiments do not justify adding a generic reuse field or replacing the roofline equation. CF-01 remains about named physical tiers, directed transfers, residency and resource accounting across execution regions.

The real Qwen timing residual belongs in GPU-01, the backend-lowering review, and GPU-02, the machine-readable result semantics review. It is not measured evidence for an SN40L tier extension. The text CLI already distinguishes its floor from a target. Empirical latency remains a separate possible product, requiring explicit engine, precision, shape, hardware and held-out validation.

Rahul's earlier positive feedback applies to the earlier proposal and requested execution-region row. These October 11 changes and the new reporting patch have not been reviewed by him. The proposal remains a reviewed earlier version with subsequent refinements awaiting disposition; no new implementation or approval is claimed.


## Local integration status

Ishaan authorized local integration on October 11. The optional `gitm/planner/tiered_memory.py` and `gitm/planner/backend_report.py` APIs and the registry reporting change are now present in this checkout with native tests. They are uncommitted and have no new maintainer approval. See [integration status](completion-status.md) for verification scope and remaining gates.

## Exact local API semantics

The design sketch above predates the local API. The implementation names the startup input `startup_floor_s`: it is an additive lower bound on service occupied on the transfer engine per nonzero transfer. It is not an arbitrary host startup latency or a delay that can overlap payload service. A platform unable to establish that meaning must supply `None`. A startup floor cannot tighten an upper bound on ideal payload service. Zero-byte transfers add no startup service.

`RegionTraffic.coverage_complete` and the residency validator’s corresponding argument default to false and require actual booleans. Unknown demands preserve their known component bytes but prevent a complete memory floor. Point rates yield conditional payload service, lower-rate bounds yield ideal payload-service upper bounds, and upper-rate bounds yield service floors. The output always withholds whole-step and attainable-region latency. Explicit scenario provenance is allowed for synthetic examples and is never relabeled measured.

The legacy adapter delegates directly to the existing roofline function; it does not reinterpret scalar traffic as known physical HBM reads. The proposed `device_hbm` default in the migration sketch is therefore a possible future adapter convention, not a mapping performed by this implementation.
