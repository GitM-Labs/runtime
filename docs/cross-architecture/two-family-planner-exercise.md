# Two family planner exercise

Status update October 13, 2026: Ishaan authorized local commits. Earlier October 11 references to uncommitted work describe that snapshot. Maintainer review, pushing, publication and external contact remain separate; none is established by creating local commits.

## Evidence update on October 11

Planner statements were rechecked on October 11 against default-branch head `16d6d44c984d5446b6fc124550c995649cf96106`: none of the three intervening commits changed `gitm/planner/`. Measurements retain their original October 6 planner pin `05abd11b866bdabe2619b5f289db52d647eb55ad`. The current kernel-identity documentation and ROCm status were separately read; see the current `docs/kernel_identity.md`, `docs/rocm_correlation.md` and `sources/repository-comparison.json`. See [validation interpretation](validation-interpretation-update.md) and [implementation review](implementation-review.md).

Completed 2026-10-06. Scope: source-derived representation exercise through a local review adapter calling the pinned runtime's unmodified `roofline()` and `Graph.serial_chain_s`. This is an executable proposal demonstration, not shipped Trainium or RDU support. No measured trace, NEFF, SN40L compiled mapping or device execution was supplied or generated.

## Shared abstraction

Physical bytes belong to named resource demands and directed transfer events. A transfer charges source read, destination write and transfer engine; consumers charge their own later reads. Demands sharing a resource add before pricing. Residency is independent of transfer volume. Unknown rates remain unknown. The same adapter is used for both families.

The script imports the actual planner modules from the read-only HTTPS archive of main `05abd11b866bdabe2619b5f289db52d647eb55ad`. It bypasses the package-level runtime/GPU imports, not planner functions. No source files in that archive were edited. This distinction makes the CPU exercise reproducible without installing the runtime or changing the working repository.

## Inputs and outputs

| Case | Source and scope | Result | What it proves |
| --- | --- | --- | --- |
| Trainium2 HBM to SBUF | AWS NKI v2.31.1 example has a 128 × 512 two-byte tile; 131,072 bytes. SBUF capacity 28MiB per physical core. AWS reports 2.9TB/s in the overview and 3.0TB/s in the NKI guide; use the faster reported peak as an optimistic envelope. | HBM service floor 43.7ns using the whole chip's optimistic peak. SBUF and DMA service remain unknown. Whole-region time is withheld. | The flow retains three resource demands and a named scratchpad residence; it does not assume one core receives chip-wide bandwidth. |
| SN40L model activation and reuse | SN40L paper v2 §V and Fig9 provide the DDR → HBM activation pattern. Select 64MB and ten later reads as a small scenario, not a deployed model. Arshia's preferred rates are DDR 100GB/s and HBM 1.6TB/s. Paper HBM capacity is explicitly 64GiB. | Conditional endpoint service: DDR 0.64ms; HBM write 0.04ms; ten HBM reads 0.40ms. Activation engine and complete-region time remain unknown. | Backing DDR is read once; repeated use belongs to HBM. It does not claim these component times add to a measured token time. |

The SN40L SRAM figure is a lower bandwidth bound (>100TB/s), so bytes divided by that value yields an upper bound on ideal service time. It cannot establish a minimum latency. Source units stay attached: paper GiB/TiB values and product GB values are not interchangeable deployment capacities. Aggregate capacity passes in the examples do not establish bank/layout feasibility or placement legality.

## Acceptance examples exercised

The script passes 15 assertions: transfer endpoint conservation, named residence, unknown transfer handling, repeated reuse, lower-rate bound direction, shared read/write contention, overlapping and adjacent lifetimes, invalid-byte rejection, the real planner's dependency-chain handling, and the unchanged scalar roofline equation. A synthetic dependent 64MB activation at 64GB/s followed by 0.2ms compute produces a 1.2ms chain floor through `Graph.serial_chain_s`.

This covers a source-derived cross-family exercise of the shared resource concept. The interpretation should be reviewed as part of requirements v1. It does not demonstrate production integration, a workload's actual placement, catalogue-wide regression compatibility, or a measured cross-family ceiling. Those are separate sources/implementation gates.

## Current planner reconciliation

At context 8,192, current V3.2 detection still returns `sparse_moe`. Its bytes are 17.28× Arshia's conditional TP ledger. The explicitly constructed GLM candidate differs by -2.43% and emits 1,273 nodes. These compare model ledgers, not measurements. Unknown collectives remain flagged.

At context 131,072, current V3.2 detection still returns `sparse_moe`. Its bytes are 127.61× Arshia's conditional TP ledger. The explicitly constructed GLM candidate differs by +14.90% and emits 1,273 nodes. These compare model ledgers, not measurements. Unknown collectives remain flagged.

At the October 6 baseline, all 15 flagged scalar K3 keys were absent from `HybridMoEModelSpec`. The later local convolution fix accepts two of them; the current reconciliation records 13 absent flagged keys and 14 total rejected scalar keys. The router precision override has existing support; the two submitted state override names remain unmatched. GPU `WeightExecution`, `TierTraffic` and the GLM HBM `MemoryFit` are now useful precedents, but do not supply named memory-tier topology or a spatial plan.

Arshia's unmodified `v32_sn_ceiling.py check` independently reruns successfully: 66/66 checks. Her full September gap probe initially could not import runtime dependencies on the available Python installations; its failed output is not a result. The local script directly re-exercises family detection and both graph paths with her original config and ledger, without substituting planner functions. It does not claim to rerun every September probe.

## Reproduce and evidence

The commands below reproduce the October 11 integrated example and Arshia calculator. The 15-assertion October 6 adapter is historical and is preserved in the separate original review archive; these commands do not rerun that adapter.

From the repository root in an environment with project dependencies, use Python 3.10 or newer:

```text
python -m scripts.cross_architecture_memory
python docs/cross-architecture/sources/arshia/deliverable-2-v32-decode-ceiling/v32_sn_ceiling.py check
```

The captured results are `sources/planner-exercise-results.json` and `sources/arshia-checks.txt`. The planner pin, supplied archive hashes, current source and code differences from September are retained beside them. The historical exercise used no Git command. The present local integration used a read-only clone and local patch application with Ishaan’s approval; no commit, push or external message occurred.

- [AWS Trainium2 hardware overview v2.31.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/about-neuron/arch/neuron-hardware/trainium2.html), Memory table.
- [AWS NKI Trainium2 guide v2.31.1](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/architecture/trainium2_arch.html), NeuronCore-v3 memory and HBM2SBUF DMA example. The overview and NKI guide differ on some rounded rates and CC-core counts; this exercise avoids those unresolved counts.
- [SN40L architecture paper v2](https://arxiv.org/html/2405.07518v2), Fig9 and §V. This documents a CoE runtime pattern, not DeepSeek V3.2 placement.
- Arshia source base v2.1, P-HBM-BW, P-DDR-BW, P-SRAM-BW; source-based scenario rates, not newly measured throughput.

## October 11 measured evidence boundary

Additional GPU experiments are now available: two A100 sessions for recurrence/pipeline kernels, a real GLM catalogue byte trace, and real Qwen checkpoint decode measurements plus a numerical diagnostic. These do not alter the Trainium2/SN40L source-derived exercise or convert its inputs into measurements. The Qwen model is measured on one execution family and is not a deployed instance of the tiered adapter.

Retain the 15 semantic assertions and Arshia's 66 calculator checks as their original CPU results. Neither count establishes physical traffic or measured cross-family model prediction accuracy.


## Local integration status

Ishaan authorized local integration on October 11. The optional `gitm/planner/tiered_memory.py` and `gitm/planner/backend_report.py` APIs and the registry reporting change are now present in this checkout with native tests. They are uncommitted and have no new maintainer approval. See [integration status](completion-status.md) for verification scope and remaining gates.

## Integrated exercise acceptance

The current example calls `gitm.planner.tiered_memory` under normal package initialization. Its three native tests are `tests/test_cross_architecture_memory.py`; the resource-accounting rules are additionally covered in `tests/test_tiered_memory.py`. These are separate from the historical 15 assertions. The [captured integrated output](sources/integrated-family-exercise.json) contains Trainium2 and SN40L inputs/results with provenance and missing-rate boundaries. No source-derived scenario is presented as a measured platform result.
