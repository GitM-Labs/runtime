# Cross-architecture ceiling work

Ishaan Gubbala’s research lane: execution concepts shared across GPU, compiler-managed and spatial families. Locally integrated October 11, 2026, against `16d6d44c984d5446b6fc124550c995649cf96106`. Local commits were authorized October 13, 2026. Maintainer review and publication remain pending; this authorization does not include pushing, opening a PR or external contact.

## Mission documents

- [Schema-gap register](schema-gap-register.md): K3 flags reconciled with current code; Arshia’s SN40L cases, evidence, consequences and proposed expressions.
- [Family-assumptions audit](family-assumptions-audit.md): invariant, parameterized and family-specific assumptions.
- [Tiered-traffic proposal](tiered-traffic-proposal.md): the confirmed CF-01 gap, semantics, migration and acceptance examples.
- [Cross-family requirements v1](cross-family-requirements-v1.md): three-family scope and the provisional shared-core/spatial-sibling position.
- [Two-family exercise](two-family-planner-exercise.md): source-derived Trainium2 and SN40L evidence, with explicit unknowns.
- [Mission acceptance record](mission-acceptance.md): criterion-by-criterion evidence and outstanding decisions.
- [Completion status](completion-status.md): preparation, local integration, approval and deployment validation are separate states.

## Local implementation

`gitm.planner.tiered_memory` provides optional memory resources, named tiers, directed transfers, provenance, service bounds and residency validation. It does not schedule a whole step. The legacy GPU adapter calls the original scalar roofline exactly; no model catalogue entry opts into named tiers automatically.

`gitm.planner.backend_report` provides a source-derived KV-expansion report for the pinned Qwen2.5-3B / Transformers 4.46.3 / PyTorch 2.4.1 / SDPA / BF16 / StaticCache decode case. It maps the intermediate to canonical attention nodes while retaining unknown physical HBM traffic, launch count and latency. Prefill, speculation and unsupported configurations are rejected.

The registry now identifies its aggregate as a serial resource proxy in text and JSON. JSON retains its existing fields and adds `prediction_semantics`; strict consumers that reject extra fields and text parsers expecting the old literal “floor” need interface review. Numeric predictions are unchanged in the recorded compatibility matrix.

All code and document changes were agent-written. Human approval of local integration is recorded in the conversation; maintainer review of these revisions is still pending. Rahul’s earlier positive response covers the earlier proposal and execution-region request, not these later implementations.

## Subsequent storage fixes

[Partial gap fixes](partial-gap-fixes.md) documents explicit hybrid convolution channels/state dtype and independent GLM index-key dtype/scale bytes. These optional fields preserve existing defaults. The latest selected suite passes 512 tests; historical integration counts remain labeled in their original records.

## Reproduce

Use Python 3.10+ with the project dependencies and pytest installed. From the repository root:

```text
python -m scripts.reconcile_cross_architecture_k3
python -m scripts.cross_architecture_memory
python -m pytest tests/test_tiered_memory.py tests/test_backend_report.py tests/test_prediction_semantics.py tests/test_cross_architecture_memory.py tests/test_cross_architecture_gap_fixes.py
python docs/cross-architecture/sources/arshia/deliverable-2-v32-decode-ceiling/v32_sn_ceiling.py check
```

The example calls the integrated planner API with normal package initialization. Trainium2 has unknown SBUF/DMA rates; SN40L uses selected activation/reuse quantities and conditional rates. Complete region timing and whole-step latency remain unavailable. Neither case is a deployed model prediction.

## Evidence

The [evidence index](sources/README.md) links the supplied Arshia artifacts, historical reconciliation results, current-source comparison and real Qwen timing/numerical summaries. Original full GPU traces, failed-run records and reproducibility bundles remain preserved in the separate local review package; they are not copied into the repository as large trace files. Historical numbers retain their original pins.

Local integration does not establish a shared live register, maintainer acceptance, team adoption or measured cross-family deployment validation. The work is recorded in local commits. No push, pull request or external message has occurred.

## Local commit organization

The local branch `ishaan/cross-architecture-ceilings` separates result semantics, the pinned backend report, named memory-resource accounting, independent state/cache storage, and mission documents with source evidence. Commit messages describe what changed, why and verification scope. Code and authored documents are agent-written; the original K3 and Arshia sources are unchanged. Human authorization to commit does not claim independent verification or maintainer acceptance.

The 18 Python files in the prepared change set were checked against the recorded 512-test verification snapshot before committing. No test suite was rerun for this documentation and commit operation. Earlier review archives remain historical snapshots and were not rewritten.

## Reconciled with current main

On October 13 the branch was rebased onto `12dd8b4002125447777dcb2ea20f9cd5fa772032`. All five commits replayed without conflicts. The full local Python 3.12 suite passes with 2400 passed and 6 skipped; all 14 legacy comparisons match. Ruff 0.12.11 passes, excluding only the unchanged Arshia source-artifact directory. See [reconciliation record](reconciliation-2026-10-13.md) for scope, skipped cases and the preview of related PR #176. The earlier 512-case result remains historical, not additional coverage.
