# Local implementation and review boundary

Status update October 13, 2026: Ishaan authorized local commits. Earlier October 11 references to uncommitted work describe that snapshot. Maintainer review, pushing, publication and external contact remain separate; none is established by creating local commits.

Three logical code changes are integrated locally, with a fourth group containing the mission documents and compact evidence:

1. Registry reporting adds explicit result semantics and conditions the serial resource proxy. The legacy values are retained; text/JSON consumer compatibility still needs review.
2. Optional Qwen backend reporting describes KV-head materialization, preserves canonical graph values and lists unpriced components. Physical HBM traffic and empirical latency remain unknown.
3. Optional tiered-memory accounting defines named resources, physical traffic/unknown inputs, directed transfers and residency, with an exact legacy adapter and native tests. It provides memory service bounds, not a step scheduler.
4. The register, assumptions audit, proposal, requirements and source-derived exercises live under `docs/cross-architecture/`. They retain source and review boundaries.

These implementations do not replace hardware peak rates with fitted multipliers, force spatial traffic through a GPU graph, or make K3 falsely loadable. Existing catalogue entries retain their current path. Backend reporting rejects prefill/speculation because the inspected evidence is for a cached single-token forward.

The separate GPU profile associates 72 reshape-triggered clones with 72 direct-copy kernels, totaling about 1.21 ms in the profiled eager step. Attribution to two `repeat_kv` materializations per layer is a source/trace inference: the trace lacks input shapes and Python stacks. This diagnostic is not added to unprofiled medians, and it does not explain the entire timing residual.

Maintainer review should settle the reporting interface, the optional API boundary and the use of evidence-qualified resource semantics before landing. Rahul’s older proposal feedback is not approval of these revisions. No review request has been sent. All changes remain local and uncommitted.

## Subsequent storage expression change

A fifth review group now adds explicit hybrid convolution channels/state precision and independent GLM index-key dtype/scale bytes. Defaults retain the old paths; the new consumers cover residency, reads and writes, with 29 focused tests. See [partial gap fixes](partial-gap-fixes.md). Current selected verification is 512 passing tests and 14 unchanged legacy cases. This group is also local and uncommitted.
