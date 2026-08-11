# Graceful Fallback and Runtime Wiring Audit

## Executive summary

Status: **in progress**. This ledger is the primary deliverable for the audit of
`gitm/` and `scripts/`. Findings are ranked by the likelihood that a fallback can
turn missing knowledge into a confident wrong result, with answer-deciding byte
traffic and dominant expert terms ranked above non-binding estimates.

Highest-severity masks closed: **0 so far**. Wiring gaps confirmed: **3 so far**.
Deferred findings: **none so far**.

The worktree already contained uncommitted scheduler/serve changes and two new
expert-signal files before this audit branch was created. They are preserved and
treated as pre-existing work until their ownership and relevance can be separated;
they will not be silently absorbed into an audit commit.

## Finding ledger

| Rank | Status | Severity | Location | Distorted term / contract | Surfacing state | Failure scenario | Disposition |
|---:|---|---|---|---|---|---|---|
| 1 | open | critical | `gitm/scheduler/loop.py:165-204, 576-586` | Entire sparse-MoE execution graph; especially dominant expert weight bytes and hybrid-attention/KV terms | silent | A live V4 engine is converted into the legacy `ModelSpec` and sent to `predict_graph`; the production loop never calls `spec_from_hf_config`/`predict_moe_graph`, so it can issue optimization claims against the wrong architecture. | Add one execution-graph dispatcher shared in intent with attach: recognize/normalize/validate the final config, build the sparse graph, and refuse graph-based claims with named diagnostics when it cannot be priced. |
| 2 | open | critical | `gitm/scheduler/loop.py:165-204` | Model identity and every residual | silent | Any parse bug or version-drift attribute error is caught by `except Exception`, returns `None`, and becomes the plausible Llama-2-7B default graph. | Narrow expected absence handling; return a typed resolution result/diagnostic and degrade to measurement-only rather than a default model. |
| 3 | open | high | `gitm/planner/context.py:156-169`; `gitm/scheduler/loop.py:576-590`; `gitm/serve/attach.py:477-487` | Compute and HBM denominators for every node | silent | Unknown/no GPU SKU makes `hardware_spec_for(None)` return an A100 spec; artifacts then write `hardware: A100-SXM4-80GB` as though detected on a different or absent GPU. | Preserve fail-open only with explicit hardware-fallback provenance and warnings; graph-based claims should not present the fallback SKU as observed hardware. |
| 4 | open (planner flag fixed; boundary surfacing pending) | high | `gitm/planner/roofline.py:42-54`; sparse graph byte builders | Weight/KV/activation bytes; dominant decode-binding term | planner FLAG added; production consumers pending | Unknown dtype returns 2-byte bf16. Graph peak fallback may incidentally flag an unknown compute dtype, but scalar sizing calls and mixed-byte nodes cannot identify that their byte width was substituted. | Added `weight_bytes_is_fallback`, per-node `bytes_are_fallback`, and `Graph.has_fallback_bytes` with known/unknown/mixed regression tests. Still must share final-spec validation across attach/loop and surface flags in artifacts/reports before closing. |
| 5 | open | high | `gitm/scheduler/loop.py:322-354`; `gitm/optimizer/report.py:20-29`; report template | Residual magnitude shipped on every Claim | silent clamp | A 10x/18x model error and a 2x error both render `+100%`, hiding that the model is broken and repeating one aggregate as if claim-specific. | Advisor-approved contract: preserve raw residual, derive capped display + saturation, render capped and raw values; label run-level versus target-op scope; reject/surface non-finite values. |
| 6 | open | high | `gitm/optimizer/monitor.py:83-184` | Residual population / coverage | silent drop | Unclassified kernels and classified ops absent from the graph are skipped, so the loop can report clean residuals over a small, biased fraction without matched/total coverage. | Add total/classified/matched kernel and duration coverage to `Residuals`; serialize and print/report warnings when incomplete. Test unknown and good paths. |
| 7 | open | high | `gitm/scheduler/loop.py:309-319` | Dense-path MoE expert weight bytes | silent | An unknown quantization method is ignored, leaving `weight_dtype_bytes` at the bf16 default; dominant expert traffic can be overstated while claims look fully priced. | Replace numeric-only extraction with named dtype/provenance and refuse or flag unknown methods; superseded by the sparse dispatcher where applicable. |
| 8 | open | medium | `gitm/planner/moe_graph.py:498-528`; `gitm/serve/model_config.py:274-276` | Expert weights, often the dominant term | silent substitution | Missing `expert_dtype` inherits linear `weight_dtype`; on mixed-precision checkpoints this can misprice most resident and fetched bytes. Official V4 Flash configs checked so far explicitly declare the field (Flash=`fp4`, Base=`fp8`), but foreign/uniform MoEs may omit it. | Record whether expert dtype was explicit or inherited; require it for model families/quantization layouts where mixed precision is possible, otherwise surface the inheritance in provenance. |
| 9 | open | medium | `gitm/planner/graph.py:123-133` | Zero-time byte-moving nodes | flag exists but name can lose coverage | `has_unpriced_collectives` scans all nodes, not only collectives; a future “cleanup” to match the name would silently remove the general zero-pricing net. | Rename general predicate (with compatibility alias if needed) or split general and collective-specific intent; trace all consumers. |
| 10 | open | high | `gitm/scheduler/loop.py:587-592` | Prediction trust diagnostics | unconsumed | `predicted_graph.json` writes only node count, total time, and hardware; peak fallback, byte fallback, unpriced nodes, estimates, default batch/model, and provenance do not reach the loop artifact/report. | Serialize machine-readable graph diagnostics and propagate them into the human report and any claim gate. |

Status values: `open`, `fixed`, `deferred (reason)`, or `won't fix (reason)`.

## Seed-finding verification

| Seed | Verification | Status |
|---:|---|---|
| 1 | Unknown weight dtype falls back to bf16 without a bytes-side flag. | Confirmed; incidental peak fallback is insufficient for scalar/mixed-byte paths. Advisor design recorded. |
| 2 | Loop MoE dispatch lacks the attach path's dtype priceability gate. | Confirmed and broader: current main has no sparse loop dispatcher/caller at all. |
| 3 | Missing `expert_dtype` inherits `weight_dtype`, potentially mispricing the dominant expert term. | Confirmed in parser. Official DeepSeek V4 Flash and Base configs explicitly declare differing expert dtypes; omission risk remains for other MoEs. |
| 4 | Residual percentage clamps at ±100% and loses raw error magnitude. | Confirmed; advisor design recorded. |
| 5 | Residual classification drops unmatched kernels without loop-path coverage. | Confirmed. |
| 6 | Unknown SKU becomes A100 while recording A100 as if observed. | Confirmed. |
| 7 | Broad model-config parse rescue becomes an unmarked default model. | Confirmed. |
| 8 | `has_unpriced_collectives` scans all nodes despite its narrow name. | Confirmed. |

## Sweep coverage

| Area | Phase 1 fallback sweep | Phase 2 wiring sweep | Notes |
|---|---|---|---|
| top-level runtime / API / CLI / workloads | Pending | Pending | |
| agents | Pending | Pending | |
| bench | Pending | Pending | |
| benchmarks | Pending | Pending | |
| deploy | Pending | Pending | |
| importers | Pending | Pending | |
| kernels | Pending | Pending | |
| optimizer | Pending | Pending | |
| planner | Pending | Pending | |
| routing | Pending | Pending | |
| safety | Pending | Pending | |
| scheduler | Pending | Pending | |
| serve | Pending | Pending | |
| telemetry | Pending | Pending | |
| tracer | Pending | Pending | |
| scripts | Pending | Pending | |

## Diagnostic-consumer trace

Every boolean flag, warning list, `estimated`, provenance field, and `has_*` or
`*_fallback` property discovered during the sweep is listed here and traced to a
human- or gate-visible consumer.

| Producer | Diagnostic | Downstream consumer | User/gate boundary | Status |
|---|---|---|---|---|
| — | — | — | — | Inventory pending |

## Sibling-path validation matrix

| Capability | Path A | Path B | Guard parity | Status |
|---|---|---|---|---|
| MoE config pricing | attach sidecar validates some raw config dtypes | scheduler has no sparse dispatcher and silently uses legacy dense defaults | Asymmetric | Open finding #1/#2/#7 |
| execution lifecycle | launch | attach | To inventory | Pending |
| model family | dense | MoE | To inventory | Pending |
| workloads | each dispatch branch | sibling branches | To inventory | Pending |

## Artifact-consumer trace

| Artifact writer | Artifact | Production reader / boundary | Status |
|---|---|---|---|
| — | — | — | Inventory pending |

## Completeness pass

This section must be empty before completion.

- Subpackages not swept: top-level runtime/API/CLI/workloads, agents, bench,
  benchmarks, deploy, importers, kernels, optimizer, planner, routing, safety,
  scheduler, serve, telemetry, tracer, scripts.
- Diagnostic flags/warnings not traced: inventory not yet complete.
- Asymmetric validation gates: inventory not yet complete.
- Fallbacks judged acceptable without confirming REFUSE/FLAG/WARN: inventory not
  yet complete.
