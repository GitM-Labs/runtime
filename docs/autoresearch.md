# Autoresearch

Autoresearch is the agentic half of the optimizer's search. The curated
intervention library (`gitm/kernels/library.yaml`) is finite and reviewed;
autoresearch proposes **non-catalog** levers — real runtime knobs that are not in
the library — constrained to the bottleneck class attribution already identified,
then routes each proposal through the *same* gate and rollback path as a catalog
lever.

It is a **candidate source, not a new trust path**. Nothing it proposes can
bypass the selection gate or be kept without a measured win.

Code: `gitm/agents/autoresearch.py`.

## Pipeline

```
trace ──▶ classify_bottleneck ──▶ propose ──▶ select_interventions ──▶ apply_intervention
         (idle/memory/compute)   (candidate    (safety gate +           (snapshot → apply →
                                   specs)        replay ranking)          measure → keep|rollback)
```

Every stage is one it shares with the catalog:

1. **`select_interventions`** (`gitm/agents/policy.py`) — pre-filters on the
   safety tier and qualification commit, then ranks survivors by counterfactual
   replay (`predict_delta`). A rejected proposal is recorded and dropped.
2. **`apply_intervention`** (`gitm/optimizer/apply.py`) — snapshots, applies,
   measures, and keeps the change only if the measured delta clears
   `min_keep_delta`; otherwise it restores. A proposal that doesn't measurably
   help is rolled back.

## Bottleneck classification

`classify_bottleneck(trace) -> str` maps a captured trace to one of three classes
using two signals read straight from the trace:

| Signal | Meaning | Threshold |
|--------|---------|-----------|
| serialized-concurrency fraction | kernels ran back-to-back on one stream instead of overlapping → scheduling gaps / launch-bound idle | `_SC_THRESHOLD = 0.5` |
| memcpy fraction | data movement dominates the op mix → memory bound | `_MEMCPY_THRESHOLD = 0.25` |

Each signal is scored against its threshold; the stronger signal wins. If neither
crosses its threshold the workload is **`compute_bound`**. An empty trace has no
stall/movement signal and defaults to `compute_bound`.

```
sc_score  = serialized_fraction / 0.5
mem_score = memcpy_fraction      / 0.25
class = compute_bound                         if max(sc_score, mem_score) < 1
        idle_stall                            if sc_score >= mem_score
        memory_bound                          otherwise
```

This is a deliberately simple v0 heuristic, not a tuned model — the thresholds
only have to route the search into the right candidate table. The rollback gate,
not the classifier, is what protects a wrong route: a bad proposal is measured
and reverted.

## Candidate table

Each class maps to a small, fixed table of `(knob, value, rationale)` in
`_RULES`. Every knob is a **real, current vLLM argument** (verified against
docs.vllm.ai) that is **not** in `library.yaml`:

| Class | Knob | Value | Rationale |
|-------|------|-------|-----------|
| `idle_stall` | `max_num_partial_prefills` | 4 | raise partial-prefill concurrency so prefill overlaps decode |
| `idle_stall` | `long_prefill_token_threshold` | 2048 | chunk big prompts so they interleave, closing decode gaps |
| `memory_bound` | `cpu_offload_gb` | 4 | offload cold weights to host RAM, freeing HBM for KV cache |
| `memory_bound` | `preemption_mode` | `swap` | swap preempted KV blocks instead of recomputing under pressure |
| `compute_bound` | `compilation_config` | 3 | torch.compile level 3: kernel fusion + piecewise CUDA graphs |

The **rationales are plausibility arguments, not measured claims**. Each proposal
carries `expected_delta_mean = 0.05` (`lo = 0.0`, `hi = 0.15`) — a modest,
honestly-unproven range — and a `source` that says so. Only the measured A/B
turns a proposal into a real number.

## Safety posture

- Proposals are always `moderate` tier — never `high_risk`. Topology and weight
  changes stay in the reviewed catalog.
- The rollback gate is the whole safety story for a proposal: applied behind a
  snapshot, kept only on a measured win, reverted otherwise.

## Loop integration

The 24-hour loop (`gitm/scheduler/loop.py`) runs autoresearch as **Phase 4b** on
the `vllm-decode` path, after the catalog pass:

- Guarded by remaining budget — if the catalog pass spent the budget, the phase
  is skipped (the class it *would* have searched is still reported).
- With no live engine attached the loop uses a `DryRunApplicator`, so autoresearch
  claims land **unverified** (`measured_delta = None`) — exactly like unverified
  catalog claims, never counted as a proven gain.
- Results are written to `<run_dir>/autoresearch.json`; the summary carries
  `bottleneck_class` and `n_autoresearch`; applied proposals appear in the report
  as `autoresearch:<class>:<knob>` claims and rejected ones in the rejected list.

## API

```python
from gitm.agents.autoresearch import autoresearch, classify_bottleneck, propose

# End-to-end: classify, propose, gate, apply/rollback.
run = autoresearch(trace, applicator=applicator, policy=policy)
run.bottleneck_class          # "idle_stall" | "memory_bound" | "compute_bound"
run.results                   # list[AutoresearchResult]

# Lower-level pieces.
cls   = classify_bottleneck(trace)
specs = propose(cls)          # list[InterventionSpec] (empty for an unknown class)
```

`AutoresearchResult` carries `spec`, `bottleneck_class`, `predicted_delta`,
`applicable`, `rejected_reason`, `measured_delta`, and `rolled_back`.

## v0 limits and roadmap

v0 is a static table per class. Later versions repoint at the largest *measured*
residual and learn an effect model instead of a fixed table, and produce the
bottleneck-class vocabulary from the shared attribution layer rather than the
local heuristic here.
