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

## Repoint at the largest residual

The class picks *which* levers to try; the residual target picks *where* to aim
them. `largest_residual(res)` takes the residuals the attribution phase already
computed (`gitm/optimizer/monitor.residuals` → the gap-vs-ceiling
`r_kt = (t_obs − t_pred)/t_pred`), aggregates them per op, and returns the op
whose kernels run furthest **over** the predicted ceiling — the biggest gap.

Two things this deliberately is **not**:
- not `measure_trace`'s residual, which is within-kernel-type jitter (a uniformly
  slow, dominant op has ~0 jitter);
- not the top Granger hypothesis, which is ranked by p-value (significance), not
  gap magnitude.

When a target op is found *and it matches a real kernel name* in the trace, each
proposal is scoped to it via `applies_to_kernels`, so the existing ranking gate
(`predict_delta`) weights the lever by that op's share of trace time. If the op
isn't present (the residual label comes from the predicted graph, which may not
match captured kernel names), the target is still **recorded** as the
justification but proposals are left unscoped — tagging an absent op would zero
the coverage and make a relevant lever look worthless.

**Honest ceiling:** repointing changes what the search aims at, how proposals are
prioritized, and the recorded "why" (`target_op` on every result). It does **not**
by itself produce genuinely different knobs per op — the lever set still comes
from the class table. Real per-op lever affinity would need hand-authored
op→knob mappings, which v0 does not claim.

## Proposal sources (the Proposer seam)

Candidates come from a `Proposer` behind a common seam; each returns
`list[InterventionSpec]` and feeds the *same* selection + rollback gate, so a
proposer is a candidate **source**, not a new trust path.

- **`GenerativeProposer`** (the loop's active source) — *generates* candidates by
  searching a workload's knob surface instead of reading a frozen list. It pulls
  knobs from a **`KnobSource`**, drops any that duplicate `library.yaml`, keeps
  the ones affine to the bottleneck class, and searches a small **value grid** per
  knob. This is what makes autoresearch a *search* rather than a lookup: each knob
  is tried at several values (`½×`/`2×`/`4×` of the default, a bool flip, the
  other members of an enum, or an explicit grid like `2048`/`4096`), emitted as
  `autoresearch:<class>:<knob>=<value>` — and only a measured win survives the gate.
- **`StochasticProposer`** — entropy-guided sampling of the same knob surface.
  The class affinity *weights the dice* (affine knobs get most of the mass) but a
  nonzero floor (`epsilon`) keeps every knob reachable, so the search can wander
  off-class and surface a lever the keyword heuristic would never pick. A seeded
  RNG draws the `(knob, value)` samples — reproducible for a given seed, varied by
  changing it — and the rollback gate makes unbounded entropy safe. `epsilon=0`
  collapses to pure heuristic; higher values explore more widely. (Available
  behind the seam; the loop still runs the generative proposer by default.)
- **`TableProposer`** (the fallback) — the static per-class `_RULES` table below.
  `FallbackProposer(EngineArgsProposer(), TableProposer())` uses it only when the
  active proposer has nothing for a class (e.g. an unknown class), so the reviewed
  catalog's levers still apply.

Neither source can emit a hallucinated knob: the generated names come from the
workload's real config surface, and both paths exclude anything already in
`library.yaml`.

### Fitting different workloads

Versatility comes from the `KnobSource`, **not** a `{workload: knobs}` table:

- **`VLLMKnobSource`** (used by `EngineArgsProposer`, the vLLM binding of
  `GenerativeProposer`) introspects `dataclasses.fields(EngineArgs)` when vLLM is
  importable and reads each field's **valid domain from its CLI argument** —
  argparse `choices=` (the real enum domain) and `type=` — via
  `EngineArgs.add_cli_args`, so the search only proposes values that can apply;
  list-valued args (`nargs`) and non-performance fields are skipped. Yields a
  frozen catalog otherwise, so the path runs offline and deterministically.
- **Hardware-applicability filter.** On a single-GPU box, knobs that only matter
  with more than one GPU (`tensor_parallel_size`, `pipeline_parallel_size`,
  `*_context_parallel_size`, …) are skipped — proposing one there can only fail
  the engine build (no topology to satisfy it), wasting a restart-A/B on a
  candidate that was never enactable. `VLLMKnobSource(gpu_count=...)` (and
  `EngineArgsProposer(gpu_count=...)`) accepts an explicit count; left unset it's
  autodetected (`torch.cuda.device_count()`, defaulting to 1 when it can't tell —
  undercounting only skips a possibly-valid knob, which is the safe direction).
- Another workload plugs in by yielding its own `Knob` list and a `workload`
  label: `GenerativeProposer(MyKnobSource(), workload="triton-serve")`. Candidates
  carry that label in their applicability; nothing enumerates workloads centrally.
- **Class affinity** is per-knob, not per-workload: a `KnobSource` can tag each
  `Knob` with the bottleneck classes it targets (`Knob.classes`), and only knobs
  with no tag fall back to the vLLM-flavoured keyword heuristic on the name.

## Fallback table

The static `_RULES` table — one small, fixed `(knob, value, rationale)` per
class, every knob a **real, current vLLM argument** (verified against
docs.vllm.ai) that is **not** in `library.yaml`:

| Class | Knob | Value | Rationale |
|-------|------|-------|-----------|
| `idle_stall` | *(none by default)* | � | dependent prefill/DBO knobs are filtered unless represented by reviewed catalog entries |
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
- It reuses the attribution phase's already-computed residuals (`res`) to repoint
  at the largest-residual op — no recomputation.
- Candidates are **generated** by `FallbackProposer(EngineArgsProposer(),
  TableProposer())`: the EngineArgs value-grid search is the active source, with
  the static table as the fallback.
- Results are written to `<run_dir>/autoresearch.json` (including the `target` op
  and per-result `target_op`); the summary carries `bottleneck_class` and
  `n_autoresearch`; generated candidates appear in the report as
  `autoresearch:<class>:<knob>=<value>` claims and rejected ones in the rejected
  list.

## API

```python
from gitm.agents.autoresearch import (
    EngineArgsProposer, FallbackProposer, TableProposer,
    autoresearch, classify_bottleneck, propose,
)

# End-to-end: classify, find the largest-residual op, generate, gate, apply/rollback.
# Without a proposer it uses the static table; the loop passes the generative one.
proposer = FallbackProposer(EngineArgsProposer(), TableProposer())
run = autoresearch(trace, applicator=applicator, policy=policy, residuals=res,
                   proposer=proposer)
run.bottleneck_class          # "idle_stall" | "memory_bound" | "compute_bound"
run.target                    # ResidualTarget(op, residual, n_kernels) | None
run.results                   # list[AutoresearchResult]

# Lower-level pieces.
cls    = classify_bottleneck(trace)
target = largest_residual(res)                       # ResidualTarget | None
specs  = EngineArgsProposer().propose(cls, target_op=target.op if target else None)
specs  = propose(cls, target_op=target.op if target else None)  # the static table
```

`AutoresearchResult` carries `spec`, `bottleneck_class`, `predicted_delta`,
`applicable`, `rejected_reason`, `measured_delta`, `rolled_back`, `target_op`,
and `apply_error`. `apply_error` distinguishes "the live apply/engine-build
itself raised" (rolled back, `measured_delta=None`, `apply_error` set — e.g. a
two-engine restart clash, or a knob the running config can't satisfy) from
"measured and lost" (rolled back with a real negative `measured_delta`) — both
otherwise look identical as a bare unexplained result. The loop surfaces it in
both `autoresearch.json` and the report's causal-evidence text.

## v0 limits and roadmap

The lever set is now *generated* from the `EngineArgs` surface with a value grid
per knob, not a frozen table — but the generation is still crude and honest about
it:

- **Class→knob affinity is a keyword heuristic** (`prefill` / `cache` / `compil`
  …) matched against the knob name, not a learned or hand-authored mapping.
- **Enum domains come from vLLM; numeric ranges don't.** When vLLM is importable,
  each knob's valid **choices** and **type** are read from its `EngineArgs` CLI
  argument (argparse `choices=`/`type=`), and list-valued args are skipped — so
  the search stops proposing enum values or shapes that can't apply. But argparse
  carries no numeric min/max, so unconstrained ints/floats still use the generic
  `½×`/`2×`/`4×` ladder (or a hand-seeded grid). Per-class candidate counts are
  bounded by `max_candidates`.
- **Generated EngineArgs filters are conservative.** Valid vLLM args that are
  unsupported/noisy as standalone candidates are skipped before gridding, including
  concurrent partial-prefill knobs, DBO thresholds, and WIP KV-sharing fast-prefill
  flags. Reviewed catalog entries should represent those feature families instead.
- **Live typing is otherwise best-effort** — fields with no CLI `choices`/`type`
  are classified coarsely from their annotation; ones the introspector can't type
  fall back to "no grid" and are simply skipped.
- **Hardware applicability only covers GPU count.** Multi-GPU topology knobs are
  filtered (above); knobs that need something else to be true of the environment
  — an AWQ-quantized checkpoint for `quantization=awq`, enough free VRAM for the
  restart-A/B's second (candidate) engine alongside the still-live baseline — are
  not, and can still fail an apply. Those still surface as an honest rolled-back /
  unenacted result (never a fabricated claim) at the cost of a wasted restart —
  but `apply_error` now carries the real exception message, so *why* it failed is
  diagnosable from the report without digging through raw engine logs.
- **The stochastic sampler isn't wired into the loop yet.** `StochasticProposer`
  exists behind the seam (seeded, reproducible), but Phase 4b still runs the
  generative proposer; flipping the loop onto it (or a `FallbackProposer` chain)
  is the remaining wiring.
- **No cross-run feedback yet.** The remaining next steps: an effect model that
  reweights the search from realized measured deltas — still blocked because the
  loop runs dry until a live applicator is attached — plus sourcing the
  bottleneck-class vocabulary from the shared attribution layer rather than the
  local `BOTTLENECK_CLASSES` heuristic here.
