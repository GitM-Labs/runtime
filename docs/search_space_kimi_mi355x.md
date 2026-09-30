# Search space: Kimi-K2.5 on 8x MI355X

**Status: draft, not complete.** See [Open questions](#open-questions); question 1
blocks treating any ranked number here as final.

## Pinned baseline

```
model         kimi-k2.5  (moonshotai/Kimi-K2.5)            gitm/planner/models/kimi-k2.5.yaml
revision      NOT PINNED (hub default at launch)            open question 2
engine        vllm/vllm-openai-rocm@sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1
vLLM version  NOT RECORDED (only written to MANIFEST at run time)            open question 2
topology      8x "AMD Instinct MI355X", 1 node, xGMI      gitm/planner/context.py _PEAKS/_INTERCONNECT
parallelism   TP=8  PP=1  DP=1  expert-parallel off
launch        vllm serve moonshotai/Kimi-K2.5 --trust-remote-code
                --tensor-parallel-size 8 --gpu-memory-utilization 0.92
                --max-num-batched-tokens 8192 --max-num-seqs 256
                --tool-call-parser kimi_k2 --enable-auto-tool-choice --reasoning-parser kimi_k2
              env VLLM_ROCM_USE_AITER=1                     deploy/k8s/mi355x-kimi-loop.yaml
traffic       rag: 4096 prompt / 512 output tokens          scripts/kimi_loop/run_loop.sh
concurrency   64 (decode batch modeled = 64)                scripts/kimi_loop/predict_sweep.py HEADLINE_C
context       kv_len modeled 4352 (prompt + output/2); max_model_len 262144 (model default)
precision     bf16 weights, int4 routed experts (W4A16), bf16 KV (auto), bf16 activations
loop          workload vllm-decode, top_n_interventions 5 (LoopConfig), restart_fn available
              (gitm/workloads.py:827), history None
```

Engine state the knob taxonomy reads (`gitm/optimizer/vllm_knobs.py` `_KNOBS`) is a
stub, because no live engine exists on this box. Launch flags are exact. Everything
else is a vLLM default the repo does not verify for this image. The biggest unverified
defaults are `chunked_prefill_enabled=True`, `enable_prefix_caching=True`,
`async_scheduling=False` and `swap_space_bytes=4 GiB` (open question 3).
`block_size`, `max_seq_len_to_capture` and `moe_backend` are left off, so `get_knob`
raises on them, as it would against an engine without them. The full table with
provenance is in `engine_state` and `engine_state_unknown` in the artifact.

## How these numbers were produced

```
python scripts/search_space/feasible_kimi_mi355x.py --write
  -> evidence/kimi-mi355x/search_space/feasible.json   (git_commit 21348b5)
```

The script calls the loop's own code. Line numbers refer to commit `21348b5`.

| Rule | Code |
|---|---|
| validity (workload scope) | `gitm/kernels/library.py:35-36` `load_library(workload=...)` |
| sweep expansion | `gitm/optimizer/vllm_knobs.py:298-322` `expand_relative_candidates` |
| prerequisites | `gitm/optimizer/vllm_knobs.py:251-266` `unmet_prerequisite`, plus the joint exemption at `gitm/scheduler/loop.py:895-900` |
| hardware | `gitm/optimizer/preconditions.py:37-42` via `applicable()` on a spec projected to its hardware/GPU fields |
| mutual dependency | `KNOB_PREREQUISITES` (`vllm_knobs.py:240-248`): a dependent lever is feasible only if a feasible lever supplies the prerequisite |
| deployment | `gitm/scheduler/loop.py:816` condition, using `knob_kind` (`vllm_knobs.py:226-233`) |
| applicability + safety | `gitm/agents/policy.py:52-124` `select_interventions` |
| ranking | same function, `top_n=5`; sort key at `policy.py:116-124` |
| coverage × prior | `gitm/optimizer/replay.py` `predict_delta` / `_applies` (lines 48-62) |
| policy | `qualify()` → `Policy(...)`, exactly as `gitm/scheduler/loop.py:727` |
| gate context | `gitm/planner/context.py` `build_planner_context` |
| autoresearch | `gitm/agents/autoresearch.py` `autoresearch(...)` with `FallbackProposer(EngineArgsProposer(), TableProposer())` and `DryRunApplicator` |

The ranked stages run on a **stand-in trace**: one kernel per node of
`predict_glm_graph(kimi-k2.5, MI355X, batch=64, kv=4352, tp=8)`. That gives 1,037
kernels and an 11.017 ms predicted step (artifact `standin_trace`). Open question 1
covers this.

## Raw entries vs single-lever candidates

`library.yaml` has **28 entries**; 25 are scoped to `vllm-decode`
(`counts.library_entries_*`). Sweep expansion turns them into **36 single-lever
candidates** (`counts.library_candidates_raw`). The gap of 8 comes entirely from the 5
swept entries (`sweep_expansion`):

- Each swept entry has a 3-point `value_multiplier_grid`, so 5 entries become 15 grid
  points, which is 10 more candidates than entries.
- `max_seq_len_to_capture_dynamic` collapses from 3 points to 1. The baseline value
  is unreadable, so `resolve_relative_value` (`vllm_knobs.py:280-283`) returns the
  static 8192 for every point, and `seen_values` (`vllm_knobs.py:313-319`) drops the
  two repeats. That removes 2 of the 10.
- Net: 25 scoped entries + 10 − 2 = 33 in-scope candidates, plus 3 out-of-scope
  edge/HFT entries = 36.

Swept knobs are also why candidates and bits diverge. A knob with *n* candidate
values costs *n* candidates but only ⌈log₂(n+1)⌉ bits (the +1 is "leave unchanged").
The raw pool is 36 candidates over 28 knobs, which is **32 bits**: four 3-point knobs
at 2 bits each, plus 24 binary knobs at 1 bit each. That gives 2³² = 4,294,967,296
unconstrained joint configs.

Autoresearch adds **0 candidates**. The stand-in classifies as `memory_bound`
(`classify_bottleneck`, from the roofline memory-bound fraction), and every proposer
returns nothing for that class on this box (`autoresearch.proposals_per_class`). See
the proposer-exclusion number below.

## Sequential funnel

Source: artifact `funnel`. Joint configs are feasible ones: the product over knobs of
(values + 1), minus combinations that set a dependent knob without its prerequisite
(the `joint_configs` function in the script).

| Stage | Candidates | Knobs | Bits | Joint configs (feasible / unconstrained) |
|---|---:|---:|---:|---|
| raw | 36 | 28 | 32 | 3,221,225,472 / 4,294,967,296 |
| after validity | 33 | 25 | 29 | 402,653,184 / 536,870,912 |
| after prerequisites | 32 | 24 | 28 | 268,435,456 / 268,435,456 |
| after hardware | 27 | 19 | 23 | 8,388,608 |
| after mutual dependency | 27 | 19 | 23 | 8,388,608 |
| after deployment | 27 | 19 | 23 | 8,388,608 |
| after applicability + safety | 25 | 17 | 21 | 2,097,152 |
| after ranking | 5 | 4 | 5 | 24 |

What each stage removes:

- **Validity** removes 3: `edge_fp16_autocast`, `edge_frame_batching` and
  `hft_top_of_book_fewer_scans`, which belong to other workloads.
- **Prerequisites** removes 1: `enable_eplb`, because `enable_expert_parallel` is off.
  This also clears the only dependency constraint, so feasible and unconstrained
  counts meet from here on.
- **Hardware** removes 5 levers whose `requires_hardware` lists exclude MI355X:
  `kv_cache_dtype_fp8`, `attention_backend_flashinfer`, `tensor_parallel_size_2`,
  `pipeline_parallel_size_2` and `enable_expert_parallel`.
- **Mutual dependency** and **deployment** remove nothing. The only dependent,
  `enable_eplb`, is already gone. The restart hook exists, so no structural knob is
  vetoed.
- **Applicability + safety** removes 2:
  - `kv_cache_block_size_16`: `max_kv_cache_len 4096` < 262144. The gate compares
    against `max_model_len`, not the modeled 4352 (`context.py` `_engine_kv_len`).
  - `moe_backend_deep_gemm`: `requires_dtype [fp8]`, but the model is bf16.
- **Ranking** keeps the top 5 (`policy.py:116-124`).

## Independent view: each rule against all 36 raw candidates

Source: artifact `independent`, `independent_overlaps`.

| Rule | Rejects | Rejected only by this rule |
|---|---:|---:|
| validity | 3 | 0 |
| prerequisites | 1 | 0 |
| hardware | 5 | 0 |
| mutual dependency | 1 | 0 |
| deployment | 0 | 0 |
| applicability (full `applicable()`, safety off) | 10 | 2 |
| safety (`ctx=None`, real policy) | 0 | 0 |

Overlaps:

- validity & applicability: 3
- hardware & applicability: 5
- prerequisites & mutual dependency: 1

The first two overlaps are structural: `applicable()` checks workload and hardware
itself, so the validity and hardware rules isolate conditions the full gate also
enforces. The only rejections unique to the full gate are the kv-length rejection of
`kv_cache_block_size_16` and the dtype rejection of `moe_backend_deep_gemm`.

`enable_eplb` falls to both the prerequisite rule and the mutual-dependency rule. The
only lever that could supply its prerequisite, `enable_expert_parallel`, is itself
hardware-rejected.

`enable_dbo` passes the prerequisite rule only because of the joint exemption at
`loop.py:899`. Called on its own, `unmet_prerequisite(engine, "enable_dbo")` returns
`"prerequisite 'enable_dbo' not enabled on this engine"`: the `("dbo", "enable_dbo")`
substring rule matches the flag against itself (artifact
`candidates[enable_dbo].raw_unmet_prerequisite`). Branch
`loop/prereq-on-library-levers` fixes this at the source.

## Rejection, ranking and dedup

These are three separate numbers drawn from different pools. They are **not
additive**.

- **Rejected before ranking: 11** of the 36 raw candidates (36 − 25).
- **Cut by ranking: 20** of the 25 gate survivors (25 − 5).
- **Dedup** (artifact `rejection_ranking_dedup.dedup`), each against its own pool:

| Dedup | Count | Pool | Code |
|---|---:|---|---|
| sweep-point collapse | 2 | 15 grid points of the 5 swept entries, before expansion | `vllm_knobs.py:313-319` |
| proposer catalog exclusion | 2 | 3 knobs on the `EngineArgsProposer` surface on this box (`cpu_offload_gb`, `preemption_mode`, `compilation_config`; vLLM not importable, so `_FALLBACK_KNOBS`) | `autoresearch.py:769-771` `_searchable` |
| cross-source duplicates | 0 | 36 raw library + autoresearch candidates keyed by (knob, value) | script |
| baseline-equal | 3 | 25 gate survivors | informational on main; the loop does not skip these |

The catalog exclusion is what leaves `memory_bound` empty. The only two
memory-affine knobs on the offline surface are the two excluded as catalog knobs. The
remaining searchable knob, `compilation_config`, is `compute_bound`-affine. The
`memory_bound` row of `_RULES` (`autoresearch.py:216`) is empty on purpose.

## Consistency check

Source: artifact `consistency_checks`; `consistency_ok: true`.

Between each pair of adjacent stages, the check requires:

(point gap − bit gap) = Σ over fully removed swept knobs of (n − ⌈log₂(n+1)⌉) + the
same term for partial removals.

Partial removals are allowed only at the ranking stage. Every rejection stage removes
binary knobs only (difference 0), so no swept-knob point is dropped at a stage that
should only remove whole knobs. Ranking drops 20 points and 16 bits:

- `max_num_seqs`, `swap_space` and `gpu_memory_utilization` are removed fully,
  contributing 1 + 1 + 1.
- `max_num_batched_tokens` is cut from 3 points to 2, contributing 1.
- Total 4 = 20 − 16.

## What the loop would spend its 5 slots on (main, Phase 3 → Phase 4)

Source: artifact `phase3_top_n_on_main`. This is the exact Phase 3 call
(`loop.py:743`) over the scoped, expanded library.

| Slot | Lever | Predicted delta | Note |
|---|---|---:|---|
| 1 | `speculative_decode_ngram_5` | +0.0285 | high_risk; admitted only because `qualify()` commits on the stand-in |
| 2 | `cuda_graphs_enable` | +0.0214 | **baseline already runs `enforce_eager=False`** |
| 3 | `enable_chunked_prefill` | +0.0190 | **baseline already runs it** (vLLM V1 default, unverified) |
| 4 | `max_num_batched_tokens_dynamic_x0_5` | +0.0119 | 4096 |
| 5 | `max_num_batched_tokens_dynamic_x2` | +0.0119 | 16384 |

If `qualify()` does not commit on a real capture, `speculative_decode_ngram_5` is
gated out. `max_num_batched_tokens_dynamic_x4` takes slot 5, and slots 1–2 are the two
no-ops (`sensitivity.other_qualification_value.phase3_top_n`). With no restart hook,
all 36 candidates would be vetoed (`sensitivity.no_restart_fn_additionally_vetoed`),
because every taxonomy knob is structural.

## The heuristic leaving budget on the table

**`predict_delta`'s coverage term cannot see this model's predicted step time.** The
result is that ranking collapses to sorting hand-written priors, and the one lever
family aimed at the dominant cost has no reachable coverage. The evidence below is
from this run's artifact (`coverage`), not carried over from an earlier finding.

1. **76.2% of predicted step time is invisible to every gate survivor**
   (`coverage.uncovered_share_by_gate_survivors`). Whole-step levers use `&all_ops`
   (`library.yaml:15-21`: `qkv_proj`, `attn_score_value`, `attn_out_proj`,
   `mlp_gate_up`, `mlp_down`, `lm_head`), which is the dense-transformer vocabulary.
   `_applies` (`replay.py:48-62`) matches on `classify_op`. On Kimi's predicted graph
   that vocabulary covers:
   - `attn_score_value`: 22.2%
   - `attn_out_proj`: 1.1%
   - `lm_head`: 0.3%
   - the layer-0 dense `mlp_gate_up` + `mlp_down`: 0.12%

   That is 23.8% in total. Nothing covers:
   - the MoE ops (`moe_*`, **65.4%**, of which `moe_routed` alone is **60.0%**);
   - the MLA projections `attn_q_a/q_b/kv_a/kv_b`, which `classify_op` returns `None`
     for (4.9%);
   - norms and collectives.
2. **Ranking reduces to the prior.** Of the 25 gate survivors
   (`coverage.gate_survivor_coverage`):
   - 16 have identical coverage of 0.2377;
   - 5 have 0.2221 (`attn_score_value` only);
   - 3 have 0 (empty `applies_to_kernels`);
   - 1 has 0.0156 (`quantization_awq`). With coverage constant, the sort key at
   `policy.py:116-124` orders by `expected_delta_mean`, which is hand-authored and the
   same on every model. Sweep points of one knob tie exactly, so the name tiebreak
   decides which points of `max_num_batched_tokens` get slots 4–5 and which is cut.
3. **The MoE levers aim at the wrong ops.** `enable_expert_parallel`, `enable_eplb`
   and `moe_backend_deep_gemm` all declare `applies_to_kernels: [mlp_gate_up,
   mlp_down]` (`library.yaml:449, 477, 504`). On this graph the routed-expert work is
   `moe_routed`, so each of them reaches coverage 0.0012 (`coverage.moe_levers`). All
   three are rejected before ranking anyway, by hardware, prerequisite and dtype
   respectively. Even if MI355X were added to their hardware lists, they would rank at
   essentially zero predicted delta.
4. **Two of five slots measure no-ops** (see the table above). The baseline-equal
   count is 3 of 25 survivors, and 2 of them rank in the top 5 because their priors
   (0.09, 0.08) are among the largest. Branch `loop/skip-baseline-noops` addresses
   this.

What this does *not* show: whether the surviving levers help or hurt on MI355X. No
measured deltas for this baseline are committed. The earlier H200 conclusion
("scheduling knobs are noise, structural is the lever",
`docs/kimi_mi355x_loop_runbook.md`) is neither confirmed nor refuted here.

## Cross-check: hand-launched interventions

Source: `scripts/kimi_loop/gen_pods.py` `SLOTS`. No measured outcomes for these pods
are committed, so this checks only whether the pipeline would reach each one.

| Pod | Change | Pipeline verdict | Why |
|---|---|---|---|
| `intervene` | `--kv-cache-dtype fp8` | **rejected** (hardware) | `kv_cache_dtype_fp8` has `requires_hardware [A100, H100, L40S]` (`library.yaml:60`) |
| `int-ep` | `--enable-expert-parallel` | **rejected** (hardware) | `enable_expert_parallel` has `requires_hardware [A100, H100, H200]` (`library.yaml:484`); also high_risk; coverage 0.0012 even if admitted |
| `int-moe-triton` | env `VLLM_ROCM_USE_AITER_MOE=0` | **not representable** | no library entry, not in `_KNOBS`, not on the proposer surface |
| `int-mla-triton` | `--attention-backend TRITON_MLA` | **not representable** | the only backend lever is `attention_backend_flashinfer` (env `VLLM_ATTENTION_BACKEND=FLASHINFER`), which is hardware-rejected |
| `int-rccl-ring` | env `NCCL_ALGO=Ring` | **not representable** | no collective-algorithm lever exists |
| `int-tp4` | TP=4 | **not representable** | only `tensor_parallel_size_2` exists, and it is hardware-rejected |
| `int-eager` | `--enforce-eager` | **not representable** (opposite direction) | the library only has `cuda_graphs_enable` (`enforce_eager=False`), a baseline no-op that takes slot 2 |
| `int-kvhead` | `--gpu-memory-utilization 0.97` | **kept by gates, cut by ranking** | `gpu_memory_utilization_dynamic_x1_1` (0.92 × 1.10 → clamped to 0.97), predicted +0.0095, below slot 5 |
| `int-maxseqs` | `--max-num-seqs 512` | **kept by gates, cut by ranking** | `max_num_seqs_dynamic_x2`, predicted +0.0095 |
| `int-prefill` | `--max-num-batched-tokens 16384` | **kept, ranked (slot 5)** | `max_num_batched_tokens_dynamic_x2`, predicted +0.0119; wins the slot on the name tiebreak over `_x4` |

The pipeline reaches 1 of the 10 by-hand interventions, and 3 of 10 survive the
gates. Five cannot be expressed in the catalogue or the knob taxonomy at all. Two are
blocked by `requires_hardware` lists that do not name MI355X; whether those lists
reflect a real MI355X limit or just the hardware each lever was reviewed on is open
question 4.

## Other observations (not fixed here)

- **`swap_space_dynamic` scales bytes and emits GiB.** The taxonomy reads
  `swap_space` from `cache_config.swap_space_bytes` (`vllm_knobs.py:72`), and
  `resolve_relative_value` writes the scaled result back under `swap_space`. The
  candidates are `swap_space=2147483648 / 8589934592 / 17179869184`
  (`sweep_expansion`), a value vLLM's `swap_space` argument would read as GiB. This
  assumes the engine exposes `swap_space_bytes`, as the stub does. All three points
  are cut by ranking here, so they cost no slot on this baseline.
- **The kv-length gate uses `max_model_len`, not the serving kv length.** GateContext
  `kv_cache_len` is 262144 (`context.py` `_engine_kv_len`). The modeled decode kv
  (4352) would also exceed `kv_cache_block_size_16`'s 4096 limit, so the verdict is
  the same here.
- **The autoresearch surface on this box is not the pod's.** vLLM is not importable
  here (`dedup.proposer_catalog_exclusion.vllm_importable: false`), so
  `EngineArgsProposer` uses the 3-knob `_FALLBACK_KNOBS`. In the pod it introspects
  real EngineArgs, and the `memory_bound` proposal set may be non-empty.

## Open questions

1. **No measured Kimi/MI355X residual profile is committed, so every ranked number
   above runs on the predicted graph as a stand-in. This needs sign-off.**
   - Coverage shares are shares of *predicted* time.
   - Residuals are zero by construction, so the autoresearch target op
     (`mlp_gate_up`) carries no signal.
   - `qualify()` commits only because the stand-in is not an imported trace and its
     head share is low. A real rocprof capture could flip it and change slot 1.

   The coverage finding (dense vocabulary vs MoE graph) is structural and does not
   depend on the profile. The exact shares, slot order and qualification outcome do.
2. The model revision and vLLM version are not pinned in the repo. Only the image
   digest is.
3. Several engine values are vLLM defaults assumed for this image:
   - `chunked_prefill_enabled`, `enable_prefix_caching` (both assumed True);
   - `async_scheduling` (assumed False);
   - `swap_space_bytes` (assumed 4 GiB).

   Two of the three baseline-equal levers depend on these assumptions.
4. MI355X is absent from the `requires_hardware` lists of `kv_cache_dtype_fp8`,
   `attention_backend_flashinfer`, `tensor_parallel_size_2`,
   `pipeline_parallel_size_2` and `enable_expert_parallel`. It is unverified whether
   those lists are MI355X limits or review scope.
5. `joint_configs` counts only `KNOB_PREREQUISITES` dependencies. Pairs that are
   incompatible for other reasons (for example two parallelism layouts at once) are
   not modeled, so the joint counts are upper bounds.
