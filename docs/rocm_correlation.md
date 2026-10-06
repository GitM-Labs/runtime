# Kernel identity on ROCm: the AMD port of NVTX correlation

*Branch `tracer/rocm-correlation-parity`, 2026-10-06.*

This port follows the NVIDIA design in `docs/kernel_identity.md`, as it stands after
`tracer/graph-node-identity` (#160). It covers:

- the correlation chain;
- the graph-replay rule (a launch range is not an op);
- the capture-time node map ("the Nsight approach");
- the attribute side table;
- memcpy step labels.

It meets the same requirements with ROCm's own tooling, and improves on the original
design where ROCm makes that possible. If "NVTX kernel correlation for MoE" meant a
different branch, the place to look is `docs/kernel_identity.md`, sections "CUDA
graphs" and "Attributes beyond `L{layer}/{op}`". Every requirement below is taken from
there.

## Requirements carried over

| # | Requirement (NVIDIA design) | ROCm mechanism | Status |
|---|---|---|---|
| R1 | Eager kernel → its enclosing `L{layer}/{op}` range. The match is made on the host clock and the same thread, never on the kernel's device window. | External-correlation **stamp** (below) as the primary join. HIP-API records are kept for host-time containment, which acts as validator and fallback. | done |
| R2 | A graph-replayed kernel must never take the range around its launch as its op. That range goes to `launch_range`. | Replay stamp `(exec, ordinal)`. The `hipGraphLaunch` runtime records carry a `graph_launch` flag as a second guard. | done |
| R3 | Name replayed kernels from the range open when their graph node was captured. | Capture-time `graph_node` records, linked to the executable through `EndCapture` → `Instantiate*`. | done, and validated (see below) |
| R4 | Structural records must survive the capture window. | They are written unarmed, and `read_shards` exempts `graph_node` and `graph_exec`. | done, shared with NVIDIA |
| R5 | Attributes beyond op and layer go in a side table, not in the range name. | `gitm.tracer.kernel_attributes`, plus `#k=v` range annotations. rocTX has no payload. | done, both vendors |
| R6 | Memcpys get a step-level time-series label. | `MemcpyEvent.launch_range`, from the stamp or from containment. | done, both vendors |
| R7 | Process scoping: ids are only unique within one process. | Unchanged: decode partitions by pid before anything else. | done |
| R8 | Classify the vendor. | `gitm.tracer.vendor` uses host and trace evidence, ranked by strength, and reports conflicts. | done |

## The ROCm-native mechanisms

### 1. Range stamps (R1)

rocprofiler-sdk has an external-correlation request service
(`rocprofiler_configure_external_correlation_id_request_service`). It is present in
ROCm 7.0–7.2.3 and marked experimental.

For every dispatch, copy and HIP API record it is about to produce, the service calls
the tool synchronously on the enqueuing thread. The tool returns a 64-bit value, which
the record then carries in `correlation_id.external`. `rocm_inject.c` returns the id of
the innermost rocTX range open on that thread at that instant. That id is the same
`marker_id` its push/pop halves carry.

```
kernel  range_id=R   ── same id ──>   marker {marker_id: R, name, start_ns, end_ns}
```

On NVIDIA, the join from a kernel to its range is inferred from timestamps across
three record kinds. On AMD it is recorded when the launch happens. Three consequences:

- No clock comparison is involved, so the async end-time hazard cannot occur.
- No thread matching is involved, so a launch from a helper thread lands under the
  range that helper thread pushed.
- No containment sweep is needed for eager kernels.

The HIP-API containment chain stays on in the correlation arm as an independent
validator. `CorrelationReport.stamp_containment_disagree` counts the cases where the
stamp and containment disagree. This is the first check to run on hardware.

### 2. HIP graph replays (R2, R3)

ROCm 7.2.3 dispatch records carry no graph or node id. The current sdk (`develop`,
1.4.2) adds a `HIP_GRAPH` tracing domain, and its documentation
(`callback_tracing.h`) gives a recipe for attributing dispatches to graph nodes:

1. Push `(exec, node_counter)` per thread on `hipGraphLaunch` ENTER and pop it on EXIT.
2. In the external-correlation request callback, stamp each dispatch with the top of
   that stack, then advance the counter.

That domain is in no release yet. The collector builds the same thing from primitives
that 7.2.3 already has:

- **Replay.** A HIP-API callback on `hipGraphLaunch{,_spt}` ENTER/EXIT maintains the
  per-thread exec/ordinal stack. Each dispatch or copy enqueued inside a launch is
  stamped `GRAPH | exec<<32 | ordinal+1` instead of with a range. The decoder reads
  this as `graph_id` / `graph_node_id`. The id layout is in `correlate.py` (`NODE_*`)
  and is pinned against the C side by `tests/test_rocm_collector_contract.py`.
- **Capture.** While a thread is stream-capturing (a per-thread flag set from
  `hipStreamBeginCapture*` EXIT to `hipStreamEndCapture` EXIT), every launch, copy or
  memset API call becomes a node. Each node is emitted as
  `graph_node{capture_node_id(C, k), name = innermost range, node_kind, kernel_id, grid, block}`.
  Using a per-thread flag rather than a stream match means event-forked side streams
  are covered.
- **Link.** `hipStreamEndCapture` gives `hipGraph_t → capture C`. Every
  `hipGraphInstantiate{,WithFlags,WithParams}` gives `hipGraphExec_t → exec E`, with
  sequence numbers assigned on each instantiation, so a reused pointer is never
  confused with the old graph. The collector emits
  `graph_exec{graph_id: E, capture_id: C, n_nodes}`.

**Validation.** A position in the capture only identifies a kernel if the replay runs
in capture order. That holds for a single-stream capture, which is what vLLM does, but
it is not promised for a forked one. So the decoder checks each replay (the kernels
sharing one launch `correlation_id`) and refuses the whole replay if any check fails:

- ordinals are distinct, which catches a stamp that did not advance;
- ordinals are within `n_nodes`;
- each kernel matches its node's signature:
  - same `kernel_id`, for `hipLaunchKernel` launches, mapped through the code-object
    host-symbol callback;
  - same grid and block, for every kernel launch;
  - for memcpy and memset nodes, a ROCclr blit (`__amd_rocclr_*`) or an SDMA copy
    record.

A refused replay still knows it is a replay, so it keeps `launch_range` and falls back
to name classification. Refusals are counted by reason in `CorrelationReport` and
surface as warnings from `read_shards`.

**Known limit.** Two nodes with identical signatures cannot be told apart if the
runtime swaps them. The usual case is the same projection in two layers. This needs a
forked capture and a reordering runtime, and
`test_known_limit_reordering_two_identical_nodes_is_undetectable` pins it.
`hipModuleLaunchKernel` / `hipExtModuleLaunchKernel` nodes (Triton, hipBLASLt, AITER
asm) validate on geometry alone, because no callback maps a `hipFunction_t` to a
`kernel_id`.

### 3. Attributes (R5)

`AttributeIndex` joins four sources onto the identity correlation produced:

- **Static, derived:** `layer_class`. Layers whose predicted structure is identical
  share a class, so a model's archetypes come out of its own graph. DeepSeek-V4's
  sliding-window and compressed layers, or a hybrid model's DeltaNet and attention
  layers, separate with no per-model code.
- **Static, declared:** the spec's own `layer_kind` / `mlp_layer_types`, when it has
  them.
- **Op vocabulary:** `moe_phase` (route, dispatch, exchange, expert, shared_expert,
  combine).
- **Dynamic:** range annotations (`L3/moe_routed#wave=2`). These are split off before
  the name is parsed, so they never reach the op. They are per range instance, which
  is what expert-parallel waves need.

Attribution consumes these opt-in, via `residuals(..., with_attributes=True)` and
`stratify=` on `check_invariants`, `attribute`, `attribute_dr` and `recoverable_by`.
The default (op only) is unchanged.

## Improvements over the initial design, in both vendors' paths

1. **Marker pairing is order-independent.** Halves used to pair in arrival order, so
   any range whose END was flushed before its START was dropped. Neither collector
   promises flush order. Pairing now sorts by timestamp, with START before END at
   equal timestamps.
2. **Capture-node names are normalized like markers.** A node captured under vLLM's
   dict-repr layerwise range used to keep the raw dict as its name, so it never
   parsed to `L{l}/{op}`.
3. **Structural records pass through `read_shards` unwindowed.** This was step 3 of
   the NVIDIA "Nsight approach" plan, so the CUPTI side inherits it when its
   `graph_node` emitter lands.
4. **`CorrelationReport`** counts every way identity is lost or doubted, by mechanism
   and reason. These used to be silent.
5. **AMD vocabulary.**
   - Tensile/hipBLASLt `Cijk_` GEMMs, AITER's MLA decode and paged-attention kernels,
     and vLLM's Triton decode attention all landed in `other`. On Kimi K2.5 / MI355X
     that is the dominant GEMM family and every attention kernel.
   - AITER's `topksoftmax` router was filed as sampling.
   - MoE sort and quant kernels were charged to the expert GEMM.
   - `observed_op` now keeps a router kernel's own op inside an expert range, which
     covers CUDA's `topk_softmax` as well. **This changes NVIDIA numbers on existing MoE
     captures:** router kernels inside expert ranges move from `moe_routed` to
     `moe_router`, so `moe_routed`'s recoverable time drops by their share.
6. **Causal attribution with mixed launch cardinalities.** `attribute` and
   `attribute_dr` truncated every op series to the shortest one. One op that runs once
   per step (`lm_head`) cut every per-layer series to the step count, so on a short
   window Granger could not fit and returned no hypotheses.
   - Pairs are now aligned only between series of comparable cardinality
     (`comparable`, min/max ≥ 0.9), each truncated to its own shorter length.
   - The tolerance absorbs a window's partial edge steps and refused replays, which
     make same-class series differ by a few launches.
   - A per-step op is never aligned with a per-layer one.
7. **Vendor classification is evidence-based.** It used to be "kfd present → AMD,
   else NVIDIA". It now ranks evidence: compute driver over PCI over device name over
   software. A split within the top tier is reported as a conflict, `GITM_VENDOR`
   overrides, and traces classify from the collector's in-band `meta` record or the
   kernel-name dialect.

**What flows back to NVIDIA next.** The CUPTI capture-time collector, from
`kernel_identity.md` (NVTX + RESOURCE callbacks). It should emit `graph_node` records
with the same optional signature fields (`node_kind`, grid, block), so the same
validation protects it. The emulator already renders that collector
(`cupti_node_map=True`), and the parity tests show it reaches exact identity on the
same executions.

## Roadmap

Each module is complete and tested on its own.

| Module | Files | Verified here | Needs hardware |
|---|---|---|---|
| M0 vendor classification | `gitm/tracer/vendor.py`, `injection.detect_vendor` | `tests/test_vendor_classification.py` | — |
| M1 collector | `gitm/tracer/_rocm/rocm_inject.c` | Compiles `-Wall -Wextra -Werror` against real rocm-7.2.3 and develop headers (`scripts/check_rocm_collector.py`). Source contract in `tests/test_rocm_collector_contract.py`. | the checks below |
| M2 correlation | `gitm/distributed/correlate.py`, `gitm/tracer/_cupti_decode.py`, `injection.read_shards` | `tests/test_rocm_correlation.py` | — |
| M3 attributes | `gitm/tracer/kernel_attributes.py`, `monitor`/`attribution`/`dr` `stratify=` | `tests/test_causal_attribution_parity.py` | — |
| M4 vocabulary | `kernel_taxonomy._RULES`, `deviation._OP_RULES` | `tests/test_amd_kernel_vocabulary.py` (names cited to upstream sources) | widen from a real MI355X capture |
| M5 accuracy harness | `gitm/tracer/emulate.py`, `mechanism_fixtures.Launch` | `tests/test_identity_accuracy.py`, `tests/test_causal_attribution_parity.py` | — |

**Accuracy measured on emulated executions.** Each execution is a mechanism fixture
rendered as the collector records it, and is scored against launch truth:

| Mode | Identity | Wrong |
|---|---|---|
| Eager, NVIDIA and AMD | 100% | 0 |
| AMD, helper-thread launches and a 10¹² ns device clock offset | 100% | 0 |
| AMD graphs, including blit nodes | 100% | 0 |
| NVIDIA graphs with the planned node map | 100% | 0 |
| NVIDIA graphs today | name-only | 0 |
| AMD hazards: stamp not advancing, inherited launch stamp, cross-kernel reorder, untracked exec | refused | 0 |
| AMD capture without ranges | unnamed, reported as a problem | 0 |

On the causal side, residuals, invariant violations, Granger and doubly-robust
rankings, and recoverable time are all identical to ground truth in every exact mode,
on both vendors. The naive decode (from before graph identity) files every replayed
kernel as one op, and the injected cause disappears.

### Hardware checks, in order (MI355X, ROCm 7.2.3)

Prerequisites:

- Build with `python -m gitm.tracer._rocm.build`.
- Use the arm-C env from `run_env(..., nvtx=True)`, which includes the roctx shim and
  `GITM_TRACE_NVTX=1`.

Checks:

1. **Stamp fires.** In an eager run (`--enforce-eager`), kernels inside layerwise
   ranges have `identity == "range_id"`. `stamp_containment_disagree == 0`, and
   `stamp_unresolved` comes only from ranges pushed before arming.
2. **The stamp advances per dispatch inside `hipGraphLaunch`.** This is the one
   behaviour 7.2.3 does not document, and it can fail in two ways. Both degrade
   replays to `range_op = None`, never to a wrong op:
   - **Dispatches inherit the launch's stamp (the likelier failure).** No separate
     request is made per dispatch, so each one inherits the `hipGraphLaunch` call's
     stamp. Kernels then arrive with `graph_id` 0 and `range_id` equal to the launch
     range's id. The `graph_launch` guard catches this, and it shows up as
     `graph_refused["untracked_launch"]` close to the graph-kernel count.
   - **Requests fire but the ordinal does not advance.** This shows up as
     `duplicate_ordinal`.

   In either case, upgrade to an sdk with the `HIP_GRAPH` domain.
3. **Capture links and names.**
   - Every `graph_exec` has a nonzero `capture_id`, and its `n_nodes` equals the
     replay's dispatch-plus-copy count, i.e. no `ordinal_out_of_range`.
   - `graph_node` names are mostly non-empty. `CorrelationReport.graph_unnamed` above
     half of the graph kernels raises a problem: the ranges did not run during
     capture. torch.compile can trace forward hooks away, and vLLM's
     `--enable-layerwise-nvtx-tracing` should be checked separately.
4. **Signatures.** `signature_*` refusals should be absent on single-stream vLLM
   capture. Two suspects:
   - **`signature_kernel_id` on `hipLaunchKernel` nodes.** It is not yet verified that
     the code-object host-symbol callback's `kernel_id` shares an id space with the
     dispatch record's `kernel_id`. If they differ, drop the `kernel_id` comparison for
     host-function nodes and keep geometry.
   - **`signature_grid` on `hipExtModuleLaunchKernel` nodes.** Compare the dispatch
     record's work-item grid against the call's `globalWorkSize`.
5. **Overhead.** Run e1 from `docs/mi355x_experiment_plan.md` with and without
   `GITM_TRACE_NVTX`. The HIP-API callback is filtered to 26 operations, and the stamp
   callback is a TLS read.
