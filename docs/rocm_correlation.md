# Kernel identity on ROCm

This is the AMD port of the NVTX correlation design in `docs/kernel_identity.md`
(through #160, graph-node identity), built on rocprofiler-sdk.

- Collector: `gitm/tracer/_rocm/rocm_inject.c`.
- Decoder contract: `gitm/distributed/correlate.py`.

## Mechanisms

**Eager kernels: range stamps.** rocprofiler-sdk's external-correlation request
service asks the tool, on the enqueuing thread, for a value to stamp on each
dispatch, copy and HIP API record. The collector answers with the innermost rocTX
range id on that thread, so kernels join their range by id. No timestamp or thread
matching is needed. Host-time containment remains as a fallback, and disagreements
are counted.

**Graph replays.** The sdk's `HIP_GRAPH` domain isn't in any release yet, so the
collector rebuilds its documented recipe from HIP API callbacks:

- `hipGraphLaunch` ENTER/EXIT keeps a per-thread `(exec, ordinal)` stack, and each
  dispatch or copy inside a launch is stamped with the next ordinal.
- While a thread is stream-capturing, each launch, copy or memset call becomes a
  `graph_node` record. It names the open range and carries a signature: `kernel_id`,
  grid/block and node kind.
- `hipStreamEndCapture` and `hipGraphInstantiate*` produce a
  `graph_exec{graph_id, capture_id, n_nodes}` record.

Graph records are written unarmed, and `read_shards` doesn't window them.

**Validation.** Each replay is refused whole, counted by reason in
`CorrelationReport`, if any of these fail:

- ordinals are distinct;
- ordinals are within `n_nodes`;
- each record matches its node's signature.

A `hipGraphLaunch` record flagged `graph_launch` keeps unstamped kernels off the
launch range. Known limit: two nodes with identical signatures, swapped by the
runtime, can't be told apart.

**Attributes** (`gitm/tracer/kernel_attributes.py`). The side table combines:

- `layer_class` from the predicted graph;
- the model's declared layer kinds;
- `moe_phase`;
- range annotations (`L3/moe_routed#wave=2`). rocTX has no payload, so the name is
  the carrier.

**Memcpys** carry `launch_range` on both vendors. **Vendor**
(`gitm/tracer/vendor.py`) is decided from ranked evidence.

**NVIDIA counterpart.** `cupti_core.c` now builds the same capture-time map:

- NVTX callbacks keep a per-thread range stack;
- `GRAPHNODE_CREATED` names the node, skipping copies made inside
  `cudaGraphInstantiate`;
- `GRAPHNODE_CLONED` links clones to their original.

The decoder merges records that share an id. Along a clone chain, the name nearest
the captured node wins.

## Does ROCm 7.2.3 stamp each kernel of a graph launch? (source review)

For vLLM's capture, yes, under the default settings.

1. **The request fires per packet.** rocprofiler-sdk `hsa/queue.cpp`
   (`WriteInterceptor`) loops over every AQL packet in a queue write. For each
   kernel-dispatch packet it calls `populate_external_correlation_ids(...,
   KERNEL_DISPATCH, ...)`. That function overwrites the stamp each time
   (`tracing/tracing.hpp`) and invokes the tool's request callback
   (`external_correlation.cpp`, `get()` → `invoke_callback`). Barrier packets are
   skipped. The stamp's thread is the one owning the current correlation ID.
2. **CLR writes the graph's packets synchronously on the calling thread.** In CLR
   `hipGraphLaunch`, a single-stream graph (`max_streams_ == 1`, which is what vLLM
   captures) runs `GraphExec::EnqueueGraphWithSingleList`
   (`hip_graph_internal.cpp`). It walks `topoOrder_`, which for a linear capture is
   capture order. Kernel nodes go through `dispatchAqlPacketBatch`, one batch
   behind one doorbell; other nodes are enqueued individually in the same loop.
   This happens inside the HIP API call, so the dispatches share its correlation
   ID, and the tool's per-thread graph stack is live.
3. **`kernel_id` validation is sound.** The host-symbol `kernel_id` is copied from
   the device symbol's (`code_object.cpp:939`, `host_data.kernel_id =
   sym_data.kernel_id`). That is the same id the dispatch record carries.

What breaks it:

- **`AMD_DIRECT_DISPATCH=0`.** A worker thread submits the graph's packets, outside
  the launch's correlation scope. The collector records the setting in its `meta`
  record, and `read_shards` warns. Kernels come back unnamed, not misnamed.
- **A node that emits more than one packet, or the first-launch hidden-heap init.**
  Ordinals shift, so that replay is refused.
- **Forked captures (`max_streams_ > 1`).** These run `RunNodes` across streams.
  Signature validation guards them, within the known limit above.

## Verification

- `python scripts/check_collectors.py [--vendor amd|nvidia|all] [--ref develop]
  [--cuda 13]` compiles both collectors `-Werror` against real headers, on any
  machine:
  - ROCm 7.2.3 and develop;
  - CUDA 12 and 13 (CUPTI with the node map).
- `gitm/tracer/emulate.py` renders fixture ground truth as each collector records
  it. On every exact mode, for both vendors:
  - identity is 100% correct, with 0 wrong;
  - residuals, violations and Granger/DR rankings match ground truth;
  - every hazard (including worker-thread dispatch) gives 0 wrong.

## On hardware

**MI355X, ROCm 7.2.3, `run_env(..., nvtx=True)`:**

1. `meta.direct_dispatch == 1`.
2. Eager: `identity == "range_id"`, and `stamp_containment_disagree == 0`.
3. Graphs: no `duplicate_ordinal` / `untracked_launch` refusals, and every
   `graph_exec` has a `capture_id`.
4. `graph_unnamed` is not reported. If it is, torch.compile may have traced the
   range hooks away.
5. Overhead: run e1 (`docs/mi355x_experiment_plan.md`) with and without
   `GITM_TRACE_NVTX`.

**NVIDIA:** run `test_replayed_kernels_take_their_capture_range_end_to_end` with
`run_env(..., nvtx=True)` exported.
