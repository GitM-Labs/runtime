# Kernel identity on ROCm

The AMD port of the NVTX correlation design in `docs/kernel_identity.md` (through
#160, graph-node identity), built on rocprofiler-sdk. Collector:
`gitm/tracer/_rocm/rocm_inject.c`. Decoder contract: `gitm/distributed/correlate.py`.

## Mechanisms

**Eager kernels: range stamps.** rocprofiler-sdk's external-correlation request
service asks the tool, on the enqueuing thread, for a value to stamp on each
dispatch, copy and HIP API record. The collector answers with the innermost rocTX
range id on that thread, so kernels join their range by id. No timestamp or thread
matching is needed. Host-time containment over the HIP API records remains as a
fallback, and disagreements are counted.

**Graph replays.** ROCm 7.2.3 dispatch records have no graph identity, and the
sdk's `HIP_GRAPH` domain isn't in a release yet. The collector rebuilds its
documented recipe from HIP API callbacks:

- **Replay.** `hipGraphLaunch` ENTER/EXIT keeps a per-thread `(exec, ordinal)`
  stack, and each dispatch or copy inside a launch is stamped with the next ordinal.
- **Capture.** While a thread is stream-capturing, each launch, copy or memset call
  becomes a `graph_node` record. It names the open range and carries a signature:
  `kernel_id`, grid/block and node kind.
- **Link.** `hipStreamEndCapture` and `hipGraphInstantiate*` produce a
  `graph_exec{graph_id, capture_id, n_nodes}` record.

Graph records are written unarmed, because capture happens before any window
opens, and `read_shards` doesn't window them.

**Validation.** Position is only an identity if the replay runs in capture order.
Each replay is refused whole, counted by reason in `CorrelationReport`, if any of
these fail:

- ordinals are distinct;
- ordinals are within `n_nodes`;
- each record matches its node's signature: same symbol and geometry for a kernel
  node; a ROCclr blit or SDMA copy for a memcpy/memset node.

A refused replay keeps `launch_range` and falls back to name classification. A
`hipGraphLaunch` record flagged `graph_launch` also keeps unstamped kernels off the
launch range. Known limit: two nodes with identical signatures, swapped by the
runtime, can't be told apart. vLLM's single-stream capture doesn't reorder.

**Attributes** (`gitm/tracer/kernel_attributes.py`). The side table combines:

- `layer_class` from the predicted graph;
- the model's declared layer kinds;
- `moe_phase`;
- range annotations (`L3/moe_routed#wave=2`). rocTX has no payload, so the name is
  the shared carrier.

Attribution uses them only when asked (`stratify=`).

**Memcpys** carry `launch_range`, a step label that survives graph replay, on both
vendors.

**Vendor** (`gitm/tracer/vendor.py`). Evidence is ranked: driver beats PCI beats
device name beats torch build. A split within the top tier is a conflict, and
`GITM_VENDOR` overrides. Traces classify from the collector's `meta` record, or
from kernel-name dialect.

## Fixes to the shared path

- Marker halves pair by timestamp, not by arrival order.
- Capture-node names are normalized the same way as marker names.
- AMD vocabulary: Tensile `Cijk_` GEMMs, AITER MLA/paged attention, router, sort
  and quant kernels.
- `observed_op` keeps router kernels out of `moe_routed`. This also changes NVIDIA
  numbers on existing MoE captures.
- Granger and DR pair only series of comparable launch cardinality. Truncating
  every series to the shortest one let `lm_head` starve every per-layer pair.

## Verification

- `python scripts/check_rocm_collector.py [--ref develop]` compiles the collector
  `-Werror` against a release's real headers, on any machine.
- `gitm/tracer/emulate.py` renders fixture ground truth as each collector records
  it. On every exact mode, for both vendors:
  - identity is 100% correct, with 0 wrong;
  - residuals, violations and Granger/DR rankings match ground truth;
  - every hazard refuses with 0 wrong.
- The pre-graph-identity decode loses the injected cause; a test shows it.

## Hardware checks (MI355X, ROCm 7.2.3, `run_env(..., nvtx=True)`)

1. Eager (`--enforce-eager`):
   - `identity == "range_id"` inside layerwise ranges;
   - `stamp_containment_disagree == 0`.
2. Graphs: the stamp must advance per dispatch inside `hipGraphLaunch`. It fails in
   one of two ways:
   - dispatches inherit the launch's stamp, which shows up as `untracked_launch`;
   - the ordinal doesn't advance, which shows up as `duplicate_ordinal`.

   Either way, replays come back unnamed, never wrong.
3. Capture:
   - every `graph_exec` has a `capture_id`;
   - no `ordinal_out_of_range`;
   - `graph_unnamed` is not reported as a problem. If it is, torch.compile may have
     traced the range hooks away.
4. Signatures: `signature_kernel_id` on `hipLaunchKernel` nodes would mean the
   host-symbol and dispatch `kernel_id` spaces differ. In that case, compare
   geometry only.
5. Overhead: run e1 (`docs/mi355x_experiment_plan.md`) with and without
   `GITM_TRACE_NVTX`.
