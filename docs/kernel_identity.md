# Kernel identity: from name-substring tags to NVTX/correlation-id ranges

## Problem

Pairing an observed kernel to its predicted graph node (`gitm.optimizer.deviation.classify_op`)
currently works by case-insensitive substring match against the kernel's mangled
`name` string — `"qkv"` → `qkv_proj`, `"flash_fwd"` → `attn_score_value`, etc.

This is a name *guess*, not an identity, and it fails silently on exactly the
kernels that matter most: bare cuBLAS/cutlass GEMMs (e.g.
`ampere_fp16_s16816gemm_fp16_128x128_...`) carry no projection tag in their
name at all. Confirmed against a real vLLM/L4/CUPTI trace
(`tests/test_deviation_alignment.py::test_classify_op_matches_real_vllm_kernel_names`):
these are ~35% of launches, reused identically across qkv/out/gate_up/down/
lm_head, and land as `<unmodeled>` — every departure on the dominant GEMM cost
is invisible to the deviation monitor. It's also why `KernelResidual.layer` is
hardcoded `None` today (`gitm/optimizer/monitor.py:50`): a name has no layer
index to recover.

## What correlation_id actually gives us (and what it doesn't)

CUPTI's `correlationId` links a device-side `KERNEL` activity record to the
host-side API call (`cudaLaunchKernel`/`cuLaunchKernel`) that issued it — both
records carry the same id. **It is not, by itself, an op/layer identity.**
Today's shim (`gitm/tracer/_cupti/cupti_shim.c`) only enables
`CONCURRENT_KERNEL`, `MEMCPY`, `SYNCHRONIZATION` — it never enables the
`RUNTIME` activity kind that would surface the host-side launch record, so
`correlation_id` is currently decoded (`_cupti_decode.py:68`) but dangling:
nothing on the host side to correlate it *against*.

The actual identity source has to be a **name we control on the host side** —
an NVTX range pushed around each op's forward call
(`torch.cuda.nvtx.range_push(f"L{layer}/{op}")` ... `range_pop()`).
`correlation_id` is the plumbing that lets a kernel find *which* NVTX range
was open when it was launched; it is not the name itself. That's a scope
correction from "just use correlation_id" — the real work is the NVTX
instrumentation, correlation_id is how it gets attached to the kernel.

## The correlation chain

Three record kinds, two different clocks:

```
NVTX range   (host clock)  "L3/qkv_proj"  [range_start_ns, range_end_ns]
     |  contains (host time)
RUNTIME rec  (host clock)  cudaLaunchKernel  correlation_id=X  [host_start_ns, host_end_ns]
     |  same correlation_id
KERNEL rec   (device clock) correlation_id=X  [start_ns, end_ns]   <- what we capture today
```

**Correlation must chain kernel → RUNTIME (by `correlation_id`) → range (by
host-timestamp containment of the RUNTIME record, not the kernel record).**
Kernel timestamps are device-clock; range timestamps are host-clock. A kernel
frequently finishes executing well after its enclosing range has popped
(async launch) — testing containment of the kernel's own `[start_ns, end_ns]`
against the range's host window will look correct on a synthetic same-clock
test and produce silently wrong attributions on a real async trace. The
RUNTIME record is the only one that shares a clock domain with the range.

Also watch thread affinity: if the launching code is multi-threaded, ranges
pushed on one thread must not swallow launches from another — match on the
RUNTIME record's thread id, not just timestamp overlap.

## Design

### 1. Shim changes (`cupti_shim.c`) — needs a CUDA/GPU box, not buildable here

- Enable `CUPTI_ACTIVITY_KIND_RUNTIME` (host-side launch-API records: name,
  `correlationId`, host `start`/`end`, `threadId`).
- Enable `CUPTI_ACTIVITY_KIND_MARKER` (+ `MARKER_DATA`) to capture NVTX
  push/pop ranges as activity records (name, host timestamp, id).
- New `gitm_record` kinds `REC_RUNTIME`, `REC_MARKER`; extend `rec_to_dict`.
  Keep `ingest()`'s existing comment/lesson in mind — verify on real hardware
  whether enabling these introduces the kind of duplicate/zeroed-timestamp
  record pairing that `CONCURRENT_KERNEL` vs `KERNEL` did.

### 2. Host-side range instrumentation — needs the attach mechanism verified first

Register a pre/post-forward hook on each named submodule of the model
(`model.layers.{i}.self_attn.qkv_proj`, `.o_proj`, `.mlp.gate_up_proj`, ...)
that does `nvtx.range_push(f"L{i}/{op}")` / `range_pop()`. Near-zero overhead
(host-side only), and the layer index falls out of the module's qualified
name for free — the thing `classify_op` can never recover.

This is trivial for the **in-process** path (`gitm/runtime_driver.py` already
runs `work()` inside the process holding the model — hooks register directly
on the live `nn.Module` tree before `capture()` starts). It is *not* trivial
for the **out-of-process attach** path (`gitm/deploy/attach.py`): that module's
docstring describes "install removable telemetry shim into the job's
user-space env" but the actual injection mechanism into an already-running
PID isn't implemented in what I've read — `attach_job` only resolves a PID
and returns a plan. Land this against the in-process path first; treat attach
injection as a separate, explicitly-scoped follow-up once that mechanism
exists (I've asked a background research agent to confirm this reading before
we commit to it).

### 3. Decode/correlate (`_cupti_decode.py`)

New function, pure Python, unit-testable without a GPU:

```python
def correlate_kernels_to_ranges(records: list[dict]) -> list[dict]:
    """Attach range_op/range_layer to kernel dicts via correlation_id.

    kernel.correlation_id -> RUNTIME record (same correlation_id) -> host
    [start_ns, end_ns] -> innermost open MARKER range containing that host
    window, matched on thread_id. Kernels with no match keep
    range_op=None (falls back to name-based classify_op downstream).
    """
```

### 4. Schema (`gitm/tracer/schema.py`)

Add two optional fields to `KernelEvent`, default `None` — backward compatible
with existing trace JSONL files:

```python
range_op: str | None = None
range_layer: int | None = None
```

### 5. Pairing (`gitm/optimizer/deviation.py`, `gitm/optimizer/monitor.py`)

`classify_op` stays as-is (it's still the fallback for unlabeled traces — HFT,
edge, or any capture without NVTX instrumentation). Both `residuals()` and
`deviating_kernel_indices()` change their op lookup to:

```python
op = ok.range_op or classify_op(ok.name)
layer = ok.range_layer  # real layer index when range-identified, else None
```

This is a strict superset of current behavior: no range → identical to today.
Range present → exact identity (recovers the ~35% unmodeled GEMMs, since
they're inside a hooked module's forward regardless of their opaque name) and
a real `layer`, which `KernelResidual.layer` and `Violation.layer` can finally
carry instead of always `None`.

## What's testable on this machine vs. not

| Piece | Testable here (no GPU) |
|---|---|
| `KernelEvent.range_op`/`range_layer` schema fields | Yes |
| `correlate_kernels_to_ranges()` — synthetic RUNTIME/MARKER dicts | Yes — extend `tests/test_deviation_alignment.py`'s `_k()` helper |
| Pairing fallback logic in `deviation.py`/`monitor.py` | Yes |
| CUPTI shim RUNTIME/MARKER activity kinds | No — needs CUDA toolkit + GPU |
| NVTX forward-hook instrumentation, in-process | No — needs `torch`+CUDA to actually exercise, but the hook logic itself can be written and reviewed here |
| Out-of-process attach injection | Blocked on confirming the attach mechanism exists at all |

## Plan

1. Schema fields + `correlate_kernels_to_ranges()` + pairing fallback, with
   synthetic-record unit tests — buildable and testable on this box.
2. NVTX forward-hook module (in-process path) — written here, exercised on a
   GPU box.
3. Shim `RUNTIME`/`MARKER` activity kinds — written here, built and verified
   on a GPU box (duplicate/zeroed-timestamp check per the existing
   `CONCURRENT_KERNEL` lesson).
4. Attach-path injection — separate follow-up, scoped after confirming the
   attach mechanism.

Starting with (1) now.

## CUDA graphs: the launch-enclosing range is not the op

*Added 2026-10-05, branch `tracer/graph-node-identity`.*

Everything above assumes a kernel's launch is its own API call. Under CUDA graphs
it isn't. A replay is one `cudaGraphLaunch`, and every kernel in it carries that
launch's `correlation_id`. So `kernel → runtime → enclosing range` gives every kernel
in the replay the same range: whatever was open around the launch. That might be a
step, a layer group, or nothing. Production vLLM runs piecewise or full graphs by
default. Only `--enforce-eager` captures avoid this.

Before this branch, the collector dropped CUPTI's `graphId` and `graphNodeId`. So
nothing downstream could tell a replayed kernel from an eager one. If a layer or op
range happened to be open around the launch, every kernel in the piece took that op
and layer. `observed_op` prefers a known `range_op` over the name guess, so the
wrong identity won. That is a silent mis-attribution, not a gap.

### What this branch does

- **Collector** (`cupti_core.{h,c}`, `cupti_shim.c`, `cupti_inject.c`): copies
  `graphId` and `graphNodeId` off `CUpti_ActivityKernel9` into both emitters. Both
  are 0 for an eager launch. *Not compiled on a CUDA box yet.*
- **Schema** (`KernelEvent`): adds `graph_id`, `graph_node_id` and `launch_range`.
  All are optional, so old traces decode unchanged, and 0 decodes to `None`.
- **Correlation** (`correlate_kernels_to_ranges`): for a kernel with a nonzero
  `graph_id`:
  - The launch-enclosing range goes to `launch_range`. That is a step or time-series
    label, which is what it actually is.
  - `range_op` and `range_layer` come only from a `graph_node` record:
    ```
    graph_node {kind:"graph_node", graph_node_id, name, cloned_from}
    ```
    `cloned_from` is followed back to the captured node.
  - With no node map, the op is `None` and falls back to name classification.
    It no longer silently takes the step's range.

### The Nsight approach, next on the collector

Nsight Systems projects NVTX ranges onto replayed kernels by recording, at graph
*capture*, which range was open when each node was created. Graph nodes are created
during stream capture, so the forward pass does run under its ranges once. The
collector can do the same with standard CUPTI. No vLLM patch is needed.

1. Subscribe `CUPTI_CB_DOMAIN_NVTX`. Push and pop callbacks are synchronous, so the
   collector can keep a per-thread stack of open range names. The activity
   `MARKER` records arrive later in buffers, which is too late for this.
2. Subscribe `CUPTI_CB_DOMAIN_RESOURCE`:
   - On `CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED`, read the node id and emit
     `graph_node{id, name=top of this thread's stack}`.
   - On `..._GRAPHNODE_CLONED` (instantiate or clone), emit
     `graph_node{id=clone, cloned_from=original}`.
3. `injection.merge` must pass `graph_node` records through without windowing them,
   the same way it does `marker`/`runtime`. Capture happens at engine start, long
   before the trace window opens. This was left out of this branch to stay clear
   of #159.

Verify on hardware before relying on it:

- Does `Kernel9.graphNodeId` name the captured node or the instantiated clone?
  The join handles either, but the clone callback must actually fire.
- Do the vLLM NVTX ranges fire during capture under `torch.compile`?
  `instrument_model`'s forward hooks may be traced away inside compiled regions.
  vLLM's `--enable-layerwise-nvtx-tracing` should be checked separately.
- What is the overhead of the NVTX callback domain on top of `GITM_TRACE_NVTX`'s
  existing cost?

### Attributes beyond `L{layer}/{op}`: a side table, not a longer name

Heterogeneous models need more than an op and a layer to classify a kernel.
DeepSeek-V4 has two examples:

- **KV cache.** Only layers 2, 8, 14 and 20 compress and cache. The rest read a
  source layer's cache, at different compress ratios. See
  `docs/deepseek-v4.1-flash/DESIGN-NOTE.md`.
- **Expert-parallel waves.** These run dispatch, expert GEMM and combine
  concurrently across waves.

Don't encode these into the range name that `parse_range_name` parses. Join them in
Python as a map keyed by what correlation already recovers:

| key | attributes | source | when |
|---|---|---|---|
| `range_layer` | archetype, KV source layer, compress ratio, reads-from | planner catalogue | static: this is next |
| `(range_layer, range_op)` | MoE phase (dispatch / expert / combine) | op vocabulary | static |
| range instance | wave index, expert group | NVTX payload (`nvtxEventAttributes_t.payload` and category, read through CUPTI `MARKER_DATA`) | dynamic: later |
| kernel | `stream_id`, overlap with other streams, `graph_node_id`, grid/block | the kernel record | already captured |

Wave identity is dynamic, so it can't come from the layer. The standard carrier is
the NVTX payload and category, not the name string. Until something pushes it,
stream id plus time overlap is the lead for separating concurrent waves.

### Memcpy and step boundaries

A memcpy's `correlation_id` links it to its own API call, not to an NVTX range.
It reaches a range the same way a kernel does: by host-time containment on the
same thread. vLLM's per-step H2D input preparation and D2H sampled-token copy run
outside the graph. That makes them likely step boundaries that survive graph
replay when kernel ranges don't. This is a candidate time-series label alongside
`launch_range`, still to be measured.

## Status after the ROCm port (2026-10-06)

See `docs/rocm_correlation.md`. Landed for both vendors: unwindowed
`graph_node`/`graph_exec` in `read_shards`, the attribute side table
(`gitm.tracer.kernel_attributes`, with `#k=v` range annotations as the
dynamic carrier), and memcpy `launch_range`. Still to do: the CUPTI
capture-time collector. It should emit the same optional node signature fields
(`node_kind`, `grid`, `block`) so replay validation covers it too.
