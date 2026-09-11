# ROCm / MI355X port — tracer + Kimi K2.5 deployment

How the GITM experiments run on AMD (MI300X/MI355X), what maps to what, and the
bring-up order on a fresh box. The NVIDIA path is unchanged; everything here is
additive.

## Mechanism map

| NVIDIA | AMD | Notes |
|---|---|---|
| CUPTI Activity API | rocprofiler-sdk buffer tracing | ROCm ≥ 6.2; MI355X ships ROCm 7.x. `roctracer`/`rocprofiler` v1 are the deprecated predecessors — we do not use them. |
| `CUDA_INJECTION64_PATH` | `ROCP_TOOL_LIBRARIES` | Same follows-children property: the HIP runtime (via rocprofiler-register) dlopens the tool in **every** process that initializes HIP, so vLLM/SGLang engine-core children are covered. Colon-separated list, unlike the single NVIDIA path. |
| `libgitm_inject.so` | `libgitm_rocm_inject.so` | Built by `python -m gitm.tracer._rocm.build`. Emits the identical per-pid JSONL shards; `injection.read_shards` and `_cupti_decode` are shared, unmodified. |
| NVTX + `NVTX_INJECTION64_PATH` | rocTX + `LD_PRELOAD` of the roctx shim | The *collection* side needs no extra variable — markers are a tracing service of the same tool — but AMD has its own B200-class emission gotcha, found live on ROCm 7.2.3/MI355X: PyTorch links `torch.cuda.nvtx` to the **legacy** `libroctx64` (roctracer lineage), whose calls never reach rocprofiler-sdk's marker service (an sdk-roctx push produced a marker record; a libroctx64 push produced nothing, silently). The correlation arm therefore preloads `libgitm_roctx_shim.so` (built alongside the tool) into the *emitting* process, which interposes the legacy symbols and forwards them to the sdk's roctx. `run_env(nvtx=True)` sets it automatically. |
| `CUPTI_ACTIVITY_KIND_RUNTIME` + `DRIVER` | `HIP_RUNTIME_API` service | HIP has no runtime/driver split; hipBLASLt and torch both launch through the HIP runtime, so one service covers what needed two kinds (and one painful lesson) on NVIDIA. |
| kernel name on the activity record | `CODE_OBJECT` callback at load | Dispatch records carry only a `kernel_id`; the tool keeps an id→name map from code-object load callbacks. A tool attached after load would see anonymous kernels — irrelevant under injection, which is registered before HIP init. |
| `cuptiGetTimestamp` (window clock) | `rocprofiler_get_timestamp` via ctypes | `injection.clock_now()` dispatches on the active vendor. Same-domain guarantee is load-bearing: a wrong clock silently windows out the whole trace. |
| SYNCHRONIZATION records | — none | rocprofiler-sdk has no sync activity kind. AMD traces carry no `sync` events; consumers already tolerate absence. |
| `registers_per_thread` | arch VGPR count | Taken from the code-object symbol; the AMD occupancy-limiting analog. |
| dropped records: no signal | `drop_count` per buffer | The counter CUPTI never gave us. Emitted in-band as `{"kind":"meta","dropped_records":N}`; `read_shards` warns if non-zero, so a lossy AMD trace is *detectable* (the NVIDIA NVTX-lossiness question stays open). |

Semantics normalized in the tool so downstream sees one schema:

* **grid** — dispatch records report work-items (HSA convention); the tool
  divides by workgroup size so `KernelEvent.grid_*` means blocks on both vendors.
* **roctx has no range ids** — push/pop pairing is a per-thread stack in the
  tool, ids synthesized from one atomic counter; `pair_markers` joins the halves
  exactly as on NVIDIA.
* **copy kinds** — mapped onto the CUPTI `CUpti_ActivityMemcpyKind` ints the
  decoder already speaks.

## Bring-up on a fresh MI355X box (bare, no k8s)

```bash
# 0) sanity: GPUs + sdk present
rocm-smi
ls /opt/rocm/include/rocprofiler-sdk/rocprofiler.h /opt/rocm/lib/librocprofiler-sdk.so*

# 1) build the tool
python -m gitm.tracer._rocm.build

# 2) render the run env (note: no NVTX_INJECTION64_PATH on AMD)
python - <<'EOF'
from gitm.tracer.injection import run_env
for k, v in run_env("/tmp/gitm/trace.jsonl", nvtx=True).items():
    print(f"export {k}={v}")
EOF

# 3) smoke test: export the env, run any small torch/HIP workload while a
#    second shell arms the window via gitm.tracer.capture(); confirm the shard
#    $GITM_TRACE_OUT.<pid> fills with kernel records whose names resolve.
```

Verify during the smoke test, in order:

1. **Tool loads**: the shard file appears at HIP init (even empty). If not,
   check `ROCP_TOOL_LIBRARIES` is absolute and the process actually initializes
   HIP.
2. **Clock domain**: `injection.clock_now()` in the parent must land inside the
   span of the shard's kernel timestamps. If the window filter returns empty on
   a shard that clearly has records, this is the first suspect.
3. **Correlation**: with `GITM_TRACE_NVTX=1` and a `torch.cuda.nvtx` range
   around the launches, kernels inside the range must come back with `range_op`
   set. This proves the roctx → marker-service path end to end.
4. **Loss**: no `meta`/`dropped_records` warnings at merge.

## Kubernetes

`deploy/k8s/mi355x-kimi.yaml` is the apply unit: SGLang serving Kimi K2.5 on
`amd.com/gpu: 8` plus the gitm harness as a sidecar, one pod, shared `/scratch`.
Read the header comments — the placeholders (image digests, storage class,
model id) must be filled, the AMD device plugin must be installed once per
cluster, and `shareProcessNamespace: true` is required for correctness, not
hygiene (shard-liveness probing crosses containers).

The serving image needs no gitm install: an init container copies the built
tool into `/scratch/lib`, and `ROCP_TOOL_LIBRARIES` points there. Keep the
ROCm minor version of the gitm image and the serving image aligned — the tool's
rocprofiler-sdk ABI must match the runtime that loads it.

Run an experiment:

```bash
kubectl apply -f deploy/k8s/mi355x-kimi.yaml
kubectl -n gitm-system wait --for=condition=ready pod -l app=kimi-k25 --timeout=60m
kubectl -n gitm-system exec -it deploy/kimi-k25-mi355x -c gitm -- \
    python -m gitm.bench --endpoint http://localhost:8000 ...
```

Speculative decode: the manifest enables Kimi's MTP head through SGLang's
NextN-style flags. The streamed `avg_decoded_tokens_per_iter` is the live
acceptance signal — **1.0 means the spec-decode path is not engaged** (that is
plain autoregressive decode, whatever the flags say), >1.0 is the accepted
draft rate the decode experiments measure.

## Known caveats

* `rocm_inject.c` is compiled against the installed rocprofiler-sdk headers;
  like `cupti_core.c`'s versioned-struct pins, a field rename in a future sdk
  fails the build loudly rather than corrupting offsets silently. It has not
  yet been compiled on the MI355X box — expect the first
  `python -m gitm.tracer._rocm.build` to be the real review of the sdk's
  current field spellings.
* In-process (non-injected) capture has no AMD backend; `capture()` degrades to
  a well-formed no-op outside injection. Injection is the only mode that sees a
  serving engine's children anyway, so this costs nothing for the experiments.
* `gitm bench profile` on AMD shells out to legacy `rocprof` CSV stats
  (`gitm/bench/profile.py`); on a ROCm 7 box prefer the injected tracer, and
  treat that CLI path as due for a `rocprofv3` migration.
