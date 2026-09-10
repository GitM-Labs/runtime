# GLM 5.2 runtime validation

This is a preparation and offline attribution scheme, not evidence that GLM ran
on H200. The local checks exercise Python dispatch, graph construction and
synthetic CUPTI records. The native collector, installed vLLM implementation,
checkpoint compatibility and real performance require the benchmark server.

## Audit findings and repairs

| Path | Failure | Repair |
| --- | --- | --- |
| `gitm capture serve --model ...` | Parser did not accept `--model`; default command served another pinned experiment. | Explicit model builds `vllm serve MODEL`; conflicting full command is refused. |
| `serve/model_config.py` | GLM was detected, then passed to the DeepSeek config gate/reader. | Dispatch GLM to its own reader. |
| `serve/attach.py` | Sidecar predictor only distinguished hybrid from DeepSeek. | Dispatch GLM to `predict_glm_graph`. |
| `tracer/injection.py` → `_cupti_decode.py` | Production shards merged before marker pairing and correlation; worker-local IDs could collide. | Preserve shard PID, partition before pairing/correlation, retain PID on kernels. |
| `tracer/vllm_stats.py` | Missing MLA/indexer/norm/embedding mappings; shared-expert children mapped as dense FFN. | Add GLM mappings and ancestor-aware expert mappings. |
| `_cupti_decode.py` | vLLM module labels recognized only Python's single-quoted repr. | Accept JSON double-quoted labels too. |
| `optimizer/deviation.py`, `monitor.py` | Any enclosing range overrode the kernel, including arbitrary containers and standalone quant/norm/collective helpers. | Common resolver preserves recognized op ranges, falls back for unknown labels, separates named helpers. |
| `deviate --by-phase` | Returned before graph construction and ignored JSON mode. | With a model, include graph comparison and a separate phase diagnostic; JSON includes both. |
| Multi-worker deviation | Sum of all worker time compared against one rank's floor. | Require one worker/device selection for graph comparisons; no cross-worker phase propagation. |

No undefined-name import errors were found by the Python static check. Existing
dense-family `NotImplementedError` paths are outside GLM dispatch. The planner is
an analytical graph, not executable inference kernels. Its approximate terms do
not certify that a serving backend implements the same graph.

## Match the server to the prediction

The catalogue's `glm-5.2-fp8` entry assumes FP8 KV storage as well as FP8 weights,
with BF16 exceptions and FP32 routing. Merely selecting the FP8 checkpoint does
not select FP8 KV storage. Enable it explicitly when comparing to this entry.
Save the exact checkpoint revision/config, vLLM/torch/CUDA versions, GPU topology,
launch command, logs, run manifest and serving summary with every result.

On the Linux GPU server, first exercise the existing preflight with the actual
serve command. This checks the installed environment; do not bypass rejected
flags to obtain a nominally successful run.

```bash
gitm capture serve --dry-run --tp 8 --out validation/preflight -- \
  vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --max-num-seqs 32
```

Then take a bounded eager identity capture. Explicit command form exposes all
serving decisions; `gitm capture serve --model zai-org/GLM-5.2-FP8` is supported
but its other defaults do not reproduce the benchmark shape by themselves.

```bash
gitm capture serve --nvtx --tp 8 --concurrency 32 --requests 64 \
  --input-tokens 8192 --output-tokens 128 --out validation/eager -- \
  vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --max-num-seqs 32 --enforce-eager
```

The input length is a client target, not proof of tokenized prompt length. The
window includes prefill, ramp-up, decode and drain. Concurrency 32 does not prove
that every scheduler step has batch 32, and KV length grows during generation.
Record scheduler steps and actual token counts. Do not derive `N` from completed
requests, kernel count, or output tokens divided by 32.

`capture attach` requires a server started with the injection environment; it
cannot retrofit CUPTI into an arbitrary live process. For another window on a
server launched here, use `--keep-server`, then `gitm capture attach --list` and
select that server's PID. NVTX must also have been enabled at startup.

## Gate attribution before reading ratios

Inventory workers and attribution (the first invocation intentionally fails the
single-worker gate on a merged trace, but writes the inventory):

```bash
python -m gitm.optimizer.validate_trace validation/eager/trace.jsonl \
  > validation/inventory.json

# Choose a kernel-producing worker PID from inventory.json, not the frontend PID.
python -m gitm.optimizer.validate_trace validation/eager/trace.jsonl \
  --pid WORKER_PID --device 0 > validation/worker-audit.json
```

Use the reported worker-local device ID; it may not be 0. Repeat for all eight
workers. Re-capture older merged traces that lack PIDs: identical local device
IDs cannot reconstruct lost worker identities or undo incorrect correlation.
The defaults target H200, batch 32, KV 8192, TP8/EP8, eight workers,
90% modeled device time and 80% canonical NVTX device time. Thresholds are
explicit review criteria, not calibrated performance guarantees. A failure
returns exit code 1; unreadable input/model returns 2. Exit 0 passes attribution
criteria only. Missing ops and the largest raw NVTX/name disagreements remain
review items even when coverage passes.

Check especially:

* IndexShare: no indexer projection/score on shared layers. The validator checks
  canonical range op/layer pairs against the predicted graph.
* MLA q/kv projections: anonymous GEMMs need specific ranges. An enclosing
  attention block does not establish which projection ran.
* Shared versus routed expert work, and standalone activation quantization,
  norms, permute/combine and collectives inside larger module ranges.
* Fused projections and absorbed MLA: the catalogue models unabsorbed MLA.
  An absent `attn_kv_b` is a possible backend difference, not automatic speedup.
* Name-only mappings are guesses. Do not transfer a bare GEMM name from eager
  mode to graph mode as a unique identity; the same name can serve many ops.

## Compare a measured step window

```bash
gitm plan glm-5.2-fp8 --gpu H200 --batch 32 --kv-len 8192 \
  --tp 8 --ep 8 --json > validation/plan.json

gitm deviate decode-window.jsonl --model glm-5.2-fp8 --gpu H200 \
  --batch 32 --kv-len 8192 --tp 8 --ep 8 --steps N \
  --pid WORKER_PID --device 0 --by-phase --json > validation/deviation.json
```

`decode-window.jsonl` must be a window selected using measured scheduler step
boundaries, with its actual shape and `N`. The capture client does not currently
produce that fixed-shape decode slice automatically. For variable shapes, price
each observed step's batch/context/prefill separately; do not multiply a single
static step by an unrelated count. `--by-phase` is a diagnostic breakdown, not
a decode filter: unknown names and nearest-anchor inference cannot prove a
homogeneous decode step, especially with chunked prefill. Multi-worker phase
reports use direct kernel-name evidence only.

Finally repeat the same workload in normal graph mode with CUPTI and without
`--nvtx`/`--enforce-eager`, plus a `--no-trace` baseline. Compare throughput and
latency to quantify instrumentation overhead. Eager identity timings cannot
validate graph-mode launch overhead or fusion. Keep missing attribution visible
and retain the raw trace; a coverage gap is not optimization headroom.

The live sidecar uses batch 1 and maximum model length as a placeholder shape;
it is useful for config dispatch inspection, not a measured scheduler-step floor.
Use the explicit plan above for the requested reference. Per-kernel deviation
filters also cannot validate multi-kernel fused/grouped nodes; use summed op
totals over measured steps for this scheme.
