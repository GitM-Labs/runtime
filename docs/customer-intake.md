# Customer profiler intake

How GTM asks a customer for profiler artifacts, what commands they run, and what
gitm can (and cannot) read from each format. No install of gitm on their side is
required for this path — they send files; we run `gitm analyze` on our side.

## What to ask for

Prefer **one of**:

1. **Nsight Systems** report (best for CUDA kernel timelines)
   - File: `*.nsys-rep`, **or** a pre-exported `*.sqlite` from nsys
   - Supported export versions: **2024.x and 2025.x** only
2. **PyTorch profiler** chrome-trace JSON
   - File: `*.json`, `*.json.gz`, or `*.pt.trace.json`

A directory dump is fine. We walk it recursively and skip unrecognized files
(listed in the report appendix).

Optional context that improves the report (not required):

- GPU SKU string if not embedded in the file (e.g. `NVIDIA H100 80GB HBM3`)
- Workload name / serving stack (e.g. vLLM decode, training step name)
- Approximate model size / batch if known

## Exact commands (customer side)

### Nsight Systems

Profile a representative run (adjust the launch command):

```bash
nsys profile -o workload_report --trace=cuda,nvtx,osrt --force-overwrite=true \
  <their-launch-command>
```

That produces `workload_report.nsys-rep`. They can send that file as-is.

If they cannot send `.nsys-rep` (size/policy), ask them to export sqlite with a
**2024 or 2025** nsys:

```bash
nsys export --type sqlite --output workload_report.sqlite workload_report.nsys-rep
```

Send `workload_report.sqlite`.

> We do **not** bundle or download nsys. If we only receive `.nsys-rep` and nsys
> is not on the analysis machine, we return the export command above for the
> customer to run.

### PyTorch profiler

Wrap a representative step:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # ... their step ...
    torch.cuda.synchronize()

prof.export_chrome_trace("workload_trace.json")
# or: gzip the file before sending
```

Send `workload_trace.json` (gzip is fine).

## What we run on our side

```bash
gitm analyze customer_dump/ --out report.md --json summary.json
# optional:
#   --sku "NVIDIA H100"          # override SKU label
#   --workload-id serving-prod   # single-file only
#   --device 0                   # multi-GPU: pick one device (never merges)
#   --keep-traces ./traces       # eng-side JSONL + internal headroom blocks
#   --strict                     # any bad file aborts
```

Output:

- `report.md` — customer-ready headroom assessment (one section per file)
- `summary.json` — machine-readable per-workload ceiling distance, gap classes,
  confidence, caveats (for decks)

## What each format can support

| Signal | Nsight Systems | PyTorch chrome trace |
|--------|----------------|----------------------|
| Kernel names / durations / streams | Yes | Yes |
| Grid / block dims | Yes | Often (defaults to 1 if missing) |
| Memcpy bytes + endpoints | Yes | Partial (bytes; direction from name) |
| Sync events | Yes | **No** — attribution is coarser |
| Per-kernel DRAM counters | **No** | **No** |
| Power / clocks / throttling (NVML) | **No** | **No** |
| FLOP counters | **No** | **No** |

Imported traces are always **trace-only** confidence. They **never** qualify for
the guaranteed optimization floor. Floor commitment requires a gitm-captured run.

## Known limitations (be honest with the customer)

- **Kernel classification** is name-keyword based today. Unfamiliar libraries or
  mangled names may land as unmodeled work until telemetry-based classification
  lands — deviation/gap labels on exotic SKUs inherit that weakness.
- **Catalogue peak rates** are illustrative (A100 defaults when the SKU is
  unknown or not in the table). The report says so rather than inventing a peak.
- We **do not** invent per-kernel bytes, FLOPs, or NVML samples from names. Gaps
  are stated as caveats instead of filled with estimates.
- Multi-GPU files are analyzed **per device** by default (never merged into one
  timeline). A node-level rollup reports skew, collective time, and exposed
  communication. Pass `--device N` to analyze a single device only.

## What not to accept (this path)

Nsight Compute, TensorBoard, Perfetto, JAX/XLA profiles, ROCm/rocprof dumps —
out of scope for intake today. Ask for nsys or torch chrome-trace instead.
