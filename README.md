# gitm-labs

<img width="1062" height="356" alt="image" src="https://github.com/user-attachments/assets/ffee3fc3-c42f-4fe5-9e31-c6a62a245f44" />


Behavioral compiler + intervention runtime for GPU-intensive workloads. Given a workload and a time budget, gitm-labs autonomously profiles, attributes, and applies kernel-level interventions to hit a target performance improvement — and produces a provenance report showing exactly what it changed and why.

## Install

```bash
pip install gitm-labs
```

NVIDIA GPU support:

```bash
pip install "gitm-labs[nvidia]"
```

Optional extras: `bench` (HFT/biotech/edge benchmark harness), `prometheus`, `otlp`, `s3`.

**Requires:** Python 3.10+, NVIDIA (NVML + CUPTI) or AMD (ROCm SMI + rocprof) GPU.

## Quick start

```bash
export GITM_S3_ROOT="s3://your-bucket/gitm"   # durable store for datasets + run outputs
export GITM_SCRATCH="/mnt/nvme/gitm"           # local ephemeral run dir (defaults to ~/.cache/gitm)

gitm run --workload hft-lob --budget 24h --target 15%
```

Workloads: `hft-lob` (HFT order-book), `af2` (AlphaFold2 protein inference), `kitti` (3D LiDAR detection).

`--budget` is the wall-clock time limit. `--target` is the performance improvement fraction gitm-labs commits to delivering, or issues a diagnostic explaining why the floor could not be met.

Verify your environment first:

```bash
gitm doctor
```

### Embedded API

```python
from gitm import optimize

result = optimize(engine, budget="24h", target=0.15)
```

## The 24-hour loop

gitm-labs runs a five-phase autonomous loop within the allotted budget:

| Phase | Hours | What happens |
|---|---|---|
| 1. Profile | 0–2 | Capture event + state telemetry; fingerprint workload; build predicted execution graph |
| 2. Attribute | 2–6 | Compute residuals against predicted graph; run causal attribution |
| 3. Rank | 6–12 | Query intervention library; rank candidates via counterfactual replay |
| 4. Apply | 12–20 | Apply top-N interventions with rollback gates |
| 5. Report | 20–24 | Stabilize; write provenance report (claim → evidence → intervention → delta) |

## Architecture

gitm-labs separates the **empirical** half (what happened) from the **predicted** half (what should have happened). Everything downstream operates on residuals — the difference between the two.

### Two telemetry planes

#### State telemetry (`gitm.telemetry`)

Point-in-time samples of GPU state at ~1 Hz: utilization, memory, power, clocks, temperature, throttle reasons, NVLink throughput, ECC counters.

Source: NVML (NVIDIA) / ROCm SMI (AMD). Cost: ~microseconds per sample.

#### Event telemetry (`gitm.tracer`)

Per-kernel activity records with start/end timestamps, stream IDs, and memory transfer events.

Source: CUPTI (NVIDIA) / rocprof (AMD). Required for kernel-time invariant checks.

### Deviation invariants

The monitor checks observed-vs-predicted against three invariants:

1. **Kernel-time** — per-kernel duration must lie within roofline bounds.
2. **Memory-traffic** — per-kernel bytes-moved must match predicted.
3. **Stream-concurrency** — predicted-concurrent kernels must overlap.

See [docs/invariants.md](https://github.com/GitM-Labs/runtime/blob/main/docs/invariants.md).

### Module responsibilities

| Module | Responsibility |
|---|---|
| `gitm.telemetry` | Vendor-backend autodiscovery, NVML/ROCm SMI samples, pluggable sinks |
| `gitm.tracer` | Event-telemetry capture (CUPTI/rocprof), trace schema, context manager |
| `gitm.planner` | Behavioral Compiler — roofline-based predicted execution graph |
| `gitm.optimizer.monitor` | Deviation monitor — residuals against 3 invariants |
| `gitm.optimizer.attribution` | Granger + doubly-robust on residual subgraph |
| `gitm.optimizer.replay` | Counterfactual replay for predicted intervention delta |
| `gitm.optimizer.qualification` | Workload fingerprint gate (commit / diagnose) |
| `gitm.optimizer.report` | Provenance chain renderer (claim → evidence → intervention → delta) |
| `gitm.kernels` | Curated intervention library — 15–20 levers with applicability + safety |
| `gitm.agents` | Autonomous policy — selects interventions, drives rollback |
| `gitm.scheduler` | 24-hour loop phase orchestration |

## Data layout

Two environment variables control where data lives:

```bash
export GITM_S3_ROOT="s3://your-bucket/gitm"   # canonical store (datasets + run outputs)
export GITM_SCRATCH="/mnt/nvme/gitm"           # local ephemeral dir (defaults to ~/.cache/gitm)
```

Layout under `$GITM_S3_ROOT`:

```
datasets/{hft,biotech,edge}/    # benchmark inputs (immutable, sha256-pinned)
runs/                            # baseline + run outputs
traces/                          # captured event-telemetry traces
telemetry/                       # state-telemetry samples
```

Local scratch is ephemeral and synced to S3 after each run.

## Primary interfaces

```python
# tracer
with gitm.tracer.capture(out_path: Path) -> ContextManager[Trace]: ...

# planner
graph = gitm.planner.predict_graph(model: ModelSpec, hw: HardwareSpec, batch: BatchConfig) -> Graph

# monitor
residuals = gitm.optimizer.monitor.residuals(trace: Trace, graph: Graph) -> Residuals
violations = gitm.optimizer.monitor.check_invariants(residuals, invariants) -> list[Violation]

# attribution
hypotheses = gitm.optimizer.attribution.attribute(residuals: Residuals, graph: Graph) -> RankedHypotheses

# report
report_md = gitm.optimizer.report.write(claims: list[Claim], provenance: Provenance) -> str
```
