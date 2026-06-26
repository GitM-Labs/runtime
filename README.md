# Git.M Runtime

<img width="668" height="384" alt="gitm_logo" src="https://github.com/user-attachments/assets/9fff0bb8-f87a-4c8b-a9c8-576b9218160d" />



Git.M is a job-level runtime that makes already-placed GPU workloads run closer to their hardware ceiling. It reads runtime and hardware telemetry, models how the workload should execute, attributes the gap between expected and actual performance to a cause, and applies runtime configuration changes to close that gap. No source code, model weights, or training data are accessed.

It sits one layer below orchestration. Kubernetes, Slurm, or your own scheduler places the job on a node; Git.M optimizes execution on that node once the job is running.

-----

## Contents

- [How it works](#how-it-works)
- [Deployment and installation](#deployment-and-installation)
- [Permissions and access](#permissions-and-access)
- [What gets tuned](#what-gets-tuned)
- [How optimizations are applied (configs, not rewrites)](#how-optimizations-are-applied)
- [Verification and the optimization floor](#verification-and-the-optimization-floor)
- [Supported environments](#supported-environments)

-----

## How it works

1. **Telemetry ingest.** Git.M reads state and event telemetry from the GPU and runtime (CUPTI and NVML on NVIDIA, ROCm telemetry on AMD). It does not read source code, model weights, or data.
1. **Predicted execution.** A GPU workload is treated as a repeating shape. From telemetry alone, Git.M models how the workload should execute and computes the achievable ceiling for that workload on that specific hardware.
1. **Causal attribution.** When actual execution deviates from the predicted path, Git.M isolates the deviating regions and attributes the gap to a cause (sync stalls, scheduling gaps, memory movement, collective communication, and similar).
1. **Constrained search.** The bottleneck class constrains the optimization search space. Within that space, Git.M selects from a library of known optimizations and runs agentic search for novel ones.
1. **Apply and validate.** Candidate optimizations are sandboxed and validated, then applied as runtime configuration. Changes are reversible.

Primary metrics are model FLOP utilization (training) and goodput at SLO / model bandwidth utilization (inference), not top-line GPU busyness.

-----

## Deployment and installation

Git.M ships as a **self-contained binary container** (a sealed artifact with a known footprint and no pip-time dependencies). A Python package is also available for development environments. The sealed container is the recommended path for environments with authorization or vulnerability-scanning boundaries.

It installs as a **node-level / job-level operator**, in the same place a Kubernetes operator or a Slurm prolog would sit. Your users do not change how they submit or run their workloads.

### Kubernetes

Git.M runs as a node-level operator. The workload is submitted to Kubernetes as usual; Git.M attaches at the job level on the node where the pod lands.

```bash
# Deploy the operator (sealed container)
kubectl apply -f gitm-operator.yaml

# Confirm the operator is running on GPU nodes
kubectl get pods -n gitm
```

### Slurm

Git.M attaches at the job level under Slurm the same way it does under Kubernetes. No scheduler replacement is required.

```bash
# Wrap the job step so the operator attaches to the allocation
srun gitm run -- <your-existing-launch-command>
```

### Standalone (no orchestrator)

Kubernetes or Slurm are not required. Telemetry at the runtime layer is sufficient for Git.M to operate on the job directly.

```bash
# Attach to a running or launching job on the local node
gitm attach --job <job-id>
```

The install itself is a few lines and adds no rewrite to the workload. Containers still pass through your normal image-scanning and authorization process before deployment.

-----

## Permissions and access

Git.M is least-privilege by design so it can run inside cloud-provider, national-lab, and regulated environments without elevated rights.

|Requirement                     |Needed?                                |
|--------------------------------|---------------------------------------|
|Root / elevated privileges      |No (default install runs in user space)|
|Kernel module                   |No                                     |
|Driver replacement              |No                                     |
|Source code access              |No                                     |
|Model weights access            |No                                     |
|Training / inference data access|No                                     |
|Outbound network / SaaS callback|No                                     |

Details:

- **User space.** Everything runs in user space inside the user’s own job. Telemetry comes from CUPTI and NVML attached to your own processes, which is standard user-space profiling and the same access the user already has to their own job.
- **Optimizations are config changes.** Optimizations are applied as environment and configuration changes plus runtime processes the user is already authorized to run. The tool ships and is shaped like any other user-space module or container image.
- **Data stays in the boundary.** Telemetry is processed in-cluster. The operator runs locally and nothing about the workload (code, weights, data) leaves the authorization boundary. There are no SaaS dependencies and no phone-home.

-----

## What gets tuned

Git.M tunes runtime settings that can change without recompiling or touching source code. Kernels are roughly 10% of the surface area; most of the headroom is in everything around them. The parameter space is organized into layers, and the bottleneck class determines which layer is searched.

- **Environment and launch.** Environment variables, library configuration, launch parameters.
- **CUDA / ROCm runtime.** Stream configuration, memory allocator settings (for better work-queue overlap).
- **Collective communication.** NCCL and MPI tuning knobs: algorithmic and protocol selection, thread counts.
- **Memory and data movement.** Staging and movement, latency hiding by overlapping with other work, restructuring around stalls.
- **Kernel variant selection.** Swapping in a better existing kernel or library implementation from available variants. This is selection, not rewriting.

Git.M does not rewrite or recompile the workload and does not edit a poorly written kernel in place. Where a kernel itself lowers the ceiling, Git.M substitutes a better available implementation or minimizes its impact (overlap, concurrency on separate streams, latency hiding) rather than pretending to configure the limit away.

-----

## How optimizations are applied

Git.M is **not a compiler**. It does not recompile or rewrite the job (no Modular-style rewrite).

The flow per workload:

1. Profile the workload from telemetry and compute the predicted ceiling (headroom).
1. Attribute each deviation from the ceiling to a cause.
1. Use the bottleneck class to constrain the search space.
1. Select candidate changes from a **library of known optimizations** (open-source mechanisms teams would otherwise hand-configure) and from **agentic auto-research** within the constrained space.
1. Sandbox and validate each candidate.
1. Apply validated changes as runtime configuration, dynamically while the job runs. Changes are reversible.

Because the same workloads recur, optimizations amortize across runs. A 20-hour job that runs weekly is ephemeral but is the same shape each time, so the profile and the applied configuration carry over to every run.

-----

## Verification and the optimization floor

Both deployment patterns start with a profiling step so you can see the gap before committing.

1. **Profile.** Git.M profiles a sample workload, shows the achievable ceiling, and reports how far the workload is from it.
1. **Floor.** Git.M commits to a guaranteed optimization floor (target 15%) on workloads that pass the headroom gate.
1. **Pay on verified gains.** Pricing is a performance fee on verified recovered throughput plus a platform fee. You pay on proven gains; if a workload does not clear the floor, it is protected.

Profiling reads telemetry only, so a blind evaluation is supported: hand Git.M a packaged workload (for example, one you believe is already well optimized and one you suspect has headroom) and it reports, per workload, how far each is from its ceiling and where the gap is.

-----

## Supported environments

- **Hardware:** NVIDIA (CUDA) and AMD (ROCm). Hardware agnostic by design.
- **Orchestration:** Kubernetes, Slurm, or standalone.
- **Serving runtimes:** Works alongside vLLM (a tracer plugs into vLLM to read its behavior) and other runtimes. Git.M optimizes execution beneath the serving layer rather than replacing the runtime.
- **Workloads:** Training and inference; text and non-text (multimodal, protein folding, high-frequency trading, robotics vision, and similar). Execution is treated as a shape, so the workload type is not a constraint.
- **Topology:** On-prem, cloud, and neocloud. All processing stays within the authorization boundary.
