# GPU headroom assessment

Diagnostic read from customer profiler exports. This is **not** a floor commitment.


---

## kineto_gpu_metrics_input

| | |
|---|---|
| **Workload** | kineto_gpu_metrics_input |
| **SKU (as reported)** | NVIDIA GPU |
| **Capture date** | 2026-07-16 19:00 UTC |
| **Source format** | torch-import |
| **Input** | `tests/fixtures/importers/real/kineto_gpu_metrics_input.json` |
| **Devices analyzed** | 1 |
| **Confidence** | **trace-only** |

### Confidence & method

This section is a **trace-only** diagnostic: computed from profiler timing only, without live device state or a gitm-attached capture.

- MBU computed from memcpy traffic only; per-kernel DRAM counters are not present in profiler exports.
- No device state plane (power, clocks, throttling); throttle-induced stalls appear as idle.
- Predicted floor uses catalogue peak rates for the reported SKU; unvalidated against live telemetry.
- Sync events absent from this trace format; sync-wait vs launch-latency attribution is coarse.
- SKU 'NVIDIA GPU' is not in the hardware catalogue; peak rates fall back to A100 defaults.


- Imported trace: diagnostic headroom read only. Floor commitment requires a gitm-captured run.




### Device 0

**Recoverable headroom: 99.6% of current wall time.**

| | |
|---|---|
| Observed wall time | 11.946 ms |
| Predicted floor | 0.048 ms |
| Busy fraction | 0.4% |
| Kernels in trace | 30 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 99.2% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 0.4% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).



### What we'd do next

A gitm-attached run confirms these reads with full telemetry (kernel events plus device state) and qualifies the workload for our guaranteed optimization floor. **Floor commitment requires a gitm-captured run, not an imported profiler file.** Import-only assessments stay diagnostic so commercial commitments track measured gains.




---

## kineto_resnet50_workers0

| | |
|---|---|
| **Workload** | kineto_resnet50_workers0 |
| **SKU (as reported)** | NVIDIA GPU |
| **Capture date** | 2026-07-16 19:00 UTC |
| **Source format** | torch-import |
| **Input** | `tests/fixtures/importers/real/kineto_resnet50_workers0.pt.trace.json.gz` |
| **Devices analyzed** | 1 |
| **Confidence** | **trace-only** |

### Confidence & method

This section is a **trace-only** diagnostic: computed from profiler timing only, without live device state or a gitm-attached capture.

- MBU computed from memcpy traffic only; per-kernel DRAM counters are not present in profiler exports.
- No device state plane (power, clocks, throttling); throttle-induced stalls appear as idle.
- Predicted floor uses catalogue peak rates for the reported SKU; unvalidated against live telemetry.
- Sync events absent from this trace format; sync-wait vs launch-latency attribution is coarse.
- SKU 'NVIDIA GPU' is not in the hardware catalogue; peak rates fall back to A100 defaults.


- Imported trace: diagnostic headroom read only. Floor commitment requires a gitm-captured run.




### Device 0

**Recoverable headroom: 40.6% of current wall time.**

| | |
|---|---|
| Observed wall time | 996.477 ms |
| Predicted floor | 591.526 ms |
| Busy fraction | 59.4% |
| Kernels in trace | 8772 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 16.5% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 24.1% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).



### What we'd do next

A gitm-attached run confirms these reads with full telemetry (kernel events plus device state) and qualifies the workload for our guaranteed optimization floor. **Floor commitment requires a gitm-captured run, not an imported profiler file.** Import-only assessments stay diagnostic so commercial commitments track measured gains.




---

## kineto_resnet50_workers4

| | |
|---|---|
| **Workload** | kineto_resnet50_workers4 |
| **SKU (as reported)** | NVIDIA GPU |
| **Capture date** | 2026-07-16 19:00 UTC |
| **Source format** | torch-import |
| **Input** | `tests/fixtures/importers/real/kineto_resnet50_workers4.pt.trace.json.gz` |
| **Devices analyzed** | 1 |
| **Confidence** | **trace-only** |

### Confidence & method

This section is a **trace-only** diagnostic: computed from profiler timing only, without live device state or a gitm-attached capture.

- MBU computed from memcpy traffic only; per-kernel DRAM counters are not present in profiler exports.
- No device state plane (power, clocks, throttling); throttle-induced stalls appear as idle.
- Predicted floor uses catalogue peak rates for the reported SKU; unvalidated against live telemetry.
- Sync events absent from this trace format; sync-wait vs launch-latency attribution is coarse.
- SKU 'NVIDIA GPU' is not in the hardware catalogue; peak rates fall back to A100 defaults.


- Imported trace: diagnostic headroom read only. Floor commitment requires a gitm-captured run.




### Device 0

**Recoverable headroom: 25.8% of current wall time.**

| | |
|---|---|
| Observed wall time | 801.550 ms |
| Predicted floor | 594.874 ms |
| Busy fraction | 74.2% |
| Kernels in trace | 8868 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 6.7% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 19.1% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).



### What we'd do next

A gitm-attached run confirms these reads with full telemetry (kernel events plus device state) and qualifies the workload for our guaranteed optimization floor. **Floor commitment requires a gitm-captured run, not an imported profiler file.** Import-only assessments stay diagnostic so commercial commitments track measured gains.




---

## synthetic_4xA100_nccl

| | |
|---|---|
| **Workload** | synthetic_4xA100_nccl |
| **SKU (as reported)** | NVIDIA GPU |
| **Capture date** | 2026-07-16 19:02 UTC |
| **Source format** | torch-import |
| **Input** | `tests/fixtures/importers/real/synthetic_4xA100_nccl.json` |
| **Devices analyzed** | 4 |
| **Confidence** | **trace-only** |

### Confidence & method

This section is a **trace-only** diagnostic: computed from profiler timing only, without live device state or a gitm-attached capture.

- MBU computed from memcpy traffic only; per-kernel DRAM counters are not present in profiler exports.
- No device state plane (power, clocks, throttling); throttle-induced stalls appear as idle.
- Predicted floor uses catalogue peak rates for the reported SKU; unvalidated against live telemetry.
- Sync events absent from this trace format; sync-wait vs launch-latency attribution is coarse.
- SKU 'NVIDIA GPU' is not in the hardware catalogue; peak rates fall back to A100 defaults.
- Cross-device dependency attribution requires captured telemetry; this report identifies skew and exposed communication, not their root cause.
- Communication classification is name-based; unknown collective libraries are not counted.


- Imported trace: diagnostic headroom read only. Floor commitment requires a gitm-captured run.

### Node summary

| | |
|---|---|
| Node recoverable headroom (duration-weighted) | **32.0%** of wall time |
| Device busy-fraction skew (max − min) | 24.0% |

#### Device skew

| Device | Busy fraction | Wall time | Recoverable headroom |
|---|---|---|---|
| 0 | 74.0% | 0.500 ms | 26.0% |
| 1 | 74.0% | 0.500 ms | 26.0% |
| 2 | 74.0% | 0.500 ms | 26.0% |
| 3 | 50.0% | 0.500 ms | 50.0% |


One or more devices spend materially more time idle than the busiest device; in multi-GPU workloads this usually indicates load imbalance or synchronization waits on the slowest rank.

#### Communication

| Device | Comm share of busy | Exposed comm (share of wall) |
|---|---|---|
| 0 | 13.5% | 5.0% |
| 1 | 13.5% | 5.0% |
| 2 | 13.5% | 5.0% |
| 3 | 20.0% | 10.0% |


Exposed communication is collective kernel time that does **not** overlap any non-collective kernel on the same device. Exposed communication is recoverable headroom: the device is waiting on the network fabric (or peers) instead of doing independent compute.


### Device 0

**Recoverable headroom: 26.0% of current wall time.**

| | |
|---|---|
| Observed wall time | 0.500 ms |
| Predicted floor | 0.370 ms |
| Busy fraction | 74.0% |
| Kernels in trace | 11 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 6.8% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 19.2% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).


### Device 1

**Recoverable headroom: 26.0% of current wall time.**

| | |
|---|---|
| Observed wall time | 0.500 ms |
| Predicted floor | 0.370 ms |
| Busy fraction | 74.0% |
| Kernels in trace | 11 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 6.8% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 19.2% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).


### Device 2

**Recoverable headroom: 26.0% of current wall time.**

| | |
|---|---|
| Observed wall time | 0.500 ms |
| Predicted floor | 0.370 ms |
| Busy fraction | 74.0% |
| Kernels in trace | 11 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 6.8% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 19.2% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).


### Device 3

**Recoverable headroom: 50.0% of current wall time.**

| | |
|---|---|
| Observed wall time | 0.500 ms |
| Predicted floor | 0.250 ms |
| Busy fraction | 50.0% |
| Kernels in trace | 10 |



#### Gap breakdown

Where the recoverable headroom sits, by class. Percentages are shares of **current wall time**.

| Class | Share of wall | What this means |
|---|---|---|
| Idle / stall | 25.0% | The GPU was not running kernels for this fraction of wall time — waiting on the host, on copies, on sync, or genuinely idle. |
| Memory-bound | 25.0% *(indicative)* | Of the recoverable gap, the portion associated with memory traffic pressure relative to the SKU's peak bandwidth. |
| Compute-bound | 0.0% *(indicative)* | Of the recoverable gap, the portion associated with arithmetic intensity rather than bandwidth. |

Memory vs compute shares are **indicative** on this export: the profiler file does not include hardware FLOP counters, so the split uses available bandwidth signal only (50/50 when that signal is weak).



### What we'd do next

A gitm-attached run confirms these reads with full telemetry (kernel events plus device state) and qualifies the workload for our guaranteed optimization floor. **Floor commitment requires a gitm-captured run, not an imported profiler file.** Import-only assessments stay diagnostic so commercial commitments track measured gains.

### Import notes

- multi-GPU input: analyzing devices [0, 1, 2, 3]; per-device kernel counts: {0: 11, 1: 11, 2: 11, 3: 10}





