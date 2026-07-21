# GPU headroom assessment

Diagnostic read from customer profiler exports. This is **not** a floor commitment.


---

## parity_workload

| | |
|---|---|
| **Workload** | parity_workload |
| **SKU (as reported)** | NVIDIA H100 |
| **Capture date** | 2024-01-15 12:00 UTC |
| **Source format** | torch-import |
| **Input** | `tests/fixtures/importers/parity_torch.json` |
| **Devices analyzed** | 1 |
| **Confidence** | **trace-only** |

### Confidence & method

This section is a **trace-only** diagnostic: computed from profiler timing only, without live device state or a gitm-attached capture.

- MBU computed from memcpy traffic only; per-kernel DRAM counters are not present in profiler exports.
- No device state plane (power, clocks, throttling); throttle-induced stalls appear as idle.
- Predicted floor uses catalogue peak rates for the reported SKU; unvalidated against live telemetry.
- Sync events absent from this trace format; sync-wait vs launch-latency attribution is coarse.


- Imported trace: diagnostic headroom read only. Floor commitment requires a gitm-captured run.




### Device 0

**Recoverable headroom: 50.0% of current wall time.**

| | |
|---|---|
| Observed wall time | 0.200 ms |
| Predicted floor | 0.100 ms |
| Busy fraction | 50.0% |
| Kernels in trace | 3 |



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






