# Current state and roadmap

This document separates what is load-bearing in the runtime today from what is
sequenced, so a technical reader can calibrate quickly. The short version: the
end-to-end loop (profile, predict, attribute, apply behind a rollback gate,
report provenance) runs today on NVIDIA for vLLM decode serving, and the two
pieces that turn estimates into measured results (the trace-driven replay engine
and the live engine applicator) are the next builds.

## What runs today

- Two telemetry planes on NVIDIA: state telemetry (NVML) and event telemetry
  (CUPTI), captured to a JSONL trace.
- Roofline planner: a per-op predicted execution graph for a transformer decode
  step (`max(t_compute, t_memory)` per op, with a vendor efficiency band).
- Deviation monitor: residuals against the predicted graph across three
  invariants (kernel-time, memory-traffic, stream-concurrency), with multi-basis
  confirmation to suppress single-sample noise.
- Causal attribution: a Granger precedence test and a doubly-robust (AIPW)
  estimator, run together so we can require agreement before acting.
- Rollback-gated apply: snapshot, apply, measure, keep-or-restore. No lever can
  leave a workload worse than its baseline.
- Provenance report: claim to evidence to intervention to delta.
- Benchmarks: HFT order-book (cuDF), AlphaFold2 inference, and KITTI 3D
  detection, each with a baseline and an output-verified optimization.

## v0 today, being hardened

- Intervention library: a curated set of vLLM decode levers, each with a cited
  source, an applicability gate, and a safety gate. Expected deltas are
  estimates from the cited source, not yet measured on our own workloads.
- Counterfactual replay: v0 estimates a lever’s delta as applicability-weighted
  trace coverage times the lever’s expected delta. The trace-driven replay
  engine (replaying the captured graph under the intervention) is the next build
  and is what produces measured, workload-specific predicted deltas.
- Predicted graph: one decode step, roofline only, no dependency edges yet.
  Multi-step and edge modeling are sequenced.
- Hardware catalogue: peak rates are illustrative A100 defaults today; a vendor
  catalogue is being populated.
- Qualification gate: a v0 heuristic (commit vs diagnose on residual headroom);
  a richer workload fingerprint is sequenced.

## On the roadmap (not yet built)

- Live vLLM/engine applicator: the apply seam (snapshot, apply, restore,
  measure) is in place; the engine-backed implementation is next. Today the loop
  runs predict-only when no engine is attached and never reports an unverified
  delta as won.
- AMD backend (ROCm SMI / rocprof): the interface exists; the implementation is
  not done. NVIDIA is the supported path today.
- Deploy modes beyond standalone: the runtime is a CLI plus embedded API today.
  Kubernetes (as an operator) and Slurm (as a job wrapper) are sequenced; the
  per-job execution contract is the same across all three.

## Workload coverage and the non-LLM story

The automated intervention library targets vLLM decode serving today. Every
lever in the library is a vLLM engine knob with a cited source.

The HFT, biotech, and edge benchmarks are hand-authored, output-verified
optimizations. They validate the method (predict, attribute, apply behind a
correctness gate, report provenance) on non-LLM workloads, and the HFT case is a
real kernel-level win (four grouped scans reduced to two, proven output-identical
before it is kept). They are bespoke proofs of the method, not output of the
automated library. Generalizing the library to non-vLLM workloads, which means a
workload-typed lever set plus the attribution-to-lever mapping for those domains,
is on the roadmap. Until then the honest framing is: automated optimization is
vLLM-serving today, and the non-LLM results are method validation done by hand.

## A note on claims

- No delta in this repository is a measured result on our own workloads yet.
  Library deltas are cited estimates. Measured numbers require the replay engine
  and live apply.
- Attribution is Granger plus doubly-robust (AIPW). We run both and look for
  agreement, then gate any action behind a live rollback test. This is
  correlational evidence ranked by effect size and confirmed empirically before
  it is kept, not a claim of proven causation.
