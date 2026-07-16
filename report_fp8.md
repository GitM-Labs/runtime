# GITM provenance report

**Workload:** `vllm-decode`
**Fingerprint:** `nvidia:c3494f4b13df235d`
**Run ID:** `77d733b66b3d41f4becff93f8a0d484f`
**git SHA:** `e762eca` &middot; **gitm:** `0.1.10`

## Summary

vLLM decode on NVIDIA H100 80GB HBM3: 5 candidate(s) evaluated, 2 rolled back. Engine scheduler: peak queue depth 36 (over 24 samples).

## Claims

Every claim below carries the full provenance chain. Incomplete chain = no
claim. Rejected and rolled-back candidates are listed in the appendix.

| # | Claim | Residual | Causal evidence | Intervention | Predicted Δ | Measured Δ |
|---|---|---|---|---|---|---|
| 1 | Raise the per-step token budget to 8192 to fill decode batches. | `kernel_time`: +279.6% | live A/B: kept (+1.3% decode throughput, via hot-swap); baseline 25850.3 → candidate 26180.0 tok/s | `max_num_batched_tokens_8192` | +0.3% | +1.3% |
| 2 | Raise gpu_memory_utilization to 0.92 to permit larger batch slots. | `kernel_time`: +279.6% | mlp_gate_up→mlp_down (p=1.9e-07), attn_score_value→mlp_down (p=1.9e-06) | `gpu_memory_utilization_092` | +0.3% | — (rolled back) |
| 3 | Allow up to 256 concurrent sequences to raise decode concurrency. | `kernel_time`: +279.6% | live A/B: kept (+1.2% decode throughput, via hot-swap); baseline 25677.1 → candidate 25992.3 tok/s | `max_num_seqs_256` | +0.3% | +1.2% |
| 4 | Switch the scheduler to priority policy for mixed-SLO traffic. | `kernel_time`: +279.6% | live A/B: kept (+0.1% decode throughput, via hot-swap); baseline 25911.9 → candidate 25926.5 tok/s | `scheduling_policy_priority` | +0.2% | +0.1% |
| 5 | Give the block manager 4 GiB CPU swap to avoid preemption stalls. | `kernel_time`: +279.6% | live A/B: rolled back (-2.2% decode throughput, via restart); baseline 25951.8 → candidate 25393.7 tok/s | `swap_space_4gib` | +0.2% | -2.2% (rolled back) |


## Appendix

- **Rejected candidates.** 0
- **Rolled back.** 2: gpu_memory_utilization_092, swap_space_4gib
- **Trace.** `/storage/scratch1/8/achawdhary3/gitm-runs/traces/77d733b66b3d41f4becff93f8a0d484f.jsonl`
- **Window.** start `1783611189106709616` &rarr; end `1783611568115705240`

---

_Trust comes from honest reports, not from optimizations themselves._
