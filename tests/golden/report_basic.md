# GITM provenance report

**Workload:** `vllm-decode`
**Fingerprint:** `nvidia:0123456789abcdef`
**Run ID:** `run_test_0001`
**git SHA:** `deadbeef` &middot; **gitm:** `0.0.1-test`

## Summary

1 verified claims, aggregate measured delta +4.8%.

## Claims

Every claim below carries the full provenance chain. Incomplete chain = no
claim. Rejected and rolled-back candidates are listed in the appendix.

| # | Claim | Residual | Evidence | Intervention | Predicted Δ | Measured Δ |
|---|---|---|---|---|---|---|
| 1 | Set PagedAttention block size to 16 | `kernel_time`: +35.0% | live A/B: kept (+4.8% decode throughput, via restart) | `kv_cache_block_size_16` | +5.0% | +4.8% |
| 2 | Raise GPU memory utilization to 0.92 | `memory_traffic`: +28.0% | Granger ran (20/20 op pairs) but is not used as evidence: series are ordered by launch, not step; pairs in residuals.json | `gpu_memory_utilization_092` | +3.0% | — |


## Appendix

- **Rejected candidates.** 1: spec_block_size_32 (policy.skip_high_risk)
- **Rolled back.** 0
- **Trace.** `/fixtures/trace.jsonl`
- **Window.** start `1700000000000000000` &rarr; end `1700000001000000000`

---

_Trust comes from honest reports, not from optimizations themselves._
