# The 3 invariants

The deviation monitor emits residuals only — observed minus predicted, evaluated
against three invariants. Storage scales with deviation, not duration.

## 1. Kernel-time invariant

For every kernel `k` in the predicted execution graph, observed duration
`t_obs(k)` must satisfy:

```
t_pred_lo(k) <= t_obs(k) <= t_pred_hi(k)
```

where `t_pred_lo` and `t_pred_hi` come from the roofline model (max of
compute-bound and memory-bound time, with vendor-specific efficiency band).

Residual: `r_kt(k) = (t_obs(k) - t_pred(k)) / t_pred(k)`.

## 2. Memory-traffic invariant

For every kernel `k`, observed bytes-moved `b_obs(k)` must match predicted
traffic within the same efficiency band. GQA-aware for decode: KV-cache reads
account for grouped-query sharing.

Residual: `r_mt(k) = (b_obs(k) - b_pred(k)) / b_pred(k)`.

## 3. Stream-concurrency invariant

Kernels the planner marked concurrent must actually overlap on distinct
streams. Predicted concurrent set `C` is violated if any pair in `C` has
strictly non-overlapping wall-clock intervals.

Residual: `r_sc(C) = serialized_fraction(C)` in `[0, 1]`.

## Severity normalization

All three residuals are mapped to a unit severity scale before attribution so
the causal engine doesn't need per-invariant logic:

```
severity = clamp(|residual| / band_width, 0.0, 1.0)
```

`band_width` is invariant-specific (kernel-time: roofline band width;
memory-traffic: traffic band; stream-concurrency: 1.0). The result is
comparable across all three.

## Tier system

- **Tier 1 — always check.** Kernel-time, memory-traffic for any decode workload.
- **Tier 2 — workload-conditional.** Stream-concurrency (only for multi-stream configs).
- **Tier 3 — aspirational.** Power/thermal residuals — added once telemetry sink stabilizes.
