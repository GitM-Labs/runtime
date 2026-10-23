# Validation interpretation after the GPU experiments

Status update October 13, 2026: Ishaan authorized local commits. Earlier October 11 references to uncommitted work describe that snapshot. Maintainer review, pushing, publication and external contact remain separate; none is established by creating local commits.

Updated October 11 2026.

## What needs correcting

The earlier summary said that the planner's latency predictions were not accurate. That is true if its output is used as a point estimate of this model's latency, but the pinned CLI explicitly says its vendor-peak floor is not a target and a residual is a lead, not a defect. We should not turn the observed residual into a demonstrated error in the roofline equation or imply that the planner already promises an empirical latency estimate.

The defensible result is narrower: on the tested Qwen2.5-3B / Transformers 4.46.3 SDPA / BF16 / A100 path, the serial resource proxy substantially understates measured forward time. It is unsuitable as a point latency estimate. All 12 mode/shape comparisons remain above that proxy, and none is within the predeclared descriptive ±20% point-accuracy threshold. That outcome is consistent with an optimistic model; it does not establish that the whole-step number is a rigorous lower bound for every schedule, backend or traffic interpretation.

The historical measurements and thresholds remain unchanged. The local reporting patch exposes that distinction rather than fitting the observations.

## What the evidence establishes

| Evidence | Established result | Remaining limit |
| --- | --- | --- |
| Actual GLM-5.2-FP8 catalogue trace | Thirteen accounting assertions pass through unmodified catalogue/graph functions. Weight amortization, distinct-expert reuse, compressed KV geometry and aggregate capacity checks exist. | No measured GLM inference or physical traffic validation. |
| Two A100 microbenchmark sessions | Repeated fusion and pipelining benefits on different physical devices. R64 fusion ratios are 20.46x and 20.54x; overlap ratios vary substantially by host. | Nominal requests are not HBM counter readings. No single fixed speedup generalizes to all workloads. |
| Actual Qwen checkpoint timing matrix | Twelve comparisons, six shapes, 30 samples per mode. Graph replay measures 9.0–35.6 ms against a 3.03–3.65 ms proxy; eager measures 28.9–38.7 ms. | One model, one measured engine/version/precision and one performance session; no production serving or answer-quality benchmark. |
| Numerical checks | Graph/eager cached-forward logits match exactly; both BF16 cached/full comparisons choose the same top-1 token. A separate FP32 check passes at both prefixes with max difference below 0.000034. | Original BF16 elementwise tolerances fail and remain failures. FP32 spot checks do not validate arbitrary prefixes or relax the BF16 threshold. |
| Trainium2/SN40L adapter exercise | Shared resource-accounting semantics exercised with source-derived inputs and unknowns preserved. | No measured Trainium/SN40L deployment, real placement artifact or production integration. |

## What to change now

Update the register, audit, requirements and proposal to distinguish physical backend lowering from model geometry. Correct the blanket launch statement: roofline defaults to zero serial launches; dense uses that default, while GLM generally passes one. A canonical node can stand for many kernels, and one physical region can fuse several logical operations.

The text CLI already carries a warning. Its JSON result should carry an equally explicit meaning, with an unavailable empirical latency estimate and unvalidated whole-step-bound status. That is the review patch in this package. Its legacy numeric outputs are unchanged in fourteen targeted comparisons.

The tiered-memory proposal retains its original motivation: named tiers, directed transfers, residency and resource contention. A GPU timing residual does not demonstrate those missing cross-family concepts by itself. Existing reuse and precision support should be credited rather than duplicated.

## What remains unfinished

There is no implemented empirical latency estimator, approved backend-lowering change, measured physical HBM byte model, new Kimi/GLM inference validation or accepted cross-family deployment result. Rahul's earlier positive feedback is not a review of the new patch or October 11 refinements. The repository has not been changed.

## Local integration status

Ishaan authorized local integration on October 11. The optional `gitm/planner/tiered_memory.py` and `gitm/planner/backend_report.py` APIs and the registry reporting change are now present in this checkout with native tests. They are uncommitted and have no new maintainer approval. See [integration status](completion-status.md) for verification scope and remaining gates.
