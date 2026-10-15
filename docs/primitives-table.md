# The 25 named slowdown causes: what each row is for

Of the 25 named causes, 10 can be shown from passive data alone (this build evaluates 5 of
them: rows 10, 11, 22 and 25, and the dropped-record half of row 24), 6 more with richer
passive records, 3 need a second capture to compare against, and 6 need something a single
capture cannot hold.

"Passive" means recorded while the server handles its normal load, with no experiment and
no change to its settings. The sources are the GPU trace (kernel, memcpy and sync records),
the server's `/metrics` counters, and device samples.

A duration cannot name a cause, because many causes change timing the same way. Each row
therefore names one cause and the signal only that cause produces.

## The 25 rows

| # | Cause | Why it is on the list | How a result is used | From passive data? |
|---|---|---|---|---|
| 1 | Prefill-decode interference | Long prefills stall other requests' decode steps | Points to `enable_chunked_prefill`, `max_num_batched_tokens_dynamic` | With richer records: trace events must carry a step id |
| 2 | Request queue time | Requests wait before admission | Points to `max_num_seqs_dynamic`, `max_num_batched_tokens_dynamic` | Yes (`/metrics`); a share of time to first token, not seconds; not read by this build |
| 3 | KV-cache preemption (recompute) | Evicted requests are recomputed | Points to `gpu_memory_utilization_dynamic`, `max_num_seqs_dynamic`; marks an A/B window untrustworthy | Yes (`/metrics`); a count, not seconds; not read by this build |
| 4 | Prefix-cache miss fraction | Misses force extra prefill | Points to `enable_prefix_caching` | Yes (`/metrics`); a fraction, not seconds; not read by this build |
| 5 | Speculative-draft rejection fraction | Rejected draft tokens are wasted work | Diagnosis only; no lever | Yes (`/metrics`), where the server exports the counters; not read by this build |
| 6 | Structured-output mask overhead | CPU-side guided decoding leaves the GPU idle | Diagnosis only; no lever | With richer records: host CPU counters and the request mix, to split it from row 9 |
| 7 | Batch-size and context-length step scaling | Expected growth, listed so it is not reported as a fault | Baseline to compare against | With richer records: per-step metadata |
| 8 | Framework-bound launch | Per-kernel launch overhead, such as a missing CUDA graph | Points to `cuda_graphs_enable`, `max_seq_len_to_capture_dynamic` | No: looks the same as row 23 without an untraced run |
| 9 | GPU starvation | The host CPU cannot feed the GPU | Points to `async_scheduling` | With richer records: the running-requests gauge, to tell it from no demand |
| 10 | Gap time under a synchronization record | The host blocks in a sync while the GPU idles | Seconds in the window; a code fix, no lever | Yes (trace) |
| 11 | Gap time under a memcpy record | Host-device copies sit in the gaps | Seconds in the window; a code fix, no lever | Yes (trace) |
| 12 | Allocator miss | `cudaMalloc` in steady state | Diagnosis only; no lever | With richer records: runtime API records, to split it from row 17 |
| 13 | Fused-path fallback, missed fusion | Slower unfused kernels ran | Points to `attention_backend_flashinfer` | Needs a reference capture; not separable from rows 14 and 15 |
| 14 | Precision / Tensor Core fallback | The fp32 path was taken | Diagnosis only; no lever | Needs a reference capture; not separable from rows 13 and 15 |
| 15 | Algorithm mis-selection | A library picked a slower algorithm | Diagnosis only; no lever | Needs a reference capture; not separable from rows 13 and 14 |
| 16 | Tile / wave quantization | Expected step effect of grid size | Baseline to compare against | No: needs a calibration capture |
| 17 | Warmup | Lazy load, autotune or recompile at first use | With row 12, marks an A/B window untrustworthy | With richer records: runtime API records, to split it from row 12 |
| 18 | Context time-slicing | Another process's context holds the device | Diagnosis only; an ops fix | No: needs the other context's activity on the device |
| 19 | SM contention | A co-tenant inflates kernel durations | Marks an A/B window untrustworthy | No: an untraced co-tenant leaves no record |
| 20 | Thermal throttling, power capping | The device is slowed by hardware limits | Marks an A/B window untrustworthy | Yes (NVML samples); not read by this build |
| 21 | Memory-bound regime against efficiency loss | Separates "at the hardware limit" from real loss | Baseline to compare against | No: needs a calibration capture with measured memory traffic |
| 22 | Exposed collective communication | Tensor-parallel all-reduce time with no compute overlapping | Seconds in the window; no lever | Yes (trace) |
| 23 | Tracer overhead | The measurement itself adds launch-scale gaps | Says how far to trust the gap rows | No: needs an untraced run |
| 24 | Kernel misroute, event dropping | Missing records read as idle time | Says whether any trace row can be ruled out | Yes (the dropped-record count in the trace header); misroute is not checked |
| 25 | Exposed expert-parallel dispatch (all-to-all) | MoE all-to-all time with no compute overlapping | Seconds in the window; points to `enable_dbo` | Yes (trace) |

Rows 2 to 5 would read the server's `/metrics` counters. This build reads only the trace, so
they are listed and not evaluated. For rows 2 and 3 the repo already has rules on the same
counters: `scheduler_attribution` reports `admission_backlog` and `kv_cache_preemption`.

In the output, a row this build cannot evaluate carries one of three reasons, and the reason
names what the build lacks, not the grouping below. Rows 1 and 21 read `no_passive_witness`;
rows 18, 19 and 20 read `channel_missing`, as does row 24 when the trace header has no
dropped-record count; rows 2 to 5 read `not_built`. Row 21 is listed with them as not evaluated,
not with the baselines (rows 7 and 16).

## The interval on a priced row

Rows 10, 11 and G1 can claim the same piece of gap time: a sync record and a memcpy record
over one gap, or either of them over a gap of 20 us or less. The trace does not say which
owns it. The time shown gives each piece to one row by a fixed precedence (memcpy, then
sync, then launch-scale), and beside it the row carries its attribution range:

- low end: the time only this row's records explain;
- high end: the time this row's records could explain at all.

A row is `estimated` only when the low end alone reaches the threshold, `absent` only when
the high end stays under it, and not identifiable (`confounded`, naming the other rows) in
between. With no overlap the range is the single value.

The range is exact arithmetic on the records, not an error bar. It does not cover timestamp
error, tracer overhead or dropped records, and it says nothing of other windows or of the
time removing the cause would recover.

## When a row is `absent`

A row's time is counted over every record in the window, so a row under its threshold
(1% of the window) is ruled out at that threshold only when no record that would raise it
can be missing. All of these must hold:

- The high end of the row's range is under the threshold.
- The trace header reports zero dropped records. The CUPTI and ROCm collectors write the
  count; an imported trace has none, so its rows are never `absent`.
- The source records the row's kind. Without sync or memcpy records the row cannot tell.
- No other traced process shares the device.
- For rows 22 and 25, the header states exactly one device and no kernel name matched.
  These rows call a kernel a collective by a substring of its name, and that classifier
  is unverified, so a priced row shows its time and is not `estimated` either.

Row 24 is never `absent`: it does not check misroute. A count above zero makes every trace
row not identifiable, priced or not, and so does a collector saying it lost records it
could not count. `absent` describes this capture only. The count does not include records
still in a collector's buffer when the capture closed.

Not yet confirmed on hardware: a traced process with nothing on disk for the window makes
the count unknown, and the process running the capture may be one on every capture. If so,
no injected capture reaches `absent` until that rule is checked against a real run.

## Passive or not, by count

| Data needed | Rows | Count |
|---|---|---|
| Passive data alone: trace, `/metrics` counters, NVML samples, drop count | 2, 3, 4, 5, 10, 11, 20, 22, 24, 25 | 10 |
| Richer passive records: step ids, runtime API records, host CPU counters, request gauge | 1, 6, 7, 9, 12, 17 | 6 |
| A second capture as reference | 13, 14, 15 | 3 |
| Not passive: an untraced run, a calibration capture, or other tenants' activity | 8, 16, 18, 19, 21, 23 | 6 |

Four sets of rows produce the same signal and can only be reported together until the
extra data arrives: 8 with 23, 6 with 9, 12 with 17, and 13 with 14 and 15.

## Pros and cons

**Pros**

- Each cause has a name an engineer already uses, and 8 rows point to a named config lever.
- A row can say it cannot tell, with the reason and the data that would settle it.
- The trace rows add up with busy time and a remainder to the full window, so unexplained
  time stays visible.
- 10 rows need no experiment and no second capture; this build evaluates 5 of them.
- Rows 3, 19 and 20, and rows 12 and 17 together, say when an A/B window should not be trusted.

**Cons**

- Only rows 10, 11, 22 and 25 give seconds, and of those only 10 and 11 (and G1) can be
  `estimated`. Rows 2 to 5 would give a ratio or a count.
- A time is associated with a cause. It is not the time removing the cause would recover.
- 17 rows have no lever (2 of them baselines, and row 21), so a result on them is a diagnosis for a person,
  not an action.
- 9 rows fall into the four sets above and cannot be told apart from passive data.
- Rows 8, 16, 18, 19, 21 and 23 are out of reach of a single passive capture.
- Thresholds are judgement calls, not derived from data.
