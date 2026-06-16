# HFT Task A (headroom report) + Task B (apply + prove) — tools and plan

Two scripts + one library that deliver Task A now (no pod) and set up Task B
(needs pod). Drop the three files in benchmarks/hft/.

## Files
- hft_library.yaml         — 3 candidate interventions in the real
                             InterventionSpec schema (review: null pending Adit)
- headroom_kernel_rank.py  — Task A: ranks them by the real predict_delta()
- (Task B uses gitm/optimizer/apply.py — already in the repo)

## Task A — run the headroom report (NO POD NEEDED)

    cd benchmarks/hft
    python3 headroom_kernel_rank.py

Output: ranked list of 3 candidate interventions, each with coverage %,
predicted fractional delta, and predicted Δ events/sec. Current ranking:

    1  hft_parquet_h2d_overlap        89.9% cover   ~22.5%   +4.65M eps
    2  hft_multistream_symbol_shards  42.1% cover    ~6.3%   +1.31M eps
    3  hft_groupby_scan_fuse          10.1% cover    ~0.8%   +0.17M eps

This satisfies Task A's "done when": >=3 candidates, each with a predicted
Δ events/sec tied to a residual (the covered kernels).

## What's measured vs estimated (say this to Adit)

MEASURED (real):
  - kernel times / coverage % — from the seed-42 nsys profile
  - the predict_delta() function — the same one the optimizer loop uses

ESTIMATED (literature, cited per spec, NOT measured on our workload):
  - expected_delta_mean (0.25 / 0.15 / 0.08) — from cuDF / NVIDIA docs
  - therefore the predicted-Δ magnitudes are estimates

So: the RANKING is defensible (driven by real coverage). The NUMBERS are
estimates until Task B measures them. The "#1 gets us to 25.4M / over target"
line depends on the 0.25 estimate holding; at the low end (0.10) it's ~22.5M,
still short. Present it as "predicted to potentially cross target, pending
measurement," not a promise.

## Task B — apply + prove the #1 intervention (NEEDS POD)

Goal: apply hft_parquet_h2d_overlap to the harness, prove it speeds things up
WITHOUT changing the derived metrics (VWAP/microprice), via a checksum gate.

The pieces that already exist in the repo:
  - gitm/optimizer/apply.py    — apply_intervention(spec, applicator,
                                 min_keep_delta): snapshot -> apply -> measure
                                 -> keep|rollback. Use this as the gate.
  - The Applicator seam         — you implement snapshot/apply/measure/restore
                                 for the harness (e.g. toggle prefetch depth).

What you build for Task B:
  1. A checksum: run harness, hash the derived-metric output (e.g. sha256 of
     the sorted VWAP/microprice arrays). This is the correctness gate.
  2. An HarnessApplicator implementing the Applicator protocol:
       - snapshot(): record current reader settings + baseline checksum
       - apply(spec): set reader_prefetch_depth=2 (overlap decode/H2D)
       - measure(spec): re-run harness, return (eps_after - eps_before)/eps_before
       - restore(): put settings back
  3. Wrap it: apply_intervention(spec, HarnessApplicator(), min_keep_delta=0.0)
     BUT also assert checksum_after == checksum_before, else force rollback.
  4. Provenance report: claim (headroom #1) -> evidence (nsys coverage) ->
     intervention (prefetch_depth=2) -> delta (measured eps before/after) +
     checksum match. If checksum mismatches, report NO speedup (correctness
     beats speed).

Task B's "done when": provenance report shows a measured speedup with
checksum-identical output, and reports no win if the checksum mismatches.

HONEST SCOPE: Task B is real engineering — implementing the overlap in the
harness (cuDF chunked/prefetched Parquet read), wiring the Applicator, and
measuring on the pod. It is NOT a copy-paste job. Do it once pod access is
back, and expect to iterate on the actual cuDF read path.

## Validation done locally
headroom_kernel_rank.py was run against the real seed-42 kernel sums and
produces the ranked table above using the repo's real predict_delta(). The
ranking logic is verified; the magnitudes carry the estimate caveat above.
