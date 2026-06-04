<!-- Copy to benchmarks/<name>/spec.md and fill all four sections (~1 page). -->
# <NAME> benchmark — spec

## 1. Input definition
<!-- What the work consumes: dataset, schema, scale target, where it lives in
     $GITM_S3_ROOT/datasets/<name>/. Point at datasets.md + manifest.yaml. -->

## 2. Work unit
<!-- One unit of work, end to end, as a pipeline of phases. Name the phases —
     they become the rows of the stall-breakdown table. Name the baseline
     harness (pinned commit + config hash). -->

## 3. Success metric
<!-- The top-line metric (e.g. events_per_second) over the warm window, the
     baseline target, and the convergence rule (three seeds within 2 %).
     List any auxiliary sanity metrics that are not targets. -->

## 4. Expected stall profile
<!-- The expected fractions per phase (CPU / data-stall / sync / GPU active).
     These must match `[expected_stall]` in bench.toml. State the saturation
     rule: if GPU active > gpu_active_ceiling, flag same day for shard fallback. -->
