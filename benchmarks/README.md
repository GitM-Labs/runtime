# Benchmarks

Three benchmarks — **HFT**, **biotech**, **edge/robotics** — exercise the GITM
runtime (planner + deviation monitor + causal attribution) against a
heterogeneous workload mix. The benchmark layer is deliberately **dumb and
identical** across domains: everything mechanical lives in the shared systems
layer [`gitm/bench/`](../gitm/bench/) and is driven from each benchmark's
`Makefile`; everything domain-specific is declared in one `bench.toml`.


## Layout

```
benchmarks/
  Makefile.common        shared make targets (manifest/verify/run/profile/baseline)
  _templates/            spec.md / datasets.md skeletons to copy when filling cells
  <name>/
    bench.toml           the ONLY domain config — metric, seeds, stall bands, work-unit
    Makefile             sets NAME, includes ../Makefile.common
    datasets.md          dataset description + seed protocol
    spec.md              4-section spec
    manifest.yaml        frozen dataset sha256s               (generated)
    results.md           stall table + GPU-active gate        (generated)
```

Datasets themselves never live in the repo or on local disk at rest. They live
in `$GITM_S3_ROOT/datasets/<name>/` and are staged into bounded local scratch
(`$GITM_SCRATCH/staging/<name>/`) only while a run needs them. Run outputs land
in `$GITM_SCRATCH/runs/`. (See [`gitm/_paths.py`](../gitm/_paths.py).)

## The shared workflow (identical for every benchmark)

```bash
cd benchmarks/<name>

# 1. freeze the staged dataset (run where the bytes were generated/staged)
make manifest          # -> manifest.yaml (sha256 + byte count per file)
make verify            # re-hash and confirm byte-identical

# 2. produce the locked baseline: all seeds -> gates -> results.md
make baseline          # the reproducibility command; exits non-zero unless signed off

# single-seed iteration while developing the harness:
make run-42            # one BaselineRun JSON
make profile-42        # nsys/rocprof + py-spy/sar bundle
```

`make baseline` is the core reproducibility command: a non-author runs
exactly this on a clean box and must hit the recorded numbers within 2 %.

## The two sign-off gates (applied to every benchmark by `gitm.bench`)

1. **spread** — three convergent seeds agree within `spread_tolerance` (2 %).
   The recorded baseline is their mean.
2. **saturation** — wall-clock-weighted GPU active % stays under
   `gpu_active_ceiling` (85 %). A saturated benchmark has no residual headroom,
   so it trips the same-day swap rule.

Optionally a **target** gate compares the recorded mean against `baseline_target`
(e.g. HFT ≥ 25 M events/s).

## The harness contract

Each benchmark's `work_unit.command` (a kernel set + ingest harness the pair
writes) must print **one JSON object** on stdout — the last such line wins —
carrying at least:

```json
{"metric_value": 25300000.0,
 "gpu_name": "A100-SXM4-80GB",
 "device_count": 1,
 "stall_breakdown": [
   {"phase": "ingest", "cpu": 0.04, "data_stall": 0.22, "sync": 0.03,
    "gpu_active": 0.71, "throughput": 25300000.0, "wall_clock_s": 41.0}
 ]}
```

`gitm.bench` wraps that line with provenance (git sha, gitm version, the dataset
manifest's own sha256) into the `<name>_baseline_N.json` contract. The
`stall_breakdown` rows are produced by `gitm.bench.profile.build_breakdown` from
per-phase timings + the profiler's GPU-busy time — see
[`gitm/bench/profile.py`](../gitm/bench/profile.py).

## Cross-cutting deliverables (per benchmark)

1. `datasets.md` + frozen `manifest.yaml` on `main`.
2. `spec.md` (4 sections).
3. Three baseline JSONs in `$GITM_SCRATCH/runs/<name>_baseline_{1,2,3}.json`,
   < 2 % spread.
4. `results.md` with the stall table + GPU active % confirmed < 85 %.
5. Reproducibility proof: a non-author ran `make baseline` in < 60 min and hit
   the numbers.
