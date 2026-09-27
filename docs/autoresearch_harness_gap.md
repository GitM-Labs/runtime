# Autoresearch harness: gap statement

Status: **code walk done; round-trip timing on MI355X still to be measured.**
Every number in the timing table below comes from reading the code and the runbook.
None of them is a measurement yet. Each row names the place its value comes from.

## Two paths, not one

A hypothesis can reach a result on the MI355X Kimi setup by two routes. Neither
route by itself goes from a hypothesis to a verdict.

| | **Path 1: `run_loop.sh e8`** (what the runbook uses) | **Path 2: `scheduler/loop.py` Phase 4 / 4b** (what `apply.py` serves) |
|---|---|---|
| Engine | `vllm serve` in the `vllm` container, TP=8 | in-process `vllm.LLM` built by the `vllm-decode` factory (`GITM_VLLM_TP`) |
| Apply | `arm.sh I $LEVER`: rewrite `arm.env`, kill the server, reload (≤ 60 min) | `LiveEngineApplicator.apply`: hot-swap, or `restart_fn` rebuild |
| Load | GuideLLM, synthetic `rag 4096/512`, c=64, 180 s | `engine.generate` on a fixed prompt set (`_engine_throughput_fn`) |
| Replay | none (BurstGPT replay exists only in the `burst` phase) | none |
| Reps / order | 1 before, 1 after, always B→I | `GITM_AB_REPS` (default 1), always baseline→candidate |
| Keep/rollback | **none**: the server is left on arm I, and a human runs `arm.sh B` | `apply_intervention`, `min_keep_delta=0.0` |
| Output | `e8_delta.md`: a single throughput `%`, no interval | `ApplyResult` + `EngineABResult` + `verification.json` |

Path 2 cannot run on the pod as deployed. `vllm serve` already holds all 8
GPUs, and a parallel restart would need two copies of the 595 GB weights, so it
has to run in `GITM_RESTART_MODE=serial` after the server is stopped.

## One round trip on Path 1, step by step

| # | Step | Actor | Expected cost (source) | Measured |
|---|---|---|---|---|
| 1 | Write the lever as a `vllm serve` flag string | human | — | _todo_ |
| 2 | `kubectl exec … INTERVENTION=… run_loop.sh e8` | human | — | _todo_ |
| 3 | `arm.sh B` (no-op if B is already serving) | script | 0 to 60 min reload (`arm.sh`) | _todo_ |
| 4 | Baseline window: capture plus GuideLLM | script | ~190 s (`run_loop.sh:228`) | _todo_ |
| 5 | `arm.sh I`: kill and reload with the lever | script | ≤ 60 min, up to 3600 s on first load (manifest `startupProbe`) | _todo_ |
| 6 | Candidate window | script | ~190 s | _todo_ |
| 7 | `analyze.sh <run>` on the laptop: tar results over `kubectl exec` | human | — | _todo_ |
| 8 | Read `e8_delta.md` and decide | human | — | _todo_ |
| 9 | Roll back: `arm.sh B` | human | ≤ 60 min reload | _todo_ |
| 10 | Record the result somewhere | human | — (no sink exists) | _todo_ |

Human actions: **5** (steps 1, 2, 7, 8, 9), plus step 10, which currently has
nowhere to write. Compute is roughly 2 × 190 s under load. Wait time is
dominated by 2 to 3 model reloads. Wall clock, compute and wait time will be
filled in from one real run.

## Gaps between today and "hypothesis in, verdict out"

| # | Missing component | Extends | Input → output | Owner / blocker |
|---|---|---|---|---|
| G1 | **Experiment contract**: a machine-readable claim, intervention, expected effect and interval, metric, gates, budget, reps, order, stopping rule | `InterventionSpec` (`gitm/kernels/spec.py`) covers only the knob and prior; the contract wraps it | YAML → validated `Experiment` | Isaiah; Tarun and Abhiram must sign off |
| G2 | **Served-engine applicator**: apply/restore through `arm.sh` against `vllm serve`, behind `apply_intervention` | the `Applicator` protocol (`optimizer/apply.py`); `arm.sh I` is the apply step | spec → server on candidate / restored to baseline | Isaiah; needs `kubectl` access to `us-mi355x-gitmachine` |
| G3 | **Replay as the measurement load**: fire the contract's workload through `ReplayPlan.bench_serve_argv` | `gitm/traffic/replay.py`; the `burst` phase already calls it | trace id + operating point → bench-serve result JSON joined to `ReplayPlan` | Isaiah; the spec's "result JSON does not carry trace identity" join is still open |
| G4 | **Repeated, interleaved measurement**: N reps, ABAB or randomized order, warmup discarded | `LiveEngineApplicator._bench_stats` (reps exist; order and warmup do not) | protocol → raw per-rep samples | Isaiah |
| G5 | **Estimator and interval**: effect with a CI and a paired or unpaired method, instead of `delta − (σ_b+σ_c)/μ_b` | `LiveEngineApplicator.measure`: the "noise band" is not an interval and is subtracted from the reported delta | samples → effect, CI | Isaiah |
| G6 | **Noise floor per operating point**, and MDE lookup | new; keyed like `history.record_for` (`gpu_sku`, fingerprint) | (hw, model, workload, metric, protocol) → MDE or `not_established` | Isaiah; **blocked on MI355X time** |
| G7 | **Correctness gate**: candidate outputs checked against baseline or eval | none exists | outputs → pass / fail | Isaiah + Tarun (the fp8 / precision tolerance is his call) |
| G8 | **Latency gate**: TTFT / ITL p50 and p95 against contract limits | GuideLLM and bench-serve already emit these; nothing gates on them | metrics → pass / fail | Isaiah |
| G9 | **Verdict record**: five terminal states, effect, CI, gates, cost, rollback, full provenance | `verification_export.py` / `ApplyResult` | run → `verdict.json` | Isaiah |
| G10 | **Provenance**: model/checkpoint revision, gitm sha, engine args, image digest, topology | `run_loop.sh manifest()` records arm, ROCm, vLLM and env only | run → provenance block | Isaiah |
| G11 | **Known-effect controls**: a no-op A/A and a controlled slowdown that block verdicts on failure | none | control batch → valid / `invalid` | Isaiah |
| G12 | **Cost accounting**: wall clock, accelerator-seconds, $ estimate, budget gate | `amdsmi.jsonl` telemetry already captured at 1 Hz | run → cost; contract → go / no-go | Isaiah; needs the $/GPU-hr rate from Seojun or Rahul |
| G13 | **Unattended driver**: queue → contract → G2 to G12 → verdict with no `kubectl exec` | `run_loop.sh` phase dispatcher | contract file → verdict file | Isaiah; Seojun for CI / in-cluster runner |

## Smaller issues found on the way

- `LiveEngineApplicator.measure` returns `delta − noise_band` as `measured_delta`.
  Downstream reports therefore show a number that is neither the observed
  effect nor a bound (`apply.py:486-496`).
- `e8` leaves the server on the intervention arm, so the next phase silently
  runs on the candidate config.
- At `reps=1` the "noise band" is 0 and every positive delta counts as
  significant (`apply.py:332`). This is the default in the loop.

## Access and blockers

- **MI355X**: access has been reported, but this laptop has no `kubectl` context
  for `us-mi355x-gitmachine` yet. The round-trip timing (above) and G6 wait on it.
- **Tarun's registered hypothesis specs**: not in the repo. G1's acceptance
  criterion needs them.
