# Experiment contract (draft for sign-off)

Status: **draft.** It still needs Tarun's and Abhiram's sign-off in a
working session, and Tarun's registered hypothesis specs have not been
validated against it yet.

Code: `gitm/experiment/`. Fixture: `tests/fixtures/experiment/kv_fp8_local.yaml`.

## Check it

```bash
python -m gitm.experiment validate <contract.yaml> [...]   # exit 0 = all valid
python -m pytest tests/test_experiment_contract.py -q
```

The test suite runs the fixture from contract validation through the evaluator
to a JSON verdict, against a deterministic fake engine. It checks the wiring and
the meaning of each terminal state. **It is not a hardware result.**

## Submission: `ExperimentContract` (`gitm.experiment.contract/v1`)

Effects are **signed relative changes where positive means better**, whatever
the metric's direction. `+0.08` means 8% more throughput or 8% less latency.

| Field | What it fixes before the run |
|---|---|
| `candidate_id` | unique id for this candidate |
| `hypothesis` | `id`, `submitter`, `claim`, `mechanism`, `expected_effect {mean, lo, hi}`, `reject_if_effect_below` |
| `intervention` | the `knob: value` change under test |
| `primary_metric` | `name`, `unit`, `direction` |
| `workload` | `trace_id` (replay identity), `source`, `operating_point`, `max_concurrency`, `duration_s` |
| `baseline`, `candidate` | full `engine_args` for each arm. **Validated:** candidate = baseline + intervention, nothing else |
| `budget` | `max_wall_clock_s`, `max_accelerator_s`, optional `max_cost_usd` |
| `correctness_gate` | `method`, `reference`, `min_score` (every candidate sample must meet it) |
| `latency_gates` | per metric: `max_regression` of the mean, as a fraction of baseline |
| `protocol` | `repetitions` (kept per arm), `warmup_reps` (discarded after **every** arm switch), `comparison_order` (`ABAB`, `ABBA`, `randomized` + `order_seed`), `pairing`, `stopping_rule` (`fixed_n` only), `alpha` |
| `noise_floor_ref` | the matching noise-floor entry, or `null` → detectability `not_established` |

Unknown fields are errors, so a typo cannot silently fall back to a default.

## Verdict: `Verdict` (`gitm.experiment.verdict/v1`)

The verdict carries:
- `state`, `reason`, `effect`, `interval`, and `method` (the estimator, named exactly)
- `n_baseline`, `n_candidate`
- the `correctness` and `latency` gate results, and `control_status`
- `claim_rejected` and `detectability`
- `cost` (wall clock, accelerator-seconds, estimated $)
- `rollback`: `kept`, `rolled_back`, `not_applied` or `restore_failed`
- `provenance`, the `contract_sha256`, and every `raw` sample, warmup included and flagged

The terminal state is decided by rules applied **in this order**:

1. an apply, sample or restore raised → `failed_to_execute`
2. a known-effect control failed for this operating point → `invalid`
3. the budget ran out before the protocol finished → `inconclusive`
4. too few samples, or a missing primary metric, correctness score or latency metric → `invalid`
5. the correctness gate or any latency gate failed → `regressed`
6. the interval lies entirely above 0 → `improved`; entirely below 0 → `regressed`
7. otherwise → `inconclusive`

The candidate stays applied **only** on `improved`; every other state restores
the baseline snapshot.

`claim_rejected` is reported separately from the state. A candidate can be
`improved` and still reject a claim that promised more than it delivered.

**Estimators:**
- **Paired:** a t-interval on the per-pair relative change `c/b − 1`.
- **Unpaired:** the ratio of means, with a delta-method standard error and Welch degrees of freedom.

## How it fits the existing code

- `evaluate()` drives any existing `Applicator` (`optimizer/apply.py`) through
  snapshot, apply and restore.
- The intervention becomes an ordinary `InterventionSpec`.
- It does not call `apply_intervention`, because that gate reduces the whole
  experiment to one scalar against `min_keep_delta`. It keeps the same
  keep-or-rollback rule, expressed as a verdict instead.
- Measurement goes through a `Sampler` seam. One call runs the workload once on
  whatever is applied. On the MI355X that is a timed replay through
  `ReplayPlan.bench_serve_argv` (gap G3 in `autoresearch_harness_gap.md`).

## Open questions for the working session

1. **Correctness failure → `regressed`, or `invalid`?** The draft says `regressed`
   (the candidate is worse). Tarun, what tolerance is right for precision changes?
2. **Budget exhausted → `inconclusive`?** Or should the budget gate refuse the
   run up front from an estimated cost (Oct 19 scope)?
3. **Warmup after every switch.** This is right for restart-applied knobs but
   costs a model reload per switch on MI355X. Is `ABBA` (half the switches) the
   default, or is the engine rebuilt per sample anyway?
4. **Latency gates on the mean over reps**, or on each rep's p95?
5. **Sequential stopping rules**: needed for v1, or is `fixed_n` enough until
   the noise floor is known?
6. **Where provenance comes from.** Today the caller passes it in; G10 would
   make the evaluator collect it.
