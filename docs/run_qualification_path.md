# MI355X Kimi run qualification path map

**Question this document answers:** at what point does a Kimi loop run become
trustworthy enough for deviation analysis, attribution, or experiment
evaluation?

**Verdict today:** `gitm.optimizer.qualify_run` (v0.2) is the first shared
machine-readable integrity verdict. It is not yet wired as a hard prerequisite
of `gitm deviate` / monitor (Oct 15). Until that wiring lands, E0 in
`scripts/kimi_loop/run_loop.sh` remains the strongest *enforced* loop gate.
The commercial floor heuristic in `gitm.optimizer.qualification` is a different
gate (whether GitM commits to an optimization floor) and must not be confused
with capture integrity.

**Audience:** Nathan (run qualification), Aravinth (signal contract), Zhu
(deployment / checkpoint / engine spec), Isaiah (evaluator consumer), Rahul /
Adit (loop consumers), senior platform (collector / k8s readiness).

Related: checked collector bring-up in
[`docs/kimi_mi355x_collector_bringup.md`](kimi_mi355x_collector_bringup.md);
`qualify_run` v0 in `gitm.optimizer.qualify_run`.

---

## Pipeline overview

```text
predict_sweep (laptop)
  → deploy mi355x-kimi-loop.yaml + arm.sh
  → capture attach (window)
  → per-pid shards /scratch/trace/kimi.jsonl.*
  → injection.read_shards + decode_records → merged JSONL
  → E0 shell asserts (optional path)
  → analyze.sh / gitm deviate / monitor   ← no shared integrity prerequisite
```

`qualify_run` v0 is the first shared verdict that can sit before downstream
analysis. Wiring it as a hard gate is an Oct 15 deliverable, not Friday's.

---

## Boundaries

For each boundary: **owner**, **input → output**, **identity retained**,
**checks**, **failure behavior**, **established by code vs manual/assumed**.

### 1. Deploy / arm

| | |
|---|---|
| **Code** | `deploy/k8s/mi355x-kimi-loop.yaml`, `scripts/kimi_loop/arm.sh` |
| **Owner** | Platform / Adit (cluster); Nathan (collector readiness procedure) |
| **Input → output** | Manifest + HF cache + image digest → Running pod (`2/2`), `/scratch/arm.env`, supervisor-launched `vllm serve` |
| **Identity retained** | Image digest in YAML only; model path on shared FS; arm letter A/B/C/I; `ROCP_TOOL_LIBRARIES`, `GITM_TRACE_NVTX`, `GITM_EXTRA_VLLM_ARGS`, `LD_PRELOAD` in `arm.env`. TP=8 is a vLLM flag in the Deployment, not rewritten per arm. |
| **Checks** | kubectl Ready; `arm.sh` waits for `/health` down then up; idempotent skip if env+health already match |
| **Failure** | Pending on market / taint → waits forever; arm flip hang → look at `/scratch/logs/server_arm*.log` |
| **Established** | Image digest pinned in YAML (code). Arm env written to disk (code). |
| **Assumed / manual** | That the running image matches the YAML digest; that HF cache holds the intended checkpoint revision; that TP/EP/DP match the planner prediction. |

### 2. Predict (context only)

| | |
|---|---|
| **Code** | `scripts/kimi_loop/predict_sweep.py`, `gitm/planner/models/kimi-k2.5.yaml` |
| **Owner** | Zhu (catalogue / what *should* execute); Isaiah consumes predicted floors |
| **Input → output** | Model catalogue + MI355X hardware → `evidence/kimi-mi355x/predicted/` |
| **Identity retained** | Model id, TP/EP, batch/kv assumptions in predicted artifacts — **not** written into the live run's capture header |
| **Checks** | Footprint vs published checkpoint size (catalogue) |
| **Failure** | Wrong catalogue → confident wrong floors; run still "succeeds" |
| **Established** | Predicted numbers on disk for join key `(config, concurrency)` |
| **Assumed** | Live deployment equals the catalogue entry used for predict |

### 3. Run MANIFEST

| | |
|---|---|
| **Code** | `manifest()` in `scripts/kimi_loop/run_loop.sh` → `$RUN/MANIFEST` |
| **Owner** | Nathan (what identity lands); Zhu (what should be declared) |
| **Input → output** | phase, arm, label, utc ts, `rocm=`, `vllm=`, every `export` line from `/scratch/arm.env` |
| **Identity retained** | Engine versions + arm env. **Not** recorded: image digest, checkpoint hash/revision, TP, traffic-manifest id |
| **Checks** | None (append-only log) |
| **Failure** | Silent omission; wrong engine still proceeds |
| **Established** | Whatever `manifest()` printed |
| **Assumed** | Completeness of deployment identity |

### 4. Serve capture sidecars

| | |
|---|---|
| **Code** | `gitm/serve/artifacts.py` (`run_manifest.json`, `serving_summary.json`, `preflight.json`, `kernel_breakdown.json`) |
| **Owner** | Nathan / serve path maintainers |
| **Input → output** | Attach/serve capture → sidecar JSON next to `trace.jsonl` |
| **Identity retained** | How the window was obtained; preflight checks; serving gauges during window |
| **Checks** | Preflight before window; status `ok` / `untraced` / `no_kernels` / `no_traffic` |
| **Failure** | Classified status; still no shared gate into `deviate` |
| **Established** | Sidecar contents when attach path ran |
| **Assumed** | Kimi loop always writes the full sidecar set (E0 uses raw JSONL glob) |

### 5. State telemetry

| | |
|---|---|
| **Code** | Pod amd-smi loop → `/scratch/telemetry/amdsmi.jsonl`; `scrape_metrics` in `run_loop.sh`; optional `gitm/telemetry/collector.py` (not the Kimi loop default) |
| **Owner** | Aravinth (signal meaning / contract); Nathan (verify presence) |
| **Input → output** | 1 Hz GPU samples and selected vLLM `/metrics` lines → files under run dir at phase end |
| **Identity retained** | Timestamps; GPU metrics. Collector `Sample` schema (`node`, `gpu_uuid`, …) is **not** what amd-smi JSONL necessarily uses |
| **Checks** | None automated for contract version |
| **Failure** | Missing file → silent for deviate |
| **Established** | File exists if supervisor wrote it |
| **Assumed** | Cadence, schema, and alignment with capture window |

### 6. Capture / shards / merge

| | |
|---|---|
| **Code** | `gitm/tracer/capture.py`, `gitm/tracer/injection.py` (`read_shards`, `decode_records`) |
| **Owner** | Nathan (integrity); platform (inject ABI / image) |
| **Input → output** | Armed window + inject tool → per-pid shards → merged JSONL (`_header` + events); markers/runtime folded into `range_op`/`range_layer` |
| **Identity retained** | Header: `workload_id`, `fingerprint`, `run_id`, `device_count`, `vendor`, `source`, timestamps. Per event: `pid` (from shard name), `device_id`, optional `correlation_id`, optional ranges. Shard path itself discarded after merge |
| **Checks** | Warnings for malformed lines, `meta.dropped_records`, unmodeled activity; empty well-formed trace on soft failure |
| **Failure** | Best-effort: empty trace or lossy merge with warnings — not a hard process exit |
| **Established** | Events that survived merge; drop counts only if meta was present and E0/tools sum them |
| **Assumed** | All ranks wrote shards; clock domains align; no silent identity repair |

### 7. Trace schema

| | |
|---|---|
| **Code** | `gitm/tracer/schema.py` |
| **Owner** | Aravinth (event contract versioning — **blocker:** no version field today); Nathan (verify runs against contract) |
| **Input → output** | Pydantic `Trace` / event models |
| **Identity retained** | As above; `source` ∈ cupti/rocprof/imports/none. **No** `signal_contract_version` |
| **Checks** | Schema validation when loaded as `Trace` |
| **Failure** | Load errors or discarded malformed lines |
| **Established** | Typed fields present on load |
| **Assumed** | Contract stability across collectors / backends |

### 8. E0 / validate_trace

| | |
|---|---|
| **Code** | E0 in `run_loop.sh`; `gitm/optimizer/validate_trace.py` |
| **Owner** | Nathan |
| **Input → output** | E0: last `**/*.jsonl` under e0 capture → PASS/FAIL print. `validate_trace`: JSONL + planner graph → attribution coverage report |
| **Identity retained** | E0 does not bind to deployment_spec. `validate_trace` takes CLI model/tp/batch — operator-supplied |
| **Checks (E0)** | Trace exists; kernels > 0; named kernels; `dropped_records == 0`; warn if no `range_op` |
| **Checks (validate_trace)** | Worker count, single-worker selection, malformed/invalid/truncated/layer tags, modeled/NVTX coverage — **attribution**, not deployment identity |
| **Failure** | E0 `assert` aborts phase. `validate_trace` exit 1/2 with JSON. Neither blocks `gitm deviate` |
| **Established** | E0 predicates for that shell invocation only |
| **Assumed** | Operator re-runs E0 before later phases; `validate_trace` used offline |

### 9. Correlation / topology

| | |
|---|---|
| **Code** | `gitm/distributed/correlate.py`, `gitm/distributed/topology.py` |
| **Owner** | Nathan (multi-rank provenance — Oct 15 hardening); platform for shard naming |
| **Input → output** | Per-pid records → `range_op`/`range_layer`; `Topology` with pid↔device↔local_rank |
| **Identity retained** | Rank ordinals derived from observed records, not launcher. Equal `correlation_id` across processes collide if correlated on a merged stream |
| **Checks** | Library APIs / tests; **not** invoked as a hard gate on the Kimi analyze path |
| **Failure** | Silent mis-attribution if merge-then-correlate |
| **Established** | What `correlate_by_rank` / `topology_from_*` return when called |
| **Assumed** | Analyze path partitions by pid before correlate |

### 10. Analyze / deviate / monitor

| | |
|---|---|
| **Code** | `scripts/kimi_loop/analyze.sh`, `gitm/optimizer/deviation.py`, `gitm/optimizer/monitor.py` |
| **Owner** | Isaiah (evaluator / reports); Rahul (customer-facing diagnosis); Nathan (gate upstream) |
| **Input → output** | Merged JSONL + predicted graph → deviation JSON / residuals / analysis markdown |
| **Identity retained** | Report paths under run id; no qualification verdict object |
| **Checks** | Multi-worker selection errors in deviate; empty-kernel guards in scheduler loop only |
| **Failure** | Confident wrong residual from half-captured or wrong-deployment trace |
| **Established** | Math on whatever events were supplied |
| **Assumed** | Trace is the declared workload, complete, and correctly identified |

---

## Where trust begins today

| Candidate | Trustworthy for downstream diagnosis? |
|---|---|
| Pod Ready | No — identity incomplete |
| MANIFEST written | No — missing digest/checkpoint/TP/traffic |
| Capture sidecars `status=ok` | Strong capture health signal, still not a shared gate |
| E0 PASS | Strongest **ad-hoc** integrity check; shell-local; not imported by deviate |
| `validate_trace` pass | Attribution coverage only |
| `qualification.qualify` commit=true | **Wrong gate** — commercial floor, not capture integrity |
| `qualify_run` v0.2 | **Yes, as a library/CLI verdict** — fingerprints + named checks; hard gate into deviate = Oct 15 |

**Conclusion:** `python -m gitm.optimizer.qualify_run` (or `gitm qualify-run`) is
the first shared point that can declare a run admissible. It is not yet
structurally unavoidable for `gitm deviate`; E0 remains the practical shell gate
on the loop until the Oct 15 `QualifiedRun` wiring.

---

## Gap register

| Gap | Code path | Failure case | Downstream consequence | Proposed owner |
|---|---|---|---|---|
| No versioned signal contract | `gitm/tracer/schema.py`, `gitm/telemetry/schema.py` | Collector silently changes fields | False qualified comparisons | **Aravinth** (define/version); Nathan verifies |
| MANIFEST omits image digest, checkpoint rev, TP, traffic id | `run_loop.sh` `manifest()` | Wrong checkpoint / image, complete trace | Confident wrong diagnosis | **Zhu** (declared deployment); Nathan (record + qualify) |
| `dropped_records` / loss not always persisted on merged header | `injection.read_shards` warnings | Lossy capture looks clean downstream | Understated residual / missing ops | Nathan |
| Shard path / node identity discarded at merge | `injection.py` merge | Two nodes, same pid namespace story | Identity corruption | Nathan (Oct 15 provenance) |
| Correlation IDs not scoped by source process on merged stream | `correlate.py` | Equal IDs across ranks joined | Wrong `range_op` | Nathan |
| E0 not shared with deviate/monitor | `run_loop.sh` vs `deviation.py` | Skip E0, analyze anyway | Unqualified diagnosis | Nathan (Oct 15 `QualifiedRun` gate); Isaiah (consume only QualifiedRun) |
| amd-smi path ≠ `gitm.telemetry.Collector` | runbook vs `collector.py` | Contract written for wrong backend | False "collector live" | Aravinth + Nathan |
| Rank not on event wire | schema / topology | Missing rank → silent inference temptation | Invalid joins | Nathan (not_established until observable) |
| Clock alignment across sources | capture + metrics + amd-smi | Window misaligned with load | Empty or partial capture | Nathan + platform |
| Commercial `qualify()` confused with run qualification | `qualification.py` | Floor commit on bad capture | Wrong refund / commitment | Nathan (keep gates separate) |

### Blockers raised (same day)

1. **Undefined signal contract (Aravinth):** no `signal_contract_version` in repo. Friday ships provisional `signal_contract.v0` under Nathan; Aravinth must replace/version.
2. **Incomplete deployment identity on disk (Zhu / platform):** image digest and checkpoint revision are not in `$RUN/MANIFEST`. Until recorded or supplied via `deployment_spec.json`, those checks are `not_established`.
