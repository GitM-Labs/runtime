# Graceful Fallback and Runtime Wiring Audit

## Executive summary

Status: **in progress**. This ledger is the primary deliverable for the audit of
`gitm/` and `scripts/`. Findings are ranked by the likelihood that a fallback can
turn missing knowledge into a confident wrong result, with answer-deciding byte
traffic and dominant expert terms ranked above non-binding estimates.

Fallback masks and wiring failures closed: **44 so far**. Wiring gaps confirmed:
**12 so far**.
Deferred findings: **none so far**.

The worktree already contained uncommitted scheduler/serve changes and two new
expert-signal files before this audit branch was created. They are preserved and
treated as pre-existing work until their ownership and relevance can be separated;
they will not be silently absorbed into an audit commit.

## Finding ledger

| Rank | Status | Severity | Location | Distorted term / contract | Surfacing state | Failure scenario | Disposition |
|---:|---|---|---|---|---|---|---|
| 1 | fixed | critical | `gitm/scheduler/loop.py` execution-graph dispatcher | Entire sparse-MoE execution graph; especially dominant expert weight bytes and hybrid-attention/KV terms | REFUSE/FLAG | A live V4 engine was converted into the legacy `ModelSpec` and sent to `predict_graph`; the production loop never called `spec_from_hf_config`/`predict_moe_graph`. | Added `_execution_graph`: partial sparse configs reach the sparse gate, valid configs dispatch to `predict_moe_graph`, and unpriceable configs refuse claims into a named measurement-only result. The full graph provenance/flags are serialized. |
| 2 | fixed | critical | `gitm/scheduler/loop.py` config resolution and refusal artifact | Model identity and every residual | REFUSE | Parse/version-drift failures returned `None` and became the plausible Llama-2-7B default graph. | Typed resolution preserves exception class/message in `prediction_refusal.json` and the report; missing engine/config refuses graph-based claims instead of defaulting. |
| 3 | fixed | high | `gitm/planner/context.py`; `gitm/planner/roofline.py`; loop and attach artifacts | Compute and HBM denominators for every node | REFUSE/FLAG/WARN | Unknown/no GPU SKU became A100 and artifacts recorded A100 as if detected. | `HardwareSpec.is_fallback`/`Graph.hardware_is_fallback` distinguish substituted pricing. The loop refuses unknown hardware; attach records observed hardware separately from fallback pricing and prints a warning. |
| 4 | fixed | high | roofline, sparse graph, shared final-spec dtype gate, loop/attach boundaries | Weight/KV/activation bytes; dominant decode-binding term | FLAG/REFUSE | Unknown dtype returned 2-byte bf16; mixed-byte nodes and scalar sizing could not identify the substituted width. | Per-node `bytes_are_fallback` and `Graph.has_fallback_bytes` now cover the graph, while shared `validate_priceable_dtypes` refuses final resolved live/command-line dtypes before customer-facing predictions. JSON and CLI/report consumers surface both directions. |
| 5 | fixed | high | `gitm/scheduler/loop.py`; `gitm/optimizer/report.py`; report template | Residual magnitude shipped on every Claim | FLAG | A 10x/18x model error and a 2x error both rendered `+100%`, hiding that the model was broken and repeating one aggregate as if claim-specific. | Raw residuals now remain on `Claim`; display capping and saturation are derived, saturated rows print raw magnitude, scopes distinguish run/target-op, and non-finite residuals are refused. Report, scheduler sibling, and ruff checks pass. |
| 6 | fixed | high | `gitm/optimizer/monitor.py`; `gitm/scheduler/loop.py`; report template | Residual population / coverage | WARN | Unclassified kernels and classified ops absent from the graph were skipped, so the loop could report clean residuals over a small, biased fraction without matched/total coverage. | `Residuals` now records total/classified/matched launch and kernel-time coverage. Incomplete coverage emits warnings into `residuals.json`, the run summary, and a printed Runtime diagnostics section; fully matched traces stay clean. Monitor/report/loop sibling tests and ruff pass. |
| 7 | fixed | high | loop dispatcher dense/sparse quantization gates | Dense-path MoE expert weight bytes | REFUSE | Unknown quantization methods were ignored, leaving bf16 byte defaults. | Sparse configs no longer use the legacy dense extraction; both sparse and dense dispatch refuse unknown quantization methods with the method named. |
| 8 | fixed | medium | sparse config resolution in loop and attach | Expert weights, often the dominant term | WARN | Missing `expert_dtype` inherited linear `weight_dtype`. Official V4 Flash configs declare it (Flash=`fp4`, Base=`fp8`), but uniform foreign MoEs may omit it legitimately. | Accepted inheritance now rides in `LiveSpec.warnings` / loop diagnostics and reaches artifacts plus human output; unpriceable inherited dtypes still refuse. |
| 9 | fixed | medium | `gitm/planner/graph.py` | Zero-time byte-moving nodes | FLAG | `has_unpriced_collectives` scanned all nodes despite its narrow name. | Split `has_unpriced_nodes` (general safety net) from the genuinely collective-specific property; production trust consumers use the general flag and artifacts retain both. |
| 10 | fixed | high | loop `predicted_graph.json`, summary, and Markdown diagnostics | Prediction trust diagnostics | FLAG/WARN | Loop artifacts omitted fallback, estimate, default, model, batch, sharding, and hardware provenance. | Artifact now carries model source, observed/pricing hardware, batch/sharding, graph flags, per-node diagnostics, and warnings; the report and run summary consume them. |
| 11 | fixed | critical | `gitm/bench/schema.py`; `gitm/bench/baseline.py` | Benchmark saturation/sign-off gate | REFUSE | A missing `stall_breakdown` became 0% GPU active and could sign off a CPU/no-telemetry run as unsaturated. | Missing coverage is now `None` and fails saturation; code, manifest, and GPU identity have a separate provenance gate and safe report rendering. |
| 12 | fixed | high | `gitm/tracer/vllm_stats.py`; `gitm/serve/vllm.py` | TPOT percentiles and SLO goodput | FLAG/WARN | SSE chunk counts undercounted multi-token chunks but entered TPOT and goodput as authoritative; missing counts passed the TPOT half of the SLO. | Usage/engine counts are authoritative, chunk estimates are excluded from TPOT/goodput, and coverage warnings reach CLI and loop reports. |
| 13 | fixed | high | `gitm/planner/roofline.py`; `gitm/planner/graph.py` | Compute/HBM denominator | FLAG/WARN | Positive work with a zero catalogue rate was priced at zero and a sibling term could hide the missing denominator. | Per-dimension unpriced flags survive on each prediction, aggregate on the graph, and surface through loop/attach artifacts and diagnostics. |
| 14 | fixed | high | `gitm/runtime_driver.py` | Trace coverage and every claimed runtime detail | REFUSE/WARN | Zero captured kernels still produced `PASS: ... all details measured`. | The driver now uses canonical measurement, emits NO DATA, records diagnostics, and exits 3 without any positive-duration kernel. |
| 15 | fixed | high | `gitm/optimizer/attribution.py`; `gitm/optimizer/dr.py`; loop/driver/report consumers | Causal evidence | WARN | Import/fit failure became `no strong causal signal`; DR nuisance-model fallbacks emitted estimates silently. | Attribution carries import, sample-coverage, pair-fit, and nuisance-model diagnostics into JSON, CLI, and Markdown. |
| 16 | fixed | high | `gitm/kernels/library.py`; scheduler caller | Intervention availability / candidate coverage | REFUSE/WARN | A missing library returned `[]`, indistinguishable from no applicable levers. | The loader refuses with the path; the scheduler emits a named candidate-coverage-unavailable measurement report and no optimization claims. |
| 17 | fixed | medium | `gitm/telemetry/collector.py`; benchmark samplers and runtime-driver consumer | State-telemetry coverage | WARN | Sampling, sink, and close failures were swallowed, making empty telemetry look like a quiet GPU. | Collector failures are deduplicated warnings and report diagnostics; benchmark sampler failures ride into JSON and stdout. |
| 18 | fixed | high | scheduler specialized HFT/OpenFold/edge intervention result paths | Residual and intervention status | FLAG/REFUSE | No CUPTI trace attached A/B speedup to fabricated `stream_concurrency=0.0`; missing A/B still reported `ok`. | All siblings use measured throughput delta without trace coverage; missing/non-finite A/B emits no claim and returns `intervention_failed`. |
| 19 | fixed | medium | `gitm/optimizer/headroom_kernel_rank.py` | Compute and memory headroom | FLAG/WARN | Memory-only samples fabricated 100% compute headroom; utilization-only samples fabricated zero memory capacity. | Each dimension is optional, absent families stay `None`, and diagnostics name missing telemetry. |
| 20 | fixed | medium | `gitm/optimizer/measure.py`; duplicated runtime-driver measurement | Kernel residual denominator | WARN/REFUSE | Zero-duration kernels used a fabricated 1 ns median and attribution filtering had no coverage diagnostic. | Invalid durations are excluded with counts, attribution abstention is diagnostic, and both consumers use canonical measurement. |
| 21 | fixed | critical | `gitm/deploy/attach.py` | Live deployment state | REFUSE | Resolving a live standalone PID returned `attached` even though the documented injection path is not implemented and no shim was installed. | The path now returns `unsupported` with the PID and a named no-injection reason; only an implemented lifecycle may claim attachment. |
| 22 | fixed | high | `gitm/cli.py` | Automation/gate status | REFUSE | `prediction_refused`, missing-candidate coverage, intervention failure, and malformed loop results all exited zero; an absent report could also look like a completed run. | Only `status=ok` exits zero. Degraded or invalid results exit 3 and always emit an explicit unavailable/diagnostic report. Report writes are UTF-8. |
| 23 | fixed | high | `gitm/scheduler/loop.py`; `gitm/agents/autoresearch.py` | Trace evidence and bottleneck class | REFUSE/FLAG | A trace containing only zero-duration events bypassed the no-data gate; an empty/invalid trace was classified `compute`, enabling plausible compute interventions without evidence. | Loop evidence requires a positive-duration kernel, and autoresearch emits `unclassified` for empty/invalid traces with no applicable rules. |
| 24 | fixed | high | shared timing predicate; workloads, runtime driver, HFT/edge benchmarks, demo script | Throughput, latency, and speedup denominators | REFUSE | Non-positive/non-finite timers were floored to 1 ns, turning timer failure or empty work into enormous plausible throughput/speedup. | Shared finite-positive duration/work predicates now refuse each affected calculation with its named context. |
| 25 | fixed | high | `gitm/bench/profile.py`; `gitm/bench/cli.py` | Profile completeness and utilization breakdown | REFUSE | Failed workload commands, sampler timeouts, missing profiler CSV, and overlapping timing could still produce a successful profile command with clamped shares. | The bundle records every failure, contradictory timing refuses rather than clamps, and incomplete profile/empty manifest commands exit nonzero. |
| 26 | fixed | medium | `gitm/agents/autoresearch.py` | Search-domain coverage | WARN | Missing/version-drifted vLLM `EngineArgs` silently substituted a frozen knob catalog; missing CLI introspection silently narrowed domains. | Both offline catalogs now emit named runtime warnings; empty introspection remains explicitly empty rather than pretending live coverage. |
| 27 | fixed | medium | `gitm/workloads.py`; HFT harness | Benchmark provenance and trace completeness | WARN/REFUSE | Missing data silently generated synthetic smoke input, missing GPU libraries silently selected CPU, and failed synchronization/pool cleanup looked like complete GPU coverage. | Synthetic/CPU/cleanup/synchronization fallbacks warn at use; existing benchmark provenance gates still refuse these runs as publishable GPU baselines. |
| 28 | fixed | medium | `gitm/safety/failopen.py` | Rollback/audit visibility | WARN/FLAG | A broken audit sink or failed signal-handler setup was swallowed, removing the very evidence meant to surface fail-open behavior. | Audit and signal failures are retained on the guard and warned immediately; reset/restore behavior is covered. |
| 29 | fixed | high | `gitm/kernels/library.py` | Candidate/intervention coverage | REFUSE | A present but malformed or empty intervention library returned no candidates just like a valid library with no applicable matches. | The loader refuses non-mapping, missing-list, and empty-list libraries with named reasons; scheduler coverage handling remains explicit. |
| 30 | fixed | medium | `gitm/routing/scorer_v0.py` | Routing score | REFUSE | Out-of-range probabilities, non-binary flags, and unknown company tiers flowed into a plausible score (unknown tier became a default weight). | The scorer validates every bounded/binary input and refuses tiers outside 1/2/3. |
| 31 | fixed | high | `gitm/importers/node_rollup.py` | Communication share and node ceiling distance | REFUSE | Zero/negative trace wall time was floored to 1 ns and zero-weight ceiling aggregation returned 0.0, fabricating clean rollup values. | Device and node rollups require positive wall time; invalid captures become named per-file analysis failures. |
| 32 | fixed | medium | `gitm/importers/analyze.py` | Kept-trace artifact wiring | REFUSE/fix | Internal identifiers containing `:` were used as filenames; on Windows the write failed and the whole input was demoted to a generic per-file failure with no trace artifact. | Artifact-only stems are sanitized portably while report identifiers retain their original value; end-to-end importer coverage verifies artifacts are written. |
| 33 | fixed | high | `gitm/tracer/vllm_stats.py` | TTFT, TPOT, and SLO goodput | REFUSE/WARN | Backwards or non-finite request timestamps were clamped to zero latency and could count as SLO-meeting traffic. | Invalid timestamp spans now produce no latency/goodput contribution and a serving warning naming excluded requests. |
| 34 | fixed | high | `gitm/tracer/vllm_stats.py`; scheduler artifact/report | Scheduler evidence coverage | WARN/FLAG | Synchronous/background scheduler reads swallowed every exception and returned an ordinary empty summary; a failed sampler looked like an idle/uninstrumented engine. | The fail-open sampler warns once per failure path, retains diagnostics in its summary/artifact, and feeds them into runtime diagnostics. Invalid/clamped intervals also surface. |
| 35 | fixed | high | NVIDIA telemetry backend and collector | Throttling, process, utilization, memory, power, clocks, and ECC state | FLAG/WARN | Per-field NVML failures became `NONE`, `{}`, `nvidia-unknown-*`, or `None`; especially, unavailable throttle reasons looked like observed no-throttle state. | Partial samples now carry field-level diagnostics; the collector deduplicates, warns, and carries them downstream. Backend close failures propagate to the collector instead of disappearing. |
| 36 | fixed | high | `gitm/optimizer/monitor.py`; `deviation.py` | Kernel-time residual and deviation population | WARN/FLAG | Zero/negative kernel durations were floored to 1 ps and entered residuals; non-positive predicted durations were likewise made comparable. | Invalid observations are excluded with launch-count diagnostics; unpriced predictions are excluded explicitly; deviation traces retain invalid kernels as departures rather than declaring them in-band. |
| 37 | fixed | high | `gitm/optimizer/metrics.py` | Busy, HFU/MFU, and MBU denominators | REFUSE | Zero trace wall time or zero peak bandwidth returned plausible 0% utilization. | Shared positive-duration validation and positive hardware-denominator gates now refuse direct callers; normal/importer siblings remain green. |
| 38 | fixed | high | `gitm/optimizer/apply.py` | Intervention verification and safety trail | WARN | Apply-only changes, broken audit sinks, activation/shutdown hooks, and GPU cleanup failures were silent; a mutation could be kept without evidence or lose rollback telemetry. | Unmeasured applies and every best-effort safety/lifecycle failure now warn while preserving fail-open behavior. |
| 39 | fixed | critical | dense scheduler graph parser; obsolete parallel parser | Dense activation/weight bytes and hybrid-attention graph shape | REFUSE/fix | Production treated every unknown dense dtype as 2-byte bf16 and ignored hybrid-attention cadence; a more capable engine parser existed only in tests and silently returned `None`. | Production parsing uses shared dtype priceability/byte widths, preserves attention cadence, and names missing/invalid fields. The dead parallel implementation and its fallback tests were removed; coverage targets the production parser. |
| 40 | fixed | high | `gitm/serve/metrics.py`; attach output | Server throughput, token count, and TPOT | WARN/REFUSE | Infinite Prometheus values entered aggregates, non-positive windows could price throughput, sampler scrape holes disappeared, and missing token/TPOT fields printed as zero. | Non-finite values are discarded, invalid windows emit no rate plus a note, sampler failures surface, and human output prints `unavailable` rather than zero. |
| 41 | fixed | critical | scheduler default live-engine throughput probe | A/B throughput and keep/rollback decision | REFUSE | A runner with no recognized work-count field silently became one generated token; a failed timer was floored to 1 ns. | The probe requires a named positive work count and positive finite duration; missing/zero evidence rolls the candidate back through the existing apply gate. |
| 42 | fixed | high | KITTI/nuScenes WorkUnit and baseline runners | Stall shares and FPS | REFUSE | A zero frame timer returned 0% for every stage, and a zero baseline timer divided into FPS. | Frame properties and baseline windows share the positive-duration gate; output writes are UTF-8. |
| 43 | fixed | medium | `gitm/planner/kitti_graph.py` | Planner wiring and hardware provenance | REFUSE | The PointPillars graph had no production caller and its example defaulted to A100; measured comparisons defaulted missing FPS/stall fields to zero. | Added a SKU-required `gitm plan-kitti` boundary that refuses catalogue misses; measured comparison refuses missing/non-positive fields. |
| 44 | fixed | medium | diagnostic/demo scripts | GPU-idle decision, real-trace residuals, and assumed hardware | WARN/REFUSE | Demo telemetry failure silently triggered the idle-GPU lever, real-trace code suppressed all warnings and floored zero medians, and serving headroom silently assumed H100/zero failures. | Scripts now refuse telemetry-less idle claims, exclude and warn on invalid timestamps, preserve warnings, and state assumed hardware/missing failure counts explicitly. |

Status values: `open`, `fixed`, `deferred (reason)`, or `won't fix (reason)`.

## Seed-finding verification

| Seed | Verification | Status |
|---:|---|---|
| 1 | Unknown weight dtype falls back to bf16 without a bytes-side flag. | Fixed with per-node/graph flags, shared final dtype refusal, and boundary consumers. |
| 2 | Loop MoE dispatch lacks the attach path's dtype priceability gate. | Fixed with shared predicates and typed sparse dispatcher/refusal. |
| 3 | Missing `expert_dtype` inherits `weight_dtype`, potentially mispricing the dominant expert term. | Fixed by explicit inheritance warnings; official V4 configs were confirmed to declare the field. |
| 4 | Residual percentage clamps at ±100% and loses raw error magnitude. | Fixed with raw + capped-display contract, saturation note, scope, and non-finite refusal. |
| 5 | Residual classification drops unmatched kernels without loop-path coverage. | Fixed with count/time coverage and JSON/summary/report warnings. |
| 6 | Unknown SKU becomes A100 while recording A100 as if observed. | Fixed with hardware fallback provenance, loop refusal, and attach warning/separate fields. |
| 7 | Broad model-config parse rescue becomes an unmarked default model. | Fixed in production dispatch with typed named refusal; no default graph. |
| 8 | `has_unpriced_collectives` scans all nodes despite its narrow name. | Fixed by splitting general and collective-specific predicates. |

## Sweep coverage

| Area | Phase 1 fallback sweep | Phase 2 wiring sweep | Notes |
|---|---|---|---|
| top-level runtime / API / CLI / workloads | Pending | Pending | |
| agents | Pending | Pending | |
| bench | In progress | In progress | Saturation and provenance sign-off gates swept/fixed; remaining CLI/results paths under review. |
| benchmarks | In progress | In progress | KITTI/edge telemetry fallbacks fixed; remaining harnesses under review. |
| deploy | Pending | Pending | |
| importers | Pending | Pending | |
| kernels | Pending | Pending | |
| optimizer | In progress | In progress | Attribution, headroom, and measurement masks fixed; apply/safety-audit paths remain under review. |
| planner | In progress | In progress | Seed and denominator paths swept/fixed; dead KITTI planner path remains under review. |
| routing | Pending | Pending | |
| safety | Pending | Pending | |
| scheduler | In progress | In progress | Main vLLM and specialized intervention siblings swept/fixed; remaining orchestration fallbacks under review. |
| serve | In progress | In progress | Launch/attach gates and token provenance swept/fixed; remaining CLI paths under review. |
| telemetry | In progress | In progress | Optional fields and collector/backend/sink failures now surface; remaining call-site consumers under review. |
| tracer | In progress | In progress | Capture backend failures warn/source-flag; scheduler sampling and request-summary fallbacks under review. |
| scripts | Pending | Pending | |

## Diagnostic-consumer trace

Every boolean flag, warning list, `estimated`, provenance field, and `has_*` or
`*_fallback` property discovered during the sweep is listed here and traced to a
human- or gate-visible consumer.

| Producer | Diagnostic | Downstream consumer | User/gate boundary | Status |
|---|---|---|---|---|
| `RooflinePrediction` / `Graph` | peak, bytes, hardware fallback; estimated; per-dimension unpriced nodes | loop and attach serializers/diagnostics | JSON + Markdown/CLI | Traced/fixed |
| `Residuals` | coverage counts/warnings | loop residual JSON, summary, report diagnostics | JSON + Markdown | Fixed |
| `ImportStats` / importer rollup | warnings, drops, caveats, SKU/time provenance | analyze summary + customer report | JSON + Markdown | Traced |
| `CaptureResult` / kernel taxonomy | warnings and capture status | serve artifacts + CLI | JSON + CLI exit | Traced |
| `ServingSummary` | TTFT/TPOT sample counts and token-provenance warnings | serve/loop artifacts | JSON + CLI/Markdown | Traced/fixed |
| `Collector` / `GpuHeadroom` | component failures and missing metric-family diagnostics | runtime driver and benchmark artifacts | warning + JSON/Markdown/stdout | Traced/fixed |
| `FailOpenGuard` | revert failures | `failures` attribute + audit log | programmatic/audit artifact | Revert failures traced; broken audit-sink fallback under review |

## Sibling-path validation matrix

| Capability | Path A | Path B | Guard parity | Status |
|---|---|---|---|---|
| MoE config pricing | attach resolves raw config then validates final dtypes | scheduler recognizes partial sparse configs, validates, and dispatches sparse graph | Shared predicates; path-specific input adapters | Fixed |
| execution lifecycle | launch | attach | To inventory | Pending |
| model family | dense | MoE | To inventory | Pending |
| workloads | each dispatch branch | sibling branches | To inventory | Pending |

## Artifact-consumer trace

| Artifact writer | Artifact | Production reader / boundary | Status |
|---|---|---|---|
| — | — | — | Inventory pending |

## Completeness pass

This section must be empty before completion.

- Subpackages not swept: top-level runtime/API/CLI/workloads, agents, bench,
  benchmarks, deploy, importers, kernels, optimizer, planner, routing, safety,
  scheduler, serve, telemetry, tracer, scripts.
- Diagnostic flags/warnings not traced: inventory not yet complete.
- Asymmetric validation gates: inventory not yet complete.
- Fallbacks judged acceptable without confirming REFUSE/FLAG/WARN: inventory not
  yet complete.
