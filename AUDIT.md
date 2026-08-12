# Graceful Fallback and Runtime Wiring Audit

## Executive summary

Status: **complete**. This ledger is the primary deliverable for the audit of
`gitm/` and `scripts/`. Findings are ranked by the likelihood that a fallback can
turn missing knowledge into a confident wrong result, with answer-deciding byte
traffic and dominant expert terms ranked above non-binding estimates.

Fallback masks and wiring failures closed: **93**. Wiring gaps confirmed:
**12**. The final pass found no deferred findings: every accepted fallback is
explicitly REFUSE, FLAG, or WARN and has a visible consumer.

The worktree already contained uncommitted scheduler/serve changes and two new
expert-signal files before this audit branch was created. They were separated into
`feat/expert-signal-eplb` (commit `452b2c8`); the Codex-only `AGENTS.md` guidance is
on `chore/agents-guidance` (commit `4796a54`). Neither is part of this audit branch.

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
| 45 | fixed | high | dense/A-B graph priceability boundary | Dense graph bytes and A/B baseline terms | REFUSE | A dense or A/B input with an unknown dtype/quantization method could still enter prediction or comparison through a caller that bypassed the main parser. | The shared priceability predicate is applied at both graph and A/B boundaries; unpriceable inputs produce a named refusal. |
| 46 | fixed | high | `gitm/serve/metrics.py` server metric aggregation | Throughput, tokens, and latency | WARN/REFUSE | Missing or non-finite server fields were treated as zero and then aggregated into a confident serving result. | Invalid fields are excluded with sampler diagnostics; invalid windows refuse rates and human output says `unavailable`. |
| 47 | fixed | medium | `gitm/serve/discover.py`; telemetry sinks | Server/model discovery and sink state | WARN | An inaccessible `/proc` or failed sink could look like no server or a quiet run. | Discovery and sink failures carry named diagnostics to the caller and artifacts; no empty discovery is presented as verified. |
| 48 | fixed | high | replay validation and verification evidence | Replay identity and validation truth | REFUSE | A replay with missing identity or contradictory validation fields could be compared as reproducible. | Replay/verification gates require complete identity, schema, package, and validation evidence; missing fields mismatch rather than default. |
| 49 | fixed | high | dense planner compute-dtype extraction | Dense activation/weight byte width | REFUSE | A compute dtype missing from a dense spec fell back to bf16 while the graph retained a plausible shape. | Dense extraction uses the shared dtype validator and refuses unknown or missing compute dtype. |
| 50 | fixed | medium | `gitm/optimizer/headroom.py` evidence split | Compute/memory headroom | WARN/FLAG | One telemetry family absent could be represented as a fabricated 50/50 split. | Missing dimensions remain absent; an indicative split is explicitly labeled and the report states the limitation. |
| 51 | fixed | high | benchmark A/B and cleanup paths | Speedup, rollback, and run completeness | REFUSE/WARN | A failed A/B sample or cleanup hook could leave an apparently successful intervention. | Non-finite/empty A/B evidence refuses claims; cleanup failures are retained as diagnostics and reports are degraded. |
| 52 | fixed | high | serving trace duration gates | Serving throughput and SLO windows | REFUSE | A zero-duration serving trace could be turned into a large rate or an all-good SLO. | Positive finite timing is required; invalid requests are excluded and the serving result is degraded/refused. |
| 53 | fixed | high | vLLM CUDA compatibility gate | CUDA/runtime compatibility | REFUSE/WARN | An incompatible or unverified CUDA build could run through the workload path as if supported. | The compatibility preflight gates the workload and records unverified build details in the report. |
| 54 | fixed | high | report delta and claim formatting | Residuals and deltas | REFUSE | NaN/inf deltas could serialize as credible percentages or pass a gate. | Non-finite values are rejected before formatting or sign-off, with a named diagnostic. |
| 55 | fixed | high | benchmark sign-off evidence | Baseline provenance and saturation | REFUSE | Missing benchmark identity or breakdown fields could pass publication gates as zero-valued evidence. | Sign-off requires complete provenance, code, manifest, GPU, and timing fields. |
| 56 | fixed | medium | planner/benchmark GPU-count discovery | Device count and topology | WARN/REFUSE | A failed GPU query became one device and produced a plausible single-GPU result. | Counts are marked fallback or refused when required; the report names the discovery failure. |
| 57 | fixed | medium | imported trace device-count handling | Multi-GPU coverage | WARN | A trace with missing device metadata could be summarized as a complete one-device capture. | Device-count discovery failures and selected-device scope are retained in importer diagnostics. |
| 58 | fixed | medium | resident-footprint provenance | Memory residency and headroom | FLAG | An inferred resident footprint was indistinguishable from a measured value. | Resident bytes carry provenance and the report distinguishes observed, inferred, and unavailable values. |
| 59 | fixed | medium | autoresearch GPU-count fallback | Search/optimization scope | WARN | Autoresearch used one GPU when discovery failed and could claim a full-device experiment. | The fallback count is warned and included in the run diagnostics. |
| 60 | fixed | high | direct sparse graph builder validation | Sparse shape, precision, and KV terms | REFUSE | Programmatic callers bypassing loop/attach could construct default-shaped or unpriced MoE graphs. | Direct builders validate structural fields and all priceable dtypes, including expert and KV widths. |
| 61 | fixed | high | injected tracer ingestion and Windows PID selection | Trace completeness and process identity | WARN/REFUSE | Malformed shard lines or ambiguous PID files could be dropped while the remaining trace looked complete. | Dropped records and selected PID scope are named in capture diagnostics; incomplete injected traces cannot claim complete evidence. |
| 62 | fixed | high | public runtime input validators | Workload timing and work counts | REFUSE | Direct public API callers could pass empty work or invalid timing and receive a throughput value. | Shared positive finite duration/work predicates gate all public runtime calculations. |
| 63 | fixed | medium | telemetry discovery and backend field probes | Utilization, power, clocks, ECC, and process state | FLAG/WARN | A missing backend field became `None`/empty and looked like a measured zero or no-throttle state. | Field-level diagnostics are retained, deduplicated, and consumed by collector/report paths. |
| 64 | fixed | high | attach window validation | Live trace interval | REFUSE | An invalid attach start/end window could produce a trace with no trustworthy temporal scope. | Attach refuses non-positive/non-finite windows and reports the named reason. |
| 65 | fixed | high | optimizer gate evidence | Optimization acceptance controls | REFUSE | Missing or non-finite gate evidence could be interpreted as a passing zero delta. | Gate controls require finite, complete evidence and refuse with a diagnostic. |
| 66 | fixed | medium | imported launch-shape metadata | Kernel launch dimensions | WARN | Missing launch dimensions were filled in without telling the importer, changing occupancy/shape interpretation. | Import diagnostics record launch-shape fallback and the customer report preserves the caveat. |
| 67 | fixed | high | dense graph parser shape/dtype fields | Dense FLOPs and bytes | REFUSE | A parser field omission could become a default shape or dtype in a production graph. | Production parser validates required shape/dtype fields and refuses incomplete graphs. |
| 68 | fixed | high | dense TP/precision/spec wiring | Sharding and weight precision | FLAG/REFUSE | Declared TP or precision could be dropped between config parsing and graph construction. | The production path carries sharding/precision into graph nodes and flags missing/unverified fields. |
| 69 | fixed | medium | live headroom CLI | GPU headroom decision | WARN/REFUSE | CLI invocation without live telemetry could still recommend an idle-GPU action. | The CLI refuses telemetry-less decisions and prints the diagnostic. |
| 70 | fixed | high | scheduler field probes | Scheduler queue/cache/load evidence | WARN | Version-drifted engine fields were silently omitted, making an empty sample look idle. | Each failed probe is named in `SchedulerSample.diagnostics` and reaches scheduler artifacts/report. |
| 71 | fixed | high | HFT/edge A/B controls | Intervention speedup and control validity | REFUSE | Invalid control values or missing A/B halves could produce a plausible keep decision. | Controls and both measured halves are validated before comparison; invalid runs are intervention failures. |
| 72 | fixed | high | benchmark timing partitions | Stall/phase shares | REFUSE | Overlapping or zero total phase timers were clamped into a clean partition. | Shared partition validation rejects contradictory/non-positive timing and names the components. |
| 73 | fixed | high | OpenFold A/B evidence | Protein inference speedup | REFUSE | Failed or incomplete OpenFold baseline/variant measurements could be reported as a speedup. | OpenFold requires positive finite paired measurements and refuses unsupported evidence. |
| 74 | fixed | high | utilization windows | GPU utilization and memory bandwidth | REFUSE | Backwards or overlapping utilization windows could be normalized into a valid-looking percentage. | Window ordering, duration, and overlap are validated before utilization is computed. |
| 75 | fixed | high | scheduler telemetry numeric values | Queue/cache/token counters | REFUSE/WARN | Negative, non-finite, or out-of-range scheduler values entered summaries as ordinary measurements. | Numeric probes reject invalid values or attach field diagnostics; summaries expose degraded coverage. |
| 76 | fixed | medium | CUDA build/version discovery | Build provenance | WARN | An unavailable CUDA version was rendered as a verified component version. | Build versions are labeled unverified and are visible in preflight/report output. |
| 77 | fixed | medium | roofline BF16 peak controls | Hardware peak denominator | FLAG | A declared bf16 peak could be silently replaced by the catalogue default. | The declared peak is honored and hardware fallback provenance remains separate from measured/declared values. |
| 78 | fixed | medium | host flamegraph capture | Profile artifact completeness | WARN | A failed py-spy capture produced a profile bundle with no visible host artifact. | The bundle lists missing flamegraph output and the profile report surfaces it. |
| 79 | fixed | medium | importer cleanup | Kept-trace/report completeness | WARN | Temporary-file cleanup failures were swallowed after a seemingly successful import. | Cleanup losses are retained in `ImportStats.warnings` and customer diagnostics. |
| 80 | fixed | medium | profile bundle artifact manifest | Profile evidence completeness | FLAG/WARN | Missing GPU CSV, host profile, or profiler output was not distinguishable from a complete bundle. | Every expected artifact and missing item is serialized and printed by the profile CLI. |
| 81 | fixed | high | live apply fail-open wiring | Mutation and rollback evidence | WARN/FLAG | Apply/rollback lifecycle failures could be swallowed while a mutation remained active. | Fail-open guard state, audit-sink failures, and lifecycle failures are retained and surfaced. |
| 82 | fixed | high | auto-revert evidence | Keep/rollback decision | REFUSE | Invalid verification evidence could trigger or suppress auto-revert as if it were measured. | Auto-revert requires complete finite evidence and refuses malformed verification. |
| 83 | fixed | medium | CUDA sibling parity | Launch vs attach CUDA verification | WARN | One lifecycle path reported an unverified CUDA build while its sibling treated the same field as verified. | Shared CUDA diagnostics are used by both paths. |
| 84 | fixed | high | GPU trace parser drops | Trace event population | WARN | Malformed GPU records were discarded without a count, biasing utilization and residual coverage. | Parser drop counts and reasons reach capture status and reports. |
| 85 | fixed | high | sparse KV sizing artifact wiring | KV bytes and memory floor | FLAG/REFUSE | Sparse KV sizing was computed but omitted from prediction artifacts, hiding a dominant memory term. | KV width/size and fallback provenance are serialized in graph and scheduler artifacts. |
| 86 | fixed | high | stream-concurrency evidence | Intervention causal evidence | REFUSE | Unsupported stream-concurrency telemetry became an A/B claim with a fabricated zero/one value. | Unsupported evidence returns measurement-only/intervention-failed status with a named reason. |
| 87 | fixed | medium | runtime schema fields | Report/artifact contract | REFUSE/fix | Producers emitted fields that consumers never honored, making a supposed diagnostic ineffective. | Unhonored fields were removed or wired to a real consumer; schema tests cover the contract. |
| 88 | fixed | high | interconnect/collective topology | Communication cost and sharding | REFUSE | Topology inferred from incomplete metadata could price collectives as if measured. | Inferred topology is refused for graph claims; explicit topology provenance is required. |
| 89 | fixed | critical | sparse dtype and default KV pricing | Dominant expert/KV byte terms | REFUSE/FLAG | Unknown expert/activation/KV dtypes silently became bf16/2-byte defaults in direct and live paths. | Final resolved specs use shared dtype priceability; graph nodes retain byte fallback flags and boundaries refuse unpriceable predictions. |
| 90 | fixed | high | `gitm/runtime_driver.py` work-unit extraction | Events/frames throughput numerator | REFUSE | A runner summary omitted the named work counter and `.get(..., 0)` produced a zero/invalid throughput path; the no-kernel report also formatted `None` as a number. | `_work_units` requires a finite positive counter and returns exit 3 with a named failure; no-kernel formatting is explicitly unavailable. |
| 91 | fixed | high | `gitm/importers/nsys.py` enum mapping | Memory endpoints, copy kind, and sync type | WARN | Missing/unknown CUPTI enums were assumed to be device/stream defaults without reaching `ImportStats`, corrupting overlap/topology interpretation. | Non-strict imports retain deduplicated named diagnostics; strict imports still refuse unknown enums. |
| 92 | fixed | high | `scripts/compare_results.py` identity comparison | Reproducibility verification | REFUSE | Two incomplete reports compared equal because `.get` defaults made absent schema/package identity look identical. | Required schema, identity, and exact package versions now mismatch when missing/unavailable. |
| 93 | fixed | medium | `gitm/tracer/_cupti_decode.py` decoder defaults | Kernel launch shape/name and CUPTI enum meaning | WARN | ABI-drifted or malformed records silently became 1x1x1, anonymous, device-copy, or device-sync events. | Decoder warnings name each fallback while preserving safe parsing; tests cover missing shape/name and unknown enums. |

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
| top-level runtime / API / CLI / workloads | Swept — fixed/clean | Swept — traced | Positive timing/work gates, degraded exit status, and workload provenance are covered by ranks 22, 24, 27, 41, 42, 62, 90. |
| agents | Swept — fixed/clean | Swept — traced | Search-domain and GPU-count fallbacks are explicit warnings (ranks 26, 59). |
| bench | Swept — fixed/clean | Swept — traced | Saturation, provenance, timing partitions, and profile artifact completeness are covered by ranks 11, 25, 50, 55, 72, 78, 80. |
| benchmarks | Swept — fixed/clean | Swept — traced | HFT, edge, OpenFold, KITTI, and shared baseline/A-B paths are covered by ranks 18, 24, 27, 42, 51, 53, 71, 73. |
| deploy | Swept — fixed/clean | Swept — traced | Unsupported live attach and attach-window gates are covered by ranks 21 and 64. |
| importers | Swept — fixed/clean | Swept — traced | Rollups, artifact stems, cleanup, launch metadata, parser drops, and NSYS enum diagnostics are covered by ranks 31, 32, 57, 66, 79, 84, 91. |
| kernels | Swept — fixed/clean | Swept — traced | Candidate-library coverage and kernel metadata/decoder fallbacks are covered by ranks 16, 29, 61, 84, 93. |
| optimizer | Swept — fixed/clean | Swept — traced | Attribution, headroom, measurement, gate, apply, and rollback evidence are covered by ranks 15, 19, 20, 38, 50, 51, 65, 81, 82. |
| planner | Swept — fixed/clean | Swept — traced | Hardware, dense/sparse pricing, topology, KV, and direct-builder paths are covered by ranks 3, 4, 7, 9, 13, 39, 43, 49, 60, 67, 68, 77, 85, 88, 89. |
| routing | Swept — fixed/clean | Swept — traced | Bounded routing inputs and unknown tiers refuse (rank 30). |
| safety | Swept — fixed/clean | Swept — traced | Fail-open audit/signal and rollback visibility are covered by ranks 28, 38, 81. |
| scheduler | Swept — fixed/clean | Swept — traced | Dispatch, prediction refusal, residual coverage, scheduler probes, topology, artifacts, and all intervention siblings are covered by ranks 1, 2, 5, 6, 10, 18, 34, 41, 45, 70, 71, 85, 86, 87, 89. |
| serve | Swept — fixed/clean | Swept — traced | Launch/attach gates, token provenance, metrics, model discovery, CUDA parity, and graph artifacts are covered by ranks 12, 21, 33, 35, 40, 46, 47, 52, 53, 83, 89. |
| telemetry | Swept — fixed/clean | Swept — traced | Backend fields, collector/sink failures, sampler probes, and runtime consumers are covered by ranks 17, 35, 40, 63, 70, 75. |
| tracer | Swept — fixed/clean | Swept — traced | Capture, injected traces, vLLM summaries, CUPTI decoding, and drop coverage are covered by ranks 12, 14, 23, 33, 34, 61, 84, 93. |
| scripts | Swept — fixed/clean | Swept — traced | Demo, headroom, report, comparison, and profile outputs are covered by ranks 44, 54, 69, 78, 80, 92. |

## Diagnostic-consumer trace

Every boolean flag, warning list, `estimated`, provenance field, and `has_*` or
`*_fallback` property discovered during the sweep is listed here and traced to a
human- or gate-visible consumer.

| Producer | Diagnostic | Downstream consumer | User/gate boundary | Status |
|---|---|---|---|---|
| `RooflinePrediction` / `Graph` | peak, bytes, hardware fallback; estimated; per-dimension unpriced nodes | loop and attach serializers/diagnostics | JSON + Markdown/CLI | Traced/fixed |
| `RooflinePrediction` / `Graph` | `bytes_are_fallback`, `has_fallback_bytes`, `has_fallback_peaks`, `has_unpriced_nodes` | predicted graph payload, attach payload, scheduler summary | Prediction gate + JSON + Markdown/CLI warning | Traced/fixed |
| `HardwareSpec` / graph context | observed vs pricing SKU and hardware provenance | planner graph, loop refusal, attach warning | Prediction gate + artifact/report | Traced/fixed |
| `Residuals` / `Claim` | raw residual, display cap, saturation, scope | loop residual JSON, scheduler claims, report template | JSON + Markdown diagnostic | Traced/fixed |
| `Residuals` | total/classified/matched launch and kernel-time coverage | residual JSON, run summary, report diagnostics | WARN in report; no clean claim on incomplete coverage | Traced/fixed |
| `ImportStats` / importer rollup | warnings, drops, caveats, SKU/time provenance | analyze summary + customer report | JSON + Markdown | Traced/fixed |
| `ImportStats` / NSYS enum mapper | missing/unknown memory, copy, and sync enums | `stats.warnings`, analyze report | Import WARN and report caveat | Traced/fixed |
| `CaptureResult` / kernel taxonomy | warnings, dropped records, capture status | serve artifacts + CLI | JSON + CLI exit | Traced/fixed |
| CUPTI decoder | missing shape/name and unknown enum warnings | tracer caller/tested capture path | Runtime warning + trace diagnostics | Traced/fixed |
| `ServingSummary` | TTFT/TPOT sample counts and token-provenance warnings | serve/loop artifacts | JSON + CLI/Markdown | Traced/fixed |
| Serving metrics sampler | invalid windows, scrape failures, unavailable fields | `metrics_before/after`, samples, report | WARN/REFUSE at serving boundary | Traced/fixed |
| `Collector` / `GpuHeadroom` | component failures and missing metric-family diagnostics | runtime driver and benchmark artifacts | warning + JSON/Markdown/stdout | Traced/fixed |
| Scheduler sampler | field-probe failures and degraded reads | `scheduler_stats.json`, summary, report | WARN and degraded summary | Traced/fixed |
| Fail-open guard | revert failures, audit-sink failures, signal-handler failures | `failures` attribute, audit log, report diagnostics | Programmatic state + audit artifact | Traced/fixed |
| Benchmark/profile bundle | missing artifacts, command failures, inferred provenance | profile manifest and bench report | Nonzero CLI / WARN for optional artifact | Traced/fixed |
| Runtime driver | no-data, synchronization, telemetry, and work-unit diagnostics | stdout, measure JSON, Markdown report | Exit 3 for degraded/no-data | Traced/fixed |
| Verification/comparison | missing identity/schema/package fields, dirty tree | comparator mismatch output and exit | REFUSE reproducibility claim | Traced/fixed |

## Sibling-path validation matrix

| Capability | Path A | Path B | Guard parity | Status |
|---|---|---|---|---|
| MoE config pricing | attach resolves raw config then validates final dtypes | scheduler recognizes partial sparse configs, validates, and dispatches sparse graph | Shared predicates; path-specific input adapters | Fixed |
| execution lifecycle | launch preflight, live engine, trace capture | attach preflight, target discovery, live metrics | Shared CUDA/trace/timing/metric gates; attach-specific unsupported PID path is explicit | Fixed |
| model family | dense parser and graph builder | MoE sparse builder and expert/KV sizing | Shared dtype/shape priceability; family-specific fields are refused or warned | Fixed |
| serving lifecycle | vLLM workload launch | attach to existing vLLM server | Shared model/config, token provenance, metrics, and artifact writers | Fixed |
| workload timing | HFT event loop | edge/KITTI frame loop | Same positive work/duration predicates and degraded exit convention | Fixed |
| intervention A/B | generic optimizer loop | HFT, edge, OpenFold, and stream-concurrency siblings | Paired finite throughput/timing and trace-evidence gates in every branch | Fixed |
| telemetry | NVML collector/backend | scheduler sampler and Prometheus serving metrics | Field-level diagnostics and invalid-window handling preserved in each consumer | Fixed |
| tracing | in-process capture | injected/CUPTI shard capture | Positive-duration/no-data gate plus malformed-drop diagnostics | Fixed |
| importers | NSYS SQLite importer | torch Chrome trace importer | Normalization, invalid-event drops, warnings, and kept-artifact reporting | Fixed |
| graph hardware | catalogue/declared peak | live detected/observed hardware | Pricing fallback is separate from observed identity; unknown topology refuses | Fixed |
| report verification | report producer | `compare_results.py` consumer | Required schema/identity/package fields are enforced at comparison boundary | Fixed |

## Artifact-consumer trace

| Artifact writer | Artifact | Production reader / boundary | Status |
|---|---|---|---|
| `gitm/scheduler/loop.py` | `predicted_graph.json` | scheduler summary/report, CLI prediction diagnostics, replay/tests | Traced/fixed |
| `gitm/scheduler/loop.py` | `prediction_refusal.json` | measurement-only scheduler result, report, CLI degraded exit | Traced/fixed |
| `gitm/scheduler/loop.py` | `scheduler_stats.json` | run summary/report and scheduler diagnostics | Traced/fixed |
| `gitm/scheduler/loop.py` | `qualification.json` | qualification/report path and run summary | Traced/fixed |
| `gitm/scheduler/loop.py` | `residuals.json` | report template, summary diagnostics, residual consumers | Traced/fixed |
| `gitm/scheduler/loop.py` | `deviations.json`, `deviation_trace.jsonl` | deviation report/measurement and optimizer attribution | Traced/fixed |
| `gitm/scheduler/loop.py` | `ranked_candidates.json` | intervention report and apply/verification paths | Traced/fixed |
| `gitm/scheduler/loop.py` | `verification.json` | optimizer gate and auto-revert evidence | Traced/fixed |
| `gitm/scheduler/loop.py` | `measurement.json` | measurement-only report and caller summary | Traced/fixed |
| `gitm/scheduler/loop.py` | `apply_result.json`, `audit.jsonl` | apply result/report and safety audit reader | Traced/fixed |
| `gitm/serve/attach.py` / `gitm/serve/vllm.py` | `predicted_moe_graph.json` | serving report, attach CLI, replay/analyze artifact directory | Traced/fixed |
| `gitm/serve/artifacts.py` | `preflight.json`, `kernel_breakdown.json`, `serving_summary.json`, `run_manifest.json` | serving report, artifact manifest, and downstream replay/analyze tools | Traced/fixed |
| serving metrics sampler | `metrics_before.txt`, `metrics_after.txt`, `metrics_samples.jsonl` | serving comparison/report and metric diagnostics | Traced/fixed |
| `gitm/runtime_driver.py` | trace JSONL, telemetry JSONL, `*_measure.json`, `*_report.md` | runtime CLI output, report readers, and verification scripts | Traced/fixed |
| `gitm/importers/analyze.py` | kept traces, summary JSON, customer Markdown | CLI return/report consumer and customer artifact | Traced/fixed |
| `gitm/bench/profile.py` / `gitm/bench/cli.py` | profile bundle, GPU CSV, host SAR/flamegraph, manifest | bench results/report and profile completeness gate | Traced/fixed |
| benchmark baseline runners | baseline JSON and spread report | benchmark comparison/sign-off and report | Traced/fixed |
| `gitm/optimizer/verification_export.py` | verification JSON | optimizer gate, comparator, and report | Traced/fixed |
| `gitm/safety/audit.py` | `report.md`, `audit.jsonl` | safety/apply report and audit reader | Traced/fixed |
| telemetry sinks | JSONL/OTLP sink records | configured external collector/sink boundary; no local reader is claimed | Traced — external boundary explicit |

## Completeness pass

No outstanding items.

## Verification and review readiness

- Targeted tests for touched runtime/importer/CUPTI/comparison paths: **73 passed**;
  targeted planner/routing tests after the final alias fix: **75 passed**.
- Full repository run: **1067 passed, 1 skipped**, with only named runtime warnings
  for optional GPU/vLLM/CUDA telemetry and intentional fallback demonstrations.
  Importer fixtures were restored after the run; `git status --short` is clean.
- Ruff passed on all touched modules and tests; `git diff --check` is clean.
- The audit branch is `audit/graceful-fallbacks` and contains only audit fixes plus
  this ledger. The clean expert-signal branch is `feat/expert-signal-eplb` from
  `main` (`ca289d6`), containing the signal module and its tests; caller wiring is
  intentionally the next feature step, not an audit change. The pre-existing
  wiring snapshot remains preserved on `feat/expert-signal-eplb-stacked` (`452b2c8`)
  for the maintainer to port or review.
- Review recommendation: one audit PR is reviewable by theme because each commit
  is a bounded REFUSE/FLAG/WARN fix and this ledger is the summary. Keep the
  expert-signal work as a separate follow-up PR; do not open multiple audit PRs
  without maintainer selection.
