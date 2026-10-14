# DeepSeek V3.2 H200 experiment plan — for hardware/deployment review

**Status (October 13, 2026): proposed; no GPU run or qualified trace yet.** PR [#132](https://github.com/GitM-Labs/runtime/pull/132) currently bundles the static work. Separate checkpoint, planner and contract branches have been prepared for smaller reviews. This plan is the scope of a later run and results PR.

## Purpose and hardware known so far

Test whether GitM's predicted memory, prefill, decode, expert routing and collectives describe **one actual DeepSeek deployment** on H200s. The hardware owner supplied an `nvidia-smi` snapshot showing **eight H200s on one machine**, each reporting **143,771 MiB** of memory, with driver **595.91.07** and no running GPU processes at that moment. This does not identify the second node, GPU interconnect, checkpoint, engine, serving flags or the run author's access.

The PR's public checkpoint audit gives **689,471,107,200 stored tensor bytes**. This is a static input, not a measured GPU footprint or a promise the model fits with the chosen TP/EP/DP shape.

## Inputs to confirm before reserving the run

| Input | Owner to confirm | Why it matters |
| --- | --- | --- |
| Actual served checkpoint path/repository and immutable revision; does it match the public pin in PR #132? | the serving owner, with the evidence verifier | A different checkpoint invalidates the current graph and byte ledger. |
| Engine source revision, container image digest, full launch command/environment, KV dtype, MLA and expert backend settings | the serving owner / the hardware owner | Determines which operations and memory layout execute. |
| Node access, `nvidia-smi topo -m`, number of nodes, rank map, TP/EP/DP and reservation process | the hardware owner, with the experiment contract | Determines per-rank memory and collectives; eight visible GPUs do not prove the full topology. |
| Approved traffic source and manifest, replay format and token totals | the serving owner / the author | A synthetic smoke test cannot stand in for engagement traffic. |
| Collector command, expected ranks, capture rules and qualification gate | the qualification owner | A trace must cover the whole declared deployment before comparison. |

## Proposed test sequence

| Step | Test | Record and decision |
| --- | --- | --- |
| 0. Preflight, before GPU time | Run the checkpoint verifier on the **actual** config/index; validate the deployment contract; save the graph and ceiling hashes. Compare checkpoint, engine and launch identities to the declared contract. | Stop the performance claim if the identities disagree or a required field remains `UNVERIFIED`. Keep the exact missing field and owner. |
| 1. Bring-up and smoke | Launch the approved engine on the agreed node/ranks. Record `nvidia-smi topo -m`, GPU inventory, image digest, command/env, per-rank memory after loading, and one short request. | Confirm the model loads, returns a response and stays within memory. Keep load/OOM errors; do not silently change the model or parallelism. |
| 2. Controlled shapes | After warm-up, run small fixed cases isolating prefill (for example 128 input / 1 output tokens), decode (128 / 128), longer-context attention (4096 / 64), and concurrent requests (eight at 128 / 128). Use actual tokenizer counts; repeat each successful case three times. | Record request and token totals, latency distribution, per-rank peak memory and GPU telemetry. These are **synthetic diagnostic cases**, not customer traffic. The hardware owner can adjust sizes to match capacity and engine limits before the run. |
| 3. Representative replay | If an approved traffic manifest exists, version its hash/source kind and replay command. Use GitM's traffic adapter and vLLM `timed_trace` when that is the chosen engine; otherwise agree on an equivalent replay that preserves arrivals, lengths and prefix identity. Repeat a warmed run three times. | Compare emitted versus completed requests, input/output tokens, timing and any dropped rows. Keep this result separate from the diagnostic cases. |
| 4. Qualified trace | Capture kernels, transfers and collectives for **every expected rank** with the trace collector. Exercise mismatch/missing-rank rejection using fixtures or a safe dry run. | the qualification owner decides whether the actual run qualifies. GitM's offline attribution checker alone does not qualify it. |
| 5. Reconcile and fix | Map each material planner region to observed work. For each, report predicted lower/central/upper time, observed time and variance, predicted/observed traffic where available, and residual. | Label model error, engine divergence, measurement uncertainty or unidentified work; retain unmatched work. Fix the largest supported contradiction in the production path and add a regression test. |

## Incremental review order

- **Part 1, checkpoint audit:** pinned public checkpoint evidence, exact header-byte ledger, fail-closed static verifier and tests. The catalogue refuses predictions until Part 2 is merged.
- **Part 2, planner integration:** MTP/storage accounting, live serving route, graph and negative tests. Its output remains exploratory without an engine/deployment pin.
- **Part 3, workload contract:** gap map, onboarding, contract validator and generated coverage, with absent deployment inputs marked `UNVERIFIED`.
- **Part 4, H200 results PR:** confirmed deployment contract with private artifacts referenced by hash/location, exact commands and versions, test/replay manifest, the qualification owner's qualification result, predicted-versus-observed table, and any justified code fix. If access or inputs block a stage, record the missing artifact, owner and affected claim; do not report a GPU measurement that did not happen.

**Decision requested (hardware/deployment):** Is this one-node, eight-H200 starting scope appropriate? Which checkpoint and engine launch should we use, how should the run author access the node, and who provides the approved workload and trace collector?
