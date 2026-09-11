# Kimi K2.5 / 8x MI355X — the full loop, end to end

The presentation run: **predicted graph → execution → deviations → monitoring
→ interventions**, on nationalcompute (`us-mi355x-gitmachine`), Kimi K2.5
served by vLLM ROCm at TP=8. Experiment numbering follows
`docs/mi355x_experiment_plan.md`; this runbook is the exact command sequence
and where each deliverable lands.

## The pieces

| stage | tool | where it runs |
|---|---|---|
| predicted graph | `scripts/kimi_loop/predict_sweep.py` (planner: `kimi-k2.5` @ MI355X, TP=8) | laptop |
| execution | `deploy/k8s/mi355x-kimi-loop.yaml` + `scripts/kimi_loop/run_loop.sh` (GuideLLM sweeps, BurstGPT replay, GITM tracer arms) | cluster |
| deviations | `gitm deviate` per captured window + `analyze_join.py` | laptop |
| monitoring | amd-smi 1 Hz (in-pod, always on) + 1 Hz `/metrics` scrapes + deviation invariants (`gitm.optimizer.monitor`) | pod + laptop |
| interventions | `run_loop.sh e8` with `INTERVENTION='--kv-cache-dtype fp8'` (or another lever), before/after measured | cluster |

Arms per config — **A** clean (no tool), **B** GITM-traced, **C** GITM +
rocTX correlation (`GITM_TRACE_NVTX=1` + vLLM `--enable-layerwise-nvtx-tracing`;
"NVTX" on AMD *is* rocTX — `torch.cuda.nvtx` maps to `roctxRangePushA`, and the
ranges reach the same injected tool, no second env var). Arm switches go
through `arm.sh`, which restarts the server **in-pod** — never a pod rollout,
because on this cluster a new pod is a new marketplace ProvisioningRequest.

## 0. Standing setup (already done, listed for reproducibility)

```bash
kubectl config use-context us-mi355x-gitmachine
# gitm wheel + loop scripts on the shared FS (re-run after local changes):
python3 -m build --wheel
kubectl cp dist/gitm_labs-*.whl default/rex-stage:/mnt/shared/gitm/wheel/
tar -C scripts/kimi_loop -cf - . | kubectl exec -i rex-stage -- tar -C /mnt/shared/gitm/scripts -xf -
kubectl apply -f deploy/k8s/mi355x-kimi-loop.yaml
kubectl get pods -l app=kimi-loop -w     # SchedulingGated -> market -> Running 2/2
```

Weights: already on `/mnt/shared/hf-cache` (594.9 GB). Image: the digest rex
served Kimi with — vLLM + rocprofiler-sdk in one image, so the tracer tool's
ABI matches the runtime by construction.

## 1. Predicted graph (laptop, no GPU needed)

```bash
python3 scripts/kimi_loop/predict_sweep.py
```

→ `evidence/kimi-mi355x/predicted/`: 18 (config × concurrency) points with
ITL floor, TTFT floor, throughput floor, plus per-op decode tables. Headline
prediction: decode floor 11.0 ms/step at rag c=64 (`moe_routed` 86%,
memory-bound) — the number the measured sweep gets compared against.
Footprint validation is in the catalogue entry itself: predicted 595.00 GB vs
the published checkpoint's 595.15 GB (−0.03%).

## 2. Execution (inside the gitm sidecar)

```bash
kubectl exec -it deploy/kimi-k25-loop -c gitm -- bash
export GITM_RUN=$(date -u +%Y%m%d-%H%M%S)   # one id for all phases
bash /mnt/shared/gitm/scripts/run_loop.sh e0     # gate: do not proceed on FAIL
bash /mnt/shared/gitm/scripts/run_loop.sh e1
bash /mnt/shared/gitm/scripts/run_loop.sh e2
bash /mnt/shared/gitm/scripts/run_loop.sh e3e4
bash /mnt/shared/gitm/scripts/run_loop.sh burst
```

* **e0** — arm C, 8 drive requests, hard gates: kernels > 0, names resolve,
  zero `dropped_records`; warns if `range_op` is empty (then check
  `grep layerwise /scratch/logs/server_armC.log` before e3e4).
* **e1** — overhead: A/B/C × 3 reps of the fixed load, tracer *collecting*
  during B/C (a dormant tool would measure A twice). Decides the pilot's
  traced-vs-sampled question. H200 reference: 1.72x / 3.46x.
* **e2** — the sweep: {chat 1024/256, rag 4096/512, long 8192/1024} ×
  c ∈ {1,4,16,64,128,256}, arm A, GuideLLM concurrent profile, 240 s per
  point, 1 Hz `/metrics` (KV occupancy, preemptions) per point. ~75 min of
  load plus arm switches. TTFT/ITL come from the GuideLLM JSONs.
* **e3e4** — the layer-by-layer deliverable: arm C, saturated decode
  (c=256), 120 s window after the prefill wave passes. Every kernel in the
  window carries `range_op`/`range_layer` from the rocTX chain.
* **burst** — BurstGPT (real Azure OpenAI arrival process) through
  `gitm.traffic`: fit, timed replay via `vllm bench serve`, arms A and B.

Everything lands in `/mnt/shared/gitm/results/$GITM_RUN/` with a MANIFEST
(arm, ROCm version, vLLM version, env per phase — no anonymous numbers).

## 3. Deviations + reports (laptop)

```bash
scripts/kimi_loop/analyze.sh <run-id>
# then the E7 deviate commands it prints, e.g.:
python -m gitm.cli deviate <trace.jsonl> --model kimi-k2.5 --gpu MI355X \
    --tp 8 --batch 64 --kv-len 4352 --json > .../deviation_rag_c64.json
```

→ `evidence/kimi-mi355x/runs/<run-id>/analysis/`:

* `overhead_table.md` — E1, the measured MI355X tracer cost
* `measured_vs_predicted.md` — E2 joined to the predicted floors; the
  residual column is the slide
* `layer_kernels.md` — per-layer, per-op kernel table with real names
* `taxonomy.md` — kernel-time by class; unclassified names = task cards
* `deviation_*.json` — predicted-graph subtraction per window; does MI355X
  decompose like the H200 (1.62x on-device, rest eager host dispatch)?

## 4. Intervention (closing the loop)

Pick the lever the deviation names. The expected one, from the H200 finding
("scheduling knobs are noise, structural is the lever"): the KV cache. MLA's
latent is the decode-attention traffic term, and fp8 halves it.

```bash
kubectl exec -it deploy/kimi-k25-loop -c gitm -- \
  env INTERVENTION='--kv-cache-dtype fp8' GITM_RUN=<same-run-id> \
  bash /mnt/shared/gitm/scripts/run_loop.sh e8
scripts/kimi_loop/analyze.sh <run-id>       # renders e8_delta.md
```

The predicted side of the same lever: `load_spec("kimi-k2.5")` with
`kv_dtype="fp8"` reprices `attn_score_value` — predicted delta next to
measured delta is the loop closed on one slide.

## Failure modes worth knowing at the podium

* Pod Pending "waiting for the market" → capacity, not config (the taint
  toleration is in the manifest). `kubectl get provisioningrequests`.
* Empty capture window with a busy server → clock domain; E0 catches it. The
  dlopen deps for `clock_now()` are baked into the image check
  (`hsa-amd-aqlprofile`, `libdw1t64` — the silent `timestamp() -> None` bug).
* `range_op` null everywhere on arm C → either vLLM build lacks
  `--enable-layerwise-nvtx-tracing` (check server log) or rocTX isn't
  reaching the marker service; E0 warns, E4 documents-or-skips per the plan.
* Arm flip hangs > 60 min → look at `/scratch/logs/server_arm*.log`; the
  supervisor relaunches on exit, so a crash-looping arm shows repeated
  banner lines with the same arm name.
* GuideLLM at c=256 on Kimi: ~53 s/request first wave — never use an
  over-saturation constraint (it reads the ramp as overload; rex RUNLOG §25).
