# vLLM-decode structural experiment — runbook

Start-to-end procedure for the fp8-KV structural result on a rented H100. Every
step here is something we actually hit; the troubleshooting table at the bottom is
not hypothetical.

The one-line summary of what makes this work: the tracer sees vLLM's **child**
`EngineCore` process via the CUDA driver's injection hook, the install stack is
**pinned to the host driver's CUDA major**, and the run **oversubscribes the KV
cache** so fp8 has something to relieve.

---

## 0. Rent the pod

- **GPU:** H100 80GB.
- **CUDA:** either 12.x or 13.x works — `gpu_setup.sh` installs the matching stack.
  If the deploy screen lets you filter CUDA version, pin it so you don't get
  rescheduled mid-experiment (it happens).
- The NVIDIA driver belongs to the **host**, not the container. You cannot change
  it from inside. A pod can come back on a different host with a different driver
  after a restart — re-run `gpu_setup.sh` if that happens.

## 1. Connect — direct TCP, not the proxy

The RunPod SSH **proxy** (`ssh.runpod.io`) times out intermittently and does not
support file transfer. Use the **direct TCP** endpoint from the dashboard:

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

## 2. Ship the code (from your Mac)

```bash
scripts/ship_to_pod.sh root@<ip> <port>
```

Ships the working tree (uncommitted changes included) as a tarball over SSH. It
**wipes and replaces** `/workspace/runtime`, and excludes `*.so` — so you rebuild
the native bits on the pod (next step). `/workspace` persists across container
resets; pip-installed packages do **not**.

## 3. Set up the pod

```bash
cd /workspace/runtime
bash scripts/gpu_setup.sh
```

This reads the host driver's CUDA major and installs the **pinned stack** for it,
then builds both CUPTI targets (`_cupti_shim` + `libgitm_inject.so`):

| Driver CUDA | vLLM   | torch          |
|-------------|--------|----------------|
| 13.x        | 0.25.1 | 2.11.0 + cu130 |
| 12.x        | 0.19.1 | 2.10.0 + cu128 |

**The two rows are different engines** (scheduler, kernels, defaults). Every vLLM
≥ 0.20.0 is a CUDA 13 build, so a CUDA 12 host can only run 0.19.1. Record the vLLM
version with any result; do not compare numbers across the two.

Confirm the stack is compatible with the driver (2 seconds, no GPU work):

```bash
python -m gitm.cuda_env      # must print "OK — every component runs on this driver"
```

If the container was reset (new container id, pip installs gone), re-run
`gpu_setup.sh`. If only torch was reinstalled, at minimum rebuild CUPTI:

```bash
python -m gitm.tracer._cupti.build   # relinks against the driver-matched libcupti
```

## 4. Hugging Face access

`meta-llama/*` is gated. Either use the ungated mirror (identical weights) —

```bash
export GITM_VLLM_MODEL=NousResearch/Meta-Llama-3-8B
```

— or authenticate for the official repo:

```bash
read -rs HF_TOKEN && export HF_TOKEN         # keeps it out of shell history
export GITM_VLLM_MODEL=meta-llama/Meta-Llama-3-8B
hf auth whoami                               # confirm the account is granted access
```

## 5. Environment — the scripts set it for you

You do **not** export anything by hand. Both run scripts call
`set_decode_run_defaults()` in-process (before CUDA init), which sets the injection
path, trace output, model, and the KV-pressure knobs. It uses `setdefault`, so any
value you *did* export still wins.

The defaults it sets, and why each matters:

| Var | Default | Why |
|---|---|---|
| `CUDA_INJECTION64_PATH` | the built `libgitm_inject.so` | driver injects the collector into the child EngineCore |
| `GITM_TRACE_OUT` | `/root/.cache/gitm/traces/vllm.jsonl` | where per-pid shards land |
| `GITM_VLLM_MODEL` | `NousResearch/Meta-Llama-3-8B` | ungated Llama-3-8B mirror |
| `GITM_VLLM_GPU_MEM` | `0.45` | baseline + restart candidate both fit at once |
| `GITM_VLLM_PROMPTS` | `512` | with MAX_TOKENS, ~8x oversubscribes the ~151k KV cache |
| `GITM_VLLM_MAX_TOKENS` | `2048` | without this pressure, fp8 measures pure noise |

To override one (e.g. the official gated model, with a granted `HF_TOKEN` from
step 4), just export it before running — the rest still fill in:

```bash
export GITM_VLLM_MODEL=meta-llama/Meta-Llama-3-8B
```

Do **not** set `VLLM_ENABLE_V1_MULTIPROCESSING=0`. It folds `EngineCore` back onto
the frontend's GIL and injects idle gaps into the exact stall/idle signal you are
measuring. The injection tracer exists precisely so you don't have to.

## 6a. The direct fp8 answer (fast, ~5 min)

The single question — does fp8 KV raise decode throughput under pressure — with no
scheduler candidates, so nothing in vLLM's scheduler can deadlock the run:

```bash
python scripts/fp8_ab.py
```

Run it as a **file**, never `python -c` or a stdin heredoc: the fp8 restart may
build its second engine under `spawn`, which re-imports `__main__`, which must be a
real importable file guarded by `if __name__ == "__main__":`. (The code also keeps
the parent CUDA-free so `fork` is used instead — belt and suspenders.)

Output — the percentage on the last line is the result:

```
run env: {'GITM_VLLM_MODEL': 'NousResearch/Meta-Llama-3-8B', 'GITM_VLLM_GPU_MEM': '0.45', ...}
baseline bf16 KV: 8,492 tok/s
fp8 KV:           X,XXX tok/s   (+X.X%)
```

## 6b. The full optimizer loop (slower, produces a provenance report)

```bash
python scripts/run_vllm_optimize.py     # a spawn-safe wrapper around optimize()
# report path prints at the end; cat it for the claims table
```

Known issue: a scheduler-knob candidate can deadlock vLLM's scheduler under heavy
oversubscription (GPU drops to 0% util, `generate()` never returns). The A/B probe
has no wall-clock timeout yet, so the loop hangs. Until that lands, prefer 6a, or
watch the run and kill+restart if a decode pass stops advancing (see below).

## 7. Sanity-check the trace (optional)

```bash
python - <<'EOF'   # read-only; fine from stdin
from gitm.tracer import injection
print("shards:", [p.name for p in injection.shard_paths()])   # expect >=2 pids
EOF
```

Two shards with different pids = parent + `EngineCore` both traced. Kernel names
are FlashAttention 3 (`FlashAttnFwd`) and `reshape_and_cache_flash`, **not**
`paged_attention` — the library's `applies_to_kernels` are keyed to the real names.

---

## Troubleshooting — everything we actually hit

| Symptom | Cause | Fix |
|---|---|---|
| `ssh: connect to host 100.65.x.x ... timed out` | RunPod SSH proxy overlay down | Use the direct TCP endpoint (`ssh root@<ip> -p <port>`) |
| `The NVIDIA driver on your system is too old (found 12080)` | torch/vLLM built for a newer CUDA major than the host driver | `python -m gitm.cuda_env` for the exact fix; `gpu_setup.sh` installs the pinned stack |
| `libcudart.so.13 / libnvrtc.so.13: cannot open shared object file` | vLLM/torch is a CUDA 13 build on a CUDA 12.8 driver | Install the CUDA 12 row (vLLM 0.19.1); the driver is host-owned and can't move |
| `GatedRepoError: 403 ... meta-llama/Meta-Llama-3-8B` | gated HF repo | `GITM_VLLM_MODEL=NousResearch/Meta-Llama-3-8B`, or set a granted `HF_TOKEN` |
| Trace comes back empty; `shards: []` | in-process shim can't see the child engine | Use injection (`CUDA_INJECTION64_PATH`); never disable V1 multiprocessing |
| `shards: []` even under injection | `clear_shards` unlinked a live process's shard | Fixed — `clear_stale_shards` only reaps dead pids |
| `FileNotFoundError: '<stdin>'` during fp8 restart | second engine built under `spawn`, `__main__` not importable | Run from a script file with a `__main__` guard; the parent-CUDA-free fix uses fork |
| fp8 measures ~1% (noise) | KV cache not saturated | Oversubscribe: `GITM_VLLM_PROMPTS=512 GITM_VLLM_MAX_TOKENS=2048` |
| Decode pass frozen, GPU util 0%, one engine alive | vLLM scheduler deadlock under oversubscription (no A/B timeout yet) | `kill -9 <EngineCore pid>`, confirm VRAM frees, re-run 6a; timeout fix pending |
| fp8 candidate never appears in the report | old `applies_to_kernels` named kernels that don't exist → scored 0.0, truncated | Fixed — names re-pointed at the real trace; `top_n_interventions=12` |
| Container id changed, imports fail | pod rescheduled to a fresh container; pip installs gone (`/workspace` survived) | Re-run `gpu_setup.sh` |

## What to record with any result

- vLLM version (0.19.1 on CUDA 12 vs 0.25.1 on CUDA 13 — **not comparable**)
- driver / CUDA version, GPU SKU
- `GITM_VLLM_GPU_MEM`, `GITM_VLLM_PROMPTS`, `GITM_VLLM_MAX_TOKENS`
- KV cache size the engine reported, and the resulting oversubscription ratio
