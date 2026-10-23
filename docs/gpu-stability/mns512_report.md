# `MAX_NUM_SEQS=512` GPU wedge on alpha01 — full report

**Host:** alpha01, 8× MI300X (gfx942)
**Image:** `4599ea638415` (vLLM `v0.30.0`)
**Model:** Kimi-K2.6 · **Client:** InferenceX `d2819bd1`, TP8, 8k/1k, conc 4/8/16/32/64, `--ignore-eos`
**Scope:** startup wedges tied to `max_num_seqs=512` (mns512), 2026-10-08 → 2026-10-13 sweep window
**Times:** host time is PDT; vLLM server logs are UTC (+7 h).

Companion files: `bug_analysis.md` (master writeup), `logs/README.md` (evidence index), `logs/01_*`, `logs/02_*`, `logs/03_*`.

---

## 0. Environment / exact versions

### vLLM + ROCm container (image `sha256:4599ea63…`, built 2026-09-21)

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| ROCm | 7.2.3 (HIP 7.2.53211) |
| vLLM | 0.30.0 |
| PyTorch | 2.12.0+git6bbd260 (HIP 7.2.53211) |
| Python | 3.12.13 |
| AITER | amd-aiter 0.1.21.post2 |
| Triton | 3.7.1+gitf0b55c07 |
| Transformers | 5.17.0 |
| Docker | 29.1.3 (client and server) |

### Host (alpha01)

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+ |
| amdgpu driver | 6.16.13 |
| GPUs | 8× MI300X (gfx942) |

**ROCm split:** the host has **no ROCm userspace version file** — all ROCm userspace (7.2.3 / HIP) comes from the container; the host supplies only the **kernel-mode amdgpu driver (6.16.13)**. So the KIQ-fence / page-fault faults sit at the boundary between container userspace (ROCm 7.2.3, AITER 0.1.21.post2) and the host kernel driver (amdgpu 6.16.13) — both are in-scope variables for the wedge, and neither is pinned by anything the container alone controls.

---

## 1. Summary

Three startups have wedged a GPU on **three different physical cards** (7 → 2 → 1) during the sweep. The only common denominator is the config knob `MAX_NUM_SEQS=512`. The surface symptom differs by *which startup phase* the wedge hits, but all three end the same way: a GPU stops responding and the run dies.

| # | Date | GPU | Run / config | Phase when it died | Kernel signature |
|---|------|-----|--------------|--------------------|------------------|
| 01 | Oct 9  | **7** | `a2-eagle3-aiter-mns512` — AITER MLA, EAGLE3×3 | Early init: quick-reduce/all-reduce → NCCL all-gather timeout | `timeout waiting for kiq fence` → NCCL watchdog |
| 02 | Oct 12 | **2** | `a1-ref-mns512` — Triton MLA, no DCP, no EAGLE3 | CUDA-graph capture / post-capture memory profiling | `retry page fault`, `PERMISSION_FAULTS: 0xb`, SQC, TLB flush fail |
| 03 | Oct 13 | **1** | `a5-dcp8-mns512` — Triton MLA, **DCP=8**, gpu-mem 0.94 | Quick-reduce / DCP communicator init (~2 min in) | `timeout waiting for kiq fence` on `0000:0a:00.0` |

**mns512 is not deterministic.** The same knob also *succeeded*:

| Run | When | Notes |
|-----|------|-------|
| a2-eagle3-aiter-mns512 try 3 | Oct 10, 18:02–19:14 | Identical server config to Oct 9 failures; only new pre-launch guards. 0 faults. |
| a1-ref-mns512 | Oct 13, 05:00–06:29 | Full 5-point bench. Immediately followed by the a5 KIQ fault on GPU 1. |

So mns512 raises the *probability* of the wedge; it does not guarantee it.

---

## 2. Exact errors in the logs (verbatim)

### 2.1 — 01 · Oct 9 · GPU 7 (`a2-eagle3-aiter-mns512`)

Kernel (`logs/01_*/dmesg_gpu7_bursts.txt`) — first ever amdgpu error on this card, ~520 lines in 10 s from 16:13:12:
```
amdgpu 0000:2f:00.0: amdgpu: timeout waiting for kiq fence
amdgpu: TLB flush failed for PASID 32799            (16:13:25)
amdgpu: TLB flush failed for PASID 32791            (×173 at 16:35:28, during teardown)
amdgpu: failed to write reg 2894 wait reg 28a6      (and 12894/18bd4/38bd4/58bd4/78bd4 variants)
```

Server (`logs/01_*/server_try1.log`) — all-gather never completes; NCCL watchdog fires at 600 s:
```
[E1009 23:34:17 ProcessGroupNCCL.cpp:757] [Rank 0] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=1, OpType=_ALLGATHER_BASE, NumelIn=19468800, NumelOut=155750400,
  Timeout(ms)=600000) ran for 600009 milliseconds before timing out.
[E1009 23:34:17 ProcessGroupNCCL.cpp:2395] failure detected by watchdog at work sequence id: 1
  PG status: last enqueued work: 1, last completed work: -1
[E1009 23:35:17 ProcessGroupNCCL.cpp:832] To avoid data inconsistency, we are taking the entire process down.
```
ROCm runtime fault on the wedged queue:
```
:0:rocdevice.cpp :3854: Callback: Queue 0x7f99a1a00000 aborting with error :
  HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION: The agent attempted to execute an illegal
  shader instruction. code: 0x2a
```
Backend in play right before the fault: `Using ['QUICK_REDUCE', 'AITER_CUSTOM', 'PYNCCL'] all-reduce backends ... for group 'tp:0'`.

### 2.2 — 02 · Oct 12 · GPU 2 (`a1-ref-mns512`)

Kernel (`logs/02_*/dmesg.txt`) — the permission page fault, full block:
```
amdgpu: [gfxhub0] retry page fault (src_id:0 ring:0 vmid:3 pasid:32794)
amdgpu:  Process VLLM::Worker pid 3448334 thread VLLM::Worker pid 3448334
amdgpu:   in page starting at address 0x0000f356131c8000 from IH client 0x1b (UTCL2)
amdgpu: VM_L2_PROTECTION_FAULT_STATUS:0x003012B1
amdgpu: 	 Faulty UTCL2 client ID: SQC (inst) (0x9)
amdgpu: 	 MORE_FAULTS: 0x1
amdgpu: 	 WALKER_ERROR: 0x0
amdgpu: 	 PERMISSION_FAULTS: 0xb
amdgpu: 	 MAPPING_ERROR: 0x0
amdgpu: 	 RW: 0x0
amdgpu: TLB flush failed for PASID 32794.           (×286 afterward)
```

Server (`logs/02_*/server.log`) — repeating shm_broadcast stalls, then worker death:
```
INFO 10-12 15:04:10 [shm_broadcast.py:801] No available shared memory broadcast block found in
  60 seconds. This typically happens when some processes are hanging or doing some time-consuming
  work (e.g. compilation, weight/kv cache quantization).
  (repeats every 60 s: 15:04 → 15:10)
ERROR 10-12 15:11:38 [multiproc_executor.py:314] Worker proc VllmWorker-2 died unexpectedly
  (exit code: None), shutting down executor.
ERROR 10-12 15:11:47 [core.py:1366] EngineCore failed to start.
ERROR 10-12 15:11:47 [core.py:1366] RuntimeError: cancelled
```
Died inside `_initialize_kv_caches` → `compile_or_warm_up_model` → `collective_rpc`. Capture list confirms the big buckets: `cudagraph_capture_sizes: [1,2,...,256,272,288,...,512]`, `max_cudagraph_capture_size: 512`.

> Continuous telemetry for Oct 12 is **not** in this workspace (`TELEMETRY_MISSING.txt`; csv ends Oct 10). Kernel evidence is complete.

### 2.3 — 03 · Oct 13 · GPU 1 (`a5-dcp8-mns512`)

Kernel (`logs/03_*/dmesg.txt`) — same KIQ signature as Oct 9, on PCI `0000:0a:00.0`, 520 lines from 06:32:13:
```
[Tue Oct 13 06:32:13 2026] amdgpu 0000:0a:00.0: amdgpu: timeout waiting for kiq fence   (×520)
amdgpu: failed to write reg 2894 wait reg 28a6      (18bd4/38bd4/58bd4/78bd4/12894 variants)
```

Sweep fault guard (`logs/03_*/sweep.log`) — caught the wedge in ~4 s and stopped the run:
```
=== 2026-10-13 06:30:05 a5-dcp8-mns512: QR=INT4 ATTN=TRITON_MLA DCP=8 GPU_MEM=0.94 MAX_NUM_SEQS=512
!!! 2026-10-13 06:32:17 new GPU fault lines in dmesg (0 -> 215), stopping:
[Tue Oct 13 06:32:17 2026] amdgpu 0000:0a:00.0: amdgpu: timeout waiting for kiq fence
!!! a5-dcp8-mns512 stopped by the GPU fault guard during load
```
Died during DCP communicator init — server log shows `Using ['PYNCCL'] all-reduce backends ... for group 'dcp:0'` at the fault, before weight load or capture.

Post-kill process state (`logs/03_*/stuck_procs.txt`, `stuck_worker_stack.txt`) — one worker stuck in uninterruptible `D` state, the rest zombies:
```
1967059 D   [VLLM::Worker_TP]      ← stuck, kernel stack:
  [<0>] __synchronize_srcu.part.0+0x87/0xb0
  [<0>] synchronize_srcu+0x110/0x130
  [<0>] __mmu_notifier_release+0xd0/0x1f0
  [<0>] exit_mmap+0x36c/0x390
  [<0>] do_exit+0x305/0xad0
1967060-66 Zl [VLLM::Worker_TP] <defunct>    ← zombies
```
Process exit blocked tearing down GPU memory mappings on the wedged device.

### 2.4 — Error classes consolidated

| Class | Exact string | Where | Meaning |
|---|---|---|---|
| KIQ firmware hang | `timeout waiting for kiq fence` | 01, 03 | GPU didn't complete a driver command (map/unmap/fence) during communicator init |
| Permission page fault | `PERMISSION_FAULTS: 0xb` + `Faulty UTCL2 client ID: SQC` + `retry page fault` | 02 | Shader core hit an address the page table rejects (`MAPPING_ERROR: 0x0` → permission, not missing mapping) |
| TLB flush failure | `TLB flush failed for PASID <n>` | all | Driver can't invalidate TLB on the dead device (follows the wedge) |
| Register write failure | `failed to write reg <x> wait reg <y>` | 01, 03 | Driver can't reach the GPU during teardown |
| NCCL collective timeout | `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=_ALLGATHER_BASE ...) ran for 600009 ms` | 01 | Healthy ranks wait 600 s on the dead rank, then abort the process |
| ROCm illegal instruction | `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION ... code: 0x2a` | 01 | Runtime-level fault on the wedged queue |
| shm_broadcast stall | `No available shared memory broadcast block found in 60 seconds` | 02 | **Symptom** — EngineCore waiting on a hung worker's CPU IPC, not root cause |
| Worker death / engine abort | `Worker proc VllmWorker-2 died unexpectedly` → `EngineCore failed to start` → `RuntimeError: cancelled` | 02 | vLLM tears down after the worker's GPU wedges |
| Stuck teardown | `D`-state worker in `synchronize_srcu → __mmu_notifier_release → exit_mmap` | 03 | Process can't exit; blocked unmapping memory on a dead GPU |

---

## 3. Mechanism

By default vLLM couples cudagraph capture size to `max_num_seqs`. Setting it to 512 makes startup record capture buckets **272…512** (confirmed verbatim: `cudagraph_capture_sizes: [...256, 272, 288, ..., 512]`, `max_cudagraph_capture_size: 512`) that mns256 runs never touch. Custom AMD kernels (aiter MLA / fused quant / rope, and the communicator-setup paths) run at batch shapes they are otherwise never exercised at → an illegal GPU access or a firmware-queue stall → the device stops responding. Everything else is cascade:

```
MAX_NUM_SEQS=512
  → cudagraph_capture_sizes / max_cudagraph_capture_size include 272…512
  → custom kernels run at batch sizes mns=256 never exercises
  → illegal GPU memory access and/or driver/firmware wedge
  → worker hangs (retry page fault and/or KIQ fence timeout)
  → shm_broadcast stalls (engine waiting on workers)
  → and/or NCCL collective timeout if the hang is inside a collective
  → and/or dmesg page fault / kiq fence depending on phase
```

### Two faces, same fault family

- **KIQ face** (Oct 9, Oct 13): `timeout waiting for kiq fence` during quick-reduce / AITER custom all-reduce / DCP `PYNCCL` init — **before** graph capture.
- **Page-fault face** (Oct 12): `PERMISSION_FAULTS: 0xb` + SQC — **during/after** CUDA-graph capture.

The KIQ face strikes *before* capture runs, so the "272–512 bucket" explanation does not directly cover it — this is the one ambiguity the diagnosis has not yet closed (see §5).

### `max_num_seqs` vs `max_cudagraph_capture_size`

| Knob | Role | On overflow |
|---|---|---|
| `max_num_seqs` | Scheduler capacity: how many sequences may be admitted/kept (KV) | Queue / reject new requests |
| `max_cudagraph_capture_size` | Largest batch size for which a CUDA graph was recorded | Step still runs, but eager (no graph) |

They are coupled by default, not a correctness identity — which is what makes the workaround in §4 possible.

---

## 4. Final diagnosis

**A software/driver path that `MAX_NUM_SEQS=512` exercises more often — not a bad card — intermittently wedges a GPU during vLLM startup.** Config-triggered and probabilistic, not deterministic, and not a single faulty GPU.

What the evidence forces:
- **Three wedges, three physical GPUs, one shared knob** → kills "one bad GPU" and "pure hardware" as the primary story.
- **The same config also succeeds** (Oct 10 try 3; Oct 13 a1-ref) → mns512 raises probability, doesn't guarantee the wedge.

Confounders the analysis accounts for and discounts:
- Oct 9 had a **SIGKILLed** prior container 34 s before launch (first KIQ error 1 s after peer-memory-mapping all-reduce setup). On its own this looked like "GPU 7 hardware" or "dirty teardown"; the **Oct 12/13 recurrences on other cards** overturned both and shifted the verdict to the mns512 software path.
- 2 corrected PCIe errors on GPU 7's upstream port hours earlier (08:13, 13:56); ECC clean; **not** implicated.

**Status: strongly evidenced but not formally closed.** No logged run yet applies the decouple workaround and proves the wedge disappears, and the KIQ face is not yet proven to share a root cause with the page-fault face.

### Mitigation in effect (customer guardrail, 2026-10-15)

**The customer has asked that nothing run at `max_num_seqs=512`. Operating guardrail: cap at `max_num_seqs=256`.** So the operational mitigation right now is simply *stay at 256* plus clean teardown (see §5). mns256 arms have run clean in every logged case.

The root-cause confirmation path below stays documented but is **blocked** until the customer lifts the 512 restriction, because every variant requires setting `max_num_seqs` above 256:

```bash
# BLOCKED — needs max_num_seqs > 256, not permitted right now
vllm serve ... \
  --max-num-seqs 512 \
  --compilation-config '{"max_cudagraph_capture_size": 256}'
```
Idea: hold up to 512 live sequences; only graph-accelerate batches ≤ 256; larger steps run eager; startup never records 272…512. Bisect with `--enforce-eager` (skips capture entirely). Parked for when 512 is allowed again.

---

## 5. Testing plan under the mns256 guardrail

**Constraints (customer review, 2026-10-15):** do not run anything at `max_num_seqs=512`; cap at **`max_num_seqs=256`**; the reviewer's primary concern is the **Docker SIGKILL teardown** — processes should be allowed to exit on their own.

Consequence: we **cannot confirm the mns512 root cause right now** (that needed runs above 256 — see the blocked mitigation in §4). The work pivots to what the Oct 9 timeline already flagged as the confounder — **teardown hygiene** — and to proving mns256 is a safe guardrail. Because the fault is intermittent, each arm is repeated and reported as **wedges / N**.

### Ground rules
- **N ≈ 20 launches per arm** (startup-only; the wedge hits during load — no full bench needed). N = 40 for a tighter bound. 0/20 gives a ~15 % 95 %-upper-bound.
- Every launch starts from a **verified-clean state**: all GPUs `gpu_busy_percent = 0`, dmesg quiet. Never relaunch onto a wedged card (the Oct 9 try-2 mistake).
- All arms at **`max_num_seqs=256`**. Run on at least the a1-ref config and one of the a2/a5 configs.

### Matrix (all allowed — mns256 only)

| # | Test | Config | Addresses | Pass criteria |
|---|------|--------|-----------|---------------|
| T1 | **Graceful shutdown** | mns256, `docker stop -t 120`; SIGTERM reaches the vLLM engine so it tears down its own workers **before** docker's grace expires (no executor SIGKILL ~9 s in) | Rahul's main concern — the dirty SIGKILL | Container exits **0**, not 137; no residual VRAM; all 8 GPUs `gpu_busy_percent → 0` within the grace window |
| T2 | **Idle-gate before launch** | mns256; block next container start until all GPUs `gpu_busy_percent == 0` + VRAM at baseline, else abort+alert | Never launching onto a dirty/wedged card | No launch ever starts on a non-idle GPU across the run |
| T3 | **mns256 stability baseline** | mns256 normal sweep, N≈20, with T1+T2 in place | Is 256 actually a safe guardrail? | **0 GPU wedges, every teardown clean** |
| T4 | **Teardown characterization** (diagnostic) | current pre-fix shutdown on allowed mns256 runs | Quantify the SIGKILL risk + prove T1 removes it | Measured: does executor SIGKILL workers? residual VRAM/util after `docker stop`? time-to-idle? — before vs after T1 |

### Blocked until the customer lifts the 512 restriction
These are the only path to **root-cause closure**, but each needs `max_num_seqs > 256`:
- Decouple test — `--max-num-seqs 512 --compilation-config '{"max_cudagraph_capture_size":256}'`
- `--enforce-eager` at mns512
- Capture-bucket bisection 288 → 512

Parked, not abandoned. Current posture is **operational mitigation (cap 256 + clean teardown), not a proven root-cause fix**.

### Instrumentation per trial (so each run self-diagnoses)
- `dmesg` — the face discriminator, if anything ever faults at 256: `kiq fence` (KIQ) vs `PERMISSION_FAULTS`/`SQC` (page fault).
- Fault-guard dmesg-delta (the Oct 13 guard caught it in ~4 s) — keep it.
- Per-GPU telemetry @ 5 s — watch for the wedge signature: **high `gfx_util_pct` with near-zero `vram_used_mb` / no process** (logger `status` may still read `ok` — add an anomaly check for this).
- Teardown trace: container exit code, worker-process exit path (clean vs SIGKILL), residual VRAM, time-to-idle.
- **Fix the telemetry gap** — Oct 12 had no continuous csv; don't lose it again.

### Caveat to carry forward
Capping at 256 is a **mitigation, not a confirmed root cause** — the logs show 256 arms ran clean, but we never proved *why* 512 wedges. If T3 ever produces a wedge at 256, the "mns512 software path" theory weakens and teardown/firmware becomes the primary suspect — which is exactly why T1/T4 are worth doing regardless.
