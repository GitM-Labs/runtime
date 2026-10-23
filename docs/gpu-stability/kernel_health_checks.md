# GPU / kernel health checks — alpha01 (MI300X, amdgpu 6.16.13)

Practical checks to tell a **healthy** GPU from a **wedged/degraded** one — before a launch, during a run, and after a fault. Built from the three mns512 wedge incidents (`mns512_report.md`). The failures were all at the amdgpu kernel-driver boundary (KIQ fence, page fault, TLB flush), so these are GPU-driver health checks, not app checks.

**Where to run:** the host supplies only the kernel driver (amdgpu 6.16.13); ROCm userspace (`amd-smi`, ROCm 7.2.3) lives in the container. Run `dmesg`/sysfs on the **host**; run `amd-smi` from a container (or install host userspace). The incident logger polled `sudo amd-smi` every 5 s.

---

## 0. The signatures these checks look for

| Signature (dmesg / log) | Means | Phase seen |
|---|---|---|
| `timeout waiting for kiq fence` | GPU didn't complete a driver command (map/unmap/fence) — firmware queue hung | communicator init (Oct 9, 13) |
| `retry page fault` + `PERMISSION_FAULTS: 0xb` + `Faulty UTCL2 client ID: SQC` | shader hit an address the page table rejects | graph capture (Oct 12) |
| `TLB flush failed for PASID <n>` | driver can't invalidate TLB on a dead device (follows the wedge) | all |
| `failed to write reg <x> wait reg <y>` | driver can't reach the GPU during teardown | all |
| `Memory access fault by GPU node-N` / `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` | userspace-visible fault on a wedged queue | all |
| NCCL `Watchdog caught collective operation timeout` / `No available shared memory broadcast block` | **symptoms** — ranks waiting on a dead rank; not root cause | all |

**Telemetry blind spot (important):** a wedged GPU shows **high `gfx_util_pct` with near-zero `vram_used_mb` and no process attached**, yet the logger `status` may still read `ok`. Util alone is not health. Checks below encode this explicitly.

---

## 1. Pre-launch gate — "is every GPU clean?"

Run before starting any container. **Abort the launch if any check fails** (this is the idle-gate from `mns512_report.md` §5, expanded).

| Check | Command | Pass |
|---|---|---|
| All GPUs idle | `for c in /sys/class/drm/card*/device/gpu_busy_percent; do cat $c; done` | every value `0` |
| Clocks at idle | `amd-smi metric -g all --clock` (or sysfs `pp_dpm_sclk`) | ~132 MHz, **not** stuck ~2109 MHz |
| VRAM at baseline | `amd-smi metric -g all --mem-usage` | low / baseline, no orphaned allocation |
| No stray GPU processes | `amd-smi process -g all` / `fuser -v /dev/kfd /dev/dri/*` | none except intended |
| dmesg quiet | `sudo dmesg -T \| tail -200 \| grep -iE 'kiq fence\|page fault\|TLB flush\|failed to write reg\|ras'` | no recent amdgpu fault lines |
| ECC clean | `amd-smi metric -g all --ecc` / `amd-smi bad-pages -g all` | 0 uncorrectable; no new bad pages |
| XGMI links up | `amd-smi xgmi` | all links present/healthy |
| Device responsive | `amd-smi metric -g all` returns promptly | no hang/timeout reading the device |

**Composite "clean" rule:** `gpu_busy_percent == 0` **AND** idle clocks **AND** no process **AND** no new dmesg fault lines. (The Oct 9 try-2 failure was relaunching onto a card at 25 % busy / 2109 MHz / no process — util>0 with no process is already a red flag.)

---

## 2. Live monitoring — "did a GPU wedge mid-run?"

Run continuously during a job. Any trip = capture the forensic snapshot (§3) then stop.

- **dmesg fault-delta guard** (caught Oct 13 in ~4 s): count amdgpu fault lines; if the count jumps `0 → N`, stop the run.
  ```bash
  sudo dmesg -T | grep -icE 'kiq fence|retry page fault|TLB flush failed|failed to write reg'
  ```
- **Telemetry anomaly** (closes the blind spot): flag a GPU where `gfx_util_pct` is high **and** `vram_used_mb` is near-zero **and** no process is attached — regardless of logger `status`.
- **Stall counters** (app-side corroboration): ≥6 consecutive `No available shared memory broadcast block found in 60 seconds`, or any NCCL `Watchdog caught collective operation timeout` → a worker's GPU is hung (see `mns512_report.md` §2). Treat as hang signal, confirm against dmesg.
- **Clock/power sanity:** a GPU at 100 % util but low power (~251 W seen on Oct 9) is **spinning, not computing** — a wedge, not work.

---

## 3. Post-fault forensic snapshot — capture BEFORE reset

When a fault trips, grab this **before** resetting the GPU (Oct 12 deliberately left the card wedged for inspection — do the same):

```bash
sudo dmesg -T > dmesg_$(date +%H%M).txt                 # full ring buffer
amd-smi metric -g all   > amdsmi_metric.txt             # util/power/clocks/mem
amd-smi monitor         > amdsmi_monitor.txt
amd-smi process -g all  > amdsmi_process.txt            # who holds the GPU
amd-smi ecc -g all      > amdsmi_ecc.txt
amd-smi bad-pages -g all> amdsmi_badpages.txt
amd-smi xgmi            > amdsmi_xgmi.txt
amd-smi firmware -g all > amdsmi_firmware.txt           # KIQ = firmware queue — version matters
amd-smi static -g all   > amdsmi_static.txt
# per-card sysfs
for d in /sys/class/drm/card*/device; do echo "== $d =="; \
  cat $d/gpu_busy_percent $d/pp_dpm_sclk 2>/dev/null; done > sysfs_cards.txt
```

**Stuck-process forensics** (the Oct 13 `D`-state worker blocked in `exit_mmap`):
```bash
ps -eo pid,stat,etime,wchan:32,comm | grep -iE ' D |VLLM|vllm'   # find D-state / zombies
sudo cat /proc/<pid>/stack                                        # kernel stack of the stuck pid
```
A worker in `D` with a stack through `synchronize_srcu → __mmu_notifier_release → exit_mmap` = process can't exit because it's unmapping memory on a dead GPU. That process will not die without a GPU reset.

---

## 4. Recovery verification — after a reset

A GPU reset usually resets the whole XGMI hive. Re-run the §1 pre-launch gate and confirm:
- faulted GPU back to **idle ~132 MHz**, `gpu_busy_percent == 0`;
- `dmesg` quiet (no new amdgpu lines after the reset);
- **8-GPU parity** — all cards match on clocks/power/util at idle (an outlier is suspect);
- ECC still clean, XGMI links all up.
Only then allow the next launch. (Oct 10: a host reboot reset GPU 7 and the identical config then passed every prior failure point.)

---

## 5. Baseline / environment facts to record once
- **Firmware versions** (`amd-smi firmware`) — KIQ hangs are firmware-queue failures; pin the baseline so a future drift is visible.
- **amdgpu driver** 6.16.13, **kernel** `6.6.29+main+3.0.0r1-...` — host-level, not container-controlled.
- **CUPTI profiling counters** need `NVreg_RestrictProfilingToAdminUsers=0` (reboot) — unrelated to health but affects what you can measure.
- Steady corrected PCIe `RxErr` (~60/h) on some ports is **background noise** here, not a wedge indicator — don't false-alarm on it (Oct 9 notes).

---

## 6. Minimal health-gate (drop into the launch wrapper)

```bash
# returns 0 = clean, 1 = NOT clean (abort launch)
gpu_clean() {
  for c in /sys/class/drm/card*/device/gpu_busy_percent; do
    [ "$(cat "$c")" = "0" ] || { echo "BUSY: $c=$(cat "$c")"; return 1; }
  done
  if sudo dmesg -T | tail -300 | grep -qiE 'kiq fence|retry page fault|TLB flush failed|failed to write reg'; then
    echo "FAULT lines in recent dmesg"; return 1
  fi
  return 0
}
gpu_clean || { echo "GPU not clean — aborting launch"; exit 1; }
```

---

## Cross-reference
- Wedge incidents, exact log strings, diagnosis: `mns512_report.md`
- Performance (not health) diagnosis: `perf_diagnosis.md`
- Raw artifacts and the logger columns: `logs/README.md`
