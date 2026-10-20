# H200 node health — signal list (checked)

Pre/post job gate for the Slurm H200 queue (`deploy/h200`). Every signal is
**command → healthy output → threshold that flips it**. Thresholds are the
single source of truth in `deploy/h200/gitm-node-health` (`THRESHOLDS`); print
them with:

```bash
gitm-node-health --print-thresholds
```

Wired into `gitm-submit` / `gitm-job`: runs on the host before and after every
`srun`, writes one window row per node per run (`node_health_window.json`), and
feeds `qualify_run`. An unhealthy window discards results and re-queues the job.

Companion: [`deploy/h200/README.md`](../deploy/h200/README.md).

---

## Signals

| Signal | Command | Healthy output | Flips when |
|---|---|---|---|
| thermal | `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits` | every GPU `< 83` °C | any GPU `≥ 83` °C |
| clocks | `nvidia-smi --query-gpu=clocks_throttle_reasons.sw_thermal_slowdown,hw_thermal_slowdown,hw_slowdown,hw_power_brake_slowdown --format=csv,noheader` | every field `Not Active` | any field `Active` (hex `.active` bitmask fallback also decoded) |
| power cap state | `nvidia-smi --query-gpu=power.limit,power.max_limit,clocks_throttle_reasons.sw_power_cap --format=csv,noheader,nounits` | `power.limit == power.max_limit` and SW power-cap throttle inactive on every GPU | limit below max **or** SW power-cap active |
| dmesg XID | `sudo -n /usr/local/libexec/gitm-dmesg-xid` (else `dmesg -T`); lines matching `NVRM: Xid` | no new XID lines vs pre in any sample/post | any new XID mid-window, or log unreadable (never passes) |
| vLLM liveness | in-window sampler: real server cmdline (`python…vllm` / `vllm::`), not `srun`/`bash -c` | process present for every sample after serve start | any post-start sample with no real server |
| `/dev/shm` | `df -BG --output=size,avail,pcent /dev/shm \| tail -1` | size `≥ 16G` and used `< 90%` | size `< 16G` or used `≥ 90%` |
| NVLink | `nvidia-smi nvlink -s` and `nvidia-smi topo -m` | all reported links Active; every GPU pair among 0–7 is `NV*` | any link inactive/down, or any pair not `NV*` |

---

## Window row

One JSON object per node per job attempt:

```bash
gitm-node-health pre  --job "$SLURM_JOB_ID" --out "$OUT_HOST/node_health.jsonl"
# … srun …
gitm-node-health post --job "$SLURM_JOB_ID" --out "$OUT_HOST/node_health.jsonl"
gitm-node-health summarize --job "$SLURM_JOB_ID" --out "$OUT_HOST/node_health.jsonl"
# → $OUT_HOST/node_health_window.json
```

| Field | Meaning |
|---|---|
| `healthy` | `true` only if every signal stayed healthy for the whole window |
| `failed_signals` | names from the table above that flipped |
| `job_id` / `node` | Slurm job id and hostname |

`qualify_run` with the H200 signal contract requires source `node_health`
(`node_health_window.json`). `healthy: false` → check fail → verdict
`invalid_run`. Unhealthy attempts move `$OUT_HOST` to
`$RESULTS/discarded/<jobid>/` and `scontrol requeue` (max 2 requeues).
