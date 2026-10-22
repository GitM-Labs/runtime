# Shared H200 queue on Slurm: one containerized job per node

Two 8x H200 machines in one Slurm cluster (controller: gitmachine-2). Each job
gets a whole node (all 8 GPUs), runs in a throwaway container (Pyxis + Enroot)
so CUPTI tracing and pip installs never touch the host, and waits its turn in
one queue. Live status: <https://gitm-labs.github.io/gpu-status/>.

```
 teammates ──ssh──► gitmachine-2  (slurmctld + slurmd, 8x H200)  ──git push──► github: gpu-status
                        │  Slurm (munge)                                        │ (branch `status`)
                        ▼                                                       ▼
                    node 2        (slurmd, 8x H200)                 gitm-labs.github.io/gpu-status/
```

| File | Run where | What it does |
|---|---|---|
| `setup-controller.sh` | gitmachine-2 | Adds node 2 to `slurm.conf`, makes the `gpu` partition whole-node (`OverSubscribe=EXCLUSIVE`), turns on job history, installs containers here, makes the join bundle, starts the status pusher |
| `setup-join.sh` | node 2 | Installs matching Slurm + munge, copies accounts with the same UIDs, joins the cluster, installs containers |
| `setup-node.sh` | (called by both) | nvidia-container hook, Enroot 4.2.1, Pyxis v0.24.0, pre-imported vLLM image, reporter |
| `gitm-job` | everywhere | Team CLI; also the status pusher |
| `gitm-node-report` | every node | GPU readings + job log tails for the page |
| `gitm-slack-notify` | every node | Live Slack feed: queue events + off-scheduler GPU use |

## Setup

On node 2, note its name and internal IP: `hostname -s; hostname -I`.

```bash
# 1. gitmachine-2
sudo ./setup-controller.sh --add-node NODE2NAME=NODE2_IP --status-repo GitM-Labs/gpu-status

# 2. node 2 (copy this folder + the bundle it printed)
sudo ./setup-join.sh --bundle ~/gitm-join.tgz && rm ~/gitm-join.tgz

# 3. gitmachine-2: one container job per node, each must see 8 GPUs
sudo ./setup-controller.sh --smoke
```

Then add the printed deploy key (with write access) to the `gpu-status` repo and
`sudo systemctl restart gitm-status`. Slurm tokens (JWT) keep working: only the
controller checks them, and jobs can now land on either node.

## Interns: submit with a Slurm token (no SSH)

Interns never log into the machines. They get a Slurm token and the
`gitm-submit` script (this folder; Python 3.9+, no packages needed):

```bash
# admin, on gitmachine-2: one token per person (90 days)
sudo scontrol token username=tarun lifespan=7776000

# intern, on their laptop
export SLURM_JWT=eyJ...                 # their token
export SLURM_API=https://<cluster REST address>
./gitm-submit run --name qwen3-smoke --model Qwen/Qwen3-0.6B \
    --capture-args "--requests 64 --concurrency 32"
./gitm-submit ls
./gitm-submit status <jobid>
```

Results show up on the status page under *Finished recently*: latency, goodput,
where GPU kernel time went, and the files written. Until the REST API has a
public HTTPS address, test it through an SSH tunnel:
`ssh -N -L 6820:localhost:6820 gitmachine-2` and `SLURM_API=http://localhost:6820`.

## Using it

```bash
gitm-job submit --name kimi-k26 --model moonshotai/Kimi-K2.6
gitm-job ls                 # running / waiting with position / recent
gitm-job logs 123 -f
gitm-job cancel 123
```

Default job: inside the container, install gitm (`--gitm-ref`, default `main`)
and run `gitm capture serve --out /out -- vllm serve $MODEL --tensor-parallel-size 8`.
`--vllm-args`, `--capture-args`, `--image`, `--max-hours`, `--env K=V`,
`--node`, or a custom command after `--`. `--dry-run` prints the batch script.
Results: `/var/lib/gitm/results/<jobid>/` (+ `<jobid>.log`) on the node that ran
it. Raw `sbatch --exclusive --container-image=...` works too and shows on the page.

## Live Slack feed

`gitm-slack-notify` (stdlib Python, no deps) posts cluster activity to Slack via
an Incoming Webhook at `/etc/gitm-gpu/slack_webhook` (one line, `0600 root`).
Event-driven (30 s poll, posts only on changes, seeds silently on restart):

```bash
install -m 0755 gitm-slack-notify /usr/local/bin/
install -m 0644 gitm-slack-scheduler.service gitm-slack-direct.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gitm-slack-direct.service                      # every node
systemctl enable --now gitm-slack-scheduler.service                   # controller only
```

- **scheduler** (controller): diffs `squeue` → queued / started / finished /
  failed / cancelled; failure alerts attach a 25-line log tail.
- **direct** (every node): flags GPU processes with no Slurm cgroup =
  off-scheduler / SSH use.
- On-demand snapshot: `gitm-slack-notify snapshot`. Webhook check:
  `gitm-slack-notify test`.

Integrating this feed with the node-health gate (`gitm-node-health`, PR #189) —
thermal / throttle / XID / NVLink surfaced live — is reviewed in
[`node-health-integration.md`](node-health-integration.md).

## Notes

* The page's log tail is public. Common secret formats are redacted; don't
  print secrets anyway.
* GPUs held by processes outside Slurm (e.g. a hand-started vLLM) make queued
  jobs fail; the setup scripts list them.
* CUPTI hardware counters need `NVreg_RestrictProfilingToAdminUsers=0` on the
  driver (reboot); kernel tracing works without it.
* Every `slurm.conf` / `gres.conf` change is backed up next to the file, and
  restored automatically if slurmctld fails to start.
