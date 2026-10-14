# Shared H200 queue: one 8-GPU job per node

Two 8x H200 GCP VMs shared by the team through one queue. Each job gets an
entire node (all 8 GPUs), jobs start in submission order on whichever node frees
up first, and a public page shows what is running:
<https://aditchawdhary.github.io/gpu/>.

```
          you / teammates (ssh)
                  │
        ┌─────────▼──────────┐      status.json (every ≤60s)     ┌──────────────────────┐
        │  control VM (CPU)  │ ───────── git push ─────────────► │ github: gpu-status   │
        │  k3s server, Kueue │                                   │ branch `status`      │
        │  gitm-job, pusher  │                                   └──────────┬───────────┘
        └───┬────────────┬───┘                                              │ fetch
            │ k3s        │ k3s                                   ┌──────────▼───────────┐
     ┌──────▼─────┐ ┌────▼───────┐                               │ aditchawdhary.github │
     │ h200-a 8GPU│ │ h200-b 8GPU│                               │ .io/gpu/  (static)   │
     └────────────┘ └────────────┘                               └──────────────────────┘
```

Nothing on the cluster accepts inbound traffic from the internet. The pusher
only makes outbound `git push` calls with a deploy key scoped to one repo.

| File | What it is |
|---|---|
| `setup-control.sh` | Control VM: k3s server, NVIDIA device plugin, Kueue, the queue, team access, CLI, pusher |
| `setup-gpu-node.sh` | Each H200 VM: NVIDIA container toolkit, joins the cluster as a GPU worker |
| `cluster.yaml` | Queue (16 GPUs = 2 nodes x 8, strict FIFO), RBAC, admission policy, GPU-metrics DaemonSet |
| `gitm-job` | Team CLI (`submit`, `ls`, `logs`, `cancel`); also the status pusher |
| `gitm-status.service` | systemd unit running the pusher on the control VM |

## Set up (once)

**1. Create the status repo.** A new **public** GitHub repo,
`aditchawdhary/gpu-status`. Empty is fine. It holds a single file on a single
commit that gets replaced, so it never grows.

**2. Create the control VM** in the same VPC and zone as the H200s. It needs
no GPU:

```bash
gcloud compute instances create gitm-control \
  --zone=<H200 zone> --network=<H200 VPC> --subnet=<H200 subnet> \
  --machine-type=e2-standard-4 --boot-disk-size=50GB \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --tags=gitm-k3s
gcloud compute instances add-tags <h200-vm-1> <h200-vm-2> --zone=<zone> --tags=gitm-k3s
```

**3. Open the firewall between the three VMs** (internal traffic only):

```bash
gcloud compute firewall-rules create gitm-k3s-internal --network=<H200 VPC> \
  --allow=tcp:6443,tcp:10250,udp:8472 \
  --source-tags=gitm-k3s --target-tags=gitm-k3s
```

**4. Control VM:** copy this folder over, then

```bash
sudo ./setup-control.sh --status-repo aditchawdhary/gpu-status \
     --hf-cache /var/lib/gitm/hf-cache --results-dir /var/lib/gitm/results
```

It ends by printing the exact join command for step 5, a deploy key for step 6,
and how to add teammates.

**5. Each H200 VM:** copy this folder over and run the printed command with
`--name h200-a` on the first and `--name h200-b` on the second. Then back on
the control VM:

```bash
sudo k3s kubectl get nodes          # control + h200-a + h200-b, all Ready
sudo ./setup-control.sh --smoke     # one job per node, each must see 8 GPUs
```

**6. Status page feed:** add the printed public key at
`github.com/aditchawdhary/gpu-status/settings/keys` with **Allow write access**,
then `sudo systemctl restart gitm-status` and watch `journalctl -u gitm-status -f`
for `pushed: ...`.

**7. Teammates:** each needs a Linux account on the control VM (OS Login, or
`sudo adduser`), then `sudo usermod -aG gitm-users <user>`.

## Using it

```bash
gitm-job submit --name kimi-k26-baseline --model moonshotai/Kimi-K2.6
gitm-job ls                         # running / queued with position / recent
gitm-job logs kimi-k26 -f           # any unique part of the name works
gitm-job cancel kimi-k26
```

A default job installs `gitm` from this repo (`--gitm-ref`, default `main`) into
the vLLM image and runs

```
gitm capture serve --out $OUT -- vllm serve $MODEL --tensor-parallel-size 8 --trust-remote-code
```

Tune it with `--vllm-args "--max-model-len 65536"` and `--capture-args
"--requests 1024"`, or replace the command entirely after `--`
(`gitm-job submit --name x --model m -- python my_script.py`; gitm is still
installed, and `$MODEL` and `$OUT` are set). Other flags: `--image`,
`--max-hours` (default 12, max 48), `--env KEY=VAL`, `--node h200-b`,
`--dry-run` (print the Job without submitting).

Jobs read models from the HF cache path and write traces to
`<results-dir>/<job-name>/`, both **on the GPU node the job ran on** (the `NODE`
column in `gitm-job ls`). Both paths must be the same on every GPU node.
Finished jobs stay listed for 7 days.

## How "one job per node, all 8 GPUs" is enforced

* **Kueue** admits at most 16 GPUs' worth of jobs. Every job asks for exactly 8,
  so at most two run, and the rest wait suspended (no pods) in strict FIFO order.
* **Kubernetes scheduling** puts an 8-GPU pod only on a node with all 8 free, so
  the two running jobs always land on different nodes.
* **An admission policy** rejects any Job in namespace `gitm` that skips the
  queue, asks for a GPU count other than 8, or sets `NVIDIA_VISIBLE_DEVICES`.
  Teammates' credentials can create Jobs only, not bare Pods.
* **The NVIDIA toolkit** only hands GPUs to unprivileged containers through the
  device plugin, so an env var can't grab GPUs the queue thinks are free.

## The status page

The pusher on the control VM collects jobs, queue order, `nvidia-smi` readings
from each node and the last 25 log lines of live jobs, then force-pushes
`status.json` to the `status` branch. It pushes immediately when a job starts,
finishes or joins the queue, at most once a minute for log and GPU changes, and
every 5 minutes as a heartbeat. The page shows a warning if the data is more
than 10 minutes old.

The page reads the branch's commit through GitHub's API (60 calls per hour per
network; repeat checks with nothing new are free), then the file by commit,
which the CDN can't serve stale. If an office network uses up the allowance, the
page switches to a CDN copy that can lag a few minutes and says so.

**The log tail is public.** The pusher redacts common secret shapes (HF, GitHub,
OpenAI, AWS, Google keys, JWTs, bearer tokens, `*TOKEN=`/`*PASSWORD=` values, URL
credentials), but don't print secrets in job output and rely on that.

## Operating notes

* **Disk:** Kimi K2-family checkpoints are around 600 GB. Put the HF cache on a
  disk that fits every model the team traces, on each GPU node.
* **Adding a third GPU node:** run `setup-gpu-node.sh` on it, raise
  `nominalQuota` in `cluster.yaml` to 24, re-run `setup-control.sh`.
* **Taking a node out for maintenance:** `sudo k3s kubectl drain h200-a
  --ignore-daemonsets` (a running job is killed); `uncordon` to return it.
  While it is out, the queue still admits two jobs, and the second waits in
  "Starting" until the node is back.
* **Pinned versions:** NVIDIA device plugin v0.17.1, Kueue v0.20.1, k3s from
  the stable channel at install time (GPU nodes join with the same version).
