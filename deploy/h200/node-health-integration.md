# Review — integrating the live monitor with Nathan's node-health checks

**Scope:** fold the node-health signals from
[`GitM-Labs/runtime` PR #189](https://github.com/GitM-Labs/runtime/pull/189)
(`deploy/h200/gitm-node-health`) into the live monitor (`gitm-slack-notify` +
the `gpu-status` page), so the cluster surfaces **node health continuously** —
not only as a pass/fail gate around each job.

**Verdict:** worth doing, low-risk, and the two pieces are genuinely
complementary. But they don't connect as-is — there's one real impedance
mismatch and one safety boundary to respect. Recommendation below needs a
~40-line read-only addition to `gitm-node-health` and a new watcher mode in
`gitm-slack-notify`. **No change to PR #189's setup/gate wiring is required for
this** (and that wiring still carries the P1s from the earlier PR review — keep
it out of scope).

---

## 1. What each side is

**Nathan's `gitm-node-health`** — a **job-windowed gate**. Subcommands `pre` /
`sample` / `post` / `summarize` / `gate`, every one requiring `--job` and
`--out`. It writes `node_health.jsonl` (per-sample rows) and `summarize` emits
`node_health_window.json` = `{healthy, failed_signals, job_id, node}`, which
`qualify_run` consumes: `healthy:false` → `invalid_run` → discard + requeue.
Signals (threshold = single source of truth in `THRESHOLDS`, printable via
`gitm-node-health --print-thresholds`):

| Signal | Instantaneous? | Notes |
|---|---|---|
| thermal (`≥83 °C`) | ✅ | one `nvidia-smi` query |
| clock throttle reasons | ✅ | sw/hw thermal, hw slowdown, power-brake |
| power-cap state | ✅ | `power.limit == power.max_limit`, SW power-cap inactive |
| `/dev/shm` (`≥16G`, `<90%`) | ✅ | `df` |
| NVLink | ✅ | `nvidia-smi nvlink -s` + `topo -m`, all pairs `NV*` |
| dmesg XID | ⚠️ **windowed** | "new XID lines **vs pre**" — needs a baseline |
| vLLM liveness | ⚠️ **job-scoped** | only meaningful during a serve job |

**Our `gitm-slack-notify`** — a **continuous event watcher**. Modes `scheduler`
(diff `squeue`) and `direct` (off-scheduler GPU procs), 30 s poll, posts only on
*transitions*, state in `/var/lib/gitm-gpu/slack-notify/`. Plus `gitm-status`
pushes `status.json` to the page. **It carries zero health signals today** — a
GPU can be thermal-throttling or throwing XIDs and the feed says nothing until a
*job* fails the gate.

## 2. The mismatch

`gitm-node-health` has **no side-effect-free "check now" entrypoint**. Every
subcommand except `--print-thresholds` requires a `--job`/`--out` and writes
window files. To poll health live you'd otherwise have to manufacture fake job
windows (`sample --job _live --out /tmp/...`) and scrape the last JSONL row —
which writes junk, and whose XID signal is meaningless without a prior `pre`
baseline. That's the wrong seam.

Two signals are also **not instantaneous**: XID is "new since the window
started," and vLLM liveness only applies inside a serve job. A live, node-wide
health view wants point-in-time state for the 5 instantaneous signals, plus
"XID seen recently / since last poll" tracked by the *caller*.

## 3. Recommendation

**(a) Add a read-only `check` subcommand to `gitm-node-health`** (upstream into
PR #189, so thresholds stay single-source):

```
gitm-node-health check [--json] [--signals thermal,clocks,power,nvlink,shm]
```

- Runs the same per-signal functions, **no `--job`, no file writes**.
- Emits `{ "node": "...", "healthy": bool, "failed_signals": [...],
  "signals": { "thermal": {"ok":true,"value":...}, ... }, "ts": "..." }`.
- Excludes vLLM liveness (job-scoped) by default; reports **current** XID
  presence (`any NVRM: Xid in the last N min`) rather than "new vs pre" — the
  live watcher does its own since-last-poll diffing.
- Reuses `THRESHOLDS` verbatim, so the live feed and the qualify gate can never
  disagree about what "unhealthy" means.

This is the key move: one signal implementation, two consumers (the gate and the
live monitor).

**(b) New `health` mode in `gitm-slack-notify`** (symmetric with `direct` —
per-node, runs on both):

- `gitm-slack-notify health --interval 60` → each tick runs
  `gitm-node-health check --json`, diffs against state in
  `/var/lib/gitm-gpu/slack-notify/health.json`, posts **only on transitions**:
  - healthy → unhealthy: `🔴 *Node gm1 unhealthy* — failed: thermal, nvlink (GPU3 82→85 °C)` + the offending values.
  - unhealthy → healthy: `🟢 Node gm1 recovered (was: thermal)`.
  - new XID: always post (`⚠️ XID 79 on gm2 GPU5` — these are rare and serious).
- Seeds silently on restart (same pattern as the other modes), so it won't
  replay a standing condition.
- New unit `gitm-slack-health.service` on **both** nodes.

**(c) Status page** — add a `health` block to `status.json` per node
(`{healthy, failed_signals, gpus:[{temp, throttle, xid}]}`) so
`gitm-labs.github.io/gpu-status` shows a per-node health badge next to the
queue. The pusher already runs on gm2; it would call `gitm-node-health check`
for gm2 locally and for gm1 over the existing `gitm-node-report`/`report_key`
channel (no new trust path).

## 4. Cross-node shape

Health is **per-node**, so the watcher runs on each node (like `direct`) and
posts about its own box — gm2 doesn't need to reach into gm1 for the Slack feed.
Only the **status page** aggregates both nodes, and it already has the
`gitm-node-report` command-pinned key to read gm1 from gm2. Extend
`gitm-node-report` to include a `health` section rather than opening a new
channel.

## 5. Risks & cautions

1. **Do NOT pull in PR #189's `setup-node.sh` / `setup-controller.sh` /
   `gitm-submit` / `gitm-job` changes for this.** Those are the gate-wiring
   (pre/post around `srun` + requeue) and still carry the 2 P1s flagged in the
   earlier PR review. Live monitoring needs **only** the `gitm-node-health`
   script (read-only `nvidia-smi`/`dmesg`) + the new `check` subcommand.
2. **PR #189 is OPEN and "verified locally for now."** `gitm-node-health`
   itself is side-effect-free and safe to run, but it isn't merged. Pin a
   known-good copy of just that script on the nodes (don't point at adit's
   working clone), and track upstream so the `check` addition lands in the PR.
3. **XID baseline.** The live watcher must track "XIDs seen" in its own state
   and diff per-poll; don't rely on the gate's pre/post baseline, which doesn't
   exist outside a job.
4. **vLLM liveness is job-scoped** — exclude it from the node-wide live check
   (outside a serve job there's no server to expect; it would false-alarm).
5. **`nvidia-smi` cost.** `nvlink -s` + `topo -m` are ~1 s. Poll health at
   **≥60 s** (not the 30 s the queue watcher uses), and `check` should be a
   single batched `nvidia-smi` query where possible. A health poll is CPU/driver
   only — it does **not** contend with a running GPU job.
6. **Noise control.** Transition-only posting + silent seed-on-restart is
   essential — a GPU that sits at 84 °C must post **once**, not every minute.
   Consider a cooldown so a flapping signal can't spam the channel.
7. **Root + dmesg.** The monitor services run as root, so the XID read
   (`dmesg -T` / `gitm-dmesg-xid`) works without the setup-script's sudoers
   additions. Good — another reason we don't need the setup wiring.

## 6. Phased plan

| Phase | Work | Touches nodes? | Needs go-ahead |
|---|---|---|---|
| 1 | Add `check --json` to `gitm-node-health` (PR #189 follow-up) | no | PR comment only |
| 2 | Add `health` mode + `gitm-slack-health.service` to this repo | no (repo only) | — |
| 3 | Pin `gitm-node-health` + deploy the `health` watcher on gm1/gm2, test with `--once` | **yes** | **yes — Rishith** |
| 4 | Add `health` block to `status.json` + page badge | yes (gm2) | yes |

Phases 1–2 are code-only and reviewable here. Phase 3 is the first thing that
touches the live boxes and needs explicit sign-off (new service on production
nodes).

## 7. Open questions

- **Slack routing:** health alerts in the **same channel** as queue events, or a
  separate `#gpu-health`? (XID/thermal are higher-urgency than "job queued".)
- **Thresholds for *alerting* vs *gating*:** the gate flips at `≥83 °C`. Do we
  want an earlier **warn** tier for the live feed (e.g. 78 °C) that doesn't fail
  jobs but gives a heads-up? If so, `check` should return raw values (it will)
  and the watcher owns the warn threshold.
- **Upstream vs fork:** land `check` in PR #189, or keep a thin wrapper in this
  repo until #189 merges? (Recommend: propose it on the PR; it's small and makes
  the gate + monitor share one signal source.)
