# Safety primitives

The guardrails every live apply runs behind. The day-one posture is
**detect-revert-page**, not confidence-gated auto-act — the runtime mutates only
reversible runtime knobs, records everything it does, and reverts on any exit or
regression.

Four primitives make that posture real (`gitm/safety/`):

| Primitive | Role | File |
|-----------|------|------|
| Audit log | durable record of every change and revert | `audit.py` |
| Fail-open guard | revert every live mutation on any exit | `failopen.py` |
| Auto-revert | detect a regression vs baseline over a window | `autorevert.py` |
| Gated rollout | stage in shadow until a manual, confirmed promote | `rollout.py` |

## 1. Audit log

An **append-only** JSONL trail: one event per line, `fsync`'d on write, never
rewritten. When something runs unattended we must be able to answer "what did the
runtime change, when, why, and did it revert?" from a durable record — a crash
mid-run still leaves a readable, append-only trail.

```python
log = AuditLog(path)
log.record_apply("kv_cache_dtype_fp8", knob="kv_cache_dtype", value="fp8", cause="headroom")
log.record_revert("kv_cache_dtype_fp8", reason="TPOT regression", cause="auto-revert")
log.entries()   # -> list[AuditEvent] in write order
```

Each `AuditEvent` is `ts_ns, event, intervention, cause, detail`. `record(...)`
is the general form; `record_apply` / `record_revert` are the two typed helpers.
A fresh `AuditLog` over an existing path **appends** — it never truncates.

## 2. Fail-open guard

The runtime's death must leave the workload untouched. Every live mutation
registers a revert with the guard; on any exit — normal, exception, or a
catchable signal (SIGTERM/SIGINT) — the guard runs all reverts in **LIFO** order.
`disarm(name)` marks a change as intentionally kept (it cleared the gate) so it is
not rolled back.

```python
with FailOpenGuard(audit=log) as guard:
    apply_knob()
    guard.register("knob", revert_knob, cause="applied live")
    ...            # on any exit, revert_knob() runs
```

**Best-effort, but never silent.** One revert failing must not block the others —
but a failure is the most page-worthy event here (the workload may be left
mutated), so it is **not** swallowed:

- a failing revert is recorded to the audit trail as a `revert_failed` event
  (with the exception repr), and
- its name is surfaced on `guard.failures`. A non-empty `failures` list means
  fail-open did **not** fully clean up.

Audit writes inside `fire()` are themselves best-effort — a broken audit sink can
never abort the reverts. `SIGKILL` / power loss can't run code; for those the
guarantee comes from the mutations being non-persistent (in-memory engine knobs,
removable hooks), which is why the live applicator only hot-swaps reversible
knobs.

## 3. Auto-revert

The *detect* half of detect-revert-page. After a change is applied, a
higher-is-better metric (throughput, goodput) is sampled over a short window; if
the windowed mean drops below baseline by more than `tolerance`, `observe`
signals a revert. A **full window** is required before any decision so a single
noisy sample can't trip it.

```python
ar = AutoRevert(baseline=100.0, tolerance=0.05, window=5)
d = ar.observe(sample)      # -> AutoRevertDecision
d.should_revert             # True once the windowed mean regresses beyond tolerance
d.relative_delta            # (windowed_mean - baseline) / baseline
```

Auto-revert is **decision-only**: it reports `should_revert`; the caller performs
the revert (typically via the fail-open guard or the rollback gate).

## 4. Gated rollout

Not confidence-gated auto-act. A change is staged in **shadow** (recorded, never
applied live), and only a manual `promote(confirm=True)` moves it live. `abort`
discards it. Every transition is audited, so the trail shows what was staged,
what was promoted (and why), and what was dropped.

```python
r = GatedRollout(audit=log, guard=guard)
r.stage("lever", knob="k", value=1, apply_fn=apply, revert_fn=revert)
r.is_live("lever")          # False — shadow
r.promote("lever", confirm=True)   # applies, registers revert with the guard
r.is_live("lever")          # True
```

When a `guard` is supplied and a staged change carries an `apply_fn` (and
optionally a `revert_fn`), `promote` **genuinely applies** it and registers the
revert with the fail-open guard — so a promoted change is both live and fail-open
protected. The apply happens *before* the state flips: a failing apply leaves the
change in shadow (and raises) rather than falsely reporting it live. Without an
`apply_fn`, `GatedRollout` is bookkeeping only (state + audit), which is all the
shadow-accounting callers need. An unconfirmed promote aborts the change.

## Integration: the apply seam

The primitives attach to live applies through a single seam,
`apply_intervention` (`gitm/optimizer/apply.py`):

```python
apply_intervention(spec, applicator, *, min_keep_delta=0.0, audit=None)
```

When an `audit` log is supplied, every live mutation and rollback is recorded
(`apply` on keep; `apply` + `revert` on a regression or a failed measure). It is
recorded **best-effort** — a broken audit sink never blocks the apply. Pass an
audit only where the applicator mutates a real target; a dry-run leaves it `None`
so the trail stays free of no-op entries.

In the loop (`gitm/scheduler/loop.py`) the real-apply workload paths
(hft / openfold / edge) pass an `AuditLog(run_dir / "audit.jsonl")`, so a live A/B
apply and any rollback land on a durable per-run trail. The `vllm-decode` catalog
path runs dry (no live engine), so it writes no trail.
