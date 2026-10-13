# Checkpoint inventory

`gitm/planner/inventories/kimi.yaml` records, for every Kimi checkpoint, exactly what a
deployment would load: repository, immutable commit, the quantization recipe the
checkpoint declares, the SHA-256 of every file that defines it, and every
`deploy/` manifest that loads it. It holds checkpoint *facts* only. Engine
behaviour lives in the planner catalogue (`gitm/planner/models/`), measured
performance in benchmark results.

```
gitm inventory add moonshotai/Kimi-K3 [--family kimi] [--ref main | --revision <sha>]   # network
gitm inventory sync                                                                   # offline
gitm inventory check [--offline] [--strict] [--json]
```

* **Add a checkpoint** with `add`; there is no schema to change. It resolves the ref
  to a commit SHA, hashes the files (LFS shards from the tree API, no download;
  small files fetched and tied to their git blob id), and reads the declared
  quantization recipe verbatim. `sync` then fills catalogue status and consumers.
* **`check` exits** 0 clean, 1 drift, 2 could not check (hub unreachable, invalid file).
  It reports: the tracked branch moved (`revision-drift`, with the files that
  differ); files at the pinned commit no longer match (`artifact-drift`); a
  manifest missing from, or stale in, a record; a deployment pinned to a different
  commit (`pin-mismatch`); catalogue coverage in both directions. A deployment
  with no `--revision` is a *warning* (`unpinned`); `--strict` fails on it.
* **`UNVERIFIED`** is a valid value and is never replaced by a guess. `revision:
  main` is a schema error. Identity is repository + revision + hashes, never
  `model_type` (Kimi-K2.6's config says `kimi_k25`).
* **`catalogue.status`**: `covered` when a catalogue entry's `name:` equals the
  repository in full, owner included; otherwise `not-in-catalogue`.
* **What `check` cannot see:** which revision a node's cache actually holds. A
  manifest that pins nothing resolves to whatever the hub or `HF_HOME` holds, so
  its consumer revision stays `UNVERIFIED` until someone pins it.

The generic snapshot/diff layer is `gitm/planner/revisions.py`; it knows nothing
about Kimi.
