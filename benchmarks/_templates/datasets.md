<!-- Copy to benchmarks/<name>/datasets.md. This is the reproducibility contract:
     anyone following it must land byte-identical data. -->
# <NAME> benchmark — datasets

## Sources
<!-- Each upstream source, exact version/tag, and how to fetch it. Pin
     generator commands (generator version + seeds) or download URLs. -->

## Fields & units
<!-- Per-record schema, dtypes, units. Paste a first-row sample. -->

## Scale target
<!-- Row/event count at the scale target, raw + compressed size on disk. -->

## Seed protocol
<!-- Which seeds (e.g. {42, 43, 44}) and how each maps to a frozen dataset
     directory under $GITM_S3_ROOT/datasets/<name>/. -->

## Freeze & verify
```bash
# generate/stage into bounded local scratch, then freeze:
make manifest      # -> manifest.yaml (sha256 + byte count per file)
make verify        # re-hash and confirm byte-identical
```
Manifest sha256 (`python -m gitm.bench`-generated): <!-- paste after first freeze -->
