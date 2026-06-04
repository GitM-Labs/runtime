# Biotech benchmark — spec

> Owner: Michael — baseline + profiling + spec doc.

## 1. Input definition
50 000 UniProt sequences (lengths 50–512 aa) + CASP14/15 targets, combined into
`proteins_50k.fasta` with precomputed `.a3m` MSAs. See [datasets.md](datasets.md).
Bytes in `$GITM_S3_ROOT/datasets/biotech/`, frozen by [manifest.yaml](manifest.yaml).

## 2. Work unit
One protein, single-seed AlphaFold2 inference (5 recycles, 1 model) via
**OpenFold v1.0.1** — pinned commit, pinned weight hashes. Phases:
`MSA load + featurize → Evoformer → structure module (recycles)`.
<!-- TODO: paste pinned OpenFold commit + weight hashes. -->

## 3. Success metric
`structures_per_hour` over a 1 000-protein warm window, restricted to length
≤ 384 to bound run time. Three seeds must agree within 2 %. Auxiliary (sanity,
**not** a target): median plDDT.

## 4. Expected stall profile
Matches `[expected_stall]` in [bench.toml](bench.toml):

| | CPU | Data-stall | Sync | GPU active |
| --- | --- | --- | --- | --- |
| Expected | < 5 % | 30–50 % | 5–10 % | 40–55 % |

Data-stall is MSA load + featurization; sync is recycle barriers. **The whole
point is GPU active % well under 85 %** — that residual headroom is the
optimization story. If GPU active comes in high, flag Adit immediately: this
benchmark must have headroom or it is swapped.
