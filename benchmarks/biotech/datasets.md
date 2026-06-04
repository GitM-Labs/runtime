# Biotech benchmark — datasets

> Owner: Nicholas — dataset + reproducibility. Fill the TODOs as data lands.

Pipeline driver: [`fetch.py`](fetch.py) (`fetch_casp` / `filter_uniprot` /
`build_msas`). Steps 2–3 need `mmseqs2` + the OpenFold MSA tooling and the
UniRef30/BFD DBs (~2.2 TB NVMe), so they run on the staging box. `make smoke`
synthesizes a tiny local dataset to drive the freeze/verify/reproduce loop.

## Sources
- **CASP14 targets** — 110 sequences (`fetch.py fetch_casp`, pinned URL).
- **CASP15 targets** — 94 sequences (`fetch.py fetch_casp`, pinned URL).
- **UniProt subset** — filtered to lengths 50–512 aa, 50 000 sequences.
- **Combined FASTA:** `proteins_50k.fasta`, reproducible from one `mmseqs2 filter`
  command. <!-- TODO: paste the pinned mmseqs2 command into fetch.filter_uniprot. -->
- **MSAs:** precomputed against UniRef30 + BFD with the OpenFold MSA tooling,
  frozen as `.a3m`. <!-- TODO: pin OpenFold MSA commit in fetch.build_msas. -->

## Fields & units
FASTA records + per-sequence `.a3m` MSA. <!-- TODO: paste a header sample + length histogram. -->

## Scale target
50 000 + CASP sequences; MSAs dominate on-disk size.
<!-- TODO: record .a3m total size after generation. -->

## Seed protocol
`{42, 43, 44}` are inference seeds (single-seed AF2, 5 recycles, 1 model). The
dataset itself is fixed; seeds vary the run, not the bytes.
Layout: `$GITM_S3_ROOT/datasets/biotech/{casp14,casp15,uniprot_50k}/`.

## Freeze & verify
```bash
make manifest      # -> manifest.yaml (sha256 + byte count per .fasta / .a3m)
make verify        # re-hash and confirm byte-identical
```
Manifest sha256: <!-- TODO: paste after first freeze -->
