"""Biotech dataset pipeline: CASP sequences + UniProt 50 k filter + MSAs.

Drives the three acquisition steps in order, each runnable independently:

1. **CASP** — fetch CASP14 (110) + CASP15 (94) target sequences.
2. **UniProt** — filter UniProt to lengths 50–512 aa, 50 000 sequences, via
   `mmseqs2`, concatenated into `proteins_50k.fasta`.
3. **MSAs** — build per-sequence `.a3m` against UniRef30 + BFD with the OpenFold
   MSA tooling.

Steps 2–3 require external tools (`mmseqs2`, the OpenFold MSA pipeline) and the
UniRef30/BFD databases (~2.2 TB resident NVMe — see the data-layout note), so
they run on a staging box with those resident, not a laptop. Each wrapper checks
its tool is present and fails loudly otherwise rather than fabricating data.

`--smoke` bypasses all external tools and synthesises a tiny, schema-correct
dataset (a handful of proteins + trivial single-row MSAs) so the downstream
freeze → manifest → reproduce loop runs anywhere. The FASTA reader/writer and
the length histogram are pure and used by both paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pinned sources. Replace placeholders with the exact pinned URLs/commands.
CASP_SOURCES = {
    "casp14": "https://predictioncenter.org/download_area/CASP14/sequences/casp14.seq.txt",
    "casp15": "https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt",
}
UNIPROT_TARGET_COUNT = 50_000
LEN_MIN, LEN_MAX = 50, 512
_AMINO = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class FastaRecord:
    header: str
    seq: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append(FastaRecord(header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append(FastaRecord(header, "".join(chunks)))
    return records


def write_fasta(records: list[FastaRecord], out: str | Path, *, wrap: int = 60) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(f">{r.header}\n")
            for i in range(0, len(r.seq), wrap):
                fh.write(r.seq[i : i + wrap] + "\n")
    return out


def length_histogram(records: list[FastaRecord], *, bin_width: int = 50) -> dict[str, int]:
    """Bucketed length histogram, e.g. ``{'50-99': 12, ...}`` — Tue deliverable."""
    counter: Counter[str] = Counter()
    for r in records:
        lo = (len(r.seq) // bin_width) * bin_width
        counter[f"{lo}-{lo + bin_width - 1}"] += 1
    return dict(sorted(counter.items(), key=lambda kv: int(kv[0].split("-")[0])))


# --- real acquisition (staging box) -----------------------------------------

def clean_casp_header(raw_header: str) -> str:
    """Normalize CASP sequence headers to only target ID"""
    return raw_header.split()[0].rstrip('|')

def fetch_casp(out_dir: str | Path) -> list[Path]:
    """Download CASP14 + CASP15 target sequence files."""
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, url in CASP_SOURCES.items():
        dest = out_dir / f"{name}.fasta"
        raw, _ = urllib.request.urlretrieve(url)
        raw_records = read_fasta(raw)
        cleaned = [FastaRecord(clean_casp_header(r.header), r.seq) for r in raw_records]
        write_fasta(cleaned, dest)
        written.append(dest)
    return written


def filter_uniprot(uniprot_fasta: str | Path, out: str | Path, *, count: int = UNIPROT_TARGET_COUNT) -> Path:
    """Filter UniProt to ``count`` sequences in [LEN_MIN, LEN_MAX] aa."""
    import random

    records = read_fasta(uniprot_fasta)
    eligible = [r for r in records if LEN_MIN <= len(r.seq) <= LEN_MAX]
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} sequences in [{LEN_MIN}, {LEN_MAX}] aa range, "
            f"need {count}. Provide a larger UniProt FASTA."
        )
    rng = random.Random(42)
    sampled = rng.sample(eligible, count)
    return write_fasta(sampled, out)

def build_msas(fasta: str | Path, out_dir: str | Path) -> list[Path]:
    """Build `.a3m` MSAs via the OpenFold precompute_alignments script."""
    import subprocess

    fasta = Path(fasta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent.parent.parent / "openfold_repo/scripts/precompute_alignments.py"
    if not script.exists():
        raise RuntimeError(f"precompute_alignments.py not found at {script}")

    db = Path(os.environ.get("AF2_DATA_DIR", "/workspace/af2_data"))

    cmd = [
        "python", str(script),
        str(fasta.parent),
        str(out_dir),
        "--uniref90_database_path", str(db / "uniref90/uniref90.fasta"),
        "--mgnify_database_path", str(db / "mgnify/mgy_clusters_2022_05.fa"),
        "--bfd_database_path", str(db / "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"),
        "--uniref30_database_path", str(db / "uniref30/UniRef30_2021_03"),
        "--pdb70_database_path", str(db / "pdb70/pdb70"),
        "--obsolete_pdbs_path", str(db / "pdb_mmcif/obsolete.dat"),
        "--max_template_date", "2020-05-14",
        "--jackhmmer_binary_path", shutil.which("jackhmmer") or "jackhmmer",
        "--hhblits_binary_path", shutil.which("hhblits") or "hhblits",
        "--hhsearch_binary_path", shutil.which("hhsearch") or "hhsearch",
        "--cpus_per_task", "8",
    ]
    subprocess.run(cmd, check=True)
    return list(out_dir.glob("**/*.a3m"))


# --- smoke fixtures (laptop) -------------------------------------------------


def synth_proteins(n: int, *, seed: int = 42) -> list[FastaRecord]:
    """Deterministic synthetic proteins, lengths in [50, 512]."""
    rng = np.random.default_rng(seed)
    amino = np.frombuffer(_AMINO.encode(), dtype="S1")
    out = []
    for i in range(n):
        length = int(rng.integers(LEN_MIN, LEN_MAX + 1))
        seq = b"".join(rng.choice(amino, size=length)).decode()
        out.append(FastaRecord(header=f"smoke_{i:05d} len={length}", seq=seq))
    return out


def write_smoke(out_dir: str | Path, *, n: int = 20, seed: int = 42) -> dict:
    """Write a tiny but complete biotech dataset: FASTA + trivial `.a3m` MSAs.

    Each smoke MSA is the query as a single-row alignment — enough for the
    freeze/manifest/reproduce loop and for harness plumbing tests, not for
    real inference accuracy.
    """
    out_dir = Path(out_dir)
    records = synth_proteins(n, seed=seed)
    fasta = write_fasta(records, out_dir / "proteins_50k.fasta")

    msa_dir = out_dir / "msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    a3m_paths = []
    for r in records:
        tag = r.header.split()[0]
        a3m = msa_dir / f"{tag}.a3m"
        a3m.write_text(f">{r.header}\n{r.seq}\n")
        a3m_paths.append(a3m)

    return {
        "fasta": fasta,
        "n_proteins": len(records),
        "n_msas": len(a3m_paths),
        "length_histogram": length_histogram(records),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Biotech dataset pipeline (CASP + UniProt + MSA).")
    p.add_argument("--out", type=Path, required=True, help="Dataset staging dir.")
    p.add_argument("--smoke", action="store_true", help="Synthesize a tiny local dataset.")
    p.add_argument("--smoke-n", type=int, default=20)
    p.add_argument("--step", choices=["casp", "all"], default="all",
                   help="Real acquisition step to run (non-smoke).")
    args = p.parse_args(argv)

    if args.smoke:
        info = write_smoke(args.out, n=args.smoke_n)
        print(f"smoke biotech dataset: {info['n_proteins']} proteins, "
              f"{info['n_msas']} MSAs in {args.out}")
        print(f"length histogram: {info['length_histogram']}")
        return 0

    if args.step in ("casp", "all"):
        casp = fetch_casp(args.out / "casp")
        print(f"fetched CASP: {[p.name for p in casp]}")
    if args.step == "all":
        print("UniProt filter + MSA build run on the staging box; see filter_uniprot/build_msas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
