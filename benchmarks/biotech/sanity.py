"""One-protein GPU sanity check for the OpenFold runner — run on the pod.

Builds the real runner (pinned weights, recycling, TF32) and folds a single
protein, printing plDDT and the per-phase ``_t_*`` split. Runs the protein twice:
the first pass is COLD (kernel JIT/autotune + allocator growth), the second is
WARM (cached) — the gap is exactly what :func:`harness.run`'s warm-up hides from
the measured window. Use this to confirm the inference path is correct *before*
committing to a full warm-window baseline.

    conda activate openfold_env
    export OPENFOLD_WEIGHTS=/workspace/af2_data/params/params_model_1.npz
    python benchmarks/biotech/sanity.py --seed 42 --stage /workspace/biotech/staging/biotech

Exit codes: 0 = folded OK, 1 = no foldable protein found, 2 = framework/setup error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from benchmarks.biotech.fetch import read_fasta
from benchmarks.biotech.harness import (
    _device_info,
    _msa_path,
    load_openfold_runner,
    select_proteins,
)


def _fmt(result: dict) -> str:
    return (
        f"plDDT={result['plddt']:.2f}  "
        f"featurize={result['_t_featurize_s']:.3f}s  "
        f"inference={result['_t_inference_s']:.3f}s  "
        f"post={result['_t_post_s']:.3f}s  "
        f"total={result['_t_total_s']:.3f}s"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="One-protein OpenFold sanity check.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--tag", default=None, help="Specific protein tag; default = first foldable.")
    p.add_argument("--stage", type=Path, default=None)
    args = p.parse_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    fasta = stage / "proteins_50k.fasta"
    if not fasta.exists():
        print(f"missing {fasta} — stage the dataset first")
        return 2

    records = select_proteins(read_fasta(fasta), max_len=args.max_len, warm=10_000)
    if args.tag is not None:
        records = [r for r in records if r.header.split()[0] == args.tag]

    # Pick the first protein whose precomputed alignments resolve.
    target = next((r for r in records if _msa_path(stage, r) is not None), None)
    if target is None:
        print(
            "no foldable protein found (need a sequence ≤ "
            f"{args.max_len} aa with alignments under {stage / 'msas'}). "
            "Precomputed MSAs must be per-tag dirs `msas/<tag>/...` "
            "(or a single `msas/<tag>.a3m`)."
        )
        return 1

    try:
        runner = load_openfold_runner(args.seed)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"setup error: {exc}")
        return 2

    gpu_name, device_count = _device_info()
    tag = target.header.split()[0]
    msa = _msa_path(stage, target)
    print(f"device: {gpu_name} x{device_count}")
    print(f"protein: {tag}  len={len(target.seq)}  alignments={msa}")

    cold = runner.predict(target, msa)
    print(f"  COLD  {_fmt(cold)}")
    warm = runner.predict(target, msa)
    print(f"  WARM  {_fmt(warm)}")

    speedup = cold["_t_total_s"] / max(warm["_t_total_s"], 1e-9)
    print(f"cold/warm total speedup: {speedup:.2f}x  (kernel cache + allocator warm-up)")
    if abs(cold["plddt"] - warm["plddt"]) > 1.0:
        print(
            f"WARN: plDDT drifted between passes ({cold['plddt']:.2f} -> "
            f"{warm['plddt']:.2f}) — check determinism/seeding."
        )
    print("PASS: OpenFold folded one protein end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
