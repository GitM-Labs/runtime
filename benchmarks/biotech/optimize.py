"""AF2 (OpenFold) verified A/B — the 'act + prove' half for the biotech workload.

The runtime trace showed AF2 inference is launch-bound and GEMM-heavy, running
``cutlass_80`` (Ampere) tensor-op kernels on a Blackwell card. The candidate
lever is **bf16 inference** (``torch.autocast`` over the forward) — Blackwell
tensor cores are far faster in bf16, and AF2 plDDT typically holds within a point
or two.

Proven the GITM way, adapted to a model where bit-exactness is impossible:

    fold N proteins fp32 (baseline)  →  fold the SAME N in bf16 (candidate)
    →  compare median plDDT  →  keep the candidate ONLY if it is structure-quality
       -equivalent (|Δ plDDT| ≤ tol) AND faster, else roll back to fp32.

So a speedup is never reported on degraded structures: if bf16 moves plDDT past
the tolerance, the gate keeps fp32 and says so. This is the honest AF2 analog of
the HFT byte-identical gate — the tolerance is the only thing that differs,
because every real AF2 perf lever changes numerics.

    python -m benchmarks.biotech.optimize --seed 42 --proteins 8 \
        --stage $GITM_BENCH_STAGE --plddt-tol 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.biotech.fetch import read_fasta
from benchmarks.biotech.harness import _msa_path, load_openfold_runner


def _select(stage: Path, *, max_len: int, n: int) -> list[tuple]:
    """First ``n`` proteins (len ≤ max_len) that have precomputed MSAs, file order."""
    out: list[tuple] = []
    for r in read_fasta(stage / "proteins_50k.fasta"):
        if len(r.seq) <= max_len and (p := _msa_path(stage, r)) is not None:
            out.append((r, p))
            if len(out) >= n:
                break
    return out


def _fold_all(runner, proteins) -> tuple[float, float | None]:
    """Fold every protein, return (structures_per_hour, median_plDDT). Synced so
    GPU time is honest."""
    import torch

    plddts: list[float] = []
    t0 = time.perf_counter()
    for r, msa in proteins:
        out = runner.predict(r, msa)
        if "plddt" in out:
            plddts.append(float(out["plddt"]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    sph = len(proteins) / elapsed * 3600.0
    return sph, (statistics.median(plddts) if plddts else None)


@dataclass
class AF2ABResult:
    baseline_sph: float          # structures/hour, fp32
    candidate_sph: float         # structures/hour, bf16
    speedup: float               # candidate / baseline
    baseline_plddt: float | None
    candidate_plddt: float | None
    plddt_delta: float | None    # candidate - baseline (median)
    plddt_tol: float
    equivalent: bool             # |plddt_delta| <= tol
    kept: str                    # "candidate" | "baseline"

    @property
    def verdict(self) -> str:
        if not self.equivalent:
            return (f"rolled back — bf16 shifted median plDDT by "
                    f"{self.plddt_delta:+.2f} (> ±{self.plddt_tol} tolerance)")
        if self.kept == "candidate":
            return (f"kept candidate — bf16 verified +{(self.speedup - 1) * 100:.1f}% faster, "
                    f"median plDDT {self.plddt_delta:+.2f} (within ±{self.plddt_tol})")
        return f"kept baseline — bf16 within plDDT tolerance but not faster ({self.speedup:.2f}x)"


def optimize_af2(stage: Path, seed: int, *, n_proteins: int, max_len: int,
                 warmup: int, plddt_tol: float) -> AF2ABResult:
    """Run the fp32-vs-bf16 A/B and return a gated verdict."""
    import torch

    proteins = _select(stage, max_len=max_len, n=n_proteins)
    if not proteins:
        raise FileNotFoundError(
            f"no proteins (len<={max_len}) with precomputed MSAs under {stage}/msas"
        )

    baseline = load_openfold_runner(seed)                 # fp32
    candidate = load_openfold_runner(seed, bf16=True)     # bf16 autocast

    # Warmup both outside the timed window (first-call autotune / allocator).
    for r, msa in proteins[: min(warmup, len(proteins))]:
        baseline.predict(r, msa)
        candidate.predict(r, msa)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    base_sph, base_plddt = _fold_all(baseline, proteins)
    cand_sph, cand_plddt = _fold_all(candidate, proteins)

    speedup = cand_sph / base_sph if base_sph else 0.0
    delta = (cand_plddt - base_plddt) if (cand_plddt is not None and base_plddt is not None) else None
    equivalent = delta is not None and abs(delta) <= plddt_tol
    kept = "candidate" if (equivalent and cand_sph > base_sph) else "baseline"

    return AF2ABResult(
        baseline_sph=base_sph,
        candidate_sph=cand_sph,
        speedup=speedup,
        baseline_plddt=base_plddt,
        candidate_plddt=cand_plddt,
        plddt_delta=delta,
        plddt_tol=plddt_tol,
        equivalent=equivalent,
        kept=kept,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AF2 fp32-vs-bf16 verified A/B (OpenFold).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proteins", type=int, default=8, help="proteins to fold per pipeline")
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--plddt-tol", type=float, default=1.5,
                   help="max allowed |median plDDT shift| to accept bf16 (0-100 scale)")
    p.add_argument("--stage", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    r = optimize_af2(stage, args.seed, n_proteins=args.proteins, max_len=args.max_len,
                     warmup=args.warmup, plddt_tol=args.plddt_tol)

    print(f"  baseline (fp32): {r.baseline_sph:,.1f} structures/hour  (median plDDT {r.baseline_plddt})")
    print(f"  candidate (bf16): {r.candidate_sph:,.1f} structures/hour  ({r.speedup:.2f}x, "
          f"median plDDT {r.candidate_plddt})")
    print(f"  VERDICT: {r.verdict}")

    payload = {
        "baseline_structures_per_hour": r.baseline_sph,
        "candidate_structures_per_hour": r.candidate_sph,
        "speedup": r.speedup,
        "baseline_median_plddt": r.baseline_plddt,
        "candidate_median_plddt": r.candidate_plddt,
        "plddt_delta": r.plddt_delta,
        "plddt_tol": r.plddt_tol,
        "plddt_equivalent": r.equivalent,
        "kept": r.kept,
        "verdict": r.verdict,
        "lever": "bf16-autocast-forward",
    }
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        out = args.outdir / f"af2_seed{args.seed}_{args.proteins}p_optimize.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
