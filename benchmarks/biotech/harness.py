"""Biotech baseline harness — AlphaFold2 inference via OpenFold v1.0.1.

The work unit is one protein, single-seed AF2 inference (5 recycles, 1 model),
length ≤ 384. The metric is ``structures_per_hour`` over a warm window of
proteins. Unlike the HFT harness there is no CPU equivalent of AF2 — OpenFold is
the workload — so this module is framework-integration code that runs only on a
GPU box with OpenFold + weights installed.

To keep the *harness scaffolding* (work-unit iteration, warm-window timing,
contract emission, plDDT aggregation) testable without a GPU, the per-protein
inference is behind a small ``Runner`` seam: :func:`load_openfold_runner` builds
the real one; tests inject a fake. The runner contract is one method —
``predict(record, msa_path) -> {"plddt": float}``.

Prints the one-line harness contract on stdout: ``metric_value`` =
structures/hour, plus device info and median plDDT as an auxiliary sanity field.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Protocol

from benchmarks.biotech.fetch import FastaRecord, read_fasta

OPENFOLD_COMMIT = "v1.0.1"  # pinned; weight hashes pinned in datasets.md
RECYCLES = 5
MODELS = 1


class Runner(Protocol):
    """Per-protein inference seam. The real impl wraps an OpenFold model."""

    name: str

    def predict(self, record: FastaRecord, msa_path: Path | None) -> dict: ...


def load_openfold_runner(seed: int, *, recycles: int = RECYCLES):
    """Build the real OpenFold runner (pinned commit + weights). GPU-only."""
    try:
        import openfold  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - framework absent on laptop
        raise RuntimeError(
            "OpenFold/torch not importable — the biotech harness runs on a GPU "
            "box with OpenFold v1.0.1 installed. (The dataset + reproducibility "
            "loop is exercised via the CPU smoke harness instead.)"
        ) from exc
    # pragma: no cover below — exercised only on the GPU box.
    raise NotImplementedError(  # pragma: no cover
        "Wire OpenFold model construction here: load the pinned weights, set "
        f"recycles={recycles}, single model, seed={seed}, and return a Runner "
        "whose predict() featurizes the MSA and runs inference, returning plDDT."
    )


def _msa_path(stage: Path, record: FastaRecord) -> Path | None:
    tag = record.header.split()[0]
    cand = stage / "msas" / f"{tag}.a3m"
    return cand if cand.exists() else None


def select_proteins(records: list[FastaRecord], *, max_len: int, warm: int) -> list[FastaRecord]:
    """Length-filtered warm window, in file order (deterministic)."""
    eligible = [r for r in records if len(r.seq) <= max_len]
    return eligible[:warm]


def run(stage: Path, seed: int, *, warm: int, max_len: int, runner: Runner) -> dict:
    """Run the warm window through ``runner`` and return the contract payload."""
    fasta = stage / "proteins_50k.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"missing {fasta} — run the biotech dataset pipeline first")

    proteins = select_proteins(read_fasta(fasta), max_len=max_len, warm=warm)
    if not proteins:
        raise RuntimeError(f"no proteins with length <= {max_len} in {fasta}")

    plddts: list[float] = []
    t0 = time.perf_counter()
    for r in proteins:
        result = runner.predict(r, _msa_path(stage, r))
        if "plddt" in result:
            plddts.append(float(result["plddt"]))
    elapsed = max(time.perf_counter() - t0, 1e-9)

    structures_per_hour = len(proteins) / elapsed * 3600.0
    return {
        "metric_value": structures_per_hour,
        "n_structures": len(proteins),
        "median_plddt": statistics.median(plddts) if plddts else None,
        "harness_commit": f"openfold-{OPENFOLD_COMMIT}",
    }


def _device_info() -> tuple[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return "cpu", 0


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(description="Biotech AF2 harness (OpenFold).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-proteins", type=int, default=1000)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--stage", type=Path, default=None)
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    runner = runner or load_openfold_runner(args.seed)
    gpu_name, device_count = _device_info()

    payload = run(stage, args.seed, warm=args.warm_proteins, max_len=args.max_len, runner=runner)
    payload.update({"gpu_name": gpu_name, "device_count": device_count})

    print(f"[biotech harness:{getattr(runner, 'name', '?')}] "
          f"{payload['n_structures']} structures, "
          f"{payload['metric_value']:.1f} structures/hour")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
