"""Shared SMOKE/PLUMBING harness, not the real benchmark harness.

Each benchmark's real work-unit (CUDA LOB kernels / OpenFold / OpenPCDet) is
intern-2's deliverable and needs a GPU. This stand-in lets intern-1's
dataset + reproducibility loop (`make smoke`, `make reproduce`) run end-to-end on
a laptop: it reads the staged dataset, derives a *deterministic* pseudo-metric
and a stall breakdown sampled at the midpoint of the spec's expected bands, and
prints the one-JSON-line harness contract. Replace via each benchmark's
`harness.py` once the real kernels land.

The metric is deterministic per seed with a tiny, bounded per-seed jitter so the
three seeds converge well within the 2 % spread gate — exercising the gate logic
without pretending to be real performance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _config(config_path: Path) -> dict:
    return tomllib.loads(config_path.read_text())


def _mid(band: dict) -> float:
    return (band["lo"] + band["hi"]) / 2.0


def emit(config_path: str | Path, seed: int, work_units: int) -> dict:
    """Build the harness-contract payload for ``work_units`` of work."""
    cfg = _config(Path(config_path))
    es = cfg["expected_stall"]
    gpu = _mid(es["gpu_active"])
    sync = _mid(es["sync"])
    cpu = _mid(es["cpu"])
    data_stall = max(0.0, 1.0 - gpu - sync - cpu)

    # Deterministic base metric with sub-1% per-seed jitter -> passes spread gate.
    jitter = ((seed * 2654435761) % 1000) / 1000.0  # in [0, 1)
    base = 100.0 * max(1, work_units) ** 0.5
    metric = base * (1.0 + 0.005 * (jitter - 0.5))  # +/- 0.25%

    return {
        "metric_value": metric,
        "gpu_name": "smoke-cpu",
        "device_count": 0,
        "harness_commit": "smoke",
        "stall_breakdown": [
            {
                "phase": "all",
                "cpu": round(cpu, 4),
                "data_stall": round(data_stall, 4),
                "sync": round(sync, 4),
                "gpu_active": round(gpu, 4),
                "throughput": metric,
                "wall_clock_s": 1.0,
            }
        ],
    }


def count_work_units(stage_dir: Path, benchmark: str) -> int:
    """Inexpensive size proxy so the metric scales with the staged dataset."""
    if benchmark == "hft":
        return sum(1 for _ in stage_dir.glob("**/part-*.parquet")) or 1
    if benchmark == "biotech":
        return sum(1 for _ in (stage_dir / "msas").glob("*.a3m")) if (stage_dir / "msas").is_dir() else 1
    if benchmark == "edge":
        mani = stage_dir / "manifest.jsonl"
        if mani.is_file():
            return sum(1 for _ in mani.read_text().splitlines()) or 1
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke/plumbing harness (not the real kernels).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", type=Path, default=None,
                   help="Staged dataset dir; defaults to $GITM_BENCH_STAGE or cwd.")
    p.add_argument("--warm", type=int, default=0)  # accepted + ignored
    args, _unknown = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    work_units = count_work_units(stage, args.benchmark)
    print(f"[smoke harness] {args.benchmark} seed={args.seed} work_units={work_units}")
    print(json.dumps(emit(args.config, args.seed, work_units)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
