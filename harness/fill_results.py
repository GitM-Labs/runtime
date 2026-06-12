"""Auto-fill benchmarks/kitti/spec.md and results.md from baseline JSON outputs.

Run after all baseline seeds complete:

    python harness/fill_results.py
    # or with explicit path:
    GITM_DATA_ROOT=/workspace/edge python harness/fill_results.py

Reads $GITM_DATA_ROOT/runs/kitti_baseline_{1..6}.json and replaces every TBD
field in spec.md and results.md with measured values. Prints a diff summary
so you can review before committing.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "benchmarks" / "kitti" / "spec.md"
RESULTS_PATH = REPO_ROOT / "benchmarks" / "kitti" / "results.md"


def _load_runs(runs_dir: Path, n_seeds: int = 6) -> list[dict]:
    runs = []
    for i in range(1, n_seeds + 1):
        p = runs_dir / f"kitti_baseline_{i}.json"
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping", file=sys.stderr)
            continue
        with p.open() as f:
            runs.append(json.load(f))
    return runs


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _spread(fps_vals: list[float]) -> float:
    if len(fps_vals) < 2:
        return 0.0
    return (max(fps_vals) - min(fps_vals)) / max(fps_vals) * 100


def build_summary(runs: list[dict]) -> dict:
    seeds      = [r["seed"] for r in runs]
    fps        = [r["frames_per_second"] for r in runs]
    gpu_pct    = [r["gpu_active_pct"] for r in runs]
    data_pct   = [r["data_stall_pct"] for r in runs]
    sync_pct   = [r["sync_stall_pct"] for r in runs]
    cpu_pct    = [r["cpu_pct"] for r in runs]
    headroom   = [r.get("compute_headroom_pct") for r in runs]
    mem_free   = [r.get("mem_free_at_peak_gb") for r in runs]

    spread_pct = _spread(fps)

    # Per-seed rows
    rows = []
    for r in runs:
        rows.append({
            "seed": r["seed"],
            "fps": r["frames_per_second"],
            "gpu_pct": r["gpu_active_pct"],
            "data_pct": r["data_stall_pct"],
            "sync_pct": r["sync_stall_pct"],
            "cpu_pct": r["cpu_pct"],
            "headroom_pct": r.get("compute_headroom_pct"),
        })

    # Stage spread from first run (representative)
    stage_spread = runs[0].get("stage_spread", {}) if runs else {}

    # Environment from first run
    env = runs[0] if runs else {}

    return {
        "seeds": seeds,
        "rows": rows,
        "mean_fps": _mean(fps),
        "std_fps": _std(fps),
        "mean_gpu": _mean(gpu_pct),
        "std_gpu": _std(gpu_pct),
        "mean_data": _mean(data_pct),
        "std_data": _std(data_pct),
        "mean_sync": _mean(sync_pct),
        "std_sync": _std(sync_pct),
        "mean_cpu": _mean(cpu_pct),
        "std_cpu": _std(cpu_pct),
        "spread_pct": spread_pct,
        "converged": spread_pct <= 2.0,
        "mean_headroom": _mean([h for h in headroom if h is not None]) if any(h is not None for h in headroom) else None,
        "mean_mem_free": _mean([m for m in mem_free if m is not None]) if any(m is not None for m in mem_free) else None,
        "stage_spread": stage_spread,
        "hostname": env.get("hostname", "TBD"),
        "date": env.get("captured_at_iso", "TBD")[:10] if env.get("captured_at_iso") else "TBD",
    }


def _update_spec(path: Path, s: dict) -> str:
    text = path.read_text()

    # Build 6-seed table rows
    def row(r: dict) -> str:
        h = _fmt(r.get("headroom_pct"))
        return (
            f"| {r['seed']}   | {_fmt(r['fps'])} | {_fmt(r['gpu_pct'])}          "
            f"| {_fmt(r['data_pct'])}         | {_fmt(r['sync_pct'])}    "
            f"| {_fmt(r['cpu_pct'])}   | {h}               |"
        )

    seed_rows_new = "\n".join(row(r) for r in s["rows"])
    mean_row = (
        f"| Mean | {_fmt(s['mean_fps'])} | {_fmt(s['mean_gpu'])}          "
        f"| {_fmt(s['mean_data'])}         | {_fmt(s['mean_sync'])}    "
        f"| {_fmt(s['mean_cpu'])}   | {_fmt(s['mean_headroom'])}               |"
    )
    std_row = (
        f"| Stddev | {_fmt(s['std_fps'])} | {_fmt(s['std_gpu'])}          "
        f"| {_fmt(s['std_data'])}         | {_fmt(s['std_sync'])}    "
        f"| {_fmt(s['std_cpu'])}   | --                |"
    )

    # Replace seed rows individually
    for r in s["rows"]:
        seed = r["seed"]
        h = _fmt(r.get("headroom_pct"))
        new_row = (
            f"| {seed}   | {_fmt(r['fps'])} | {_fmt(r['gpu_pct'])}          "
            f"| {_fmt(r['data_pct'])}         | {_fmt(r['sync_pct'])}    "
            f"| {_fmt(r['cpu_pct'])}   | {h}               |"
        )
        text = re.sub(
            rf"\| {seed}\s+\| TBD.*?\|",
            new_row,
            text,
        )

    # Mean + stddev rows
    text = re.sub(r"\| Mean \| TBD.*?\|", mean_row, text)
    text = re.sub(r"\| Stddev \| TBD.*?\|", std_row, text)

    # Spread line
    conv = "YES" if s["converged"] else "NO"
    text = text.replace(
        "6-seed fps spread: TBD -- within 2%: TBD",
        f"6-seed fps spread: {_fmt(s['spread_pct'])}% -- within 2%: {conv}",
    )

    # GPU headroom table rows
    if s["mean_headroom"] is not None:
        text = text.replace(
            "| Compute headroom (100 - mean util) | >35% | TBD |",
            f"| Compute headroom (100 - mean util) | >35% | {_fmt(s['mean_headroom'])}% |",
        )
    if s["mean_mem_free"] is not None:
        text = text.replace(
            "| Memory free at peak | >10 GB | TBD |",
            f"| Memory free at peak | >10 GB | {_fmt(s['mean_mem_free'], 1)} GB |",
        )

    # Stage spread table
    ss = s.get("stage_spread", {})
    for stage_key, stage_label in [
        ("load", "load"),
        ("preprocess", "preprocess (voxelize + H2D)"),
        ("inference", "inference (backbone + BEV + NMS)"),
        ("postprocess", "postprocess (D2H)"),
    ]:
        st = ss.get(stage_key, {})
        if st:
            text = text.replace(
                f"| {stage_label}  | TBD | TBD | TBD | TBD |",
                f"| {stage_label}  | {st.get('mean_ms', 0):.1f} | {st.get('p50_ms', 0):.1f} | {st.get('p95_ms', 0):.1f} | {st.get('mean_pct', 0):.1f}% |",
            )

    # Environment
    text = text.replace("- GPU: TBD", f"- GPU: TBD (check nvidia-smi on pod)")
    text = text.replace("- Driver: TBD", f"- Driver: TBD (check nvidia-smi)")
    text = text.replace("- CUDA: TBD", f"- CUDA: TBD (check nvcc --version)")
    text = text.replace("- Date: TBD", f"- Date: {s['date']}")

    return text


def _update_results(path: Path, s: dict) -> str:
    text = path.read_text()

    for r in s["rows"]:
        seed = r["seed"]
        h = _fmt(r.get("headroom_pct"))
        new_row = (
            f"| Baseline {s['rows'].index(r)+1} | {seed} | {_fmt(r['fps'])} "
            f"| {_fmt(r['gpu_pct'])} | {_fmt(r['data_pct'])} "
            f"| {_fmt(r['sync_pct'])} | {_fmt(r['cpu_pct'])} | {h} |"
        )
        text = re.sub(
            rf"\| Baseline {s['rows'].index(r)+1} \| {seed} \| TBD.*?\|",
            new_row,
            text,
        )

    mean_row = (
        f"| Mean | -- | {_fmt(s['mean_fps'])} | {_fmt(s['mean_gpu'])} "
        f"| {_fmt(s['mean_data'])} | {_fmt(s['mean_sync'])} | {_fmt(s['mean_cpu'])} "
        f"| {_fmt(s['mean_headroom'])} |"
    )
    std_row = (
        f"| Stddev | -- | {_fmt(s['std_fps'])} | {_fmt(s['std_gpu'])} "
        f"| {_fmt(s['std_data'])} | {_fmt(s['std_sync'])} | {_fmt(s['std_cpu'])} | -- |"
    )
    text = re.sub(r"\| Mean \| -- \| TBD.*?\|", mean_row, text)
    text = re.sub(r"\| Stddev \| -- \| TBD.*?\|", std_row, text)

    conv = "YES" if s["converged"] else "NO"
    text = text.replace(
        "6-seed fps spread: TBD% -- within 2%: TBD",
        f"6-seed fps spread: {_fmt(s['spread_pct'])}% -- within 2%: {conv}",
    )
    if s["mean_headroom"] is not None:
        text = text.replace(
            "GPU headroom (compute_headroom_pct = 100 - mean NVML util): TBD%",
            f"GPU headroom (compute_headroom_pct = 100 - mean NVML util): {_fmt(s['mean_headroom'])}%",
        )
    if s["mean_mem_free"] is not None:
        text = text.replace(
            "Memory free at peak: TBD GB",
            f"Memory free at peak: {_fmt(s['mean_mem_free'], 1)} GB",
        )

    text = text.replace(
        "- Machine: RunPod y4xbh7yws2e4tu-64410cb0",
        f"- Machine: RunPod y4xbh7yws2e4tu-64410cb0 ({s['hostname']})",
    )
    text = text.replace("- Date: TBD", f"- Date: {s['date']}")

    return text


def main() -> int:
    data_root = os.environ.get("GITM_DATA_ROOT", "/workspace/edge")
    runs_dir = Path(data_root) / "runs"

    print(f"Reading baselines from {runs_dir} …")
    runs = _load_runs(runs_dir)

    if not runs:
        print("ERROR: no baseline JSON files found.", file=sys.stderr)
        print(f"  Expected: {runs_dir}/kitti_baseline_1.json … kitti_baseline_6.json", file=sys.stderr)
        return 1

    print(f"  Loaded {len(runs)} run(s).")
    s = build_summary(runs)

    print()
    print(f"  fps: {' | '.join(_fmt(r['fps']) for r in s['rows'])}")
    print(f"  spread: {_fmt(s['spread_pct'])}%  ({'PASS' if s['converged'] else 'FAIL'})")
    if s["mean_headroom"] is not None:
        print(f"  compute headroom: {_fmt(s['mean_headroom'])}%")

    # Update spec.md
    spec_new = _update_spec(SPEC_PATH, s)
    SPEC_PATH.write_text(spec_new)
    print(f"\n  Updated: {SPEC_PATH}")

    # Update results.md
    results_new = _update_results(RESULTS_PATH, s)
    RESULTS_PATH.write_text(results_new)
    print(f"  Updated: {RESULTS_PATH}")

    print()
    if s["converged"]:
        print("Convergence PASS. Commit and open PR:")
        print("  git add benchmarks/kitti/manifest.yaml benchmarks/kitti/spec.md benchmarks/kitti/results.md")
        print("  git commit -m 'KITTI: fill measured baseline numbers'")
        print("  git push")
    else:
        print(f"WARNING: convergence FAIL — spread {_fmt(s['spread_pct'])}% > 2%. Flag Adit.")

    return 0 if s["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
