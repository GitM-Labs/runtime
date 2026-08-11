"""``python -m gitm.bench`` — the command surface the benchmark Makefiles call.

Subcommands map one-to-one onto the shared systems pieces:

* ``manifest build`` / ``manifest verify`` — freeze / check a dataset.
* ``edge-manifest`` — build the nuScenes+KITTI ``manifest.jsonl``.
* ``run`` — execute one seed's work-unit, emit a ``BaselineRun`` JSON.
* ``profile`` — wrap the work-unit in nsys/rocprof + py-spy/sar.
* ``gate`` — aggregate baseline runs, apply sign-off gates, render ``results.md``.

Every command prints JSON (or writes a file) so it composes in make and CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gitm.bench",
        description="Shared benchmark systems layer (manifest, profile, baseline, gate).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    man = sub.add_parser("manifest", help="Build or verify a dataset sha256 manifest.")
    man_sub = man.add_subparsers(dest="man_cmd", required=True)
    mb = man_sub.add_parser("build", help="Hash a staged dataset into manifest.yaml.")
    mb.add_argument("--root", required=True, type=Path, help="Local staged dataset dir.")
    mb.add_argument("--benchmark", required=True)
    mb.add_argument("--dataset-root", default=None, help="S3 sub-path; defaults to benchmark.")
    mb.add_argument("--out", required=True, type=Path)
    mv = man_sub.add_parser("verify", help="Re-hash a dataset and compare to manifest.")
    mv.add_argument("--manifest", required=True, type=Path)
    mv.add_argument("--root", required=True, type=Path)
    mv.add_argument("--no-extra", action="store_true", help="Ignore unexpected on-disk files.")

    em = sub.add_parser("edge-manifest", help="Build the edge manifest.jsonl from nuScenes+KITTI.")
    em.add_argument("--root", required=True, type=Path, help="Edge dataset root.")
    em.add_argument("--out", required=True, type=Path)
    em.add_argument("--nuscenes-version", default="v1.0-trainval")
    em.add_argument("--kitti-split", default="training")

    rn = sub.add_parser("run", help="Run one seed's work-unit and emit a BaselineRun JSON.")
    rn.add_argument("--config", required=True, type=Path, help="bench.toml")
    rn.add_argument("--seed", required=True, type=int)
    rn.add_argument("--out", required=True, type=Path)
    rn.add_argument("--manifest", default=None, type=Path, help="Override manifest to hash in.")

    pr = sub.add_parser("profile", help="Wrap the work-unit in the vendor profiler + host capture.")
    pr.add_argument("--config", required=True, type=Path)
    pr.add_argument("--seed", required=True, type=int)
    pr.add_argument("--out", required=True, type=Path, help="Profile bundle output dir.")

    bl = sub.add_parser(
        "baseline",
        help="Run every seed, gate, and render results.md — the `make baseline` path.",
    )
    bl.add_argument("--config", required=True, type=Path)
    bl.add_argument("--runs-dir", required=True, type=Path, help="Where to write run JSONs.")
    bl.add_argument("--results", default=None, type=Path, help="Write results.md here.")
    bl.add_argument("--manifest", default=None, type=Path, help="Manifest to hash + cite.")

    rp = sub.add_parser(
        "reproduce",
        help="Reproducibility test: dataset byte-identity + baseline re-run within a time budget.",
    )
    rp.add_argument("--config", required=True, type=Path)
    rp.add_argument("--stage", required=True, type=Path, help="Staged dataset to re-verify.")
    rp.add_argument("--manifest", required=True, type=Path)
    rp.add_argument("--runs-dir", required=True, type=Path)
    rp.add_argument("--limit-minutes", type=float, default=60.0)
    rp.add_argument("--no-metric", action="store_true",
                    help="Check dataset reproducibility only (skip baseline re-run).")

    gt = sub.add_parser("gate", help="Aggregate baseline runs, apply gates, render results.md.")
    gt.add_argument("--config", required=True, type=Path)
    gt.add_argument("runs", nargs="+", type=Path, help="BaselineRun JSON files.")
    gt.add_argument("--results", default=None, type=Path, help="Write results.md here.")
    gt.add_argument("--summary", default=None, type=Path, help="Write summary JSON here.")
    gt.add_argument("--manifest", default=None, type=Path, help="Manifest to cite in results.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.cmd == "manifest":
        return _cmd_manifest(args)
    if args.cmd == "edge-manifest":
        return _cmd_edge_manifest(args)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "profile":
        return _cmd_profile(args)
    if args.cmd == "baseline":
        return _cmd_baseline(args)
    if args.cmd == "reproduce":
        return _cmd_reproduce(args)
    if args.cmd == "gate":
        return _cmd_gate(args)
    return 2


def _cmd_manifest(args) -> int:
    from gitm.bench import manifest as m

    if args.man_cmd == "build":
        man = m.build_manifest(
            args.root, args.benchmark, dataset_root=args.dataset_root
        )
        m.write_manifest(man, args.out)
        print(
            f"wrote {args.out}: {man['file_count']} files, "
            f"{man['total_bytes'] / 1e9:.2f} GB"
        )
        return 0

    # verify
    result = m.verify_manifest(args.manifest, args.root, check_extra=not args.no_extra)
    print(result.summary())
    for label, items in (
        ("missing", result.missing),
        ("mismatched", result.mismatched),
        ("unexpected", result.extra),
    ):
        for it in items[:20]:
            print(f"  {label}: {it}")
    return 0 if result.ok else 1


def _cmd_edge_manifest(args) -> int:
    from gitm.bench import edge_manifest as em

    rows = em.build_manifest(
        args.root,
        nuscenes_version=args.nuscenes_version,
        kitti_split=args.kitti_split,
    )
    em.write_manifest(rows, args.out)
    n_nusc = sum(1 for r in rows if r.source == "nuscenes")
    n_kitti = sum(1 for r in rows if r.source == "kitti")
    print(f"wrote {args.out}: {len(rows)} keyframes ({n_nusc} nuScenes, {n_kitti} KITTI)")
    if not rows:
        print(
            "ERROR: no dataset keyframes were discovered; empty manifest is not usable",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_run(args) -> int:
    from gitm.bench.runner import run_seed, write_run
    from gitm.bench.schema import BenchConfig

    config = BenchConfig.from_toml(args.config)
    run = run_seed(
        config,
        args.seed,
        manifest_path=args.manifest,
        config_dir=Path(args.config).parent,
    )
    write_run(run, args.out)
    print(f"wrote {args.out}: {run.metric}={run.metric_value:.6g} seed={run.seed}")
    return 0


def _cmd_profile(args) -> int:
    from gitm.bench.profile import ProfilerTools, run_profile
    from gitm.bench.runner import build_command
    from gitm.bench.schema import BenchConfig

    config = BenchConfig.from_toml(args.config)
    command = build_command(config, args.seed)
    bundle = run_profile(config, command, args.out)
    print(json.dumps(
        {
            "out_dir": str(bundle.out_dir),
            "complete": bundle.complete,
            "missing_tools": bundle.missing,
            "gpu_csv": str(bundle.gpu_csv) if bundle.gpu_csv else None,
        },
        indent=2,
    ))
    if bundle.missing:
        print(
            f"WARNING: missing capture tools {bundle.missing} — "
            f"detected: {ProfilerTools.detect()}",
            file=sys.stderr,
        )
    return 0 if bundle.complete else 1


def _cmd_baseline(args) -> int:
    from gitm.bench.baseline import aggregate, write_summary
    from gitm.bench.results import render_results, representative_breakdown, write_results
    from gitm.bench.runner import run_seed, write_run
    from gitm.bench.schema import BenchConfig

    config = BenchConfig.from_toml(args.config)
    config_dir = Path(args.config).parent
    manifest = args.manifest or (config_dir / config.dataset.manifest)

    runs = []
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(config.seeds, start=1):
        run = run_seed(config, seed, manifest_path=manifest, config_dir=config_dir)
        out = args.runs_dir / f"{config.name}_baseline_{i}.json"
        write_run(run, out)
        print(f"  seed {seed} -> {out.name}: {run.metric}={run.metric_value:.6g}")
        runs.append(run)

    summary = aggregate(runs, config)
    write_summary(summary, args.runs_dir / f"{config.name}_summary.json")
    print(json.dumps(summary.to_dict(), indent=2))

    if args.results:
        manifest_sha = None
        if Path(manifest).exists():
            from gitm.bench.manifest import manifest_digest

            manifest_sha = manifest_digest(manifest)
        text = render_results(
            summary,
            representative_breakdown(runs),
            gpu_active_ceiling=config.gpu_active_ceiling,
            manifest_sha256=manifest_sha,
        )
        write_results(text, args.results)
        print(f"wrote {args.results}")

    return 0 if summary.passed else 1


def _cmd_reproduce(args) -> int:
    from gitm.bench.reproduce import reproduce
    from gitm.bench.schema import BenchConfig

    config = BenchConfig.from_toml(args.config)
    report = reproduce(
        config,
        stage_dir=args.stage,
        manifest_path=args.manifest,
        runs_dir=args.runs_dir,
        limit_minutes=args.limit_minutes,
        run_metric=not args.no_metric,
    )
    print(json.dumps(report.to_dict(), indent=2))
    print("REPRODUCIBLE ✅" if report.passed else "NOT REPRODUCIBLE ❌")
    return 0 if report.passed else 1


def _cmd_gate(args) -> int:
    from gitm.bench.baseline import aggregate_files, write_summary
    from gitm.bench.results import (
        load_runs_for_breakdown,
        render_results,
        representative_breakdown,
        write_results,
    )
    from gitm.bench.schema import BenchConfig

    config = BenchConfig.from_toml(args.config)
    summary = aggregate_files(args.runs, config)
    print(json.dumps(summary.to_dict(), indent=2))

    if args.summary:
        write_summary(summary, args.summary)

    if args.results:
        runs = load_runs_for_breakdown(args.runs)
        breakdown = representative_breakdown(runs)
        manifest_sha = None
        if args.manifest and Path(args.manifest).exists():
            from gitm.bench.manifest import manifest_digest

            manifest_sha = manifest_digest(args.manifest)
        text = render_results(
            summary,
            breakdown,
            gpu_active_ceiling=config.gpu_active_ceiling,
            manifest_sha256=manifest_sha,
        )
        write_results(text, args.results)
        print(f"wrote {args.results}")

    return 0 if summary.passed else 1


if __name__ == "__main__":
    sys.exit(main())
