"""``gitm`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gitm",
        description="Behavioral compiler and intervention runtime.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the autonomous optimization loop.")
    run.add_argument("--workload", required=True, help="Workload identifier, e.g. vllm-decode.")
    run.add_argument("--budget", default="24h", help="Wall-clock budget, e.g. 24h.")
    run.add_argument(
        "--target",
        default="15%",
        help="Target improvement fraction (15%% or 0.15).",
    )
    run.add_argument(
        "--scratch",
        default=None,
        help="Override $GITM_SCRATCH (local ephemeral run dir; datasets stay in S3).",
    )
    run.add_argument("--report", type=Path, default=None, help="Write report markdown here.")
    # hft-only data-selection flags (mapped onto the GITM_BENCH_* env the
    # workload factory reads). No-ops for other workloads — using them there errors.
    run.add_argument("--seed", type=int, default=None, help="hft: dataset seed.")
    run.add_argument("--stage", type=Path, default=None, help="hft: staged dataset dir.")
    run.add_argument(
        "--max-events",
        type=lambda s: int(s.replace("_", "")),
        default=None,
        help="hft: cap events processed (single-frame).",
    )
    run.add_argument(
        "--stream",
        action="store_true",
        help="hft: stream the sharded dataset in batches (for data too big for one frame).",
    )
    run.add_argument(
        "--shards-per-batch", type=int, default=None, help="hft: shards per streamed batch."
    )
    run.add_argument("--max-shards", type=int, default=None, help="hft: cap shards streamed.")

    replay = sub.add_parser("replay", help="Counterfactual replay of an intervention on a trace.")
    replay.add_argument("trace", type=Path, help="Captured trace file.")
    replay.add_argument("--intervention", type=Path, required=True, help="Intervention spec YAML.")

    apply_cmd = sub.add_parser("apply", help="Apply an intervention spec to the live workload.")
    apply_cmd.add_argument("--intervention", type=Path, required=True)
    apply_cmd.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Target config file to mutate (snapshot+rollback-gated).",
    )
    apply_cmd.add_argument(
        "--min-keep-delta",
        type=float,
        default=0.0,
        help="Roll back if the measured delta is below this fraction.",
    )

    attach = sub.add_parser("attach", help="Attach to a running job (user-space, no root).")
    attach.add_argument("--job", required=True, help="Job identifier to attach to.")
    attach.add_argument(
        "--workload", default=None, help="Optional workload hint, e.g. vllm-decode."
    )
    attach.add_argument(
        "--pid", type=int, default=None, help="Explicit target PID (else resolved locally)."
    )
    attach.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the attach without touching the live process.",
    )

    sub.add_parser("doctor", help="Probe environment, GPUs, and data locations.")

    return p


def _parse_target(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


_HFT_WORKLOADS = {"hft", "hft-lob"}


def _apply_hft_run_flags(args) -> None:
    """Map the hft-only run flags onto the ``GITM_BENCH_*`` env the workload
    factory reads. Errors if they're used with a non-hft workload, where they
    have no meaning (rather than silently ignoring them)."""
    import os

    flags = {
        "GITM_BENCH_SEED": None if args.seed is None else str(args.seed),
        "GITM_BENCH_STAGE": None if args.stage is None else str(args.stage),
        "GITM_BENCH_MAX_EVENTS": None if args.max_events is None else str(args.max_events),
        "GITM_BENCH_SHARDS_PER_BATCH": (
            None if args.shards_per_batch is None else str(args.shards_per_batch)
        ),
        "GITM_BENCH_MAX_SHARDS": None if args.max_shards is None else str(args.max_shards),
        "GITM_BENCH_STREAM": "1" if args.stream else None,
    }
    used = sorted(k for k, v in flags.items() if v is not None)
    if used and args.workload not in _HFT_WORKLOADS:
        raise SystemExit(
            "--seed/--stage/--max-events/--stream/--shards-per-batch/--max-shards apply to "
            f"--workload hft only (got {args.workload!r})"
        )
    for k, v in flags.items():
        if v is not None:
            os.environ[k] = v


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.version:
        from gitm import __version__

        print(__version__)
        return 0

    if args.cmd is None:
        _parser().print_help()
        return 0

    if args.cmd == "run":
        from gitm import optimize

        _apply_hft_run_flags(args)
        result = optimize(
            workload=args.workload,
            budget=args.budget,
            target=_parse_target(args.target),
            scratch=args.scratch,
        )
        summary = result.get("summary", {})
        if args.report is not None:
            args.report.write_text(result.get("report_md", ""))
        else:
            print(json.dumps(summary, indent=2))
        # Non-zero so automation notices a run that measured nothing (no GPU /
        # CUPTI shim, or the workload never ran) instead of seeing a fake pass.
        return 3 if summary.get("status") == "no_data" else 0

    if args.cmd == "replay":
        from gitm.optimizer.replay import predict_delta_from_files

        delta = predict_delta_from_files(args.trace, args.intervention)
        print(json.dumps({"predicted_delta": delta}, indent=2))
        return 0

    if args.cmd == "apply":
        from gitm.optimizer.apply import apply_intervention_from_file

        result = apply_intervention_from_file(
            args.intervention, config=args.config, min_keep_delta=args.min_keep_delta
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "attach":
        from gitm.deploy import attach_job

        plan = attach_job(args.job, workload=args.workload, dry_run=args.dry_run, pid=args.pid)
        print(json.dumps(plan, indent=2))
        # no_target is an operator-actionable miss, not a crash — signal it.
        return 0 if plan.get("status") in {"attached", "planned"} else 4

    if args.cmd == "doctor":
        from gitm.doctor import doctor

        report = doctor()
        print(json.dumps(report, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
