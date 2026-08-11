"""``gitm`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_capture(sub) -> None:
    """``gitm capture serve|attach`` — get a kernel trace out of a vLLM server.

    Two modes rather than one with a flag, because they take different inputs and
    have different blast radii: ``serve`` starts and owns a process, ``attach``
    touches nothing but a marker file next to a server somebody else is running.
    Collapsing them would mean a single command where half the flags are ignored
    depending on the other half, and where ``--dry-run`` means two different things.
    """
    from gitm.serve.vllm import add_serve_arguments

    cap = sub.add_parser(
        "capture",
        help="Capture GPU kernels from a vLLM server (launch one, or attach to a running one).",
    )
    cap.set_defaults(capture_help=cap.print_help)
    modes = cap.add_subparsers(dest="capture_mode")

    serve = modes.add_parser(
        "serve",
        help="Launch `vllm serve` under the collector and capture a window.",
        epilog="Pass a full serve command after `--` to override the pinned experiment.",
    )
    add_serve_arguments(serve)

    attach = modes.add_parser(
        "attach",
        help="Attach to an already-running vLLM server and capture a window.",
        epilog=(
            "The server must have been started with CUDA_INJECTION64_PATH pointing at "
            "libgitm_inject.so — the CUDA driver reads it only at CUDA init, so it "
            "cannot be added to a live process. `gitm capture attach --list` reports "
            "which servers on this box qualify."
        ),
    )
    attach.add_argument("--list", action="store_true",
                        help="List the vLLM servers on this box and whether each can be traced.")
    who = attach.add_mutually_exclusive_group()
    who.add_argument("--pid", type=int, default=None, help="Target PID (the server frontend).")
    who.add_argument("--port", type=int, default=None,
                     help="Resolve the target from whoever is listening on this port.")
    attach.add_argument("--base-url", default=None,
                        help="Server base URL (default http://127.0.0.1:<its own --port>).")
    attach.add_argument("--out", default=None,
                        help="Output dir (default $GITM_SCRATCH/traces/vllm-attach-<ts>).")
    attach.add_argument("--duration", type=float, default=30.0,
                        help="Observe mode: seconds to watch the server's own traffic.")
    attach.add_argument("--requests", type=int, default=0,
                        help="Drive mode: issue this many synthetic requests instead of observing.")
    attach.add_argument("--concurrency", type=int, default=64, help="Drive mode: in-flight requests.")
    attach.add_argument("--input-tokens", type=int, default=1024)
    attach.add_argument("--output-tokens", type=int, default=256)
    attach.add_argument("--seed", type=int, default=42)
    attach.add_argument("--no-ignore-eos", action="store_true",
                        help="Drive mode: let the model stop early.")
    attach.add_argument("--request-timeout", type=float, default=600.0)
    attach.add_argument("--metrics-interval", type=float, default=1.0,
                        help="Seconds between /metrics gauge samples during the window.")
    attach.add_argument("--dry-run", action="store_true",
                        help="Verify the target and stop; never opens a window.")


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

    _add_capture(sub)

    sub.add_parser("doctor", help="Probe environment, GPUs, and data locations.")

    plan_kitti = sub.add_parser(
        "plan-kitti", help="Render the PointPillars execution graph for a known GPU SKU."
    )
    plan_kitti.add_argument("--sku", required=True, help="GPU SKU from the hardware catalogue.")
    plan_kitti.add_argument(
        "--baseline", type=Path, default=None, help="Optional measured baseline JSON to compare."
    )

    analyze = sub.add_parser(
        "analyze",
        help="Ingest customer Nsight/PyTorch profiler dumps into a headroom report.",
    )
    analyze.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Profiler files and/or directories (scanned recursively).",
    )
    analyze.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write the combined customer markdown report here.",
    )
    analyze.add_argument(
        "--sku",
        default=None,
        help="Override GPU SKU label (else read from file metadata).",
    )
    analyze.add_argument(
        "--workload-id",
        default=None,
        help="Override workload id (single recognized input only).",
    )
    analyze.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional device filter: analyze only this device index (default: all devices).",
    )
    analyze.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="Optional machine-readable summary JSON path.",
    )
    analyze.add_argument(
        "--keep-traces",
        type=Path,
        default=None,
        help="Optional directory to write intermediate gitm JSONL traces.",
    )
    analyze.add_argument(
        "--strict",
        action="store_true",
        help="Any per-file failure aborts the run.",
    )

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


def _run_capture(args, serve_argv: list[str] | None) -> int:
    if args.capture_mode == "serve":
        from gitm.serve.vllm import launch_and_capture

        rc, _ = launch_and_capture(args, serve_argv)
        return rc

    from gitm.serve.attach import AttachOptions, attach_and_capture, print_targets

    if args.list:
        return print_targets()

    rc, _ = attach_and_capture(
        AttachOptions(
            pid=args.pid,
            port=args.port,
            base_url=args.base_url,
            out=Path(args.out) if args.out else None,
            duration_s=args.duration,
            requests=args.requests,
            concurrency=args.concurrency,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            seed=args.seed,
            ignore_eos=not args.no_ignore_eos,
            request_timeout=args.request_timeout,
            metrics_interval=args.metrics_interval,
            dry_run=args.dry_run,
        )
    )
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    # Under ``capture``, everything after ``--`` is a serve command to hand to vLLM
    # verbatim, not gitm's own flags. Split before argparse sees it: `vllm serve M
    # --port 9000` shares flag names with this parser, and letting argparse claim them
    # would silently retarget the capture at a port the server was never told about.
    # Scoped to ``capture`` because argparse's own ``--`` (end-of-flags, e.g.
    # `gitm analyze -- ./-weird-name.json`) still has to work everywhere else.
    serve_argv: list[str] | None = None
    if argv[:1] == ["capture"] and "--" in argv:
        i = argv.index("--")
        argv, serve_argv = argv[:i], argv[i + 1:]

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
        summary = result.get("summary")
        if not isinstance(summary, dict):
            summary = {
                "status": "invalid_result",
                "diagnostic": "optimization loop returned no machine-readable summary",
            }
        if args.report is not None:
            report_md = result.get("report_md")
            if not isinstance(report_md, str) or not report_md.strip():
                report_md = (
                    "# Runtime result unavailable\n\n"
                    f"{summary.get('diagnostic', 'optimization loop returned no report')}\n"
                )
            args.report.write_text(report_md, encoding="utf-8")
        else:
            print(json.dumps(summary, indent=2))
        # Only the explicit success state is a shell success. Prediction/candidate
        # refusals and failed A/Bs may still have useful measurement reports, but
        # automation must not read those degraded outcomes as a completed run.
        return 0 if summary.get("status") == "ok" else 3

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

    if args.cmd == "capture":
        if getattr(args, "capture_mode", None) is None:
            # Print help but exit non-zero: `gitm capture` on its own is an incomplete
            # command, and a script that runs it should not read that as success.
            args.capture_help()
            return 2
        return _run_capture(args, serve_argv)

    if args.cmd == "doctor":
        from gitm.doctor import doctor

        report = doctor()
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "plan-kitti":
        from gitm.planner.context import hardware_spec_for, peak_for_sku
        from gitm.planner.kitti_graph import predict_kitti_graph, render_kitti_graph

        peak = peak_for_sku(args.sku)
        if peak is None:
            print(f"prediction refused: GPU SKU {args.sku!r} is not in the hardware catalogue")
            return 3
        measured = None
        if args.baseline is not None:
            try:
                measured = json.loads(args.baseline.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"prediction refused: baseline {args.baseline} is unreadable ({exc})")
                return 3
        graph = predict_kitti_graph(hw=hardware_spec_for(peak))
        print(render_kitti_graph(graph, measured=measured))
        return 0

    if args.cmd == "analyze":
        from gitm.importers.analyze import analyze_paths

        result = analyze_paths(
            args.paths,
            out=args.out,
            sku=args.sku,
            workload_id=args.workload_id,
            device=args.device,
            json_out=args.json_out,
            keep_traces=args.keep_traces,
            strict=args.strict,
        )
        # Brief stdout summary for operators; full prose is in --out.
        print(
            json.dumps(
                {
                    "n_workloads": result.summary.get("n_workloads", 0),
                    "n_failures": result.summary.get("n_failures", 0),
                    "out": str(args.out),
                    "json": str(args.json_out) if args.json_out else None,
                },
                indent=2,
            )
        )
        return 0 if result.workloads else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
