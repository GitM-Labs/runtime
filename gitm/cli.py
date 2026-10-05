"""``gitm`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gitm.optimizer.deviation import add_deviate_arguments
from gitm.planner.registry import add_plan_arguments


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
        "--rerank", choices=("off", "recapture"), default="off",
        help="What to do with what the run learns between candidates. "
             "'recapture' traces the workload again after each applied candidate "
             "and re-ranks what is left against it; 'off' keeps the opening order.",
    )
    run.add_argument(
        "--skip-lever", action="append", metavar="PATTERN", default=None,
        help="Do not try this lever. Matches its name or the knob it sets, with "
             "shell globs and ignoring case (e.g. 'speculative*' or "
             "num_speculative_tokens). Repeatable, or comma-separated. Also read "
             "from GITM_SKIP_LEVERS. A skipped lever is recorded in the run so it "
             "is not mistaken for one that was tried and failed.",
    )
    hist = run.add_mutually_exclusive_group()
    hist.add_argument(
        "--use-history", dest="use_history", action="store_true", default=None,
        help="Rank levers from what previous runs measured on this GPU, without asking.",
    )
    hist.add_argument(
        "--no-history", dest="use_history", action="store_false",
        help="Ignore previous runs' results and score from the catalog. Deletes nothing.",
    )
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

    hist = sub.add_parser(
        "history",
        help="What previous runs measured, per lever.",
        description=(
            "Aggregate every past run's verification.json into a per-lever record: "
            "how often each lever was tried, kept, or rolled back, and by how much. "
            "Read-only — it runs no workload and touches no engine."
        ),
    )
    hist.add_argument(
        "--scratch",
        default=None,
        help="Override $GITM_SCRATCH (local ephemeral run dir; datasets stay in S3).",
    )
    hist.add_argument(
        "--gpu",
        default=None,
        help="Only count runs measured on this GPU SKU (substring must match exactly).",
    )
    hist.add_argument("--top", type=int, default=20, help="Rows to show (default 20).")
    hist.add_argument("--json", action="store_true", help="Emit the records as JSON.")

    add_plan_arguments(sub.add_parser(
        "plan",
        help="Predicted roofline floor for a checkpoint — no GPU, no server needed.",
        epilog="Takes a catalogue entry name (see --list) or a path to a config.json.",
    ))


    add_deviate_arguments(sub.add_parser(
        "deviate",
        help="Subtract a predicted graph from a captured trace.",
        epilog="Streams the trace, so a multi-GB capture is fine.",
    ))

    prop = sub.add_parser(
        "propose",
        help="Emit the next batch of experiments from a baseline capture.",
        epilog="The other half of 'gitm ingest': this writes the arms, that reads "
               "their results back. Every arm is the baseline's own server argv "
               "plus one lever, so the results round-trip by construction.",
    )
    prop.add_argument("--baseline", required=True, metavar="DIR",
                      help="A capture to propose against. Supplies the server argv "
                           "the arms are built from, the model, the load shape, and "
                           "the trace the ranking reads.")
    prop.add_argument("--out", default=None, metavar="FILE",
                      help="Where to write the sweep. Defaults to experiments.json "
                           "inside the baseline capture.")
    prop.add_argument("--max-arms", type=int, default=None,
                      help="Cap the sweep. One arm is one cluster job.")
    prop.add_argument("--top-n", type=int, default=25,
                      help="How many candidates to rank before planning arms "
                           "(default: the whole catalogue).")
    prop.add_argument("--run-id", default=None, help="Run id to record in the sweep.")
    prop.add_argument("--gpu-sku", default=None,
                      help="The GPU the arms will run on, e.g. 'AMD Instinct MI355X'. "
                           "Does two things: screens out levers that do not apply to "
                           "that box, and lets the ranking read what previous sweeps "
                           "measured on it. Without it, both fall back to the "
                           "catalogue's estimates.")
    prop.add_argument("--dtype", default=None,
                      help="Serving dtype, e.g. bf16. A capture does not record it, so "
                           "without this the levers that require a specific dtype are "
                           "not proposed, and the sweep says so.")
    prop.add_argument("--fingerprint", default=None,
                      help="Workload fingerprint to rank history under. Defaults to "
                           "the baseline trace's own.")
    prop.add_argument("--no-history", action="store_true",
                      help="Rank from the catalogue's estimates only, ignoring what "
                           "previous sweeps measured.")
    prop.add_argument("--scratch", default=None,
                      help="Scratch root holding runs/, where ingested results live.")

    ing = sub.add_parser(
        "ingest",
        help="Read cluster harness results into the history the ranking reads.",
        epilog="One baseline and the arms measured against it. Writes "
               "runs/<run-id>/verification.json, which 'gitm run --use-history' "
               "then reads like any local result.",
    )
    ing.add_argument("--baseline", required=True, metavar="DIR",
                     help="The baseline arm's capture directory.")
    ing.add_argument("--candidate", required=True, action="append", metavar="DIR",
                     help="An arm measured against it. Repeat for a sweep.")
    ing.add_argument("--run-id", default=None,
                     help="Run id to file these under. Defaults to a digest of the "
                          "arms, so re-ingesting the same sweep collides instead of "
                          "silently double-counting it.")
    ing.add_argument("--gpu-sku", default=None,
                     help="The GPU these ran on, e.g. 'AMD Instinct MI355X'. "
                          "Required for the ranking to use them: a result measured "
                          "on another box is not evidence about this one, and "
                          "history drops records with no SKU.")
    ing.add_argument("--fingerprint", default=None,
                     help="Workload fingerprint. Defaults to one computed from the "
                          "first candidate's own trace.")
    ing.add_argument("--scratch", default=None,
                     help="Scratch root holding runs/. Defaults to the usual one.")
    ing.add_argument("--dry-run", action="store_true",
                     help="Print what would be written and exit.")

    inst = sub.add_parser(
        "install",
        help="Prepare a CUDA host: driver-matched CUPTI, pinned vLLM/torch, tracer shim.",
    )
    inst.add_argument("--dry-run", action="store_true",
                      help="Print the plan and exit without executing it.")
    inst.add_argument("--skip-stack", action="store_true",
                      help="Do not install or replace vLLM/torch.")
    inst.add_argument("--skip-apt", action="store_true",
                      help="Do not install system build dependencies.")
    inst.add_argument("--with-gpu-extras", action="store_true",
                      help="Additionally install RAPIDS cuDF and CuPy (HFT harness only).")

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


def _ask_use_history(n_runs: int, *, timeout_s: float = 60.0, stream: Any = None,
                     tty: bool | None = None) -> bool:
    """Ask whether this run should be scored from what previous runs measured.

    Lives here, and not in the loop, because a prompt is a property of being run
    by a person at a terminal. ``gitm.optimize`` never touches stdin, so an
    embedded caller cannot be blocked by a question it did not ask for.

    No answer means yes, for two reasons. An unattended run must not sit on a
    prompt forever, and of the two answers using the record is the one that
    discards nothing: declining only skips it for this run. Nothing is deleted
    either way, since every run writes into its own ``runs/<uuid4>/`` and never
    touches another run's export.

    Without a terminal there is nobody to ask, so it takes the same default at
    once rather than waiting out the timeout against a pipe that will not reply.
    """
    import select

    stream = sys.stdin if stream is None else stream
    interactive = tty if tty is not None else bool(getattr(stream, "isatty", lambda: False)())
    if not interactive:
        return True

    print(
        f"\n{n_runs} previous run(s) left measured results."
        "\n  [Y] rank this run from them   [n] ignore them and score from the catalog"
        f"\n  Nothing is deleted either way. No answer within {timeout_s:.0f}s uses them."
        "\n> ",
        end="", flush=True,
    )
    try:
        ready, _, _ = select.select([stream], [], [], timeout_s)
    except (OSError, ValueError):  # not a selectable stream
        return True
    if not ready:
        print(f"\n  no answer in {timeout_s:.0f}s \u2014 using previous results.")
        return True
    answer = (stream.readline() or "").strip().lower()
    if answer.startswith("n"):
        print("  ignoring previous results for this run; they stay on disk.")
        return False
    return True


def _resolve_use_history(args: Any) -> bool:
    """What ``--use-history`` said, or the answer to the prompt.

    An explicit flag is never second-guessed, which is what keeps scripted and
    scheduled runs deterministic. With no flag and no previous results there is
    nothing to ask about and nothing to rank from.
    """
    if getattr(args, "use_history", None) is not None:
        return bool(args.use_history)
    from gitm._paths import runs_dir
    from gitm.optimizer.history import runs_with_results

    n = runs_with_results(runs_dir(args.scratch))
    return _ask_use_history(n) if n else False


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


def _run_propose(args) -> int:
    """Emit the arms for the next batch, from a baseline capture.

    The baseline is a capture rather than a pile of flags because every input
    this needs is already in one: the server argv the arms are built from, the
    model and load shape they must share to be an A/B, and the trace the ranking
    reads. Asking for them separately would be asking the operator to restate
    what the capture already says, with a chance of disagreeing with it.
    """
    from gitm._paths import runs_dir
    from gitm.agents.policy import Policy, select_interventions
    from gitm.kernels.library import load_library
    from gitm.optimizer.experiment_specs import plan_arms, write_experiments
    from gitm.optimizer.harness_results import CaptureError, fingerprint_of, read_capture
    from gitm.optimizer.history import load_history
    from gitm.optimizer.preconditions import GateContext
    from gitm.optimizer.replay import _load_trace_jsonl

    try:
        baseline = read_capture(args.baseline)
    except CaptureError as exc:
        print(f"gitm propose: {exc}", file=sys.stderr)
        return 2
    if baseline.trace_path is None or not baseline.trace_path.exists():
        print(f"gitm propose: {Path(args.baseline).name} has no trace, so there is "
              "nothing to rank against. Capture it with tracing on.", file=sys.stderr)
        return 2
    if not baseline.load:
        # Every arm must carry the baseline's load shape or `gitm ingest` refuses
        # it as not an A/B. Substituting a default would commission a sweep whose
        # results the return path then rejects wholesale.
        print(f"gitm propose: {Path(args.baseline).name} records no load shape, so "
              "the arms would have nothing comparable to run. Re-capture the "
              "baseline through 'gitm capture serve'.", file=sys.stderr)
        return 2

    trace = _load_trace_jsonl(baseline.trace_path)
    library = load_library(workload="vllm-decode")

    # Which box these arms will run on decides two separate things, and without
    # it both fall back to the catalogue: whether a lever applies at all, and
    # what previous sweeps measured for it here.
    tp = 1
    argv = list(baseline.serve_argv)
    for flag in ("--tensor-parallel-size", "--tp"):
        if flag in argv:
            try:
                tp = int(argv[argv.index(flag) + 1])
            except (IndexError, ValueError):
                pass
    ctx = GateContext(
        workload="vllm-decode", hardware=args.gpu_sku, dtype=args.dtype,
        num_gpus=tp, has_collective=tp > 1, has_interconnect=tp > 1,
    )

    fingerprint = args.fingerprint
    if fingerprint is None:
        try:
            fingerprint = fingerprint_of(baseline)
        except CaptureError:
            fingerprint = None

    # What earlier sweeps measured on this box, which is the half of the loop
    # that makes this a loop: ranking the next batch from results rather than
    # from the catalogue's estimates again. Needs the GPU, because a result from
    # another box is not evidence about this one.
    history = None
    if not args.no_history and args.gpu_sku:
        history = load_history(runs_dir(args.scratch), gpu_sku=args.gpu_sku,
                               fingerprint=fingerprint)

    # The qualification gate is deliberately open here. These arms run on the
    # cluster behind the harness's own rollback, and refusing to *propose* a
    # high-risk lever is a different decision from refusing to apply one
    # in-process; the sweep records what each arm is for whoever launches it.
    ranked = select_interventions(
        trace, library,
        Policy(require_qualification_commit=True, use_history=history is not None),
        top_n=args.top_n, ctx=ctx, history=history,
        gpu_sku=args.gpu_sku, fingerprint=fingerprint,
    )
    arms, unreachable = plan_arms(baseline.serve_argv, ranked, max_arms=args.max_arms)

    measured = sum(1 for c in ranked if getattr(c, "delta_source", "") == "measured")
    out = Path(args.out) if args.out else Path(args.baseline) / "experiments.json"
    written = write_experiments(
        out, baseline_argv=baseline.serve_argv, arms=arms, unreachable=unreachable,
        served_model=baseline.served_model or "unknown", load=baseline.load,
        run_id=args.run_id, fingerprint=fingerprint, gpu_sku=args.gpu_sku,
        notes=(f"ranked against {baseline.trace_path.name} "
               f"({len(trace.kernels())} kernels); "
               f"{measured} lever(s) scored from measured results, "
               f"the rest from catalogue estimates"),
    )

    print(f"wrote {written}")
    if fingerprint:
        # Also written into the file's ingest command, so an operator following
        # that verbatim files the results under the key this ranking reads.
        print(f"fingerprint: {fingerprint}")
    if history is not None:
        print(f"history   : {history.runs_read} run(s) read, "
              f"{measured} lever(s) scored from measurement")
    elif not args.gpu_sku:
        print("history   : none read (no --gpu-sku), so every arm is ranked from "
              "the catalogue's estimate", file=sys.stderr)
    print(f"{len(arms)} arm(s), {len(unreachable)} lever(s) not reachable "
          f"against this baseline")
    for arm in arms:
        env = f"  env {' '.join(f'{k}={v}' for k, v in arm.env.items())}" if arm.env else ""
        mark = "" if arm.ingestable else "  (result not attributable automatically)"
        print(f"  {arm.lever}{env}{mark}")
    if not arms:
        # Not an error: a baseline that already runs everything rankable is a
        # real answer. But an empty sweep looks like a failure, so say which.
        print("nothing to run: every ranked lever is already in the baseline or "
              "cannot be reached from it. See the file for the reasons.",
              file=sys.stderr)
    return 0


def _run_ingest(args) -> int:
    """Read harness captures into the history the ranking reads.

    This is the loop's return edge. ``runtime-experiment-harness`` runs arms
    across the cluster and leaves a summary and manifest in each one's
    directory; the converter for them has existed and had no caller, so every
    result measured on the cluster was invisible to the ranking that reads
    history. There is nothing to add to the loop itself — the loop drives a
    local engine and never sees a harness directory — so the operator is the
    caller, after a sweep lands.
    """
    from gitm._paths import runs_dir
    from gitm.kernels.library import load_library
    from gitm.optimizer.harness_results import (
        CaptureError,
        read_capture,
        sweep_id,
        write_comparisons,
    )

    if args.run_id is not None and (
        Path(args.run_id).name != args.run_id or args.run_id in {"", ".", ".."}
    ):
        # Appended to the runs directory, so anything that is not a single
        # directory name writes the export outside the history it is meant to
        # join — absent from where the ranking looks, and present somewhere
        # nobody asked for.
        print(f"gitm ingest: --run-id must be a single directory name, not "
              f"{args.run_id!r}", file=sys.stderr)
        return 2

    try:
        baseline = read_capture(args.baseline)
        candidates = [read_capture(c) for c in args.candidate]
    except CaptureError as exc:
        print(f"gitm ingest: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or sweep_id(baseline, candidates)
    out_dir = runs_dir(args.scratch) / run_id

    if args.dry_run:
        # The same checks the real command makes, minus the write. A dry run that
        # predicts a write the command would refuse is worse than no dry run: it
        # is consulted precisely when the operator is unsure.
        print(f"run id   : {run_id}")
        print(f"baseline : {baseline.path}")
        for c in candidates:
            print(f"candidate: {c.path}")
        export = out_dir / "verification.json"
        if export.exists():
            print(f"would refuse: {export} already exists", file=sys.stderr)
            return 1
        try:
            write_comparisons(baseline, candidates, out_dir=out_dir,
                              library=load_library(), gpu_sku=args.gpu_sku,
                              fingerprint=args.fingerprint, run_id=run_id,
                              dry_run=True)
        except CaptureError as exc:
            print(f"would refuse: {exc}", file=sys.stderr)
            return 1
        print(f"would write: {export}")
        if not args.gpu_sku:
            print("note     : no --gpu-sku, so the ranking will not read these")
        return 0

    try:
        written = write_comparisons(
            baseline, candidates, out_dir=out_dir, library=load_library(),
            gpu_sku=args.gpu_sku, fingerprint=args.fingerprint, run_id=run_id,
        )
    except CaptureError as exc:
        print(f"gitm ingest: {exc}", file=sys.stderr)
        return 1

    refused = out_dir / "ingest_refused.json"
    print(f"wrote {written}")
    if refused.exists():
        print(f"some arms could not be compared, see {refused}", file=sys.stderr)
    if not args.gpu_sku:
        # Not an error: the export is still correct and still readable by hand.
        # But history filters on the SKU, so without one these records exist and
        # are never read, which is the quieter failure of the two.
        print("warning: no --gpu-sku given, so 'gitm run --use-history' will "
              "filter these out", file=sys.stderr)
    return 0


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


def _warn_degraded(summary: dict, run_dir: str | None) -> None:
    """One stderr line when the run fell back anywhere.

    With ``--report`` the summary JSON is never printed, so without this a run
    whose residuals were scored against a default graph finishes looking like
    any other.
    """
    deg = summary.get("degradations") or {}
    if not deg.get("n"):
        return
    parts = []
    if deg.get("unreliable"):
        parts.append("unreliable: " + ", ".join(deg["unreliable"]))
    if deg.get("approximate"):
        parts.append("approximate: " + ", ".join(deg["approximate"]))
    where = f"; see {run_dir}/degradations.json" if run_dir else ""
    print(f"gitm: run degraded ({'; '.join(parts)}){where}", file=sys.stderr)


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
        # Asked here, before the loop starts any capture, so nobody answers a
        # prompt that arrived an hour into a 24h run.
        result = optimize(
            workload=args.workload,
            budget=args.budget,
            target=_parse_target(args.target),
            scratch=args.scratch,
            use_history=_resolve_use_history(args),
            rerank=args.rerank,
            skip_levers=args.skip_lever,
        )
        summary = result.get("summary", {})
        if args.report is not None:
            args.report.write_text(result.get("report_md", ""))
        else:
            print(json.dumps(summary, indent=2))
        _warn_degraded(summary, result.get("run_dir"))
        # Non-zero so automation notices a run that measured nothing (no GPU /
        # CUPTI shim, or the workload never ran) instead of seeing a fake pass.
        if summary.get("status") == "no_data":
            return 3
        if summary.get("engine_lost"):
            # The report is complete for what was tried, but the run stopped
            # early with the engine in an unknown state. A job that exits 0 here
            # reads as a finished run to whatever scheduled it.
            print(f"gitm run: stopped early, the engine could not be restored: "
                  f"{summary['engine_lost']}", file=sys.stderr)
            return 5
        return 0

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

    if args.cmd == "propose":
        return _run_propose(args)

    if args.cmd == "ingest":
        return _run_ingest(args)

    if args.cmd == "deviate":
        from gitm.optimizer.deviation import main as deviate_main

        dev_argv: list[str] = [str(args.trace)]
        for flag, spelling in (("no_graph", "--no-graph"), ("by_phase", "--by-phase"),
                               ("as_json", "--json")):
            if getattr(args, flag, False):
                dev_argv.append(spelling)
        for name, flag in (
            ("model", "--model"), ("gpu", "--gpu"), ("batch", "--batch"),
            ("kv_len", "--kv-len"), ("steps", "--steps"), ("tp", "--tp"), ("ep", "--ep"),
            ("spec_tokens", "--spec-tokens"),
            ("prefill_tokens", "--prefill-tokens"),
            ("prefill_context", "--prefill-context"),
            ("prefill_requests", "--prefill-requests"),
        ):
            val = getattr(args, name, None)
            if val is not None:
                dev_argv += [flag, str(val)]
        return deviate_main(dev_argv)

    if args.cmd == "plan":
        from gitm.planner.registry import main as plan_main

        plan_argv: list[str] = []
        if args.model:
            plan_argv.append(args.model)
        for flag in ("list", "as_json"):
            if getattr(args, flag, False):
                plan_argv.append("--" + ("json" if flag == "as_json" else flag))
        for name, flag in (
            ("gpu", "--gpu"), ("batch", "--batch"), ("kv_len", "--kv-len"),
            ("prefill_tokens", "--prefill-tokens"),
            ("prefill_context", "--prefill-context"),
            ("prefill_requests", "--prefill-requests"),
            ("tp", "--tp"), ("ep", "--ep"), ("dp", "--dp"), ("sweep", "--sweep"),
            # Parsed by add_plan_arguments but dropped here before, so `gitm plan
            # --spec-tokens 3` silently priced a plain decode step.
            ("spec_tokens", "--spec-tokens"), ("acceptance_rate", "--acceptance-rate"),
            ("launch_overhead", "--launch-overhead"),
            # Same omission a second time: added to add_plan_arguments and not
            # here, so `gitm plan --kv-cache-dtype bf16` priced the fp8 cache.
            ("kv_cache_dtype", "--kv-cache-dtype"),
            ("workspace_gb", "--workspace-gb"), ("gpu_mem_util", "--gpu-mem-util"),
        ):
            val = getattr(args, name, None)
            if val is not None:
                plan_argv += [flag, str(val)]
        return plan_main(plan_argv)

    if args.cmd == "install":
        from gitm.install import main as install_main

        argv: list[str] = []
        for flag in ("dry_run", "skip_stack", "skip_apt", "with_gpu_extras"):
            if getattr(args, flag, False):
                argv.append("--" + flag.replace("_", "-"))
        return install_main(argv)

    if args.cmd == "doctor":
        from gitm.doctor import doctor

        report = doctor()
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "history":
        from dataclasses import asdict

        from gitm._paths import runs_dir
        from gitm.optimizer.history import load_history, render_history

        history = load_history(runs_dir(args.scratch), gpu_sku=args.gpu)
        if args.json:
            print(json.dumps({
                "runs_read": history.runs_read,
                "filtered": history.filtered,
                "skipped": history.skipped,
                "records": [
                    {**asdict(r), "conflicted": r.conflicted}
                    for r in history.records.values()
                ],
            }, indent=2))
        else:
            print(render_history(history, top=args.top))
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
