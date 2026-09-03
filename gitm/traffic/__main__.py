"""CLI for the traffic library: describe a trace, emit a replay, validate, selftest.

    python -m gitm.traffic --selftest
    python -m gitm.traffic --describe burstgpt BurstGPT_1.csv
    python -m gitm.traffic --replay mooncake trace.jsonl --out replay.jsonl --model Qwen/...
    python -m gitm.traffic --sweep burstgpt BurstGPT_1.csv
    python -m gitm.traffic --replay mooncake t.jsonl --fire --model Qwen/... --result-dir runs/
    python -m gitm.traffic --gui

Everything except ``--fire`` is CPU-only and needs no vLLM. ``--replay`` writes
the file, validates the round trip and prints the ``vllm bench serve`` command;
``--fire`` then runs it, which needs vLLM >= 0.23.0 and a server. Without
``--fire`` the command is printed and not run, because this box may have neither.
"""

from __future__ import annotations

import argparse
import sys

from gitm._banner import add_banner_argument, show_banner
from gitm.traffic._selftest import run_all
from gitm.traffic.adapters import ADAPTERS
from gitm.traffic.parameterize import fit, grid
from gitm.traffic.regime import Regime, SourceKind
from gitm.traffic.replay import VLLM_MIN_VERSION, read_timed_trace, write_timed_trace
from gitm.traffic.runner import VllmUnavailable, run_replay
from gitm.traffic.validate import REPLAY_THRESHOLDS, compare


def main(argv: list[str] | None = None) -> int:
    # The validation render uses block characters; a cp1252 console would raise
    # on them mid-report rather than at the start.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="python -m gitm.traffic")
    add_banner_argument(p)
    p.add_argument("--selftest", action="store_true", help="run every check and exit")
    p.add_argument("--describe", nargs=2, metavar=("ADAPTER", "PATH"))
    p.add_argument("--replay", nargs=2, metavar=("ADAPTER", "PATH"))
    p.add_argument("--sweep", nargs=2, metavar=("ADAPTER", "PATH"))
    p.add_argument("--gui", action="store_true",
                   help="serve the localhost viewer (read-only; 127.0.0.1 only)")
    p.add_argument("--gui-port", type=int, default=8765)
    p.add_argument("--gui-root", default=None,
                   help="directory traces may be read from (default: the committed fixtures)")
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.add_argument("--fire", action="store_true",
                   help="after --replay, actually run the command (needs vLLM and a server)")
    p.add_argument("--result-dir", default=None,
                   help="where --fire saves bench serve's result JSON")
    p.add_argument("--dry-run", action="store_true",
                   help="with --fire: build and check everything, launch nothing")
    p.add_argument("--out", default="replay.jsonl")
    p.add_argument("--model", default="MODEL")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer id; needed when the served model name is not "
                        "resolvable on HuggingFace (a stub, or --served-model-name)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--kind", default="production", choices=[k.value for k in SourceKind])
    a = p.parse_args(argv)
    show_banner(suppressed=a.no_banner)

    if a.selftest:
        return run_all()

    if a.gui:
        from pathlib import Path

        from gitm.traffic.gui import serve

        return serve(port=a.gui_port,
                     root=Path(a.gui_root) if a.gui_root else None,
                     open_browser=not a.no_open)

    spec = a.describe or a.replay or a.sweep
    if spec is None:
        p.print_help()
        return 2
    adapter, path = spec
    if adapter not in ADAPTERS:
        p.error(f"unknown adapter {adapter!r}; known: {', '.join(sorted(ADAPTERS))}")
    trace = ADAPTERS[adapter](path, max_rows=a.max_rows)
    regime = Regime.from_trace(trace, source_kind=SourceKind(a.kind))

    print(trace.meta.summary())
    print(f"regime: {regime.summary()}")
    for note in trace.meta.notes:
        print(f"  note: {note}")

    if a.replay:
        plan = write_timed_trace(trace, a.out)
        print(f"\nwrote {plan.path} ({plan.requests} requests, "
              f"{plan.chunk_hash_size}-token blocks)")
        for note in plan.notes:
            print(f"  note: {note}")
        report = compare(trace, read_timed_trace(a.out), thresholds=REPLAY_THRESHOLDS)
        print()
        print(report.render())
        # not `argv` — that is main()'s own parameter, and shadowing it here reads
        # like a bug even though parse_args has already run.
        cmd = plan.bench_serve_argv(model=a.model, base_url=a.base_url,
                                    tokenizer=a.tokenizer)
        print("\n" + " ".join(cmd))
        if not a.fire:
            print(f"\n(not run — pass --fire to launch it; needs vLLM >= "
                  f"{VLLM_MIN_VERSION} and a server at {a.base_url})")
            return 0 if report.passed else 1

        # The version guard runs inside run_replay, before anything launches, so
        # a too-old vLLM is a sentence here rather than an argparse error there.
        try:
            res = run_replay(plan, model=a.model, base_url=a.base_url,
                             result_dir=a.result_dir, regime=regime,
                             tokenizer=a.tokenizer, dry_run=a.dry_run)
        except VllmUnavailable as exc:
            print(f"\nnot fired: {exc}")
            return 2
        print(f"\n{res.summary()}")
        for note in res.notes:
            print(f"  note: {note}")
        # Seam 3: the result joined to the workload that produced it.
        if res.joined is not None:
            print()
            print(res.joined.render())
        if not res.ok:
            print(f"\nstderr tail:\n{res.stderr_tail}")
        # A result that does not reconcile with its trace is not evidence, so it
        # fails the command even when bench serve itself exited 0.
        reconciled = res.joined is None or res.joined.reconciled
        return 0 if (report.passed and res.ok and reconciled) else 1

    if a.sweep:
        f = fit(trace)
        print(f"\nfitted envelope: {f.rate_rps:.4f} rps, D={f.burstiness:.2f}, "
              f"span {f.span_s:.0f}s")
        print(f"{'regime label':<48} {'req':>6}  {'rps':>8}  {'D':>6}")
        for sampled, reg in grid(f):
            print(f"{reg.label():<48} {len(sampled):>6}  {reg.rate_rps:>8.3f}  "
                  f"{reg.burstiness:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
