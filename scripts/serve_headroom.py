"""Headroom insights from a captured `vllm serve` run.

Capture happens on the pod (scripts/serve_capture.py); this reads what it wrote and
answers "how much of this run is recoverable, and where does it live". It is a
composition of modules that already exist, in the one order that makes their output
trustworthy:

  1. **coverage gate first.** Headroom is derived from GPU-busy time, so a trace
     that is missing kernels reports a large, confident, wrong number: absent work
     looks exactly like idle time, and idle time is what the report calls
     recoverable. Under CUDA-graph replay that is the expected failure. So the
     bucket breakdown runs first and its warnings gate everything below.
  2. ceiling distance + stall split (gitm.optimizer.headroom)
  3. per-family intra-kernel ROI (gitm.optimizer.headroom_kernel_rank)
  4. collective exposure (gitm.optimizer.collective_signal) — the TP=2 all-reduce
     that only exists because the model is sharded
  5. the client-side serving summary, so GPU headroom is stated next to the TTFT
     and TPOT it would actually buy

    python scripts/serve_headroom.py ~/traces/qwen-graphs
    python scripts/serve_headroom.py ~/traces/qwen-graphs --compare ~/traces/qwen-eager

``--compare`` is the eager-vs-graphs check: same load, same flags, one with
``--enforce-eager``. If the two disagree on bucket composition, the graphed trace
lost kernels and only the eager one can be reasoned about.

Exit codes: 0 = report written, 1 = the trace cannot support a headroom claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# H100 SXM. The catalogue in gitm/planner/context.py keys on a substring, so the
# full NVML-style name resolves; it is only used when --sku is not given.
DEFAULT_SKU = "NVIDIA H100 80GB HBM3"


def resolve_trace(target: Path) -> Path:
    """Accept either a capture directory or the trace file itself."""
    if target.is_dir():
        candidate = target / "trace.jsonl"
        if not candidate.exists():
            raise SystemExit(f"no trace.jsonl in {target} — is this a serve_capture out dir?")
        return candidate
    return target


def load(trace_path: Path):
    # Private, deliberately: replay.py owns the JSONL<->Trace round trip, and a
    # second reader here would drift from the writer the moment the schema moves.
    from gitm.optimizer.replay import _load_trace_jsonl

    return _load_trace_jsonl(trace_path)


def analyse(trace, sku: str, *, hardware_assumed: bool = False):
    """Everything the report needs, as plain data."""
    from gitm.importers.analyze import _predicted_floor_s, _resolve_peak
    from gitm.optimizer.collective_signal import collective_causes, worst_device_comm
    from gitm.optimizer.headroom import build_headroom
    from gitm.optimizer.headroom_kernel_rank import kernel_roi
    from gitm.optimizer.metrics import compute_metrics
    from gitm.tracer.kernel_taxonomy import summarize_kernels

    kernels = trace.kernels()
    breakdown = summarize_kernels(kernels, window_ns=trace.duration_ns)

    peak, sku_known = _resolve_peak(sku)
    metrics = compute_metrics(trace, peak)
    floor_s = _predicted_floor_s(trace, metrics.busy_fraction)
    headroom = build_headroom(
        trace,
        predicted_floor_s=floor_s,
        metrics=metrics,
        workload=trace.workload_id,
        sku=peak.name,
    )
    roi = kernel_roi((k.name, k.end_ns - k.start_ns) for k in kernels)
    comm = worst_device_comm(trace)
    causes = collective_causes(comm)
    return {
        "breakdown": breakdown,
        "peak": peak,
        "sku_known": sku_known,
        "hardware_assumed": hardware_assumed,
        "metrics": metrics,
        "headroom": headroom,
        "roi": roi,
        "comm": comm,
        "causes": causes,
    }


def fp8_caveat(sku_known: bool, sku: str) -> list[str]:
    """The roofline this report is measured against, stated plainly.

    The SKU catalogue carries one peak per GPU and it is the bf16 dense number. A
    model served in FP8 has roughly twice that arithmetic peak on H100, so any
    "compute-bound" share here is measured against a ceiling the run can exceed.
    It does not affect idle/stall attribution, which is timing-only.
    """
    out = []
    if not sku_known:
        out.append(f"SKU {sku!r} is not in the peak catalogue — a default peak was used, "
                   f"so utilization ratios are indicative only.")
    out.append("Peak FLOPs are the bf16 dense figure. This model is served FP8, whose "
               "peak is ~2x higher on H100, so compute-bound share is conservative. "
               "Idle/stall attribution is timing-only and unaffected.")
    return out


def render(a: dict, *, serving: dict | None, trace) -> str:
    from gitm.optimizer.headroom import render_headroom_md
    from gitm.optimizer.headroom_kernel_rank import render_roi_table
    from gitm.tracer.kernel_taxonomy import format_breakdown

    bd, h, comm = a["breakdown"], a["headroom"], a["comm"]
    out: list[str] = [
        f"# Headroom — {trace.workload_id}",
        "",
        f"trace: {bd.n_kernels} kernels over {trace.duration_ns / 1e9:.2f}s, "
        f"{bd.n_devices} device(s), source={trace.source}",
        "",
        "## Trace composition",
        "",
        "```",
        format_breakdown(bd),
        "```",
        "",
    ]

    problems = bd.warnings()
    if problems:
        out += ["## ⚠ Coverage warnings", "",
                "The headroom numbers below are derived from GPU-busy time. Kernels that "
                "ran but were not recorded are indistinguishable from idle, and idle is "
                "what gets reported as recoverable — so treat everything below as "
                "unreliable until these are resolved.", ""]
        out += [f"- {w}" for w in problems] + [""]

    out += ["## Ceiling distance", "", render_headroom_md(h), ""]

    out += ["## Where the recoverable time sits", "", "```",
            render_roi_table(a["roi"], top=15), "```", ""]

    out += ["## Collectives (TP)", ""]
    if comm is None:
        out += ["No communication kernels in this trace. For a tensor-parallel run that "
                "is itself a finding — either TP is not active or the all-reduces were "
                "not captured.", ""]
    else:
        out += [f"- device {comm.device_id}: comm {comm.comm_ns / 1e6:.1f} ms "
                f"({comm.comm_share_of_busy:.1%} of busy), exposed "
                f"{comm.exposed_comm_ns / 1e6:.1f} ms "
                f"({comm.exposed_comm_share_of_wall:.1%} of wall)", ""]
        # Exposed comm is the part no compute was hiding — the only part a topology
        # change can actually give back.
        out += ([f"- **{c.signal}** (severity {c.severity:.2f}): {c.note}" for c in a["causes"]]
                or ["- no ranked causes: comm is overlapped, not exposed"])
        out += [""]

    if serving:
        out += ["## What it would buy", "",
                f"- TTFT p50/p95: {_ms(serving.get('ttft_p50_s'))} / {_ms(serving.get('ttft_p95_s'))}",
                f"- TPOT p50/p95: {_ms(serving.get('tpot_p50_s'))} / {_ms(serving.get('tpot_p95_s'))}",
                f"- requests: {serving.get('n_requests')} "
                f"({serving.get('n_failed_requests', 'n/a')} failed), "
                f"goodput {serving.get('goodput_rps')}",
                "",
                "Latency is client-side (includes network); vLLM's own histograms are in "
                "metrics_before.txt / metrics_after.txt.", ""]

    caveats = fp8_caveat(a["sku_known"], a["peak"].name) + list(h.caveats)
    if a["hardware_assumed"]:
        caveats.insert(
            0,
            f"No --sku was supplied; pricing assumes {a['peak'].name}. "
            "Utilization ratios are not a detected-hardware measurement.",
        )
    out += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [""]
    return "\n".join(out)


def _ms(v) -> str:
    return f"{v * 1e3:.0f} ms" if isinstance(v, int | float) else "n/a"


def compare(a: dict, b: dict, *, label_a: str, label_b: str) -> str:
    """Side-by-side of the two runs' composition — the graphs-vs-eager check."""
    ba, bb = a["breakdown"], b["breakdown"]
    shares_a = {x.bucket: x.share for x in ba.buckets}
    shares_b = {x.bucket: x.share for x in bb.buckets}
    lines = [
        "", f"## Compare: {label_a} vs {label_b}", "",
        "```",
        f"{'':<12} {label_a[:14]:>14} {label_b[:14]:>14}",
        f"{'kernels':<12} {ba.n_kernels:>14} {bb.n_kernels:>14}",
        f"{'gpu_active':<12} {_pct(ba.gpu_active_share):>14} {_pct(bb.gpu_active_share):>14}",
    ]
    for bucket in sorted(set(shares_a) | set(shares_b)):
        lines.append(f"{bucket:<12} {_pct(shares_a.get(bucket, 0.0)):>14} "
                     f"{_pct(shares_b.get(bucket, 0.0)):>14}")
    lines += ["```", ""]

    # The check this comparison exists for: a bucket present in one run and absent
    # in the other means the run that lost it lost kernels, not work.
    missing = [b_ for b_ in ("moe", "gemm", "attention", "collective")
               if shares_a.get(b_, 0.0) == 0.0 and shares_b.get(b_, 0.0) > 0.0]
    if missing:
        lines += [f"**{label_a} is missing {', '.join(missing)} entirely** while {label_b} "
                  f"has them. That is attribution loss, not a difference in the work — "
                  f"use {label_b} for any per-kernel claim.", ""]
    return "\n".join(lines)


def _pct(v) -> str:
    return f"{v:.1%}" if isinstance(v, int | float) else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headroom insights from a serve capture.")
    ap.add_argument("target", type=Path, help="capture dir (or a trace.jsonl)")
    ap.add_argument("--compare", type=Path, default=None,
                    help="a second capture dir to diff against (e.g. the eager reference)")
    ap.add_argument("--sku", default=None)
    ap.add_argument("--out", type=Path, default=None, help="markdown path (default <dir>/headroom.md)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    trace_path = resolve_trace(args.target)
    trace = load(trace_path)
    sku = args.sku or DEFAULT_SKU
    a = analyse(trace, sku, hardware_assumed=args.sku is None)

    if a["breakdown"].n_kernels == 0:
        print("No kernels in this trace — there is no headroom claim to make.", file=sys.stderr)
        for w in a["breakdown"].warnings():
            print(f"  - {w}", file=sys.stderr)
        return 1

    serving = None
    summary_path = trace_path.parent / "serving_summary.json"
    if summary_path.exists():
        serving = json.loads(summary_path.read_text())

    md = render(a, serving=serving, trace=trace)
    if args.compare:
        other_path = resolve_trace(args.compare)
        b = analyse(load(other_path), sku, hardware_assumed=args.sku is None)
        md += compare(a, b, label_a=args.target.name, label_b=args.compare.name)

    out_md = args.out or (trace_path.parent / "headroom.md")
    out_md.write_text(md)
    (trace_path.parent / "headroom.json").write_text(json.dumps({
        "headroom": asdict(a["headroom"]),
        "breakdown": asdict(a["breakdown"]),
        "warnings": a["breakdown"].warnings(),
        "roi": [asdict(r) for r in a["roi"][:20]],
        "collective_causes": [asdict(c) for c in a["causes"]],
    }, indent=2))

    print(md)
    print(f"\nwrote {out_md} and {trace_path.parent / 'headroom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
