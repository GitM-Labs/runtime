#!/usr/bin/env python3
"""Analysis side of the Kimi/MI355X loop — runs on the laptop, no GPU needed.

    python scripts/kimi_loop/analyze_join.py <results_dir>

<results_dir> is a pulled copy of /mnt/shared/gitm/results/<run>. Writes into
<results_dir>/analysis/:

  overhead_table.md        E1: A/B/C throughput + TTFT/ITL, overhead factors
  measured_vs_predicted.md E2 joined against evidence/.../predicted_sweep.json
  layer_kernels.md         E3/E4: per-layer per-op kernel table from the C-arm
                           capture (the rocTX/'NVTX' correlation deliverable)
  taxonomy.md              E3: kernel-time by class + % unclassified
  e8_delta.md              E8: intervention before/after (when present)

`gitm deviate` (E7) is invoked separately — see analyze.sh — because it wants
one command per captured point with that point's batch/kv-len.

GuideLLM's JSON schema shifts between versions, so metric extraction probes
several key paths and records which one matched; a point where nothing matched
shows as '?' rather than silently dropping out of the table.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREDICTED = REPO / "evidence" / "kimi-mi355x" / "predicted" / "predicted_sweep.json"


def _dig(d, *paths):
    """First value found along any of the dotted paths, else None."""
    for path in paths:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, list):
                try:
                    cur = cur[int(key)]
                    continue
                except (ValueError, IndexError):
                    ok = False
                    break
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return None


def guidellm_point(path: Path) -> dict:
    """throughput/TTFT/ITL out of one GuideLLM result, schema-tolerantly."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"file": path.name, "error": str(e)}
    b = _dig(data, "benchmarks.0") or data
    out = {"file": path.name}
    s = _dig(b, "scheduler_state") or {}
    if s:
        el = (s.get("end_time") or 0) - (s.get("start_time") or 0)
        out["elapsed_s"] = round(el, 1)
        out["successful"] = s.get("successful_requests")
        out["errored"] = s.get("errored_requests")
    for name, paths in {
        "out_tok_s": ("metrics.output_tokens_per_second.successful.mean",
                      "metrics.output_token_throughput.successful.mean",
                      "metrics.tokens_per_second.successful.mean"),
        "ttft_ms_p50": ("metrics.time_to_first_token_ms.successful.median",
                        "metrics.ttft_ms.successful.median",
                        "metrics.time_to_first_token.successful.median"),
        "ttft_ms_p95": ("metrics.time_to_first_token_ms.successful.percentiles.p95",
                        "metrics.ttft_ms.successful.percentiles.p95"),
        "itl_ms_p50": ("metrics.inter_token_latency_ms.successful.median",
                       "metrics.itl_ms.successful.median",
                       "metrics.inter_token_latency.successful.median"),
        "itl_ms_p95": ("metrics.inter_token_latency_ms.successful.percentiles.p95",
                       "metrics.itl_ms.successful.percentiles.p95"),
    }.items():
        v = _dig(b, *paths)
        out[name] = round(v, 2) if isinstance(v, (int, float)) else None
    # Fallback throughput from totals when the metrics tree matched nothing.
    if out.get("out_tok_s") is None and s:
        req = _dig(b, "requests") or {}
        toks = sum((q.get("output_tokens") or 0)
                   for k in ("successful", "incomplete", "errored")
                   for q in (req.get(k) or []))
        if toks and out.get("elapsed_s"):
            out["out_tok_s"] = round(toks / out["elapsed_s"], 1)
    return out


def fmt(v):
    return "?" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def overhead_table(run: Path, out: Path):
    d = run / "e1"
    if not d.is_dir():
        return
    rows = defaultdict(list)
    for f in sorted(d.glob("guidellm_*_r*.json")):
        m = re.match(r"guidellm_([ABC])_r(\d)", f.stem)
        if m:
            rows[m.group(1)].append(guidellm_point(f))
    if not rows:
        return

    def med(arm, key):
        vals = sorted(v[key] for v in rows[arm] if v.get(key) is not None)
        return vals[len(vals) // 2] if vals else None

    lines = ["# E1 — tracer overhead (median of 3 reps, chat c=64)", "",
             "| arm | out tok/s | TTFT p50 (ms) | ITL p50 (ms) | ITL p95 (ms) | overhead (tok/s) |",
             "|---|---|---|---|---|---|"]
    base = med("A", "out_tok_s")
    label = {"A": "A clean", "B": "B GITM-traced", "C": "C GITM+rocTX"}
    for arm in ("A", "B", "C"):
        if arm not in rows:
            continue
        t = med(arm, "out_tok_s")
        ratio = f"{base / t:.2f}x" if base and t else "?"
        lines.append(f"| {label[arm]} | {fmt(t)} | {fmt(med(arm, 'ttft_ms_p50'))} "
                     f"| {fmt(med(arm, 'itl_ms_p50'))} | {fmt(med(arm, 'itl_ms_p95'))} "
                     f"| {ratio} |")
    lines += ["", "H200 reference: CUPTI 1.72x, +NVTX 3.46x. The MI355X figures "
              "above are measured by this run, not assumed."]
    (out / "overhead_table.md").write_text("\n".join(lines) + "\n")
    print(f"overhead_table.md: arms {sorted(rows)}")


def measured_vs_predicted(run: Path, out: Path):
    d = run / "e2"
    if not d.is_dir() or not PREDICTED.is_file():
        return
    pred = {(r["config"], r["concurrency"]): r
            for r in json.loads(PREDICTED.read_text())["records"]}
    lines = ["# E2 — measured vs predicted floor", "",
             "| config | c | meas tok/s | floor tok/s | residual | meas ITL p50 | ITL floor | meas TTFT p50 | TTFT floor |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(d.glob("guidellm_*_c*.json")):
        m = re.match(r"guidellm_(\w+)_c(\d+)$", f.stem)
        if not m or m.group(1) == "B":
            continue
        cfg, c = m.group(1), int(m.group(2))
        p = pred.get((cfg, c))
        g = guidellm_point(f)
        if not p:
            continue
        t, fl = g.get("out_tok_s"), p["throughput_floor_tok_s"]
        resid = f"{fl / t:.2f}x" if t else "?"
        lines.append(
            f"| {cfg} | {c} | {fmt(t)} | {fl:.0f} | {resid} "
            f"| {fmt(g.get('itl_ms_p50'))} | {p['itl_floor_ms']:.2f} "
            f"| {fmt(g.get('ttft_ms_p50'))} | {p['ttft_floor_ms']:.1f} |")
    lines += ["", "residual = floor/measured on throughput: 1.0 is vendor peak; "
              "the gap is what E7 decomposes into host vs device."]
    (out / "measured_vs_predicted.md").write_text("\n".join(lines) + "\n")
    print("measured_vs_predicted.md written")


def _load_trace(capdir: Path):
    traces = sorted(capdir.rglob("*.jsonl"))
    if not traces:
        return []
    return [json.loads(line) for line in open(traces[-1])]


def layer_kernels(run: Path, out: Path):
    d = run / "e3e4" / "capture"
    if not d.is_dir():
        return
    ev = _load_trace(d)
    kernels = [e for e in ev if e.get("kind") == "kernel"]
    if not kernels:
        print("layer_kernels: no kernels in E3/E4 capture")
        return
    per = defaultdict(lambda: {"n": 0, "ns": 0, "names": set()})
    t_total = sum((k.get("end_ns", 0) - k.get("start_ns", 0)) for k in kernels)
    t_attr = 0
    for k in kernels:
        op, layer = k.get("range_op"), k.get("range_layer")
        if not op:
            continue
        t_attr += (k.get("end_ns", 0) - k.get("start_ns", 0))
        key = (layer if layer is not None else "?", op)
        per[key]["n"] += 1
        per[key]["ns"] += (k.get("end_ns", 0) - k.get("start_ns", 0))
        per[key]["names"].add(str(k.get("name", ""))[:60])
    lines = [
        "# E4 — kernels by layer and op (rocTX correlation, arm C)", "",
        f"window: {len(kernels)} kernels, "
        f"{t_attr / max(t_total, 1):.1%} of kernel time attributed", "",
        "| layer | op | kernels | time (ms) | share | kernel names (sample) |",
        "|---|---|---|---|---|---|",
    ]
    for (layer, op), v in sorted(per.items(),
                                 key=lambda kv: (str(kv[0][0]), -kv[1]["ns"])):
        names = "; ".join(sorted(v["names"])[:2])
        lines.append(f"| {layer} | {op} | {v['n']} | {v['ns'] / 1e6:.3f} "
                     f"| {100 * v['ns'] / max(t_total, 1):.1f}% | `{names}` |")
    (out / "layer_kernels.md").write_text("\n".join(lines) + "\n")
    print(f"layer_kernels.md: {len(per)} (layer, op) buckets, "
          f"{t_attr / max(t_total, 1):.1%} attributed")

    # E3 taxonomy: classify every kernel name; unclassified become task cards.
    from gitm.optimizer.deviation import classify_op
    cls = defaultdict(lambda: {"n": 0, "ns": 0})
    unknown = defaultdict(int)
    for k in kernels:
        c = classify_op(str(k.get("name", "")))
        cls[c or "UNCLASSIFIED"]["n"] += 1
        cls[c or "UNCLASSIFIED"]["ns"] += (k.get("end_ns", 0) - k.get("start_ns", 0))
        if c is None:
            unknown[str(k.get("name", ""))[:80]] += 1
    lines = ["# E3 — decode-window taxonomy", "",
             "| class | kernels | time (ms) | share |", "|---|---|---|---|"]
    for c, v in sorted(cls.items(), key=lambda kv: -kv[1]["ns"]):
        lines.append(f"| {c} | {v['n']} | {v['ns'] / 1e6:.3f} "
                     f"| {100 * v['ns'] / max(t_total, 1):.1f}% |")
    if unknown:
        lines += ["", "## Unclassified names (intern vocabulary task cards)", ""]
        lines += [f"- `{n}` x{c}" for n, c in
                  sorted(unknown.items(), key=lambda kv: -kv[1])[:30]]
    (out / "taxonomy.md").write_text("\n".join(lines) + "\n")
    print(f"taxonomy.md: {len(cls)} classes, {len(unknown)} unclassified names")


def e8_delta(run: Path, out: Path):
    d = run / "e8"
    if not d.is_dir():
        return
    before = guidellm_point(d / "guidellm_before.json")
    after = guidellm_point(d / "guidellm_after.json")
    lever = (d / "LEVER").read_text().strip() if (d / "LEVER").is_file() else "?"
    lines = [f"# E8 — intervention: `{lever}` (headline point)", "",
             "| | out tok/s | TTFT p50 | ITL p50 | ITL p95 |", "|---|---|---|---|---|",
             f"| before (B) | {fmt(before.get('out_tok_s'))} | {fmt(before.get('ttft_ms_p50'))} "
             f"| {fmt(before.get('itl_ms_p50'))} | {fmt(before.get('itl_ms_p95'))} |",
             f"| after (I) | {fmt(after.get('out_tok_s'))} | {fmt(after.get('ttft_ms_p50'))} "
             f"| {fmt(after.get('itl_ms_p50'))} | {fmt(after.get('itl_ms_p95'))} |"]
    b, a = before.get("out_tok_s"), after.get("out_tok_s")
    if b and a:
        lines += ["", f"throughput delta: {100 * (a - b) / b:+.1f}%"]
    (out / "e8_delta.md").write_text("\n".join(lines) + "\n")
    print("e8_delta.md written")


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    overhead_table(run, out)
    measured_vs_predicted(run, out)
    layer_kernels(run, out)
    e8_delta(run, out)
    print(f"analysis -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
