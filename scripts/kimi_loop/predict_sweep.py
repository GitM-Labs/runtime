#!/usr/bin/env python3
"""Predicted-graph sweep for Kimi K2.5 on 8x MI355X — stage 1 of the loop.

Runs the planner over the SAME grid scripts/kimi_loop/run_loop.sh executes
(GuideLLM synthetic_text length configs x concurrency), at TP=8 on the MI355X
SKU, and writes per-point predictions:

  * decode-step floor (ms) and its per-op breakdown -> the ITL floor
  * prefill-chunk floor x chunk count            -> the TTFT floor
  * throughput floor (tok/s) at each concurrency

Outputs (evidence/kimi-mi355x/predicted/):
  predicted_sweep.json   one record per (config, concurrency) — the file
                         analyze.sh joins measured GuideLLM/deviation results
                         against, keyed by (config, concurrency)
  predicted_sweep.md     the deck table
  predicted_ops_<cfg>.md per-op decode breakdown at the headline concurrency

The floors are floors: vendor-peak roofline, so measured TTFT/ITL sits above
them by the implementation's residual — that residual is what E7 decomposes.
TTFT is approximated as n_chunks x one representative mid-prompt chunk
(chunked prefill, --max-num-batched-tokens 8192); stated in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.glm_graph import predict_glm_graph
from gitm.planner.model_catalogue import load_spec
from gitm.planner.roofline import BatchConfig, HardwareSpec, ShardingConfig

MODEL = "kimi-k2.5"
SKU = "MI355X"
TP = 8
CHUNK = 8192  # --max-num-batched-tokens in the deployment

# (name, prompt_tokens, output_tokens) — mirror run_loop.sh LENGTH_CONFIGS.
LENGTH_CONFIGS = [
    ("chat", 1024, 256),
    ("rag", 4096, 512),
    ("long", 8192, 1024),
]
CONCURRENCY = [1, 4, 16, 64, 128, 256]
HEADLINE_C = 64

OUT_DIR = Path(__file__).resolve().parents[2] / "evidence" / "kimi-mi355x" / "predicted"


def hardware() -> HardwareSpec:
    peak = peak_for_sku(SKU)
    assert peak is not None, f"SKU {SKU} missing from gitm.planner.context tables"
    return hardware_spec_for(peak)


def main() -> int:
    spec = load_spec(MODEL)
    hw = hardware()
    sh = ShardingConfig(tp=TP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    op_tables = {}
    for name, prompt, output in LENGTH_CONFIGS:
        # Decode step mid-generation: every stream has its prompt cached plus
        # half its output so far.
        kv_mid = prompt + output // 2

        # TTFT floor: chunked prefill of `prompt` tokens. One representative
        # chunk at mid-prompt context, times the chunk count.
        n_chunks = max(1, (prompt + CHUNK - 1) // CHUNK)
        chunk_tokens = min(prompt, CHUNK)
        gp = predict_glm_graph(
            spec, hw,
            BatchConfig(batch=0, kv_cache_len=0, prefill_tokens=chunk_tokens,
                        prefill_context=prompt // 2, prefill_requests=1),
            sh,
        )
        ttft_floor_ms = gp.total_pred_s * n_chunks * 1e3

        for c in CONCURRENCY:
            gd = predict_glm_graph(
                spec, hw, BatchConfig(batch=c, kv_cache_len=kv_mid), sh)
            itl_floor_ms = gd.total_pred_s * 1e3
            records.append({
                "config": name,
                "prompt_tokens": prompt,
                "output_tokens": output,
                "concurrency": c,
                "kv_len_modeled": kv_mid,
                "tp": TP,
                "sku": SKU,
                "decode_step_floor_ms": round(itl_floor_ms, 3),
                "itl_floor_ms": round(itl_floor_ms, 3),
                "ttft_floor_ms": round(ttft_floor_ms, 3),
                "throughput_floor_tok_s": round(c / gd.total_pred_s, 1),
                "prefill_chunks": n_chunks,
            })
            if c == HEADLINE_C:
                per_op = {}
                for node in gd.nodes:
                    d = per_op.setdefault(node.op, {"n": 0, "t_ms": 0.0})
                    d["n"] += 1
                    d["t_ms"] += node.prediction.t_pred_s * 1e3
                op_tables[name] = dict(
                    sorted(per_op.items(), key=lambda kv: -kv[1]["t_ms"]))

    (OUT_DIR / "predicted_sweep.json").write_text(json.dumps({
        "model": MODEL, "sku": SKU, "tp": TP,
        "note": ("Vendor-peak roofline floors. TTFT approximated as n_chunks x "
                 "one mid-prompt chunk. Join measured results on "
                 "(config, concurrency)."),
        "records": records,
    }, indent=2))

    lines = [
        f"# Predicted floors — {MODEL} @ {TP}x{SKU}",
        "",
        "| config | prompt/output | c | ITL floor (ms) | TTFT floor (ms) | floor tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['config']} | {r['prompt_tokens']}/{r['output_tokens']} "
            f"| {r['concurrency']} | {r['itl_floor_ms']:.2f} "
            f"| {r['ttft_floor_ms']:.1f} | {r['throughput_floor_tok_s']:.0f} |")
    (OUT_DIR / "predicted_sweep.md").write_text("\n".join(lines) + "\n")

    for name, ops in op_tables.items():
        t = [f"# Per-op decode floor — {name}, c={HEADLINE_C}", "",
             "| op | xN | t_pred (ms) | share |", "|---|---|---|---|"]
        total = sum(d["t_ms"] for d in ops.values())
        for op, d in ops.items():
            t.append(f"| {op} | {d['n']} | {d['t_ms']:.3f} "
                     f"| {100 * d['t_ms'] / total:.1f}% |")
        (OUT_DIR / f"predicted_ops_{name}.md").write_text("\n".join(t) + "\n")

    print(f"{len(records)} points -> {OUT_DIR}")
    for name, ops in op_tables.items():
        top = next(iter(ops.items()))
        print(f"  {name}: c={HEADLINE_C} top op {top[0]} at "
              f"{top[1]['t_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
