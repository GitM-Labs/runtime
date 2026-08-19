from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gitm.planner.graph import Graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ShardingConfig


def detect_family(cfg: dict[str, Any]) -> str:
    """``"hybrid"`` | ``"sparse_moe"`` | ``"dense"`` for a HuggingFace config."""
    from gitm.planner.hybrid_graph import is_hybrid_moe_config
    from gitm.planner.moe_graph import is_sparse_moe_config

    if is_hybrid_moe_config(cfg):
        return "hybrid"
    if is_sparse_moe_config(cfg):
        return "sparse_moe"
    return "dense"


def spec_from_hf_config(cfg: dict[str, Any], *, name: str | None = None):
    """Build whichever model spec the detected family uses."""
    family = detect_family(cfg)
    if family == "hybrid":
        from gitm.planner.hybrid_graph import spec_from_hf_config as _hybrid

        return _hybrid(cfg, name=name)
    if family == "sparse_moe":
        from gitm.planner.moe_graph import spec_from_hf_config as _sparse

        return _sparse(cfg, name=name)
    raise NotImplementedError(
        "no config reader for the dense family yet — build a ModelSpec directly"
    )


def predict_for_config(
    cfg: dict[str, Any],
    hw: HardwareSpec | None = None,
    batch: BatchConfig | None = None,
    sharding: ShardingConfig | None = None,
    *,
    name: str | None = None,
) -> tuple[Graph, str]:
    """``(graph, family)`` for a checkpoint config.

    Raises
    ------
    NotImplementedError
        For the dense family, which has no config reader. Deliberately an
        exception rather than a silently generic graph: a dense prediction for a
        checkpoint whose shape was never read would produce residuals against a
        model of something else.
    """
    family = detect_family(cfg)
    if family == "hybrid":
        from gitm.planner.hybrid_graph import predict_hybrid_graph
        from gitm.planner.hybrid_graph import spec_from_hf_config as _hybrid

        return predict_hybrid_graph(_hybrid(cfg, name=name), hw, batch, sharding), family
    if family == "sparse_moe":
        from gitm.planner.moe_graph import predict_moe_graph
        from gitm.planner.moe_graph import spec_from_hf_config as _sparse

        return predict_moe_graph(_sparse(cfg, name=name), hw, batch, sharding), family
    raise NotImplementedError(
        f"{name or 'this checkpoint'} is neither a hybrid linear-attention MoE nor a "
        "DeepSeek-V4-class sparse-MoE checkpoint. The dense graph models it, but has "
        "no config reader — construct a ModelSpec and call predict_graph directly."
    )


# ── ``gitm plan``: the CLI front for everything above ───────────────────────
#
# Lives here rather than in its own module because it adds no logic — it
# resolves a model reference to a family (the job of this module), prices it
# against a SKU, and renders the result. A separate module would have had to
# re-import every name below and would have drifted from the dispatch order it
# depends on.

def add_plan_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("model", nargs="?", default=None,
                    help="Catalogue entry name, or a path to a checkpoint config.json.")
    ap.add_argument("--list", action="store_true",
                    help="List catalogue entries and exit.")
    ap.add_argument("--gpu", default=None,
                    help="SKU to price against (H200, B200, A100, ...). "
                         "Default: this box's GPU, else the A100 fallback.")
    ap.add_argument("--batch", type=int, default=1, help="Sequences per decode step.")
    ap.add_argument("--kv-len", type=int, default=4096,
                    help="Tokens already cached when the step runs.")
    ap.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    ap.add_argument("--ep", type=int, default=1, help="Expert-parallel size.")
    ap.add_argument("--dp", type=int, default=1, help="Data-parallel size.")
    ap.add_argument("--sweep", default=None,
                    help="Comma-separated batch sizes to sweep instead of a node table.")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit the graph as JSON rather than a table.")
    return ap


def _hardware(sku: str | None) -> HardwareSpec:
    """Resolve a SKU name, or fall back to whatever this box reports.

    A miss is reported by the caller rather than silently accepted: an unknown
    SKU resolves to the A100 defaults, and an A100's 2.0 TB/s against an H200's
    4.8 TB/s is a 2.4x error on every memory-bound node — which is all of them
    on a decode step.
    """
    from gitm.planner.context import build_planner_context, hardware_spec_for, peak_for_sku

    if sku:
        return hardware_spec_for(peak_for_sku(sku))
    return hardware_spec_for(build_planner_context().peak)


def _load(model: str) -> tuple[Any, str, str]:
    """``(spec, family, provenance_note)`` from a catalogue name or a config path."""
    from gitm.planner.model_catalogue import available, load_entry, load_spec

    p = Path(model)
    if p.suffix == ".json" and p.is_file():
        cfg = json.loads(p.read_text())
        # Family first: the dense reader raises, and the caller wants to decline
        # with a message rather than surface a NotImplementedError.
        family = detect_family(cfg)
        if family == "dense":
            return None, family, "config.json (no provenance)"
        return (spec_from_hf_config(cfg, name=str(p)), family,
                "config.json (no provenance)")

    if model in available() or Path(model).suffix in (".yaml", ".yml"):
        entry = load_entry(model)
        prov = entry.get("provenance", {})
        est = [e.get("field") for e in prov.get("estimated", [])]
        note = f"catalogue; fitted fields: {est or 'none'}"
        return load_spec(model), entry["family"], note

    raise FileNotFoundError(
        f"no catalogue entry or config.json at {model!r}. "
        f"Available entries: {available() or 'none'}"
    )


def _predict(spec, family: str, hw, batch, sharding):
    if family == "hybrid":
        from gitm.planner.hybrid_graph import predict_hybrid_graph

        return predict_hybrid_graph(spec, hw, batch, sharding)
    from gitm.planner.moe_graph import predict_moe_graph

    return predict_moe_graph(spec, hw, batch, sharding)


def _render_table(g, hw: HardwareSpec, spec, family: str, note: str) -> str:
    agg: dict[str, list[float]] = {}
    for n in g.nodes:
        p = n.prediction
        a = agg.setdefault(n.op, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a[0] += 1
        a[1] += p.t_pred_s
        a[2] += p.t_compute_s
        a[3] += p.t_memory_s
        a[4] += p.flops
        a[5] += p.bytes

    total = g.total_pred_s
    ridge = (hw.peak_flops_bf16_per_s / hw.peak_mem_bw_bytes_per_s
             if hw.peak_mem_bw_bytes_per_s else 0.0)

    out = [
        f"model     {getattr(spec, 'name', '?')}  [{family}]",
        f"source    {note}",
        f"hardware  {hw.name}  "
        f"{hw.peak_flops_bf16_per_s / 1e12:.0f} TFLOP/s bf16, "
        f"{hw.peak_mem_bw_bytes_per_s / 1e12:.2f} TB/s",
        f"ridge     {ridge:.0f} FLOP/byte — a node below this is memory-bound",
        "",
        f"  {'op':24s} {'xN':>4s} {'t_pred':>9s} {'share':>7s} "
        f"{'t_comp':>9s} {'t_mem':>9s} {'AI':>7s}  bound",
    ]
    for op, (n, tp, tc, tm, fl, by) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        ai = fl / by if by else 0.0
        bound = "compute" if tc >= tm else "memory"
        out.append(
            f"  {op:24s} {int(n):4d} {tp * 1e3:8.3f}m {tp / total:6.1%} "
            f"{tc * 1e3:8.3f}m {tm * 1e3:8.3f}m {ai:7.1f}  {bound}"
        )

    n_compute = sum(1 for n in g.nodes if n.prediction.bound == "compute")
    out += [
        "",
        f"  floor {total * 1e3:.3f} ms/step   "
        f"{g.batch.batch / total:,.0f} tok/s at batch {g.batch.batch}",
        f"  {len(g.nodes)} nodes, {n_compute} compute-bound",
    ]
    if g.has_unpriced_collectives:
        out.append("  ! collectives unpriced — this SKU has no interconnect bandwidth "
                   "in the catalogue")
    if g.has_fallback_peaks:
        out.append("  ! priced against fallback peaks — the ceiling is low in a "
                   "known direction")
    out.append("")
    out.append("  This is a floor at vendor peak, not a target. A measured kernel is")
    out.append("  slower by whatever the implementation leaves on the table; a residual")
    out.append("  here is a lead, not a defect.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = add_plan_arguments(argparse.ArgumentParser(
        prog="gitm plan",
        description="Predicted roofline floor for a checkpoint, without running it.",
    ))
    args = ap.parse_args(argv)

    from gitm.planner.model_catalogue import available

    if args.list:
        entries = available()
        if not entries:
            print("no catalogue entries found.")
            return 1
        for name in entries:
            from gitm.planner.model_catalogue import load_entry

            e = load_entry(name)
            print(f"{name:24s} [{e['family']}]  {e.get('name', '')}")
        return 0

    if not args.model:
        ap.error("a model is required (a catalogue name or a config.json path); "
                 "use --list to see catalogue entries")

    try:
        spec, family, note = _load(args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"cannot plan: {e}")
        return 2
    if family == "dense":
        print(f"cannot plan: {args.model} resolves to the dense family, which has no "
              "config reader. Build a ModelSpec and call predict_graph directly.")
        return 2

    hw = _hardware(args.gpu)
    if args.gpu and args.gpu.lower() not in hw.name.lower():
        print(f"warning: --gpu {args.gpu!r} is not in the catalogue; priced against "
              f"{hw.name}, whose bandwidth may differ by more than 2x.")

    sharding = ShardingConfig(tp=args.tp, ep=args.ep, dp=args.dp)

    if args.sweep:
        try:
            sizes = [int(s) for s in args.sweep.split(",") if s.strip()]
        except ValueError:
            print(f"cannot parse --sweep {args.sweep!r} as comma-separated integers")
            return 2
        print(f"{getattr(spec, 'name', '?')} [{family}] on {hw.name}, "
              f"kv_len={args.kv_len}, TP={args.tp} EP={args.ep}")
        print(f"  {'batch':>7s} {'ms/step':>10s} {'tok/s':>12s} {'compute-bound':>14s}")
        for b in sizes:
            g = _predict(spec, family, hw, BatchConfig(batch=b, kv_cache_len=args.kv_len),
                         sharding)
            cb = sum(1 for n in g.nodes if n.prediction.bound == "compute")
            print(f"  {b:7d} {g.total_pred_s * 1e3:9.3f} "
                  f"{b / g.total_pred_s:12,.0f} {cb:9d}/{len(g.nodes)}")
        return 0

    batch = BatchConfig(batch=args.batch, kv_cache_len=args.kv_len)
    try:
        g = _predict(spec, family, hw, batch, sharding)
    except ValueError as e:
        print(f"cannot plan: {e}")
        return 2

    if args.as_json:
        print(json.dumps({
            "model": getattr(spec, "name", None),
            "family": family,
            "hardware": hw.name,
            "sharding": {"tp": args.tp, "ep": args.ep, "dp": args.dp},
            "batch": {"batch": args.batch, "kv_cache_len": args.kv_len},
            "total_pred_s": g.total_pred_s,
            "has_unpriced_collectives": g.has_unpriced_collectives,
            "has_fallback_peaks": g.has_fallback_peaks,
            "nodes": [
                {
                    "op": n.op, "layer": n.layer,
                    "t_pred_s": n.prediction.t_pred_s,
                    "t_compute_s": n.prediction.t_compute_s,
                    "t_memory_s": n.prediction.t_memory_s,
                    "bound": n.prediction.bound,
                    "flops": n.prediction.flops,
                    "bytes": n.prediction.bytes,
                    "estimated": n.prediction.estimated,
                }
                for n in g.nodes
            ],
        }, indent=2))
        return 0

    print(_render_table(g, hw, spec, family, note))
    return 0
