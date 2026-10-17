#!/usr/bin/env python3
"""Measure what the GitM planner does with DeepSeek-V3.2 on SambaNova, against D2's ledger.

Deliverable 3 of the SambaNova lane. Every "expected modeling error" in
``sambanova-schema-gaps.md`` that is a number comes from this script. CPU-only.

    python sambanova_gap_probe.py --runtime /path/to/GitM-Labs/runtime

Needs the runtime checkout (read at 21348b5) and ``v32_sn_ceiling.py`` (deliverable 2)
beside this file. The SN40L HardwareSpec is hand-built from the source base, because
the catalogue has no RDU entry (gap S3): HBM 1.6 TB/s (P-HBM-BW), bf16 640 TF/s
(P-BF16-PEAK), interconnect 0.0 = unknown (U4), launch overhead left at the GPU default.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent

V32_CONFIG = {  # S25 @ a7e62ac0, config.json verbatim
    "architectures": ["DeepseekV32ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 0, "eos_token_id": 1, "ep_size": 1, "first_k_dense_replace": 3, "hidden_act": "silu",
    "hidden_size": 7168, "index_head_dim": 128, "index_n_heads": 64, "index_topk": 2048,
    "initializer_range": 0.02, "intermediate_size": 18432, "kv_lora_rank": 512,
    "max_position_embeddings": 163840, "model_type": "deepseek_v32", "moe_intermediate_size": 2048,
    "moe_layer_freq": 1, "n_group": 8, "n_routed_experts": 256, "n_shared_experts": 1, "norm_topk_prob": True,
    "num_attention_heads": 128, "num_experts_per_tok": 8, "num_hidden_layers": 61, "num_key_value_heads": 128,
    "num_nextn_predict_layers": 1, "q_lora_rank": 1536, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
    "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                            "scale_fmt": "ue8m0", "weight_block_size": [128, 128]},
    "rms_norm_eps": 1e-06,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0,
                     "original_max_position_embeddings": 4096, "type": "yarn"},
    "rope_theta": 10000, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": False,
    "topk_group": 4, "topk_method": "noaux_tc", "torch_dtype": "bfloat16", "transformers_version": "4.44.2",
    "use_cache": True, "v_head_dim": 128, "vocab_size": 129280,
}


def glm_adapter(cfg: dict) -> dict:
    """The smallest change that routes V3.2 through the GLM family: every layer a full
    indexer (V3.2 has no IndexShare, MD-INDEXER), dense prefix from first_k_dense_replace."""
    L, Ld = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
    return dict(cfg, model_type="glm_moe_dsa", indexer_types=["full"] * L,
                mlp_layer_types=["dense"] * Ld + ["sparse"] * (L - Ld))


def summarize(g) -> dict:
    by, t = defaultdict(float), defaultdict(float)
    for n in g.nodes:
        by[n.op] += n.prediction.bytes
        t[n.op] += n.prediction.t_pred_s
    roof = sum(max(n.prediction.t_compute_s, n.prediction.t_memory_s) for n in g.nodes)
    return {
        "nodes": len(g.nodes),
        "bytes": sum(by.values()),
        "total_s": g.total_pred_s,
        "roofline_s": roof,
        "launch_excess_s": g.total_pred_s - roof,
        "launch_nodes": sum(1 for n in g.nodes if n.prediction.bound == "launch"),
        "by_op": dict(by),
        "unpriced": g.has_unpriced_collectives,
        "fallback": g.has_fallback_peaks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, help="path to a GitM-Labs/runtime checkout")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.runtime).resolve()))
    sys.path.insert(0, str(HERE))

    from gitm.planner.context import hardware_spec_for, peak_for_sku
    from gitm.planner.glm_graph import predict_glm_graph
    from gitm.planner.glm_graph import spec_from_hf_config as glm_spec
    from gitm.planner.registry import detect_family, predict_for_config, spec_from_hf_config
    from gitm.planner.roofline import BatchConfig, HardwareSpec, ShardingConfig

    import v32_sn_ceiling as d2

    sn40l = HardwareSpec(
        name="SN40L (hand-built from source base)",
        peak_flops_fp16_per_s=640e12, peak_flops_bf16_per_s=640e12,
        peak_mem_bw_bytes_per_s=1.6e12, interconnect_bw_bytes_per_s=0.0,
    )
    out: dict = {}

    def d2_step(**kw):
        r = d2.ledger(d2.Scenario(**kw))
        return r["traffic_chip_total"], r["step_bound_s"]

    # P1 -- family detection
    fam = detect_family(V32_CONFIG)
    s = spec_from_hf_config(V32_CONFIG, name="v32")
    out["P1_family"] = {
        "family": fam,
        "num_kv_heads": s.num_kv_heads, "head_dim": s.head_dim, "o_lora_rank": s.o_lora_rank,
        "compress_ratios": list(s.compress_ratios), "sliding_window": s.sliding_window, "kv_dtype": s.kv_dtype,
        "has_dense_layer_field": hasattr(s, "first_k_dense_replace"),
    }

    # P2 -- today's path (sparse_moe) vs D2 tp ledger
    rows = []
    for S in (8192, 131072):
        g, _ = predict_for_config(V32_CONFIG, sn40l, BatchConfig(batch=1, kv_cache_len=S), ShardingConfig(tp=16), name="v32")
        p = summarize(g)
        b, t = d2_step(plan="tp", context=S)
        rows.append({"context": S, "planner_bytes": p["bytes"], "planner_s": p["total_s"],
                     "d2_bytes": b, "d2_s": t, "bytes_ratio": p["bytes"] / b, "time_ratio": p["total_s"] / t,
                     "indexer_nodes": sum(1 for n in g.nodes if "index" in n.op),
                     "top_ops": sorted(p["by_op"].items(), key=lambda kv: -kv[1])[:3]})
    out["P2_today_vs_d2"] = rows

    # P3 -- GLM adapter vs D2 tp, both cache-dtype settings the GLM spec can express
    spec = glm_spec(glm_adapter(V32_CONFIG), name="v32-as-glm")
    rows = []
    for S in (8192, 131072):
        b, t = d2_step(plan="tp", context=S)
        for kvd in ("bf16", "fp8"):
            g = predict_glm_graph(replace(spec, kv_dtype=kvd), sn40l, BatchConfig(batch=1, kv_cache_len=S), ShardingConfig(tp=16))
            p = summarize(g)
            rows.append({"context": S, "kv_dtype": kvd, "planner_bytes": p["bytes"], "d2_bytes": b,
                         "bytes_err": p["bytes"] / b - 1, "roofline_s": p["roofline_s"], "total_s": p["total_s"],
                         "launch_excess_s": p["launch_excess_s"], "launch_nodes": p["launch_nodes"], "nodes": p["nodes"],
                         "d2_s": t})
    out["P3_glm_adapter_vs_d2"] = rows

    # P4 -- EP16 vs TP16: the planner vs D2's derived busiest chip
    g_tp = predict_glm_graph(spec, sn40l, BatchConfig(batch=1, kv_cache_len=8192), ShardingConfig(tp=16))
    g_ep = predict_glm_graph(spec, sn40l, BatchConfig(batch=1, kv_cache_len=8192), ShardingConfig(tp=16, ep=16))
    _, t_tp = d2_step(plan="tp")
    _, t_ep = d2_step(plan="ep")
    out["P4_ep_vs_tp"] = {
        "planner_tp_bytes": summarize(g_tp)["bytes"], "planner_ep_bytes": summarize(g_ep)["bytes"],
        "planner_ep_over_tp": summarize(g_ep)["roofline_s"] / summarize(g_tp)["roofline_s"],
        "d2_tp_s": t_tp, "d2_ep_s": t_ep, "d2_ep_over_tp": t_ep / t_tp,
        "d2_busiest_chip_experts_B1": d2.ep_max_load(1, 16)["max"],
        "ep_imbalance_that_would_reproduce_d2": None,  # see note: imbalance scales only the routed term
    }
    # The scalar that would make the planner match: routed-term multiplier.
    r_tp = d2.ledger(d2.Scenario(plan="tp"))
    r_ep = d2.ledger(d2.Scenario(plan="ep"))
    out["P4_ep_vs_tp"]["ep_imbalance_that_would_reproduce_d2"] = r_ep["traffic_chip"]["experts"] / r_tp["traffic_chip"]["experts"]

    # P5 -- unknown SKU
    out["P5_unknown_sku"] = {}
    for sku in ("SN40L", "SN50"):
        hw = hardware_spec_for(peak_for_sku(sku))
        out["P5_unknown_sku"][sku] = {"resolves_to": hw.name, "hbm_bw": hw.peak_mem_bw_bytes_per_s,
                                      "bf16": hw.peak_flops_bf16_per_s}
    hlb = d2.ledger(d2.Scenario())
    out["P5_unknown_sku"]["hlb_at_a100_bw_over_sn40l"] = 1.6e12 / 2.039e12

    # P6 -- sum vs max across resources, where a second tier is active
    r = d2.ledger(d2.Scenario(weights="bf16"))
    tt = r["times_s"]
    out["P6_sum_vs_max"] = {"hbm_s": tt["HBM"], "ddr_s": tt["DDR"], "max_s": max(tt["HBM"], tt["DDR"]),
                            "sum_s": tt["HBM"] + tt["DDR"], "sum_over_max": (tt["HBM"] + tt["DDR"]) / max(tt["HBM"], tt["DDR"])}

    # P7 -- fixed +-40 % residual band vs the one measured DeepSeek point
    meas = 1 / 184.0
    band = 0.4  # gitm/optimizer/invariants.py:22
    rows = []
    for plan in ("ideal", "tp", "ep"):
        b = d2.ledger(d2.Scenario(model="v31", attention="dense_mla", context=10_000, plan=plan))["step_bound_s"]
        rows.append({"reference": plan, "bound_s": b, "measured_over_bound": meas / b,
                     "flagged_as_deviation": abs(meas / b - 1) > band})
    out["P7_band"] = rows

    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    f = out["P1_family"]
    print("P1  V3.2 config -> family", f["family"], "| num_kv_heads", f["num_kv_heads"], "| head_dim", f["head_dim"],
          "| o_lora_rank", f["o_lora_rank"], "| compress_ratios", f["compress_ratios"], "| sliding_window",
          f["sliding_window"], "| kv_dtype", f["kv_dtype"], "| dense-layer field", f["has_dense_layer_field"])
    print("\nP2  today's path, TP16 on SN40L-shaped spec, vs D2 tp ledger")
    for r in out["P2_today_vs_d2"]:
        print(f"    S={r['context']:>6}: planner {r['planner_bytes']/1e9:8.2f} GB {r['planner_s']*1e3:8.2f} ms | "
              f"D2 {r['d2_bytes']/1e9:5.2f} GB {r['d2_s']*1e3:5.2f} ms | bytes x{r['bytes_ratio']:.1f}, time x{r['time_ratio']:.1f} | "
              f"indexer nodes {r['indexer_nodes']} | top {[(k, round(v/1e9,1)) for k, v in r['top_ops']]}")
    print("\nP3  GLM-family adapter, TP16, vs D2 tp ledger")
    for r in out["P3_glm_adapter_vs_d2"]:
        print(f"    S={r['context']:>6} kv={r['kv_dtype']}: planner {r['planner_bytes']/1e9:5.2f} GB vs D2 {r['d2_bytes']/1e9:5.2f} GB "
              f"({r['bytes_err']*100:+.1f} %) | roofline {r['roofline_s']*1e3:.3f} ms + launch floor {r['launch_excess_s']*1e3:.3f} ms "
              f"({r['launch_nodes']} of {r['nodes']} nodes) = {r['total_s']*1e3:.3f} ms vs D2 {r['d2_s']*1e3:.3f} ms")
    p = out["P4_ep_vs_tp"]
    print(f"\nP4  EP16 vs TP16: planner bytes {p['planner_tp_bytes']/1e9:.3f} vs {p['planner_ep_bytes']/1e9:.3f} GB "
          f"(ratio {p['planner_ep_over_tp']:.3f}) | D2 {p['d2_tp_s']*1e3:.3f} vs {p['d2_ep_s']*1e3:.3f} ms (ratio {p['d2_ep_over_tp']:.2f}); "
          f"busiest chip {p['d2_busiest_chip_experts_B1']:.2f} experts at B=1; ep_imbalance needed {p['ep_imbalance_that_would_reproduce_d2']:.2f}")
    print("\nP5  unknown SKU:", {k: v for k, v in out["P5_unknown_sku"].items()})
    q = out["P6_sum_vs_max"]
    print(f"\nP6  bf16 upcast (spill): HBM {q['hbm_s']*1e3:.3f} + DDR {q['ddr_s']*1e3:.3f} ms; max {q['max_s']*1e3:.3f}, sum {q['sum_s']*1e3:.3f} ms (x{q['sum_over_max']:.2f})")
    print("\nP7  measured V3.1 (5.435 ms/token) against each reference, fixed +-40 % band")
    for r in out["P7_band"]:
        print(f"    {r['reference']:>5}: bound {r['bound_s']*1e3:.3f} ms, measured/bound {r['measured_over_bound']:.2f}x, "
              f"flagged {r['flagged_as_deviation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
