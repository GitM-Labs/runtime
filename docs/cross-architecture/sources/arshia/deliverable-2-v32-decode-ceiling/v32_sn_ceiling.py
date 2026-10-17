#!/usr/bin/env python3
"""DeepSeek-V3.2 decode ceiling on SambaNova SN40L, and a separate SN50 projection.

Deliverable 2 of the SambaNova lane. CPU-only, Python standard library only.

    python v32_sn_ceiling.py check        # reconciliations and self-tests; exits non-zero on failure
    python v32_sn_ceiling.py report       # every table in DESIGN-NOTE-v32-sambanova.md, in order
    python v32_sn_ceiling.py point [opts] # one operating point with its full ledger
    python v32_sn_ceiling.py point --json [opts]

Every input is a ``Cited`` value that carries an evidence class and an ID in
``sn40l-source-base.md`` (v2.1). The classes are
the source base's: DOC, MODEL, VMEAS, 3PMEAS, 2HAND, DERIVED, UNKNOWN. This file
adds ASSUMED for modelling choices that no source fixes. An UNKNOWN input is
``None`` and is never replaced by a default: any term that needs it is reported
as unpriced, with the break-even value at which it would bind.

Three kinds of number come out, and every output row says which one it is:

* **HLB, the hardware lower bound.** The ``ideal`` plan: every byte spread evenly
  over every chip, 100 % of peak bandwidth, perfect overlap between tiers, free
  collectives and free kernel boundaries. No compiled plan can beat it.
* **PCB, a plan-conditional bound.** The same peaks, for a *named* placement
  (``tp`` or ``ep``, defined in ``PLANS``) with the replication and load imbalance
  that placement forces. It is still a bound, but for that plan only.
* **ANCHOR, a measurement-anchored projection.** A PCB ratio applied to the one
  measured DeepSeek decode on SN40L with a stated setup (D-V31-AA). It assumes
  V3.2 is served with the same plan and attainment as V3.1. It is not a bound.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field, replace

GB = 1e9
GiB = 2**30
MiB = 2**20
TB = 1e12


# =============================================================================
# 1. Cited inputs
# =============================================================================


@dataclass(frozen=True)
class Cited:
    value: float | int | None
    cls: str  # DOC | MODEL | VMEAS | 3PMEAS | 2HAND | DERIVED | ASSUMED | UNKNOWN
    src: str  # source-base ID(s)
    note: str = ""


# --- SN40L, per socket (source base §2.1) -----------------------------------
SN40L = {
    "hbm_bw": Cited(1.6e12, "DOC", "P-HBM-BW", "C1 resolved to 1.6 TB/s (S04 Fig 16.4.1, S03, S02)"),
    "hbm_bw_alt": Cited(2.0e12, "DOC", "P-HBM-BW", "S01 text; optimistic sensitivity only (C1)"),
    "hbm_cap": Cited(64 * GiB, "DOC", "P-HBM-CAP", "64 GB read as GiB (4 x 16 GiB HBM2E stacks), as DV-FIT does"),
    "ddr_bw": Cited(100e9, "DOC", "P-DDR-BW", "C2 resolved to 100 GB/s"),
    "ddr_cap": Cited(512e9, "DOC", "P-DDR-CAP", "C3 open: 512 GB (S05) or 1.5 TB (S04); lower value used"),
    "sram_cap": Cited(520 * MiB, "DOC", "P-SRAM-CAP"),
    "sram_bw_lower": Cited(100e12, "DOC", "P-SRAM-BW", "a LOWER bound (> 100 TB/s), so SRAM time from it is an upper bound"),
    "peak_bf16": Cited(640e12, "DOC", "P-BF16-PEAK", "optimistic end of 638-640"),
    "peak_fp8": Cited(None, "UNKNOWN", "P-FP8-PEAK", "no fp8 compute documented; fp8 weights priced at bf16 compute"),
    "p2p_bw": Cited(None, "UNKNOWN", "P-P2P-BW / U4"),
    "p2p_latency": Cited(None, "UNKNOWN", "U4"),
    "node_chips": Cited(16, "DOC", "P-NODE / D-NODE", "one SN40L-16 node per bundle"),
}

# --- SN50, per RDU (source base §2.7) -- kept separate from SN40L ------------
SN50 = {
    "hbm_bw": Cited(1.84e12, "2HAND", "P50-HBM-BW (S27)", "primary S10 paywalled; corroborated by DV-SN50-HBM"),
    "hbm_bw_observed": Cited(1.8e12, "VMEAS", "R50-EP / DV-SN50-HBM", "EP trace plateau, read from figure: peak >= this"),
    "hbm_cap": Cited(64 * GiB, "DOC", "P50-HBM-CAP"),
    "ddr_bw": Cited(100e9, "2HAND", "P50-DDR-BW (S27)", "uncorroborated; matters only when weights spill"),
    "sram_cap": Cited(432e6, "DOC", "P50-SRAM"),
    "sram_bw_lower": Cited(None, "UNKNOWN", "-", "not published for SN50"),
    "peak_bf16": Cited(1600e12, "DOC", "P50-BF16-PEAK"),
    "peak_fp8": Cited(3200e12, "DOC", "P50-FP8-PEAK", "whether decode uses it is unknown (U10); not binding either way"),
    "link_bw_dir": Cited(100e9, "DERIVED", "P50-NET", "200 GB/s bidirectional per 800G link = 100 GB/s per direction"),
    "intra_links": Cited(7, "DOC", "P50-NET", "full mesh inside an 8-RDU node"),
    "inter_links": Cited(2, "DOC", "P50-NET", "400 GB/s bidirectional per RDU between nodes"),
    "p2p_latency": Cited(None, "UNKNOWN", "-", "not published"),
    "node_chips": Cited(8, "DOC", "P50-SCALE"),
}

HW = {"SN40L": SN40L, "SN50": SN50}

# --- DeepSeek-V3.2 checkpoint (S25 @ a7e62ac0; source base §2.5) ------------
CFG = {
    "hidden": Cited(7168, "MODEL", "MD-DIMS"),
    "layers": Cited(61, "MODEL", "MD-LAYERS"),
    "dense_layers": Cited(3, "MODEL", "MD-LAYERS", "first_k_dense_replace"),
    "mtp_layers": Cited(1, "MODEL", "MD-LAYERS", "num_nextn_predict_layers"),
    "dense_inter": Cited(18432, "MODEL", "MD-DIMS"),
    "moe_inter": Cited(2048, "MODEL", "MD-DIMS"),
    "experts": Cited(256, "MODEL", "MD-MOE"),
    "topk": Cited(8, "MODEL", "MD-MOE"),
    "n_group": Cited(8, "MODEL", "MD-MOE"),
    "topk_group": Cited(4, "MODEL", "MD-MOE"),
    "heads": Cited(128, "MODEL", "MD-MLA"),
    "q_lora": Cited(1536, "MODEL", "MD-MLA"),
    "kv_lora": Cited(512, "MODEL", "MD-MLA"),
    "qk_nope": Cited(128, "MODEL", "MD-MLA"),
    "qk_rope": Cited(64, "MODEL", "MD-MLA"),
    "v_head": Cited(128, "MODEL", "MD-MLA"),
    "idx_heads": Cited(64, "MODEL", "MD-INDEXER"),
    "idx_dim": Cited(128, "MODEL", "MD-INDEXER"),
    "idx_topk": Cited(2048, "MODEL", "MD-INDEXER / MD-TOPK"),
    "vocab": Cited(129280, "MODEL", "MD-VOCAB"),
    "block": Cited(128, "MODEL", "MD-QUANT", "fp8 e4m3, 128x128 block scales"),
}
C = {k: v.value for k, v in CFG.items()}

# Published parameter counts by stored dtype, the reconciliation target.
MD_PARAMS = {"F8": 681_408_069_632, "BF16": 3_946_184_704, "F32": 42_667_040}
MD_PARAMS_TOTAL = 685_396_921_376
MD_BYTES = 689_471_107_200

# Cache layout (S29: reference inference code at the pinned commit).
CACHE = {
    # MLA: kv_cache [.., kv_lora] + pe_cache [.., qk_rope], default dtype = bf16.
    "latent_bytes": Cited((512 + 64) * 2, "MODEL", "S29", "bf16 latent + rope key: 1,152 B/token/layer"),
    # Indexer: k_cache [.., 128] float8_e4m3fn + k_scale_cache [.., 128 // 128] float32.
    "index_bytes": Cited(128 * 1 + 4, "MODEL", "S29", "fp8 key + one fp32 scale: 132 B/token/layer"),
}

# --- Measured DeepSeek decode on SN40L (the only calibration points) ---------
ANCHORS = {
    "D-V31-AA": dict(
        tok_s=184.0, context=10_000, model="v31", cls="3PMEAS", src="D-V31-AA (S15, fetched 2026-09-28)",
        setup="DeepSeek V3.1 non-reasoning, 10k input tokens; batch, speculation and precision unstated (U8, U9)",
    ),
    "D-R1-16": dict(
        tok_s=198.0, context=None, model="v31", cls="DEPLOY/VMEAS", src="D-R1-16 (S11, 2025-02-13)",
        setup="DeepSeek-R1 671B on 16 SN40L; context, batch, speculation unstated. R1 = V3.1 architecture (S12)",
    ),
}
ETA_DENSE = Cited((0.76, 0.93), "VMEAS", "M-KL-ROOF", "dense Llama 3.1 decode, % of HBM roofline on SN40L-16 / -8")


# =============================================================================
# 2. Model inventory, reconciled against the checkpoint's own dtype totals
# =============================================================================


@dataclass(frozen=True)
class Tensor:
    name: str
    params: int
    dtype: str  # F8 | BF16 | F32, as stored
    tp: str  # "shard" | "repl" under the tp/ep plans
    group: str


def _blocks(out_f: int, in_f: int) -> int:
    b = C["block"]
    return math.ceil(out_f / b) * math.ceil(in_f / b)


def _fp8(name: str, out_f: int, in_f: int, tp: str, group: str) -> list[Tensor]:
    return [
        Tensor(name, out_f * in_f, "F8", tp, group),
        Tensor(name + ".scale", _blocks(out_f, in_f), "F32", tp, group),
    ]


def attention_layer() -> list[Tensor]:
    h, heads = C["hidden"], C["heads"]
    qk = C["qk_nope"] + C["qk_rope"]
    t = [Tensor("norms", 2 * h + C["q_lora"] + C["kv_lora"], "F32", "repl", "norm")]
    # Latent producers are replicated under head-TP: one latent serves every head.
    t += _fp8("q_a", C["q_lora"], h, "repl", "attn_repl")
    t += _fp8("q_b", heads * qk, C["q_lora"], "shard", "attn")
    t += _fp8("kv_a", C["kv_lora"] + C["qk_rope"], h, "repl", "attn_repl")
    t += _fp8("kv_b", heads * (C["qk_nope"] + C["v_head"]), C["kv_lora"], "shard", "attn")
    t += _fp8("o_proj", h, heads * C["v_head"], "shard", "attn")
    return t


def indexer() -> list[Tensor]:
    # One indexer key per token, shared by all 64 indexer heads (MD-INDEXER-SEM), so
    # every chip that scores must read every key: the tp plan replicates it (as vLLM does).
    h = C["hidden"]
    t = _fp8("idx_wq_b", C["idx_heads"] * C["idx_dim"], C["q_lora"], "repl", "indexer")
    t += _fp8("idx_wk", C["idx_dim"], h, "repl", "indexer")
    t += [Tensor("idx_k_norm", 2 * C["idx_dim"], "F32", "repl", "indexer")]
    # Stored bf16 (forced by the dtype totals); the reference runs it in fp32 (S29).
    t += [Tensor("idx_weights_proj", C["idx_heads"] * h, "BF16", "repl", "indexer")]
    return t


def dense_mlp() -> list[Tensor]:
    h, i = C["hidden"], C["dense_inter"]
    return _fp8("mlp_gate", i, h, "shard", "dense_mlp") + _fp8("mlp_up", i, h, "shard", "dense_mlp") + _fp8(
        "mlp_down", h, i, "shard", "dense_mlp"
    )


def one_expert(group: str) -> list[Tensor]:
    h, i = C["hidden"], C["moe_inter"]
    return _fp8("gate", i, h, "shard", group) + _fp8("up", i, h, "shard", group) + _fp8("down", h, i, "shard", group)


def moe_nonrouted() -> list[Tensor]:
    h, e = C["hidden"], C["experts"]
    t = [Tensor("router", e * h, "BF16", "repl", "router"), Tensor("router_bias", e, "F32", "repl", "router")]
    return t + one_expert("shared")


EXPERT_BYTES = sum(t.params * {"F8": 1, "BF16": 2, "F32": 4}[t.dtype] for t in one_expert("expert"))
EXPERT_PARAMS_F8 = sum(t.params for t in one_expert("expert") if t.dtype == "F8")


def globals_main() -> list[Tensor]:
    v, h = C["vocab"], C["hidden"]
    return [
        Tensor("embed", v * h, "BF16", "shard", "embed"),
        Tensor("lm_head", v * h, "BF16", "shard", "head"),
        Tensor("final_norm", h, "F32", "repl", "norm"),
    ]


def mtp_extras() -> list[Tensor]:
    v, h = C["vocab"], C["hidden"]
    return [
        Tensor("mtp_embed", v * h, "BF16", "shard", "mtp_embed"),
        Tensor("mtp_enorm_hnorm", 2 * h, "F32", "repl", "norm"),
        Tensor("mtp_eh_proj", h * 2 * h, "BF16", "repl", "mtp_eh"),
        Tensor("mtp_head_norm", h, "F32", "repl", "norm"),
        Tensor("mtp_head", v * h, "BF16", "shard", "mtp_head"),
    ]


def inventory(model: str = "v32") -> dict[str, list[tuple[Tensor, int]]]:
    """Every stored tensor as (tensor, instance count), split into the main model and
    the MTP block. ``v31`` is the same inventory without the indexer (MD-LINEAGE)."""
    L, Ld = C["layers"], C["dense_layers"]
    Lm = L - Ld
    with_idx = model == "v32"
    main: list[tuple[Tensor, int]] = []
    main += [(t, L) for t in attention_layer()]
    if with_idx:
        main += [(t, L) for t in indexer()]
    main += [(t, Ld) for t in dense_mlp()]
    main += [(t, Lm) for t in moe_nonrouted()]
    main += [(t, Lm * C["experts"]) for t in one_expert("expert")]
    main += [(t, 1) for t in globals_main()]
    mtp: list[tuple[Tensor, int]] = []
    mtp += [(t, 1) for t in attention_layer()]
    if with_idx:
        mtp += [(t, 1) for t in indexer()]
    mtp += [(t, 1) for t in moe_nonrouted()]
    mtp += [(t, C["experts"]) for t in one_expert("expert")]
    mtp += [(t, 1) for t in mtp_extras()]
    return {"main": main, "mtp": mtp}


BYTES_PER = {"F8": 1, "BF16": 2, "F32": 4}


def params_by_dtype(model: str = "v32") -> dict[str, int]:
    out = {"F8": 0, "BF16": 0, "F32": 0}
    for part in inventory(model).values():
        for t, n in part:
            out[t.dtype] += t.params * n
    return out


def stored_bytes(model: str = "v32", part: str | None = None, weights: str = "fp8") -> float:
    """Resident bytes. ``weights='bf16'`` is the all-bf16 upcast of DV-FIT (params x 2)."""
    parts = [part] if part else ["main", "mtp"]
    total = 0.0
    for p in parts:
        for t, n in inventory(model)[p]:
            if weights == "bf16":
                total += 0 if t.name.endswith(".scale") else 2 * t.params * n
            else:
                total += BYTES_PER[t.dtype] * t.params * n
    return total


# =============================================================================
# 3. Routing: distinct experts, and the busiest chip under expert parallelism
# =============================================================================


def distinct_experts(rows: int, E: int = 256, k: int = 8) -> float:
    """E[# distinct experts] for ``rows`` tokens under uniform routing (roofline.py's formula).

    Group-limited routing (n_group 8, topk_group 4) leaves every expert's marginal
    selection probability at k/E, so the expectation is unchanged; ``check`` confirms it
    against the simulator below.
    """
    if rows <= 0:
        return 0.0
    return E * (1.0 - (1.0 - k / E) ** rows)


def _route_token(rng: random.Random) -> list[int]:
    """One token's experts under DeepSeek noaux_tc group-limited top-k, with iid scores.

    iid scores are the uniform-routing assumption (A-ROUTE) made concrete: group score =
    sum of the group's top-2 scores, keep the top-4 of 8 groups, then the top-8 experts
    inside them (MD-MOE).
    """
    E, G, TG, K = C["experts"], C["n_group"], C["topk_group"], C["topk"]
    gs = E // G
    s = [rng.random() for _ in range(E)]
    gscore = [sum(heapq.nlargest(2, s[g * gs : (g + 1) * gs])) for g in range(G)]
    groups = heapq.nlargest(TG, range(G), key=gscore.__getitem__)
    cand = [e for g in groups for e in range(g * gs, (g + 1) * gs)]
    return heapq.nlargest(K, cand, key=s.__getitem__)


_EP_CACHE: dict[tuple[int, int], dict] = {}
MC_SAMPLES = 3000
MC_SEED = 20260928


def ep_max_load(rows: int, chips: int, layout: str = "striped") -> dict:
    """E[max experts on one chip] and E[distinct] when ``chips`` each hold E/chips whole
    experts. Deterministic seed.

    ``contiguous`` puts expert e on chip e // (E/chips), so a chip sits inside one routing
    group and group-limited routing piles a token's 8 experts onto few chips. ``striped``
    puts expert e on chip e % chips, spreading every group over every chip. The ep plan
    uses ``striped``, the better of the two, because a plan-conditional bound should be
    the best layout within the plan.
    """
    key = (rows, chips, layout)
    if key in _EP_CACHE:
        return _EP_CACHE[key]
    E = C["experts"]
    per_chip = E // chips
    chip_of = (lambda e: e // per_chip) if layout == "contiguous" else (lambda e: e % chips)
    rng = random.Random(MC_SEED + 1000 * rows + chips)
    mx, ds = [], []
    for _ in range(MC_SAMPLES):
        chosen: set[int] = set()
        for _ in range(rows):
            chosen.update(_route_token(rng))
        load = [0] * chips
        for e in chosen:
            load[chip_of(e)] += 1
        mx.append(max(load))
        ds.append(len(chosen))

    def stats(xs):
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(var / len(xs))

    (m_max, se_max), (m_d, se_d) = stats(mx), stats(ds)
    out = {"max": m_max, "max_se": se_max, "distinct": m_d, "distinct_se": se_d, "samples": MC_SAMPLES}
    _EP_CACHE[key] = out
    return out


# =============================================================================
# 4. Scenario and plans
# =============================================================================

PLANS = {
    "ideal": "HLB. Every byte spread evenly over all chips; the attention cache context-parallel; "
    "collectives and kernel boundaries free.",
    "tp": "PCB. GPU-convention tensor parallelism over all chips: attention head-sharded, latent producers "
    "(q_a, kv_a), router and the whole indexer replicated, so every chip reads the full latent and "
    "indexer cache; experts sliced 1/N per chip; 2 all-reduces per layer.",
    "ep": "PCB. Attention as tp; experts whole, E/N per chip, so a layer waits for the chip that drew the "
    "most selected experts; all-reduce after attention, dispatch + combine per MoE layer.",
}

ATTENTION = {
    "dsa_sparse": "V3.2 semantics with a gathering kernel: indexer over all S keys, core over min(S, 2048) latents",
    "dsa_masked": "the reference decode path (S29): indexer over all S keys, core scores ALL S latents, top-k applied as a mask",
    "dense_mla": "indexer not implemented: dense MLA over all S latents (V3.1-style); indexer weights not read",
}


@dataclass(frozen=True)
class Scenario:
    name: str = "OP-1"
    gen: str = "SN40L"
    chips: int = 16
    model: str = "v32"  # v32 | v31 (v31 = no indexer, dense MLA)
    batch: int = 1  # sequences in the decode step = request concurrency on the node
    context: int = 8192  # tokens already cached per sequence
    attention: str = "dsa_sparse"
    plan: str = "ideal"
    mtp: int = 0  # draft depth D; 0 = MTP off (U8)
    weights: str = "fp8"  # fp8 (as stored, dequantised in the PCUs) | bf16 (upcast)
    residency: str = "hbm"  # hbm | ddr (all weights streamed from DDR)
    latent_bytes: int = CACHE["latent_bytes"].value
    index_bytes: int = CACHE["index_bytes"].value
    kv_b_bf16: bool = False  # reference caches a dequantised bf16 kv_b (S29)
    routing: str = "uniform"  # uniform | concentrated (all rows share 8 experts: the strict floor)
    hbm_bw: float | None = None  # override (sensitivity)
    hbm_reserved: float = 0.0  # bytes per chip reserved for program/buffers: UNKNOWN, 0 = optimistic
    use_fp8_compute: bool = True  # SN50 only; SN40L has none documented

    @property
    def rows(self) -> int:
        return self.batch * (1 + self.mtp)


def hw(sc: Scenario) -> dict:
    return HW[sc.gen]


# =============================================================================
# 5. The ledger
# =============================================================================


def _tensor_bytes(t: Tensor, sc: Scenario) -> float:
    if sc.weights == "bf16":
        return 0.0 if t.name.endswith(".scale") else 2.0 * t.params
    if sc.kv_b_bf16 and t.name.startswith("kv_b"):
        return 0.0 if t.name.endswith(".scale") else 2.0 * t.params
    return BYTES_PER[t.dtype] * t.params


def _expert_bytes(sc: Scenario) -> float:
    return sum(_tensor_bytes(t, sc) for t in one_expert("expert"))


def _uses_indexer(sc: Scenario) -> bool:
    return sc.model == "v32" and sc.attention in ("dsa_sparse", "dsa_masked")


def _distinct(rows: int, sc: Scenario) -> float:
    if sc.routing == "concentrated":
        return float(min(C["topk"], C["experts"]))
    return distinct_experts(rows, C["experts"], C["topk"])


def _ep_load(rows: int, sc: Scenario) -> float:
    if sc.routing == "concentrated":
        return float(math.ceil(C["topk"] / (C["experts"] // sc.chips)))
    return ep_max_load(rows, sc.chips)["max"]


def kv_per_sequence(sc: Scenario, layers: int) -> dict:
    """Cache bytes per sequence: resident, and read in one step."""
    S = sc.context
    lat_entries = min(S, C["idx_topk"]) if sc.attention == "dsa_sparse" else S
    idx_on = _uses_indexer(sc)
    resident = layers * S * (sc.latent_bytes + (sc.index_bytes if idx_on else 0))
    read_latent = layers * lat_entries * sc.latent_bytes
    read_index = layers * S * sc.index_bytes if idx_on else 0
    return {"resident": resident, "read_latent": read_latent, "read_index": read_index, "lat_entries": lat_entries}


def ledger(sc: Scenario) -> dict:
    """Capacity and per-step traffic, logical (whole deployment) and on the critical chip."""
    H = hw(sc)
    N = sc.chips
    inv = inventory(sc.model)
    L, Ld = C["layers"], C["dense_layers"]
    Lm = L - Ld
    B, D, R = sc.batch, sc.mtp, sc.rows
    idx_on = _uses_indexer(sc)
    h = C["hidden"]

    # ---- per-step weight reads, by group ------------------------------------
    # logical[group] = bytes read by the whole deployment; split into shard/repl so a
    # plan can place them.
    shard = {}
    repl = {}

    def add(d, g, v):
        d[g] = d.get(g, 0.0) + v

    def read_block(part_tensors, skip_groups, mult):
        for t, n in part_tensors:
            if t.group in skip_groups:
                continue
            per = _tensor_bytes(t, sc) * n * mult
            add(shard if t.tp == "shard" else repl, t.group, per)

    skip = {"expert", "embed", "mtp_embed"}
    if not idx_on:
        skip = skip | {"indexer"}
    # Main (verify) pass: every non-expert weight once, whatever the row count (A-WSTREAM).
    read_block(inv["main"], skip, 1)
    # Draft chain: D stages over the MTP block (reused per stage, as vLLM does for D > 1).
    if D > 0:
        read_block(inv["mtp"], skip, D)
    # Embedding gathers: rows x hidden x bf16, not the table.
    add(shard, "embed", R * h * 2 + D * B * h * 2)

    # Routed experts.
    eb = _expert_bytes(sc)
    d_main = _distinct(R, sc)
    d_draft = _distinct(B, sc)
    experts_logical = Lm * d_main * eb + D * d_draft * eb

    # Caches.
    kv_main = kv_per_sequence(sc, L)
    kv_mtp = kv_per_sequence(sc, 1)
    kv_read = B * (kv_main["read_latent"] + kv_main["read_index"]) + D * B * (kv_mtp["read_latent"] + kv_mtp["read_index"])
    per_tok_write = sc.latent_bytes + (sc.index_bytes if idx_on else 0)
    kv_write = R * L * per_tok_write + D * B * 1 * per_tok_write

    logical = {
        "weights_shardable": sum(shard.values()),
        "weights_replicable": sum(repl.values()),
        "experts": experts_logical,
        "kv_read": kv_read,
        "kv_write": kv_write,
    }
    logical_total = sum(logical.values())
    groups = {g: shard.get(g, 0) + repl.get(g, 0) for g in set(shard) | set(repl)}
    groups["expert (routed)"] = experts_logical
    groups["cache read"] = kv_read
    groups["cache write"] = kv_write

    # ---- the critical chip under the plan -----------------------------------
    if sc.plan == "ideal":
        chip = {k: v / N for k, v in logical.items()}
    else:
        chip = {
            "weights_shardable": logical["weights_shardable"] / N,
            "weights_replicable": logical["weights_replicable"],
            "kv_read": kv_read,  # head-sharded attention: every chip reads the whole selection
            "kv_write": kv_write,
        }
        if sc.plan == "tp":
            chip["experts"] = experts_logical / N
        elif sc.plan == "ep":
            chip["experts"] = (Lm * _ep_load(R, sc) + D * _ep_load(B, sc)) * eb
        else:
            raise ValueError(sc.plan)
    chip_total = sum(chip.values())

    # ---- residency: HBM capacity per chip -----------------------------------
    w_main = stored_bytes(sc.model, "main", sc.weights)
    w_mtp = stored_bytes(sc.model, "mtp", sc.weights) if D > 0 else 0.0
    w_resident = w_main + w_mtp
    # Split resident weights the same way the plan splits reads.
    res_repl = 0.0
    for part in (["main", "mtp"] if D > 0 else ["main"]):
        for t, n in inv[part]:
            if t.tp == "repl" and (idx_on or t.group != "indexer"):
                res_repl += _tensor_bytes(t, sc) * n
    res_shard = w_resident - res_repl
    kv_seq = kv_main["resident"] + (kv_mtp["resident"] if D > 0 else 0)
    cap = H["hbm_cap"].value
    if sc.plan == "ideal":
        w_chip = w_resident / N
        kv_chip_per_seq = kv_seq / N
    else:
        w_chip = res_shard / N + res_repl
        kv_chip_per_seq = kv_seq  # replicated latent + indexer cache
    free = cap - sc.hbm_reserved - w_chip
    max_seqs = math.floor(free / kv_chip_per_seq) if free > 0 and kv_chip_per_seq > 0 else 0
    used = w_chip + B * kv_chip_per_seq + sc.hbm_reserved

    # ---- where the bytes come from: HBM, or DDR (residency / spill) ----------
    hbm_chip, ddr_chip = chip_total, 0.0
    spill_frac = 0.0
    weight_chip = chip["weights_shardable"] + chip["weights_replicable"] + chip["experts"]
    if sc.residency == "ddr":
        hbm_chip, ddr_chip = chip_total - weight_chip, weight_chip
    elif w_chip > cap - sc.hbm_reserved:
        # X-ALLOC: spill the symbols with the smallest bandwidth need first: routed experts.
        # Every chip is over by the same amount, so the node-wide deficit is N times it.
        deficit = (w_chip - (cap - sc.hbm_reserved)) * N
        expert_resident = (Lm + (1 if D > 0 else 0)) * C["experts"] * eb
        spill_frac = min(1.0, deficit / expert_resident)
        ddr_chip = chip["experts"] * spill_frac
        hbm_chip = chip_total - ddr_chip

    # ---- compute -------------------------------------------------------------
    heads = C["heads"]
    lat_e = kv_main["lat_entries"]
    flops_attn = 2 * heads * lat_e * (C["kv_lora"] + C["qk_rope"] + C["kv_lora"]) * (R * L + D * B)
    flops_idx = (2 * C["idx_heads"] * C["idx_dim"] * sc.context * (R * L + D * B)) if idx_on else 0
    gemm_params_main = sum(t.params * n for t, n in inv["main"] if t.dtype != "F32" and t.group not in {"expert", "embed", "norm"} and (idx_on or t.group != "indexer"))
    gemm_params_mtp = sum(t.params * n for t, n in inv["mtp"] if t.dtype != "F32" and t.group not in {"expert", "mtp_embed", "norm"} and (idx_on or t.group != "indexer"))
    flops_gemm = 2 * R * gemm_params_main + 2 * D * B * gemm_params_mtp
    flops_exp = 2 * C["topk"] * EXPERT_PARAMS_F8 * (R * Lm + D * B)
    flops_total = flops_attn + flops_idx + flops_gemm + flops_exp
    if sc.plan == "ideal":
        flops_chip = flops_total / N
    else:  # indexer and replicated projections run in full on every chip
        flops_chip = (flops_total - flops_idx) / N + flops_idx
    peak = H["peak_bf16"].value
    if sc.gen == "SN50" and sc.use_fp8_compute and sc.weights == "fp8":
        peak = H["peak_fp8"].value

    # ---- collectives ---------------------------------------------------------
    if sc.plan == "ideal":
        n_coll, payload = 0, 0.0
    else:
        ar = R * h * 2  # bf16 hidden state
        if sc.plan == "tp":
            per_pass_main = 2 * L + 1  # 2 all-reduces per layer + logits gather
            per_stage = 2 + 1
            payload = per_pass_main * ar + D * per_stage * B * h * 2
        else:  # ep: all-reduce after attention (+ after dense MLP), dispatch + combine per MoE layer
            per_pass_main = L + Ld + 2 * Lm + 1
            per_stage = 1 + 2 + 1
            payload = (L + Ld + 1) * ar + 2 * Lm * R * C["topk"] * h * 2 + D * per_stage * B * h * 2
        n_coll = per_pass_main + D * per_stage

    # ---- times ---------------------------------------------------------------
    hbm_bw = sc.hbm_bw or H["hbm_bw"].value
    ddr_bw = H["ddr_bw"].value
    t = {
        "HBM": hbm_chip / hbm_bw,
        "DDR": ddr_chip / ddr_bw if ddr_chip else 0.0,
        "compute": flops_chip / peak,
    }
    # P2P bandwidth term, only where link bandwidth is published (SN50). A ring
    # all-reduce moves ~2x its payload through each chip; across nodes the narrower
    # inter-node links carry it. Latency stays unpriced on both generations.
    t_p2p_bw = None
    if n_coll and sc.gen == "SN50":
        links = H["inter_links"].value if N > H["node_chips"].value else H["intra_links"].value
        t_p2p_bw = 2 * payload / (links * H["link_bw_dir"].value)
        t["P2P bandwidth"] = t_p2p_bw
    sram_lower = H["sram_bw_lower"].value
    # Every HBM byte is written into and read out of PMUs at least once (A-SRAM).
    t_sram_upper = (2 * chip_total / sram_lower) if sram_lower else None
    step = max(t.values())
    binding = max(t, key=t.get)

    tokens_per_step = 1.0  # MTP acceptance is UNKNOWN; see mtp_table for the symbolic form
    out = {
        "scenario": asdict(sc),
        "kind": "HLB" if sc.plan == "ideal" else "PCB",
        "capacity": {
            "hbm_per_chip": cap,
            "weights_resident_total": w_resident,
            "weights_per_chip": w_chip,
            "kv_resident_per_seq": kv_seq,
            "kv_per_chip_per_seq": kv_chip_per_seq,
            "hbm_used_per_chip": used,
            "hbm_used_frac": used / cap,
            "max_seqs_before_spill": max_seqs,
            "weights_spill_frac": spill_frac,
            "sram_per_chip": H["sram_cap"].value,
        },
        "traffic_logical": logical,
        "traffic_logical_total": logical_total,
        "traffic_by_group": groups,
        "traffic_chip": chip,
        "traffic_chip_total": chip_total,
        "hbm_bytes_chip": hbm_chip,
        "ddr_bytes_chip": ddr_chip,
        "distinct_experts_main": d_main,
        "ep_max_load_main": (_ep_load(R, sc) if sc.plan == "ep" else None),
        "flops_total": flops_total,
        "flops_chip": flops_chip,
        "collectives": {"count": n_coll, "payload_bytes": payload},
        "times_s": t,
        "t_sram_upper_s": t_sram_upper,
        "step_bound_s": step,
        "binding": binding,
        "tok_s_user": tokens_per_step / step,
        "tok_s_node": sc.batch * tokens_per_step / step,
        "p2p_breakeven_latency_s": (step / n_coll) if n_coll else None,
        "p2p_breakeven_bw": (payload / step) if n_coll else None,
    }
    return out


# =============================================================================
# 6. Tables
# =============================================================================


def ms(x):
    return f"{x * 1e3:.3f}"


def gb(x):
    return f"{x / GB:.2f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def t_inventory() -> str:
    got = params_by_dtype("v32")
    rows = []
    for d in ("F8", "BF16", "F32"):
        rows.append([d, f"{got[d]:,}", f"{MD_PARAMS[d]:,}", f"{got[d] - MD_PARAMS[d]:+,}"])
    tot = sum(got.values())
    rows.append(["**total**", f"**{tot:,}**", f"**{MD_PARAMS_TOTAL:,}**", f"**{tot - MD_PARAMS_TOTAL:+,}**"])
    b = stored_bytes("v32")
    rows.append(["bytes", f"{b:,.0f}", f"{MD_BYTES:,}", f"{b - MD_BYTES:+,.0f}"])
    main, mtp = stored_bytes("v32", "main"), stored_bytes("v32", "mtp")
    s = md_table(["stored dtype", "rebuilt from config (params)", "MD-PARAMS / MD-BYTES", "difference"], rows)
    s += f"\n\nMain model {gb(main)} GB + MTP block {gb(mtp)} GB = {gb(main + mtp)} GB. "
    s += f"One routed expert = {EXPERT_BYTES:,} B ({EXPERT_BYTES / 1e6:.2f} MB, fp8 + fp32 block scales)."
    return s


def t_capacity(sc: Scenario) -> str:
    rows = []
    for plan in ("ideal", "tp"):
        for S in (8192, 32768, 131072):
            r = ledger(replace(sc, plan=plan, context=S))
            c = r["capacity"]
            rows.append([
                plan, f"{S // 1024}K", gb(c["weights_per_chip"]), gb(c["kv_per_chip_per_seq"]),
                f"{c['hbm_used_frac'] * 100:.1f} %", c["max_seqs_before_spill"],
            ])
    return md_table(
        ["plan", "context", "weights / chip (GB)", "cache / chip / sequence (GB)", "HBM used at B=1", "max sequences before spill (≤)"],
        rows,
    )


def t_traffic(sc: Scenario) -> str:
    r = ledger(sc)
    rows = []
    order = sorted(r["traffic_by_group"].items(), key=lambda kv: -kv[1])
    tot = r["traffic_logical_total"]
    for g, v in order:
        if v <= 0:
            continue
        rows.append([g, f"{v / GB:.3f}", f"{v / tot * 100:.1f} %"])
    rows.append(["**total**", f"**{tot / GB:.3f}**", "100 %"])
    return md_table(["group", "bytes read / step (GB, whole node)", "share"], rows)


def _row(sc: Scenario, label: str) -> list:
    r = ledger(sc)
    t = r["times_s"]
    return [
        label, r["kind"], gb(r["traffic_chip_total"]), ms(t["HBM"]), ms(t["DDR"]) if t["DDR"] else "0",
        f"{t['compute'] * 1e6:.1f} µs", ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}", r["binding"],
    ]


HDR = ["case", "kind", "bytes / critical chip (GB)", "HBM ms", "DDR ms", "compute", "step bound ms", "tok/s/user ≤", "binding"]


def t_plans(sc: Scenario) -> str:
    rows = [_row(replace(sc, plan=p), p) for p in ("ideal", "tp", "ep")]
    return md_table(HDR, rows)


def t_context(base: Scenario) -> str:
    rows = []
    for att in ("dsa_sparse", "dsa_masked", "dense_mla"):
        for S in (8192, 32768, 131072):
            for plan in ("ideal", "tp"):
                r = ledger(replace(base, attention=att, context=S, plan=plan))
                kvshare = (r["traffic_chip"]["kv_read"]) / r["traffic_chip_total"]
                rows.append([att, f"{S // 1024}K", plan, ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}",
                             f"{kvshare * 100:.1f} %", r["binding"]])
    return md_table(["attention case", "context", "plan", "step bound ms", "tok/s/user ≤", "cache share of chip bytes", "binding"], rows)


def t_discriminator(base: Scenario) -> str:
    """The observation that identifies the attention case: step time at ~30K vs ~1K context.

    SambaCloud serves V3.2 as a 32K preview (D-V32-CLOUD), so both ends are reachable
    by a customer with an API key; the ratio cancels the unknown attainment if it is
    the same at both lengths (A-ETA-FLAT).
    """
    rows = []
    for att in ("dsa_sparse", "dsa_masked", "dense_mla"):
        cells = []
        for plan in ("ideal", "tp", "ep"):
            t1 = ledger(replace(base, attention=att, context=1024, plan=plan))["step_bound_s"]
            t30 = ledger(replace(base, attention=att, context=30720, plan=plan))["step_bound_s"]
            cells.append(f"{t30 / t1:.2f}×")
        rows.append([att, *cells])
    return md_table(["attention case", "ideal: t(30K) / t(1K)", "tp", "ep"], rows)


def t_residency(base: Scenario) -> str:
    rows = [
        _row(base, "fp8 weights in HBM (OP-1)"),
        _row(replace(base, residency="ddr"), "fp8 weights streamed from DDR"),
        _row(replace(base, weights="bf16"), "bf16 upcast, experts spill to DDR"),
        _row(replace(base, chips=8), "SN40L-8: 8 chips, experts spill"),
    ]
    extra = []
    for label, sc in [("bf16 upcast", replace(base, weights="bf16")), ("SN40L-8", replace(base, chips=8))]:
        r = ledger(sc)
        extra.append(f"{label}: {r['capacity']['weights_spill_frac'] * 100:.1f} % of routed experts spilled")
    return md_table(HDR, rows) + "\n\n" + "; ".join(extra) + "."


def t_batch(base: Scenario) -> str:
    rows = []
    for B in (1, 2, 4, 8, 16, 32):
        for plan in ("ideal", "tp", "ep"):
            r = ledger(replace(base, batch=B, plan=plan))
            rows.append([B, plan, f"{r['distinct_experts_main']:.1f}", ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}",
                         f"{r['tok_s_node']:.0f}", r["binding"]])
    return md_table(["B", "plan", "distinct experts / layer", "step bound ms", "tok/s/user ≤", "tok/s/node ≤", "binding"], rows)


def t_routing(base: Scenario) -> str:
    rows = []
    for B in (1, 4):
        for routing in ("uniform", "concentrated"):
            r = ledger(replace(base, batch=B, routing=routing))
            rows.append([B, routing, f"{r['distinct_experts_main']:.1f}", ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}"])
    return md_table(["B", "routing", "distinct experts / layer", "HLB step ms", "tok/s/user ≤"], rows)


def t_ep_load(chips: int = 16) -> str:
    rows = []
    for R in (1, 2, 4, 8):
        s = ep_max_load(R, chips, "striped")
        c = ep_max_load(R, chips, "contiguous")
        f = distinct_experts(R)
        rows.append([R, f"{f:.2f}", f"{s['distinct']:.2f} ± {s['distinct_se']:.2f}",
                     f"{c['max']:.2f} ± {c['max_se']:.2f}", f"{s['max']:.2f} ± {s['max_se']:.2f}",
                     f"{max(1, math.ceil(f / chips - 1e-9))}", f"{f / chips:.2f}"])
    return md_table(["rows", "E[distinct], formula", "E[distinct], simulated", "busiest chip, contiguous layout",
                     "busiest chip, striped layout (used)", "whole-expert floor", "tp slice equivalent"], rows)


def t_mtp(base: Scenario) -> str:
    rows = []
    t0 = {p: ledger(replace(base, plan=p))["step_bound_s"] for p in ("ideal", "tp", "ep")}
    for D in (1, 2, 3):
        for plan in ("ideal", "tp", "ep"):
            tD = ledger(replace(base, plan=plan, mtp=D))["step_bound_s"]
            be = tD / t0[plan]
            grid = [f"{tau / tD:.0f}" for tau in (1.5, 2.0, 2.5, 3.0) if tau <= D + 1]
            rows.append([D, plan, ms(t0[plan]), ms(tD), f"{tD / t0[plan]:.2f}×", f"{be:.2f}", " / ".join(grid)])
    return md_table(
        ["D", "plan", "MTP off ms", "MTP step ms", "step cost", "break-even accepted tokens τ*", "tok/s/user ≤ at τ = 1.5 / 2 / 2.5 / 3"],
        rows,
    )


def t_p2p(base: Scenario) -> str:
    rows = []
    for plan in ("tp", "ep"):
        for B in (1, 4):
            r = ledger(replace(base, plan=plan, batch=B))
            c = r["collectives"]
            rows.append([plan, B, c["count"], f"{c['payload_bytes'] / 1e6:.2f}", ms(r["step_bound_s"]),
                         f"{r['p2p_breakeven_latency_s'] * 1e6:.1f}", f"{r['p2p_breakeven_bw'] / GB:.2f}"])
    return md_table(
        ["plan", "B", "dependent collectives / step", "payload / step (MB)", "PCB step ms",
         "break-even latency per collective (µs)", "break-even bandwidth (GB/s)"],
        rows,
    )


def t_sram(base: Scenario) -> str:
    rows = []
    for plan in ("ideal", "tp", "ep"):
        r = ledger(replace(base, plan=plan))
        rows.append([plan, ms(r["times_s"]["HBM"]), f"{r['t_sram_upper_s'] * 1e6:.1f} µs",
                     f"{r['t_sram_upper_s'] / r['times_s']['HBM'] * 100:.1f} %"])
    return md_table(["plan", "HBM ms", "SRAM time, upper bound", "SRAM / HBM"], rows)


def anchor_rows() -> tuple[str, dict]:
    """Implied attainment of each plan against the measured DeepSeek points."""
    rows = []
    eta = {}
    for aid, a in ANCHORS.items():
        contexts = [a["context"]] if a["context"] else [1024, 8192]
        for S in contexts:
            for plan in ("ideal", "tp", "ep"):
                sc = Scenario(model="v31", attention="dense_mla", context=S, plan=plan)
                r = ledger(sc)
                t_meas = 1.0 / a["tok_s"]
                e = r["step_bound_s"] / t_meas
                if aid == "D-V31-AA":
                    eta[plan] = e
                if e > 1.0:
                    verdict = "**not attainable** at B=1 without speculation"
                elif e > ETA_DENSE.value[1]:
                    verdict = "needs more than the best measured dense attainment (93 %)"
                else:
                    verdict = "consistent"
                rows.append([aid, f"{S:,}", plan, ms(r["step_bound_s"]), f"{a['tok_s']:.0f}", ms(t_meas),
                             f"{e * 100:.0f} %", verdict])
    return md_table(["anchor", "context", "plan", "V3.1 bound ms", "measured tok/s", "measured ms/token",
                     "implied attainment (bound ÷ measured)", "verdict"], rows), eta


def t_anchor_projection() -> str:
    """V3.2 time = measured V3.1 time x (V3.2 bound / V3.1 bound), same plan, same attainment.

    Only plans the anchor leaves attainable (T13) are projected through.
    """
    a = ANCHORS["D-V31-AA"]
    t_meas = 1.0 / a["tok_s"]
    plans = []
    for plan in ("tp", "ep"):
        ref = ledger(Scenario(model="v31", attention="dense_mla", context=a["context"], plan=plan))["step_bound_s"]
        if ref <= t_meas:
            plans.append((plan, ref))
    rows = []
    for att in ("dsa_sparse", "dsa_masked", "dense_mla"):
        for S in (8192, 32768, 131072):
            cells = []
            for plan, ref in plans:
                v32 = ledger(Scenario(model="v32", attention=att, context=S, plan=plan))["step_bound_s"]
                cells.append(f"{1.0 / (t_meas * v32 / ref):.0f}")
            rows.append([att, f"{S // 1024}K", *cells])
    return md_table(["attention case", "context", *[f"anchored tok/s/user, {p} plan (η = {r / t_meas * 100:.0f} %)" for p, r in plans]], rows)


def t_tp_floor() -> str:
    """The tp plan's replicated bytes do not shrink with N: the bound flattens."""
    rows = []
    for gen, Ns in (("SN40L", (16,)), ("SN50", (16, 32, 64, 128, 256))):
        for N in Ns:
            for S in (8192, 131072):
                r = ledger(Scenario(gen=gen, chips=N, plan="tp", context=S))
                ch = r["traffic_chip"]
                floor = ch["weights_replicable"] + ch["kv_read"] + ch["kv_write"]
                bw = HW[gen]["hbm_bw"].value
                rows.append([gen, N, f"{S // 1024}K", gb(r["traffic_chip_total"]), gb(floor), ms(r["step_bound_s"]),
                             ms(floor / bw), f"{r['tok_s_user']:.0f}", f"{1 / (floor / bw):.0f}"])
    return md_table(["gen", "chips", "context", "bytes / chip (GB)", "replicated part (GB)", "tp step bound ms",
                     "floor as N→∞ ms", "tok/s/user ≤", "ceiling as N→∞"], rows)


def t_sn50() -> str:
    rows = []
    for N in (8, 16, 32, 64, 128):
        for plan in ("ideal", "tp", "ep"):
            sc = Scenario(name="SN50", gen="SN50", chips=N, plan=plan)
            r = ledger(sc)
            t = r["times_s"]
            rows.append([N, plan, r["kind"], gb(r["capacity"]["weights_per_chip"]), f"{r['capacity']['weights_spill_frac'] * 100:.0f} %",
                         ms(t["HBM"]), ms(t["DDR"]) if t["DDR"] else "0", ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}", r["binding"]])
    return md_table(["RDUs", "plan", "kind", "weights / chip (GB)", "experts spilled", "HBM ms", "DDR ms",
                     "step bound ms", "tok/s/user ≤", "binding"], rows)


def t_sn50_p2p() -> str:
    rows = []
    for N in (16, 64):
        for plan in ("tp", "ep"):
            r = ledger(Scenario(gen="SN50", chips=N, plan=plan))
            c = r["collectives"]
            rows.append([N, plan, c["count"], f"{c['payload_bytes'] / 1e6:.2f}",
                         f"{r['times_s']['P2P bandwidth'] * 1e6:.1f} µs", ms(r["step_bound_s"]),
                         f"{r['p2p_breakeven_latency_s'] * 1e6:.1f}"])
    return md_table(["RDUs", "plan", "dependent collectives / step", "payload (MB)", "P2P bandwidth term",
                     "step bound ms", "break-even latency per collective (µs)"], rows)


def t_sn50_context() -> str:
    rows = []
    for att in ("dsa_sparse", "dense_mla"):
        for S in (8192, 131072):
            for plan in ("ideal", "tp"):
                r = ledger(Scenario(gen="SN50", chips=16, attention=att, context=S, plan=plan))
                rows.append([att, f"{S // 1024}K", plan, ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}", r["binding"]])
    return md_table(["attention case", "context", "plan", "step bound ms", "tok/s/user ≤", "binding"], rows)


def t_sn50_hbm_sens() -> str:
    rows = []
    for bw, lab in ((1.8e12, "1.80 (trace plateau, VMEAS floor on peak)"), (1.84e12, "1.84 (2HAND, S27)"),
                    (1.6e12, "1.60 (= SN40L, if S27 were wrong)")):
        r = ledger(Scenario(gen="SN50", chips=16, plan="ideal", hbm_bw=bw))
        rows.append([lab, ms(r["step_bound_s"]), f"{r['tok_s_user']:.0f}"])
    return md_table(["SN50 HBM TB/s per RDU", "HLB step ms (16 RDUs, OP-1)", "tok/s/user ≤"], rows)


def report() -> str:
    op1 = Scenario()
    parts = []
    parts.append("## T1. Parameter inventory rebuilt from config, reconciled\n\n" + t_inventory())
    parts.append("## T2. HBM capacity per chip, OP-1 (SN40L-16, fp8, B=1, DSA)\n\n" + t_capacity(op1))
    parts.append("## T3. Traffic per decode step, OP-1 at 8K (whole node)\n\n" + t_traffic(op1))
    parts.append("## T3b. Traffic per decode step, OP-1 at 128K (whole node)\n\n" + t_traffic(replace(op1, context=131072)))
    parts.append("## T4. Bounds by placement plan, OP-1 at 8K\n\n" + t_plans(op1))
    parts.append("## T4b. Bounds by placement plan, OP-1 at 128K\n\n" + t_plans(replace(op1, context=131072)))
    parts.append("## T5. Attention case x context x plan (B=1)\n\n" + t_context(op1))
    parts.append("## T5b. The observation that identifies the attention case (B=1)\n\n" + t_discriminator(op1))
    parts.append("## T6. Residency cases (8K, ideal plan)\n\n" + t_residency(op1))
    parts.append("## T7. Batch sweep (8K, DSA)\n\n" + t_batch(op1))
    parts.append("## T8. Routing assumption (HLB, 8K)\n\n" + t_routing(op1))
    parts.append("## T9. Expert-parallel load on 16 chips (simulated noaux_tc routing, seed %d, %d samples)\n\n" % (MC_SEED, MC_SAMPLES) + t_ep_load(16))
    parts.append("## T10. MTP, symbolic in accepted tokens τ (8K, B=1)\n\n" + t_mtp(op1))
    parts.append("## T11. Inter-chip communication: what P2P would have to be to bind (8K)\n\n" + t_p2p(op1))
    parts.append("## T12. SRAM cannot bind on SN40L (8K)\n\n" + t_sram(op1))
    at, _ = anchor_rows()
    parts.append("## T13. Calibration against measured DeepSeek decode on SN40L\n\n" + at)
    parts.append("## T14. Anchored projection for V3.2 on SN40L-16 (B=1), from D-V31-AA\n\n" + t_anchor_projection())
    parts.append("## T15. SN50 projection by RDU count (8K, B=1, DSA)\n\n" + t_sn50())
    parts.append("## T15b. The tp plan's replicated floor (B=1, DSA)\n\n" + t_tp_floor())
    parts.append("## T15c. SN50 P2P, bandwidth priced and latency not (8K, B=1)\n\n" + t_sn50_p2p())
    parts.append("## T16. SN50, 16 RDUs, context and attention case\n\n" + t_sn50_context())
    parts.append("## T17. SN50 HBM bandwidth sensitivity\n\n" + t_sn50_hbm_sens())
    return "\n\n".join(parts) + "\n"


# =============================================================================
# 7. Checks
# =============================================================================


def run_checks() -> list[tuple[str, bool, str]]:
    res = []

    def ok(name, cond, detail=""):
        res.append((name, bool(cond), detail))

    got = params_by_dtype("v32")
    for d in MD_PARAMS:
        ok(f"MD-PARAMS {d} reproduced exactly", got[d] == MD_PARAMS[d], f"{got[d]:,} vs {MD_PARAMS[d]:,}")
    ok("MD-BYTES reproduced exactly", stored_bytes("v32") == MD_BYTES, f"{stored_bytes('v32'):,.0f}")
    ok("C12: index total_size = params x 2", 2 * MD_PARAMS_TOTAL == 1_370_793_842_752)
    fit = MD_BYTES / (16 * 64 * GiB)
    ok("DV-FIT 62.7 %", abs(fit - 0.627) < 0.0005, f"{fit * 100:.2f} %")
    # DV-SWAP: fp8 copy over aggregate DDR
    swap = MD_BYTES / (16 * SN40L["ddr_bw"].value)
    ok("DV-SWAP fp8 copy >= 0.43 s", abs(swap - 0.431) < 0.001, f"{swap:.3f} s")
    # Hand arithmetic, independent of the ledger code: routed + shared at B=1.
    hand = 58 * 9 * 3 * 7168 * 2048
    ok("DV-DDR-DECODE 23.0 B params", abs(hand - 22.98e9) < 0.01e9, f"{hand:,}")
    ok("expert bytes = 3 x 2048 x 7168 + 2688 scales x 4", EXPERT_BYTES == 3 * 2048 * 7168 + 4 * 3 * 16 * 56)
    r = ledger(Scenario())
    ok("OP-1 routed bytes at B=1 = 58 x 8 x expert", abs(r["traffic_logical"]["experts"] - 58 * 8 * EXPERT_BYTES) < 1)
    ok("OP-1 cache read at 8K = 61 x (2048 x 1152 + 8192 x 132)",
       r["traffic_logical"]["kv_read"] == 61 * (2048 * 1152 + 8192 * 132))
    # Simulator: expectation must match the closed form (marginals are k/E under group limits).
    for R in (1, 4):
        m = ep_max_load(R, 16)
        f = distinct_experts(R)
        ok(f"simulated distinct experts (rows={R}) within 4 SE of formula", abs(m["distinct"] - f) <= 4 * max(m["distinct_se"], 1e-9),
           f"{m['distinct']:.3f} ± {m['distinct_se']:.3f} vs {f:.3f}")
    # Falsification: the hardware lower bound may never exceed a measured time.
    for aid, a in ANCHORS.items():
        S = a["context"] or 1024
        hlb = ledger(Scenario(model="v31", attention="dense_mla", context=S, plan="ideal"))["step_bound_s"]
        ok(f"HLB <= measured time for {aid}", hlb <= 1.0 / a["tok_s"], f"{hlb * 1e3:.3f} ms vs {1e3 / a['tok_s']:.3f} ms")
    # Unknowns stay unknown.
    ok("SN40L P2P bandwidth is not defaulted", SN40L["p2p_bw"].value is None)
    ok("SN40L fp8 peak is not defaulted", SN40L["peak_fp8"].value is None)
    ok("SN50 SRAM bandwidth is not defaulted", SN50["sram_bw_lower"].value is None)
    # Every input has a source ID.
    for table in (SN40L, SN50, CFG, CACHE):
        for k, v in table.items():
            ok(f"input {k} is cited", bool(v.src) and v.cls in {"DOC", "MODEL", "VMEAS", "3PMEAS", "2HAND", "DERIVED", "ASSUMED", "UNKNOWN"})
    return res


# =============================================================================
# 8. CLI
# =============================================================================


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("report")
    q = sub.add_parser("point")
    d = Scenario()
    q.add_argument("--gen", choices=list(HW), default=d.gen)
    q.add_argument("--chips", type=int, default=d.chips)
    q.add_argument("--model", choices=["v32", "v31"], default=d.model)
    q.add_argument("--batch", type=int, default=d.batch)
    q.add_argument("--context", type=int, default=d.context)
    q.add_argument("--attention", choices=list(ATTENTION), default=d.attention)
    q.add_argument("--plan", choices=list(PLANS), default=d.plan)
    q.add_argument("--mtp", type=int, default=d.mtp)
    q.add_argument("--weights", choices=["fp8", "bf16"], default=d.weights)
    q.add_argument("--residency", choices=["hbm", "ddr"], default=d.residency)
    q.add_argument("--latent-bytes", type=int, default=d.latent_bytes, help="per token per layer; 1152 = bf16 (S29)")
    q.add_argument("--index-bytes", type=int, default=d.index_bytes, help="per token per layer; 132 = fp8 + fp32 scale (S29)")
    q.add_argument("--kv-b-bf16", action="store_true")
    q.add_argument("--routing", choices=["uniform", "concentrated"], default=d.routing)
    q.add_argument("--hbm-bw", type=float, default=None, help="bytes/s per chip, overrides the cited value")
    q.add_argument("--hbm-reserved-gb", type=float, default=0.0)
    q.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    if a.cmd == "check":
        res = run_checks()
        bad = [r for r in res if not r[1]]
        for name, good, detail in res:
            print(f"{'PASS' if good else 'FAIL'}  {name}  {detail}")
        print(f"\n{len(res) - len(bad)}/{len(res)} checks pass")
        return 1 if bad else 0
    if a.cmd == "report":
        print(report())
        return 0

    sc = Scenario(
        name="cli", gen=a.gen, chips=a.chips, model=a.model, batch=a.batch, context=a.context,
        attention=a.attention, plan=a.plan, mtp=a.mtp, weights=a.weights, residency=a.residency,
        latent_bytes=a.latent_bytes, index_bytes=a.index_bytes, kv_b_bf16=a.kv_b_bf16, routing=a.routing,
        hbm_bw=a.hbm_bw, hbm_reserved=a.hbm_reserved_gb * GB,
    )
    r = ledger(sc)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
        return 0
    H = hw(sc)
    print(f"{sc.gen} x{sc.chips} | {sc.model} | B={sc.batch} S={sc.context} | {sc.attention} | plan={sc.plan} ({r['kind']}) | "
          f"MTP D={sc.mtp} | weights {sc.weights} in {sc.residency}")
    print(f"  plan: {PLANS[sc.plan]}")
    print(f"  attention: {ATTENTION[sc.attention]}")
    c = r["capacity"]
    print("\nCAPACITY (per chip)")
    print(f"  HBM {gb(c['hbm_per_chip'])} GB [{H['hbm_cap'].cls} {H['hbm_cap'].src}]: weights {gb(c['weights_per_chip'])} GB, "
          f"cache {gb(c['kv_per_chip_per_seq'])} GB/sequence -> {c['hbm_used_frac'] * 100:.1f} % used; "
          f"<= {c['max_seqs_before_spill']} sequences before spill; experts spilled {c['weights_spill_frac'] * 100:.1f} %")
    print(f"  SRAM {c['sram_per_chip'] / 1e6:.0f} MB [{H['sram_cap'].cls}] holds no weights across steps (A-ACT)")
    print("\nTRAFFIC per step (GB)")
    for k, v in r["traffic_logical"].items():
        print(f"  {k:<20} node {v / GB:9.3f}   critical chip {r['traffic_chip'][k] / GB:8.3f}")
    print(f"  HBM on critical chip {gb(r['hbm_bytes_chip'])} GB; DDR {gb(r['ddr_bytes_chip'])} GB")
    print("\nTIME TERMS (ms, critical chip)")
    for k, v in r["times_s"].items():
        print(f"  {k:<8} {v * 1e3:9.4f}")
    if r["t_sram_upper_s"] is not None:
        print(f"  SRAM   <= {r['t_sram_upper_s'] * 1e3:.4f} (upper bound; cannot bind)")
    else:
        print("  SRAM     not numerically bounded (bandwidth unpublished)")
    col = r["collectives"]
    if col["count"]:
        print(f"  P2P      UNPRICED: {col['count']} dependent collectives, {col['payload_bytes'] / 1e6:.2f} MB; binds if each costs "
              f"> {r['p2p_breakeven_latency_s'] * 1e6:.1f} µs on the critical path, or if bandwidth < {r['p2p_breakeven_bw'] / GB:.2f} GB/s")
    print(f"\nSTEP BOUND {ms(r['step_bound_s'])} ms  ->  <= {r['tok_s_user']:.0f} tok/s/user, <= {r['tok_s_node']:.0f} tok/s/node  "
          f"[binding: {r['binding']}]  ({r['kind']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
