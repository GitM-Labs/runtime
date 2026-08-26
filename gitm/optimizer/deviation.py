"""Deviation-only tracing — keep only the kernels that depart from prediction.

The behavioral compiler predicts a per-op execution graph; most kernels land
inside their roofline band and are *uninteresting* — they behaved exactly as
predicted, so storing them buys nothing. The optimization signal lives in the
*departures*: kernels slower (or heavier) than predicted, and kernels with no
predicted counterpart at all (unmodeled work). This module reduces a captured
trace to just those, so trace storage scales with deviation, not duration (the
monitor's design principle, applied to the trace itself).

    dev = deviating_kernel_indices(trace, graph)   # which observed kernels departed
    reduced = deviation_trace(trace, graph)         # a Trace of only those kernels

The band check mirrors :func:`gitm.optimizer.monitor.check_invariants` (same
``INVARIANTS`` band widths) but is applied per *observed kernel index* so each
departure maps back to its original event — the residual pass loses that link
because it keys by predicted op name (many kernels share one op).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gitm.optimizer.invariants import INVARIANTS, Invariant
from gitm.planner.graph import Graph, PredictedNode
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace

# Ordered kernel-name → predicted-op rules (first match wins), case-insensitive
# substring. None = no modeled op (norms, activations, copies, launch overhead)
# — unmodeled work, kept as departures.
#
# The projection GEMMs (qkv/out/gate_up/gate_down/lm_head) only classify when the
# kernel name carries a projection tag; a bare cuBLAS/cutlass GEMM (e.g.
# `ampere_fp16_s16816gemm_*`) carries none and stays unmodeled — confirmed
# against a real vLLM/L4/CUPTI trace, where these are ~35% of launches and
# reused across every projection. Needs shape-matching or launch-order
# instrumentation to fix, not a vocabulary tweak. The attention/KV-cache
# needles below ARE confirmed against that trace: FlashAttention's real
# kernel is `flash_fwd_splitkv_kernel` (`flash_attn` alone misses it), and
# vLLM's `reshape_and_cache_flash_kernel`/`_compute_slot_mapping_kernel`
# weren't covered before.
_OP_RULES: dict[str, tuple[str, ...]] = {
    # ── collectives, first of all ────────────────────────────────────────────
    # NCCL names every kernel `ncclDevKernel_<Op>`, so a generic "nccl" needle
    # would swallow the all-to-all — and expert-parallel dispatch is the one
    # collective whose cost a MoE deployment exists to trade against, so it must
    # not disappear into a bucket labelled all-reduce.
    "moe_all_to_all": ("alltoall", "all_to_all", "_a2a", "dispatch_combine"),
    "tp_all_reduce": ("nccl", "allreduce", "all_reduce", "custom_ar", "cross_device",
                      "one_shot", "two_shot", "reduce_scatter", "all_gather"),

    # ── gated DeltaNet / linear attention (hybrid checkpoints) ───────────────
    # High because these names are unambiguous and long, and because the generic
    # attention needles below must never claim them: a GDN layer keeps a
    # fixed-size recurrent state rather than a KV cache, so folding it into
    # `attn_score_value` would compare a constant-traffic op against a prediction
    # that grows with context and report the difference as a deviation.
    #
    # The convolution is its own entry because its own kernel runs
    # (`causal_conv1d_update`); merging it into the recurrent entry would leave
    # that kernel permanently unattributed.
    "linattn_conv": ("causal_conv1d", "post_conv"),
    "linattn_in_proj": ("in_proj_qkvz", "in_proj_ba", "qkvz"),
    # `deltarule` carries no underscore on purpose: the CuTeDSL path is
    # `_FullyFusedDeltaRuleSm90`, and CamelCase lowercases to a single token, so
    # every underscored delta needle misses the SM90 fast path entirely.
    # "linear_attention" is listed separately from "linear_attn": the shorter
    # string is NOT a prefix of the longer one ("...att**n**" vs "...att**e**n"),
    # so the needle written for one misses the other entirely. vLLM's
    # torch.compile splitting op is `vllm::linear_attention`, and on a B200 run
    # every GDN layer fell through to the softmax-attention rule because of it.
    #
    # "gdn" is load-bearing for the same reason: `vllm::qwen_gdn_attention_core`
    # contains "attention", so without an earlier claim it is misfiled as
    # `attn_score_value` — a constant-traffic op compared against a prediction
    # that grows with context, reporting a false deviation on every long step.
    "linattn_recurrent": ("fused_recurrent", "gated_delta", "delta_rule", "deltarule",
                          "deltanet", "chunk_fwd", "chunk_scaled_dot", "recompute_w_u",
                          "solve_tril", "wy_fast", "linear_attn", "linear_attention",
                          "gdn", "mamba_mixer", "short_conv"),

    # ── sparse-MoE / compressed attention ────────────────────────────────────
    # Before the generic entries below, because their vocabularies are subsets of
    # them: `moe_align_block_size` contains "moe" but is routing, not an expert
    # GEMM; `shared_expert` contains "expert" but is unconditional work with a
    # completely different cost curve; and the indexer must never fall through to
    # a bare "index" rule, which is how it gets misfiled as elementwise in the
    # coarse taxonomy.
    "attn_index_score": ("indexer", "lightning_index", "index_topk", "topk_indices"),
    "moe_shared": ("shared_expert", "moe_shared"),
    # `topkGating` is vLLM's fused routing kernel. Without it the generic "moe"
    # needle below claims it as an expert GEMM, which puts routing cost — cheap,
    # and bound by something else entirely — inside the entry whose weight
    # traffic dominates the step.
    "moe_router": ("moe_align", "topk_softmax", "topkgating", "gating", "router",
                   "routing", "sinkhorn", "expert_bias"),
    "moe_routed": ("moe", "expert", "grouped_gemm", "group_gemm", "groupedgemm"),
    "dspark": ("dspark",),

    # ── softmax attention ────────────────────────────────────────────────────
    # MLA and generic FlashAttention needles share one entry: they resolve to the
    # same op and nothing sits between them, so the two tuples that used to be
    # separate are merged here with no change in behaviour.
    # The FlashInfer names are here as well as in the coarse taxonomy: which
    # attention backend vLLM selects is a per-SKU decision (FLASH_ATTN on the
    # H200, FLASHINFER on the B200 in the same vLLM build), so a rule set that
    # covers only one of them silently loses the whole attention path on the
    # other machine.
    "attn_score_value": ("flash_mla", "mla_sparse", "sparse_mla", "cutlass_mla",
                         "sparse_fwd", "flash_attn", "flashattn", "flash_fwd",
                         "paged_attention", "paged_attn", "fmha", "attention",
                         "attn_score", "reshape_and_cache", "slot_mapping",
                         "batchprefill", "batchdecode", "pagedkv"),
    # Norm + rotary + cache insert. Every graph family emits this as *one* node
    # because vLLM runs it as one fused kernel; without an entry the kernel stayed
    # unmodeled on both the sparse-MoE and hybrid paths even though the node
    # existed to receive it. "mrope" is the multimodal variant these hybrid
    # checkpoints use (`_triton_mrope_forward`).
    "attn_qnorm_rope_insert": ("qnorm", "q_norm", "qk_norm", "mrope", "rope", "rotary"),

    # ── projections ──────────────────────────────────────────────────────────
    "attn_q_a": ("q_a_proj", "q_lora", "q_down"),
    "attn_q_b": ("q_b_proj", "q_up"),
    # `kv_b_proj` is absent on purpose: in the absorbed decode form it is folded
    # into the query and output projections, so there is no node to map it to and
    # a guess would attribute real work to the wrong op.
    "attn_kv_a": ("kv_a_proj", "kv_lora", "kv_down", "compress_kv"),
    "qkv_proj": ("qkv",),
    "attn_out_proj": ("o_proj", "out_proj", "attn_out"),
    "mlp_gate_up": ("gate_up", "gate_proj", "up_proj", "swiglu", "silu_and_mul"),
    "mlp_down": ("down_proj", "mlp_down"),
    "lm_head": ("lm_head", "logits", "vocab_proj", "embed"),
}


def classify_op(kernel_name: str) -> str | None:
    """Map a raw kernel name to a predicted-graph op, or ``None`` if unmodeled.

    Case-insensitive substring match, first entry wins — see the ordering note on
    :data:`_OP_RULES`. ``None`` = the kernel maps to no op in the predicted graph
    (a norm/activation/copy, or a bare GEMM whose name doesn't carry its
    projection) → treated as unmodeled work.
    """
    n = kernel_name.lower()
    for op, needles in _OP_RULES.items():
        if any(k in n for k in needles):
            return op
    return None

@dataclass
class DeviationResult:
    """Which observed kernels departed from the predicted graph, and how much it compresses."""

    kept_indices: list[int]  # indices into trace.kernels() that departed
    n_observed: int
    n_predicted: int

    @property
    def n_kept(self) -> int:
        return len(self.kept_indices)

    @property
    def reduction(self) -> float:
        """Fraction of observed kernels dropped as in-band (0.0 if none observed)."""
        return 1.0 - (self.n_kept / self.n_observed) if self.n_observed else 0.0


def _departs(
    ok: KernelEvent,
    node_pred_s: float,
    node_pred_bytes: float,
    inv_kt: Invariant | None,
    inv_mt: Invariant | None,
) -> bool:
    """True if observed kernel ``ok`` is out-of-band vs its predicted node."""
    t_obs = max((ok.end_ns - ok.start_ns) / 1e9, 1e-12)
    t_pred = max(node_pred_s, 1e-12)
    r_kt = (t_obs - t_pred) / t_pred
    if inv_kt is not None and abs(r_kt) > inv_kt.band_width:
        return True
    if (
        inv_mt is not None
        and ok.bytes_read is not None
        and ok.bytes_written is not None
        and node_pred_bytes > 0
    ):
        r_mt = ((ok.bytes_read + ok.bytes_written) - node_pred_bytes) / node_pred_bytes
        if abs(r_mt) > inv_mt.band_width:
            return True
    return False


def deviating_kernel_indices(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> DeviationResult:
    """Indices of observed kernels that depart from the predicted graph.

    Each observed kernel is matched to a predicted op **by identity** — its name is
    classified (:func:`classify_op`) to an op and compared against that op's
    predicted roofline node. This replaces the old positional ``i % len(pred)``
    pairing, which was meaningless once CUDA graphs reorder/fuse the kernel stream
    (it flagged ~everything, uniformly across ops). A kernel *departs* when its
    kernel-time or memory-traffic residual is out-of-band; a kernel that classifies
    to no modeled op (or to an op the graph didn't predict) is unmodeled work and is
    kept. With no predicted graph at all, every kernel is kept.
    """
    obs = trace.kernels()
    pred = graph.nodes
    inv_kt = next((i for i in invariants if i.id == "kernel_time"), None)
    inv_mt = next((i for i in invariants if i.id == "memory_traffic"), None)

    if not pred:
        # Nothing predicted → all observed kernels are unmodeled departures.
        return DeviationResult(kept_indices=list(range(len(obs))), n_observed=len(obs),
                               n_predicted=0)

    # One representative predicted node per op — per-layer nodes share the same
    # roofline prediction, so we match by op identity, not ordinal position.
    by_op: dict[str, PredictedNode] = {}
    for pn in pred:
        by_op.setdefault(pn.op, pn)

    kept: list[int] = []
    for i, ok in enumerate(obs):
        op = ok.range_op or classify_op(ok.name)
        pn = by_op.get(op) if op is not None else None
        if pn is None:
            kept.append(i)  # unmodeled op → keep as a departure
            continue
        if _departs(ok, pn.prediction.t_pred_s, pn.prediction.bytes, inv_kt, inv_mt):
            kept.append(i)

    return DeviationResult(kept_indices=kept, n_observed=len(obs), n_predicted=len(pred))


def deviation_trace(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> Trace:
    """Return a copy of ``trace`` keeping only kernels that depart from prediction.

    Non-kernel events (memcpy/sync) are dropped — the predicted graph models
    kernels, so deviation is only defined over them. The header (workload id,
    fingerprint, run id, duration) is preserved so the reduced trace is still a
    well-formed, self-describing :class:`Trace`.
    """
    obs = trace.kernels()
    dev = deviating_kernel_indices(trace, graph, invariants)
    kept_events = [obs[i] for i in dev.kept_indices]
    return trace.model_copy(update={"events": kept_events})


def deviation_summary(
    trace: Trace, graph: Graph, invariants: tuple[Invariant, ...] = INVARIANTS
) -> dict:
    """Compact summary of the deviation filter — for the run dir / report.

    ``kept_ops`` counts departures per op — the kernel's NVTX-range identity
    when the capture has it, else its :func:`classify_op` name guess — so the
    report says which ops actually departed. ``<unmodeled>`` for kernels that
    map to no predicted op either way.
    """
    obs = trace.kernels()
    dev = deviating_kernel_indices(trace, graph, invariants)
    kept_ops: dict[str, int] = {}
    for i in dev.kept_indices:
        ok = obs[i]
        op = ok.range_op or classify_op(ok.name) or "<unmodeled>"
        kept_ops[op] = kept_ops.get(op, 0) + 1
    return {
        "n_observed": dev.n_observed,
        "n_predicted": dev.n_predicted,
        "n_kept": dev.n_kept,
        "reduction": dev.reduction,
        "kept_ops": kept_ops,
    }

def write_deviation_jsonl(reduced: Trace, path: str | Path) -> None:
    """Write a deviation-only trace as JSONL via the canonical trace writer.

    Delegates to :func:`gitm.tracer.capture.write_trace_jsonl` so the reduced
    trace uses the exact same on-disk format as a full capture (one definition,
    no drift) and round-trips through the same loaders.
    """
    write_trace_jsonl(path, reduced)

def stream_observed(path: str | Path) -> tuple[dict[str, list], int, int, int]:
    """``(per_op, n_kernels, total_ns, span_ns)`` from a trace, without loading it.

    ``per_op`` maps a predicted-op name to ``[count, total_ns]``, with
    ``"<unmodeled>"`` collecting every kernel that classifies to no node. The
    NVTX range identity wins when present, exactly as in
    :func:`deviation_summary`, so the two agree on what a kernel is.
    """
    import json

    per_op: dict[str, list] = {}
    n = total = 0
    t_min: int | None = None
    t_max: int | None = None
    cache: dict[str, str | None] = {}

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") != "kernel":
                continue
            start, end = d.get("start_ns"), d.get("end_ns")
            if start is None or end is None:
                continue
            name = d.get("name") or ""
            op = d.get("range_op")
            if not op:
                if name not in cache:
                    cache[name] = classify_op(name)
                op = cache[name]
            slot = per_op.setdefault(op or "<unmodeled>", [0, 0])
            dur = max(0, end - start)
            slot[0] += 1
            slot[1] += dur
            n += 1
            total += dur
            t_min = start if t_min is None or start < t_min else t_min
            t_max = end if t_max is None or end > t_max else t_max

    span = (t_max - t_min) if (t_min is not None and t_max is not None) else 0
    return per_op, n, total, span


def phase_anchors(path: str | Path) -> list[tuple[int, str]]:
    """``[(start_ns, phase)]`` for every kernel that names its own phase, sorted.

    These are the fixed points the rest of the timeline is inferred from. Gated
    DeltaNet runs in 30 of Qwen3.6's 40 layers and its kernels differ by phase,
    so anchors land every few microseconds — dense enough that most untagged
    kernels sit between two that agree.
    """
    import json

    from gitm.tracer.kernel_taxonomy import classify_phase

    cache: dict[str, str | None] = {}
    out: list[tuple[int, str]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            # Cheap reject before paying for json.loads. Deliberately not
            # '"kind":"kernel"' — pydantic writes compact JSON but json.dumps
            # spaces after the colon, and a prefilter that silently matches only
            # one spelling finds zero anchors and disables propagation without
            # any error.
            if '"kernel"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            name, start = d.get("name") or "", d.get("start_ns")
            if start is None:
                continue
            if name not in cache:
                cache[name] = classify_phase(name)
            if cache[name]:
                out.append((int(start), cache[name]))
    out.sort()
    return out


def stream_by_phase(path: str | Path, *, propagate: bool = True):
    """``(by_phase, stats)`` — kernels bucketed by phase and taxonomy.

    ``by_phase`` is ``{phase: {bucket: [count, ns]}}``. ``stats`` reports how the
    phase was decided, which is the part that governs how much to trust it.

    Only some kernels name their phase; MoE, the GEMMs, norms and elementwise
    work are byte-identical in both and are roughly three quarters of device
    time. With ``propagate`` those inherit the phase of the nearest kernel that
    *does* name one, on the reasoning that kernels adjacent in time belong to the
    same engine step.

    The assumption fails where a step genuinely mixes phases — chunked prefill
    schedules a prefill chunk alongside decode requests — so ``stats`` carries
    the median and worst distance to an anchor. A few microseconds means the
    neighbours are in the same step; milliseconds means the inference spans a
    step boundary and should not be believed.
    """
    import bisect
    import json
    import statistics

    from gitm.tracer.kernel_taxonomy import classify_kernel, classify_phase

    anchors = phase_anchors(path) if propagate else []
    times = [a[0] for a in anchors]

    by_phase: dict[str, dict[str, list]] = {}
    ph_cache: dict[str, str | None] = {}
    bk_cache: dict[str, str] = {}
    gaps: list[int] = []
    n_direct = n_inferred = n_unknown = 0

    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("kind") != "kernel":
                continue
            start, end = d.get("start_ns"), d.get("end_ns")
            if start is None or end is None:
                continue
            name = d.get("name") or ""
            if name not in ph_cache:
                ph_cache[name] = classify_phase(name)
                bk_cache[name] = classify_kernel(name)

            phase = ph_cache[name]
            if phase is not None:
                n_direct += 1
            elif times:
                i = bisect.bisect_left(times, start)
                cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
                j = min(cands, key=lambda j: abs(times[j] - start))
                phase = anchors[j][1]
                gaps.append(abs(times[j] - start))
                n_inferred += 1
            else:
                phase = "unknown"
                n_unknown += 1

            slot = by_phase.setdefault(phase, {}).setdefault(bk_cache[name], [0, 0])
            slot[0] += 1
            slot[1] += max(0, end - start)

    stats = {
        "n_direct": n_direct,
        "n_inferred": n_inferred,
        "n_unknown": n_unknown,
        "n_anchors": len(anchors),
        "gap_median_ns": int(statistics.median(gaps)) if gaps else 0,
        "gap_max_ns": max(gaps) if gaps else 0,
    }
    return by_phase, stats


def render_by_phase(by_phase, stats) -> str:
    """Prefill and decode side by side, bucketed, with the inference qualified."""
    totals = {ph: sum(v[1] for v in b.values()) for ph, b in by_phase.items()}
    grand = sum(totals.values()) or 1

    out = [f"  {'phase':9s} {'ms':>10s} {'share':>7s} {'kernels':>11s}"]
    for ph in ("prefill", "decode", "unknown"):
        if ph not in totals:
            continue
        n = sum(v[0] for v in by_phase[ph].values())
        out.append(f"  {ph:9s} {totals[ph] / 1e6:10.1f} {totals[ph] / grand:7.1%} {n:11,}")

    for ph in ("prefill", "decode", "unknown"):
        buckets = by_phase.get(ph)
        if not buckets:
            continue
        out.append(f"\n  {ph}")
        sub = totals[ph] or 1
        for bk, (n, ns) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
            out.append(f"    {bk:14s} {ns / 1e6:9.1f} ms {ns / sub:7.1%} {n:10,}")

    total_k = stats["n_direct"] + stats["n_inferred"] + stats["n_unknown"]
    if total_k:
        out.append(
            f"\n  phase named by the kernel itself: {stats['n_direct'] / total_k:.1%} "
            f"({stats['n_anchors']:,} anchors)"
        )
    if stats["n_inferred"]:
        out.append(
            f"  inferred from the nearest anchor:  {stats['n_inferred'] / total_k:.1%}, "
            f"median gap {stats['gap_median_ns'] / 1e3:.1f} us, "
            f"worst {stats['gap_max_ns'] / 1e6:.2f} ms"
        )
        out.append(
            "    A gap of microseconds means the neighbour is in the same engine step.\n"
            "    Milliseconds means the inference crossed a step boundary — under chunked\n"
            "    prefill a single step mixes both phases, and no name can separate those."
        )
    return "\n".join(out)


def predicted_per_op(graph: Graph) -> dict[str, float]:
    """Predicted seconds per op, summed over layers — the shape ``stream_observed``
    produces, so the two can be differenced directly."""
    out: dict[str, float] = {}
    for node in graph.nodes:
        out[node.op] = out.get(node.op, 0.0) + node.prediction.t_pred_s
    return out


def render_deviation(
    per_op: dict[str, list],
    pred: dict[str, float] | None,
    *,
    n_kernels: int,
    total_ns: int,
    span_ns: int,
    steps: int | None,
    band: float,
) -> str:
    """The op-level subtraction as a table.

    Two categories that must not be conflated. A **modelled** op reports observed
    against floor as a ratio; above 1 is the implementation leaving something on
    the table, which is normal, and only its size is interesting. **Unmodelled**
    work is not a deviation at all — it is the graph's coverage gap, and reading
    it as headroom is the error this separation exists to prevent.
    """
    obs_s = total_ns / 1e9
    busy = f" over a {span_ns / 1e9:.1f} s window ({total_ns / span_ns:.1%} busy)" if span_ns else ""
    out = [f"observed  {n_kernels:,} kernels, {obs_s:.3f} s device time{busy}"]
    if steps:
        out.append(f"window    {steps:,} steps -> {obs_s / steps * 1e3:.3f} ms/step observed")
    out.append("")

    if pred is None:
        out.append(f"  {'op':24s} {'kernels':>12s} {'time_s':>9s} {'share':>7s}")
        for op, (c, ns) in sorted(per_op.items(), key=lambda kv: -kv[1][1]):
            out.append(f"  {op:24s} {c:12,} {ns / 1e9:9.3f} {ns / total_ns:6.1%}")
        return "\n".join(out)

    scale = steps or 1
    out.append(f"  {'op':24s} {'obs_ms':>9s} {'floor_ms':>9s} {'ratio':>7s} "
               f"{'kernels':>11s}  verdict")
    for op, (count, ns) in sorted(per_op.items(), key=lambda kv: -kv[1][1]):
        obs_ms = ns / 1e6
        if op == "<unmodeled>":
            out.append(f"  {op:24s} {obs_ms:9.1f} {'-':>9s} {'-':>7s} {count:11,}  "
                       "not in the graph")
            continue
        floor_ms = pred.get(op, 0.0) * scale * 1e3
        if floor_ms <= 0:
            out.append(f"  {op:24s} {obs_ms:9.1f} {'-':>9s} {'-':>7s} {count:11,}  "
                       "observed but not predicted")
            continue
        ratio = obs_ms / floor_ms
        verdict = "within band" if ratio <= 1.0 + band else f"{ratio:.1f}x over floor"
        out.append(f"  {op:24s} {obs_ms:9.1f} {floor_ms:9.1f} {ratio:7.2f} "
                   f"{count:11,}  {verdict}")

    missing = sorted(set(pred) - set(per_op))
    if missing:
        out.append("")
        out.append(f"  predicted but never observed: {', '.join(missing)}")
        out.append("    Either the op did not run, or no kernel name classifies to it —")
        out.append("    the second is a taxonomy gap, not a finding about the model.")

    unmod = per_op.get("<unmodeled>", [0, 0])[1]
    if unmod:
        out.append("")
        out.append(f"  unmodeled work is {unmod / total_ns:.1%} of device time: the graph's")
        out.append("  coverage gap, not headroom. The floor never claimed to predict it.")
    return "\n".join(out)


def add_deviate_arguments(ap):
    ap.add_argument("trace", type=Path, help="A captured trace.jsonl.")
    ap.add_argument("--model", default=None,
                    help="Catalogue entry name or config.json. Required unless --no-graph.")
    ap.add_argument("--gpu", default=None, help="SKU to price the floor against.")
    ap.add_argument("--batch", type=int, default=1, help="Sequences per decode step.")
    ap.add_argument("--kv-len", type=int, default=4096, help="Tokens cached per sequence.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Decode steps in the window; scales the floor so observed and "
                         "predicted are directly comparable.")
    # ── prefill ──────────────────────────────────────────────────────────────
    # Zero means a pure decode step, which is what every caller wanted before
    # prefill was modelled. A step is a *chunk*, bounded by
    # --max-num-batched-tokens (8192 by default), not by prompt length.
    ap.add_argument("--prefill-tokens", type=int, default=0,
                    help="Query tokens being prefilled this step. 0 = pure decode.")
    ap.add_argument("--prefill-context", type=int, default=0,
                    help="Context already cached before this chunk (0 for a first chunk).")
    ap.add_argument("--prefill-requests", type=int, default=1,
                    help="How many prompts those tokens belong to — sets lm_head rows.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--no-graph", action="store_true",
                    help="Report observed op totals only, with no prediction.")
    ap.add_argument("--by-phase", dest="by_phase", action="store_true",
                    help="Split kernels into prefill and decode instead of comparing "
                         "against a predicted floor.")
    ap.add_argument("--json", dest="as_json", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from gitm.planner.roofline import BatchConfig, ShardingConfig

    ap = add_deviate_arguments(argparse.ArgumentParser(
        prog="gitm deviate",
        description="Subtract a predicted graph from a captured trace.",
    ))
    args = ap.parse_args(argv)

    if not args.trace.is_file():
        print(f"cannot read trace: {args.trace}")
        return 2
    # --by-phase reports what ran, not how it compares to a floor, so it needs no
    # model — and requiring one would make the cheapest view the most awkward.
    if not args.model and not args.no_graph and not args.by_phase:
        ap.error("--model is required unless --no-graph or --by-phase")

    if args.by_phase:
        print(render_by_phase(*stream_by_phase(args.trace)))
        return 0

    per_op, n_kernels, total_ns, span_ns = stream_observed(args.trace)
    if not n_kernels:
        print(f"no kernel records in {args.trace} — nothing to subtract.")
        return 1

    pred: dict[str, float] | None = None
    if not args.no_graph:
        from gitm.planner.registry import _hardware, _load, _predict

        try:
            spec, family, _note = _load(args.model)
        except (FileNotFoundError, ValueError) as e:
            print(f"cannot build a graph: {e}")
            return 2
        if family == "dense" or spec is None:
            print(f"cannot build a graph: {args.model} resolves to the dense family.")
            return 2
        try:
            g = _predict(spec, family, _hardware(args.gpu),
                         BatchConfig(batch=args.batch, kv_cache_len=args.kv_len,
                                     prefill_tokens=args.prefill_tokens,
                                     prefill_context=args.prefill_context,
                                     prefill_requests=args.prefill_requests),
                         ShardingConfig(tp=args.tp, ep=args.ep))
        except ValueError as e:
            print(f"cannot build a graph: {e}")
            return 2
        pred = predicted_per_op(g)

    band = next((i.band_width for i in INVARIANTS if i.id == "kernel_time"), 0.4)

    if args.as_json:
        scale = args.steps or 1
        print(json.dumps({
            "trace": str(args.trace),
            "n_kernels": n_kernels,
            "device_time_s": total_ns / 1e9,
            "window_s": span_ns / 1e9,
            "steps": args.steps,
            "band_width": band,
            "ops": {
                op: {
                    "kernels": c,
                    "observed_s": ns / 1e9,
                    "floor_s": (pred.get(op, 0.0) * scale)
                    if pred and op != "<unmodeled>" else None,
                }
                for op, (c, ns) in sorted(per_op.items(), key=lambda kv: -kv[1][1])
            },
        }, indent=2))
        return 0

    print(render_deviation(per_op, pred, n_kernels=n_kernels, total_ns=total_ns,
                           span_ns=span_ns, steps=args.steps, band=band))
    return 0
