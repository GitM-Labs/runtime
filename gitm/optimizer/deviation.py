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
# The projection GEMMs (qkv/out/gate_up/down/lm_head) only classify when the
# kernel name carries a projection tag; a bare cuBLAS/cutlass GEMM (e.g.
# `ampere_fp16_s16816gemm_*`) carries none and stays unmodeled — confirmed
# against a real vLLM/L4/CUPTI trace, where these are ~35% of launches and
# reused across every projection. Needs shape-matching or launch-order
# instrumentation to fix, not a vocabulary tweak. The attention/KV-cache
# needles below ARE confirmed against that trace: FlashAttention's real
# kernel is `flash_fwd_splitkv_kernel` (`flash_attn` alone misses it), and
# vLLM's `reshape_and_cache_flash_kernel`/`_compute_slot_mapping_kernel`
# weren't covered before.
_OP_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("flash_attn", "flashattn", "flash_fwd", "paged_attention", "paged_attn", "fmha",
      "attention", "attn_score", "reshape_and_cache", "slot_mapping"), "attn_score_value"),
    (("qkv",), "qkv_proj"),
    (("o_proj", "out_proj", "attn_out"), "attn_out_proj"),
    (("gate_up", "gate_proj", "up_proj", "swiglu", "silu_and_mul"), "mlp_gate_up"),
    (("down_proj", "mlp_down"), "mlp_down"),
    (("lm_head", "logits", "vocab_proj", "embed"), "lm_head"),
)


def classify_op(kernel_name: str) -> str | None:
    """Map a raw kernel name to a predicted-graph op, or ``None`` if unmodeled.

    Case-insensitive substring match, first rule wins. ``None`` = the kernel maps
    to no op in the predicted graph (a norm/activation/copy, or a bare GEMM whose
    name doesn't carry its projection) → treated as unmodeled work.
    """
    n = kernel_name.lower()
    for needles, op in _OP_RULES:
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
