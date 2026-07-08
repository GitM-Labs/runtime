"""Deviation monitor — emits residuals only.

    residuals(trace, graph) -> Residuals
    check_invariants(residuals, INVARIANTS) -> list[Violation]

Storage scales with deviation, not duration. Severity normalized across
invariants so attribution doesn't need per-invariant logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gitm.optimizer.invariants import INVARIANTS, Invariant, Violation
from gitm.optimizer.multibasis import confirmed_positions
from gitm.planner.graph import Graph
from gitm.tracer.schema import KernelEvent, Trace


@dataclass
class KernelResidual:
    op: str
    layer: int | None
    r_kt: float  # kernel-time residual
    r_mt: float | None  # memory-traffic residual (None if bytes unavailable)
    t_obs_s: float | None = None
    t_pred_s: float | None = None


@dataclass
class Residuals:
    """Residuals against predicted graph. Per-kernel + per-stream-set."""

    per_kernel: list[KernelResidual] = field(default_factory=list)
    serialized_concurrency_fraction: float = 0.0


def residuals(trace: Trace, graph: Graph) -> Residuals:
    """Pair observed kernels to predicted nodes by ordinal position.

    v0 alignment: predicted nodes and observed kernels are matched by index
    after filtering. The attribution layer doesn't require perfect alignment
    — bad pairings show as outsized residuals that the Granger pass localizes.
    """
    obs = trace.kernels()
    pred = graph.nodes

    res = Residuals()
    n = min(len(obs), len(pred))
    for i in range(n):
        ok: KernelEvent = obs[i]
        pn = pred[i]
        t_obs = max((ok.end_ns - ok.start_ns) / 1e9, 1e-12)
        t_pred = max(pn.prediction.t_pred_s, 1e-12)
        r_kt = (t_obs - t_pred) / t_pred

        if ok.bytes_read is not None and ok.bytes_written is not None and pn.prediction.bytes > 0:
            b_obs = ok.bytes_read + ok.bytes_written
            r_mt: float | None = (b_obs - pn.prediction.bytes) / pn.prediction.bytes
        else:
            r_mt = None

        res.per_kernel.append(
            KernelResidual(
                op=pn.op, layer=pn.layer, r_kt=r_kt, r_mt=r_mt,
                t_obs_s=t_obs, t_pred_s=t_pred,
            )
        )

    res.serialized_concurrency_fraction = _serialized_fraction(obs)
    return res


def _serialized_fraction(obs: list[KernelEvent]) -> float:
    """Fraction of adjacent kernel pairs that executed serialized.

    Sort observed kernels by start time; a consecutive pair is *serialized* when
    the later kernel starts after the earlier one ends (no temporal overlap)
    while sharing a stream — concurrency a well-tuned pipeline would have
    achieved was lost. 0.0 = fully overlapped, 1.0 = fully sequential. Computed
    from the real trace (stream IDs + ns timestamps), not assumed.
    """
    if len(obs) < 2:
        return 0.0
    s = sorted(obs, key=lambda k: k.start_ns)
    pairs = serialized = 0
    for a, b in zip(s, s[1:], strict=False):
        pairs += 1
        overlapped = b.start_ns < a.end_ns
        if not overlapped and a.stream_id == b.stream_id:
            serialized += 1
    return serialized / pairs if pairs else 0.0


def check_invariants(
    residuals_: Residuals,
    invariants: tuple[Invariant, ...] = INVARIANTS,
    *,
    multi_basis: bool = True,
) -> list[Violation]:
    """Emit a Violation per out-of-band residual.

    With ``multi_basis`` (default), a *kernel-time* deviation is reported only
    when it is confirmed in 2+ bases (a transient anomaly — see
    :mod:`gitm.optimizer.multibasis`) or systematic for its op (median residual
    over band). This suppresses single-basis noise without dropping systematic
    shifts. Memory-traffic and stream-concurrency use the direct band check.
    """
    out: list[Violation] = []
    inv_kt = next((i for i in invariants if i.id == "kernel_time"), None)
    inv_mt = next((i for i in invariants if i.id == "memory_traffic"), None)
    inv_sc = next((i for i in invariants if i.id == "stream_concurrency"), None)

    # Kernel-time confirmed-anomaly set: multi-basis transient ∪ systematic shift.
    confirmed: set[tuple[str, int]] | None = None
    if multi_basis and inv_kt is not None:
        series_by_op: dict[str, list[float]] = {}
        for kr in residuals_.per_kernel:
            series_by_op.setdefault(kr.op, []).append(kr.r_kt)
        confirmed = confirmed_positions(series_by_op)
        for op, vals in series_by_op.items():
            if abs(float(np.median(vals))) > inv_kt.band_width:  # systematic
                confirmed.update((op, i) for i, v in enumerate(vals) if abs(v) > inv_kt.band_width)

    op_idx: dict[str, int] = {}
    for kr in residuals_.per_kernel:
        i = op_idx.get(kr.op, 0)
        op_idx[kr.op] = i + 1

        if inv_kt is not None and abs(kr.r_kt) > inv_kt.band_width:
            if confirmed is None or (kr.op, i) in confirmed:
                out.append(
                    Violation(
                        invariant="kernel_time",
                        node_op=kr.op,
                        layer=kr.layer,
                        residual=kr.r_kt,
                        severity=min(abs(kr.r_kt) / inv_kt.band_width, 1.0),
                    )
                )
        if (
            inv_mt is not None
            and kr.r_mt is not None
            and abs(kr.r_mt) > inv_mt.band_width
        ):
            out.append(
                Violation(
                    invariant="memory_traffic",
                    node_op=kr.op,
                    layer=kr.layer,
                    residual=kr.r_mt,
                    severity=min(abs(kr.r_mt) / inv_mt.band_width, 1.0),
                )
            )

    if (
        inv_sc is not None
        and residuals_.serialized_concurrency_fraction > inv_sc.band_width * 0.5
    ):
        out.append(
            Violation(
                invariant="stream_concurrency",
                node_op="<stream-set>",
                layer=None,
                residual=residuals_.serialized_concurrency_fraction,
                severity=min(residuals_.serialized_concurrency_fraction / inv_sc.band_width, 1.0),
            )
        )
    return out
