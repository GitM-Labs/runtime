"""Deviation monitor — emits residuals only.

    residuals(trace, graph) -> Residuals
    check_invariants(residuals, INVARIANTS) -> list[Violation]

Storage scales with deviation, not duration. Severity normalized across
invariants so attribution doesn't need per-invariant logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gitm.optimizer.deviation import classify_op
from gitm.optimizer.invariants import INVARIANTS, Invariant, Violation
from gitm.optimizer.multibasis import confirmed_positions
from gitm.planner.graph import Graph, PredictedNode
from gitm.tracer.schema import KernelEvent, Trace


@dataclass
class KernelResidual:
    op: str
    layer: int | None
    r_kt: float  # kernel-time residual
    r_mt: float | None  # memory-traffic residual (None if bytes unavailable)
    t_obs_s: float | None = None
    t_pred_s: float | None = None
    bound: str | None = None  # the matched op's roofline bound: "compute" | "memory"
    #: How many structurally distinct predictions the op has across layers. 1 for
    #: every dense model and for most ops of a heterogeneous one. Above 1 without
    #: a resolved ``layer`` means this residual was measured against an interval
    #: rather than a point — see :func:`residuals`.
    n_classes: int = 1

    @property
    def interval_based(self) -> bool:
        """True when the op's layers disagree and this kernel's layer is unknown.

        Such a residual is *conservative*: zero anywhere inside the span of the
        op's per-layer predictions. It cannot be read as "this kernel matched
        prediction", only as "it was not outside every prediction".
        """
        return self.n_classes > 1 and self.layer is None


@dataclass
class Residuals:
    """Residuals against predicted graph. Per-kernel + per-stream-set."""

    per_kernel: list[KernelResidual] = field(default_factory=list)
    serialized_concurrency_fraction: float = 0.0
    total_kernels: int = 0
    classified_kernels: int = 0
    matched_kernels: int = 0
    total_kernel_time_ns: int = 0
    classified_kernel_time_ns: int = 0
    matched_kernel_time_ns: int = 0

    @staticmethod
    def _ratio(part: int, whole: int) -> float:
        return part / whole if whole > 0 else 0.0

    @property
    def classification_coverage(self) -> float:
        return self._ratio(self.classified_kernels, self.total_kernels)

    @property
    def match_coverage(self) -> float:
        return self._ratio(self.matched_kernels, self.total_kernels)

    @property
    def classified_time_coverage(self) -> float:
        return self._ratio(self.classified_kernel_time_ns, self.total_kernel_time_ns)

    @property
    def matched_time_coverage(self) -> float:
        return self._ratio(self.matched_kernel_time_ns, self.total_kernel_time_ns)

    @property
    def coverage_warnings(self) -> list[str]:
        """Human-facing notes for work excluded from residual conclusions."""
        if self.total_kernels == 0:
            return ["residual coverage unavailable: trace contains no kernels"]
        warnings: list[str] = []
        if self.classified_kernels < self.total_kernels:
            warnings.append(
                "residual coverage: classified "
                f"{self.classification_coverage:.1%} of launches and "
                f"{self.classified_time_coverage:.1%} of kernel time "
                f"({self.classified_kernels}/{self.total_kernels} kernels)"
            )
        if self.matched_kernels < self.total_kernels:
            warnings.append(
                "residual coverage: matched to the predicted graph "
                f"{self.match_coverage:.1%} of launches and "
                f"{self.matched_time_coverage:.1%} of kernel time "
                f"({self.matched_kernels}/{self.total_kernels} kernels)"
            )
        return warnings


def _class_key(pn: PredictedNode) -> tuple[float, float]:
    """What makes two per-layer nodes of the same op structurally interchangeable.

    Nodes computed by the same formula come out bit-identical, so exact equality
    would do; the rounding is insurance against a future term that introduces
    float drift between layers that are meant to be the same.
    """
    return (round(pn.prediction.t_pred_s, 15), round(pn.prediction.bytes, 6))


def _interval_residual(obs: float, lo: float, hi: float) -> float:
    """Signed residual against a prediction *interval* instead of a point.

    Zero anywhere inside ``[lo, hi]``; outside, the distance to the nearest edge
    relative to that edge. Deliberately conservative — when the op's layers
    disagree and the kernel's layer is unknown, the honest prediction is the span
    they cover, and only an observation outside all of it is evidence of
    anything.
    """
    if obs < lo:
        return (obs - lo) / lo if lo > 0 else 0.0
    if obs > hi:
        return (obs - hi) / hi if hi > 0 else 0.0
    return 0.0


def residuals(trace: Trace, graph: Graph) -> Residuals:
    """Pair observed kernels to predicted nodes by op identity, not position.

    The old ordinal pairing matched a handful of early kernels against unrelated
    ops (orders of magnitude fewer predicted nodes than real kernels), producing
    runaway r_kt ratios. Each kernel is classified by its NVTX-range identity when
    the capture has one (``range_op``/``range_layer`` — see
    :mod:`gitm.tracer.nvtx_correlate`), else by name
    (:func:`gitm.optimizer.deviation.classify_op`).

    **Pairing is per structural class, not one representative per op.** A dense
    transformer repeats one layer, so any node stands for all of them — that
    assumption is why this used to keep a single node per op. It does not survive
    a heterogeneous stack. On a DeepSeek-V4-class model, layers 0-1 are
    sliding-window (128 tokens, no indexer) while 2-42 are compressed (640
    tokens), and the compressed layers themselves split 32x apart at the indexer.
    Taking the first node per op made layer 0 the yardstick for all 43, which
    scores a perfectly healthy compressed layer at ``r_kt = +4.0`` against a
    ±0.4 band — and ``check_invariants`` treats a systematic offset as
    *confirmation*, so multi-basis filtering amplifies the artefact instead of
    rejecting it.

    Three cases, in order:

    * **Layer known** (NVTX capture): the exact ``(op, layer)`` node. Point
      residual, and ``layer`` is carried through.
    * **Layer unknown, op uniform**: the single class. Identical to the previous
      behaviour, which is every dense model and 8 of V4's 10 per-layer ops.
    * **Layer unknown, op heterogeneous**: an interval over the op's classes (see
      :func:`_interval_residual`), flagged by ``n_classes > 1``. Real deviations
      outside the whole span still surface; deviations *within* it are given up
      rather than guessed at, because guessing is what produced the artefact.

    The third case degrades to the first the moment NVTX ranges land, with no
    change here.
    """
    obs = trace.kernels()
    pred = graph.nodes

    durations = [max(ok.end_ns - ok.start_ns, 0) for ok in obs]
    res = Residuals(
        total_kernels=len(obs),
        total_kernel_time_ns=sum(durations),
    )
    by_op_layer: dict[tuple[str, int], PredictedNode] = {}
    classes: dict[str, dict[tuple[float, float], PredictedNode]] = {}
    for pn in pred:
        if pn.layer is not None:
            by_op_layer.setdefault((pn.op, pn.layer), pn)
        classes.setdefault(pn.op, {}).setdefault(_class_key(pn), pn)

    for ok, duration_ns in zip(obs, durations, strict=True):
        op = ok.range_op or classify_op(ok.name)
        if op is None:
            continue
        res.classified_kernels += 1
        res.classified_kernel_time_ns += duration_ns
        cls = list(classes.get(op, {}).values())
        if not cls:
            continue
        res.matched_kernels += 1
        res.matched_kernel_time_ns += duration_ns

        t_obs = max((ok.end_ns - ok.start_ns) / 1e9, 1e-12)
        b_obs = (
            ok.bytes_read + ok.bytes_written
            if ok.bytes_read is not None and ok.bytes_written is not None
            else None
        )

        pn: PredictedNode | None = None
        if ok.range_layer is not None:
            pn = by_op_layer.get((op, ok.range_layer))
        if pn is None and len(cls) == 1:
            pn = cls[0]

        if pn is not None:
            t_pred = max(pn.prediction.t_pred_s, 1e-12)
            r_kt = (t_obs - t_pred) / t_pred
            r_mt = (
                (b_obs - pn.prediction.bytes) / pn.prediction.bytes
                if b_obs is not None and pn.prediction.bytes > 0
                else None
            )
            layer, bound = ok.range_layer, pn.prediction.bound
        else:
            ts = sorted(max(c.prediction.t_pred_s, 1e-12) for c in cls)
            r_kt = _interval_residual(t_obs, ts[0], ts[-1])
            bs = sorted(c.prediction.bytes for c in cls if c.prediction.bytes > 0)
            r_mt = _interval_residual(b_obs, bs[0], bs[-1]) if b_obs is not None and bs else None
            # Report the class the observation actually sits nearest, so the
            # bound and t_pred in the record describe a real layer rather than an
            # average of layers that don't resemble each other.
            nearest = min(cls, key=lambda c: abs(c.prediction.t_pred_s - t_obs))
            t_pred = nearest.prediction.t_pred_s
            layer, bound = None, nearest.prediction.bound

        res.per_kernel.append(
            KernelResidual(
                op=op, layer=layer, r_kt=r_kt, r_mt=r_mt,
                t_obs_s=t_obs, t_pred_s=t_pred, bound=bound, n_classes=len(cls),
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
