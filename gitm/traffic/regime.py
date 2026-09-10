"""Workload regimes — the coordinates a playbook row is stated in.

A :class:`Regime` is what turns "this knob won once" into "this knob wins in
decode-heavy bursty traffic". Every generated workload carries one, the harness
writes :meth:`Regime.label` into every result row, and deliverable 4 keys the
playbook on it.

Two representations, deliberately both:

* the **raw axes** (floats), which deliverable 4's match semantics measure a
  distance on — live traffic never lands exactly on a measured point;
* the **label** (:meth:`Regime.label`), coarse and bucketed, which is what a
  result row carries and a human reads. A label built from raw floats would
  never match twice.

The burstiness axis is the **index of dispersion** — variance over mean of the
arrival counts per bin — not the coefficient of variation of interarrival times.
Both are 1 for a Poisson process, but two traces with the same mean rate and
different bunching collapse to one CV under a fixed mean, and bunching is exactly
the axis the customer's traffic varies on.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gitm.traffic.schema import Trace

#: Default counting-bin width for the burstiness statistic, seconds. The index of
#: dispersion is only meaningful relative to a bin: 1 s reads sub-second bunching
#: as burst, 60 s reads it as smooth. Pinned here so every regime is comparable.
DEFAULT_BIN_S = 1.0


class SourceKind(str, Enum):
    """Where a workload came from — a schema field, not a naming convention.

    ``SCOREBOARD`` exists so Artificial Analysis's fixed-length benchmark can
    never be read as production traffic. It is the public scoreboard condition;
    a playbook row measured under it says something about the scoreboard, and the
    distinction has to survive being copied into a spreadsheet.
    """

    PRODUCTION = "production"  # replayed from a real production trace
    SYNTHETIC = "synthetic"  # sampled from a fitted envelope
    SCOREBOARD = "scoreboard"  # fixed-length public benchmark condition


def _bucket_tokens(n: float) -> str:
    """Round a token count down to a power of two and render it compactly."""
    if n < 1:
        return "0"
    exp = int(np.floor(np.log2(n)))
    v = 1 << exp
    return f"{v // 1024}k" if v >= 1024 else str(v)


def _bucket_ratio(r: float) -> str:
    if r <= 0:
        return "io0"
    exp = int(np.floor(np.log2(r)))
    return f"io{2**exp}" if exp >= 0 else f"io1-{2 ** -exp}"


def _bucket_burst(d: float) -> str:
    if d < 0.8:
        return "burst-flat"  # more regular than Poisson (paced or rate-limited)
    if d < 1.5:
        return "burst-poisson"
    if d < 5.0:
        return "burst-mod"
    return "burst-hi"


def index_of_dispersion(arrivals: list[float], *, bin_s: float, span_s: float) -> float:
    """Variance / mean of arrival counts per ``bin_s`` bin. 1.0 for Poisson.

    Returns 1.0 when the statistic is undefined — fewer than two bins, or a zero
    mean — rather than raising, so a degenerate trace still produces a regime.
    The caller learns it was degenerate from :attr:`Regime.burstiness_defined`.
    """
    if span_s <= 0 or len(arrivals) < 2:
        return 1.0
    nbins = max(int(np.ceil(span_s / bin_s)), 1)
    if nbins < 2:
        return 1.0
    counts = np.bincount(
        np.minimum((np.asarray(arrivals) / bin_s).astype(np.int64), nbins - 1),
        minlength=nbins,
    )
    mean = counts.mean()
    return float(counts.var() / mean) if mean > 0 else 1.0


class Regime(BaseModel):
    """One point in workload space, plus the identity of what produced it."""

    model_config = ConfigDict(extra="forbid")

    source_kind: SourceKind
    trace: str  # trace identity — adapter name, or the synthetic generator's id
    requests: int
    rate_rps: float
    io_ratio: float  # total input tokens / total output tokens
    input_p50: int
    input_p95: int
    output_p50: int
    output_p95: int
    burstiness: float  # index of dispersion at bin_s
    bin_s: float = DEFAULT_BIN_S
    burstiness_defined: bool = True
    concurrency: int | None = None  # offered concurrency cap; None = open-loop
    in_envelope: bool = True  # False when sampled beyond any observed trace
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        source_kind: SourceKind = SourceKind.PRODUCTION,
        bin_s: float = DEFAULT_BIN_S,
        concurrency: int | None = None,
        in_envelope: bool = True,
        trace_id: str | None = None,
    ) -> Regime:
        if not trace.requests:
            raise ValueError("cannot tag an empty trace with a regime")
        inp = np.asarray(trace.input_tokens, dtype=float)
        outs = trace.output_tokens
        out = np.asarray(outs, dtype=float) if outs else np.zeros(1)
        span = trace.meta.span_s
        arrivals = trace.arrivals
        return cls(
            source_kind=source_kind,
            trace=trace_id or trace.meta.source,
            requests=len(trace.requests),
            rate_rps=trace.rate_rps(),
            io_ratio=float(inp.sum() / out.sum()) if out.sum() > 0 else 0.0,
            input_p50=int(np.percentile(inp, 50)),
            input_p95=int(np.percentile(inp, 95)),
            output_p50=int(np.percentile(out, 50)),
            output_p95=int(np.percentile(out, 95)),
            burstiness=index_of_dispersion(arrivals, bin_s=bin_s, span_s=span),
            bin_s=bin_s,
            burstiness_defined=span > 0 and len(arrivals) >= 2,
            concurrency=concurrency,
            in_envelope=in_envelope,
            notes=[] if outs else ["no output lengths in the source; io_ratio is 0"],
        )

    def label(self) -> str:
        """The stable, coarse string the harness writes into every result row.

        Bucketed on purpose: two runs of the same workload must produce the same
        label, and raw quantiles never repeat. An out-of-envelope point is
        suffixed ``/xenv`` so a sampled extrapolation can never be mistaken for a
        measured production point.
        """
        parts = [
            {"production": "prod", "synthetic": "syn", "scoreboard": "board"}[
                self.source_kind.value
            ],
            _bucket_ratio(self.io_ratio),
            f"in{_bucket_tokens(self.input_p50)}",
            f"out{_bucket_tokens(self.output_p50)}",
            _bucket_burst(self.burstiness),
            f"c{self.concurrency}" if self.concurrency else "copen",
        ]
        if not self.in_envelope:
            parts.append("xenv")
        return "/".join(parts)

    def summary(self) -> str:
        return (
            f"{self.label()}  [{self.requests} req, {self.rate_rps:.3f} rps, "
            f"in p50/p95 {self.input_p50}/{self.input_p95}, "
            f"out p50/p95 {self.output_p50}/{self.output_p95}, "
            f"D={self.burstiness:.2f}@{self.bin_s:g}s]"
        )
