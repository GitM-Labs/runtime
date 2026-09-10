"""Validation — prove the pipeline preserves the trace, and show the proof.

The brief asks for this as a deliverable, not a nicety: the replayed stream's
arrival-rate and length distributions compared against the source's, **shown**,
with any mismatch explained. Everything downstream — every regime label, every
promoted playbook row — inherits whatever distortion this step fails to catch.

What gets compared is the file the benchmark will actually consume. The replay
emitter writes a vLLM ``timed_trace`` JSONL;
:func:`gitm.traffic.replay.read_timed_trace` reads that same file back into
canonical form, and :func:`compare` puts it beside the trace the adapter
produced. So "the pipeline preserves the trace" is a measurement of the artifact,
not an argument about the code.

The same function serves the second, looser use: a parameterized sample against
the trace it was fitted on. There the thresholds are wider — a sample is drawn
from the envelope, not copied from it — and :data:`SAMPLED_THRESHOLDS` says so
explicitly rather than leaving a reader to wonder which standard was applied.
"""

from __future__ import annotations

import sys

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gitm.traffic.regime import DEFAULT_BIN_S, index_of_dispersion
from gitm.traffic.schema import Trace


class Thresholds(BaseModel):
    """What counts as preserved. Every number here is a decision, not a default."""

    model_config = ConfigDict(extra="forbid")

    name: str
    #: Two-sample KS statistic on arrival times, and on each length distribution.
    ks: float = 0.02
    #: Relative error on mean request rate.
    rate: float = 0.01
    #: Relative error on the burstiness axis (index of dispersion).
    burstiness: float = 0.10
    #: Relative error on request count. 0 = exact.
    count: float = 0.0
    #: Whether to compare the arrival *timeline* at all. True for a replay, which
    #: must reproduce it. False for a parameterized sample, which reproduces the
    #: rate and the dispersion by construction and the timeline by nothing — a KS
    #: on arrival times there measures only that a sample is not a copy, which is
    #: the point of sampling.
    check_arrival_times: bool = True


#: A replay must reproduce the trace, not resemble it: the emitter is a format
#: change, so every statistic should come back identical and the tolerances are
#: there to absorb float rounding of the timestamps, nothing else.
REPLAY_THRESHOLDS = Thresholds(name="replay", ks=0.001, rate=1e-6, burstiness=1e-6, count=0.0)

#: A parameterized sample is a *draw* from the fitted envelope. Sampling error at
#: a few hundred requests is real, so these are the finite-sample tolerances —
#: loose enough not to fire on noise, tight enough that a broken inverse-CDF or a
#: mis-set dispersion target does fire.
SAMPLED_THRESHOLDS = Thresholds(
    name="sampled", ks=0.15, rate=0.25, burstiness=0.60, count=0.35, check_arrival_times=False
)

#: Arrival times are compared at microsecond resolution. Two reasons, and both
#: are about not lying: a replay's timing fidelity below 1 us is meaningless next
#: to millisecond network jitter, and the emitted JSONL rounds to 6 decimals, so
#: an un-quantized KS reports the 6e-16 s residue of ``5999 / 1000`` as a real
#: distribution gap. It found exactly that on the Mooncake fixture.
ARRIVAL_RESOLUTION_S = 1e-6


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    statistic: float
    threshold: float
    passed: bool
    detail: str = ""


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standard: str
    source: str
    replayed: str
    checks: list[Check] = Field(default_factory=list)
    source_hist: list[int] = Field(default_factory=list)
    replayed_hist: list[int] = Field(default_factory=list)
    hist_bin_s: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        """The shown comparison: the check table plus both arrival-rate profiles.

        ASCII on purpose — this lands in a terminal, a markdown log and a commit
        message, and none of those render a PNG. A plotting dependency would buy
        prettier and lose all three.
        """
        w = max((len(c.name) for c in self.checks), default=4)
        lines = [
            f"trace validation [{self.standard}]  {self.source} -> {self.replayed}",
            f"{'check'.ljust(w)}  {'value':>12}  {'threshold':>12}  result",
            f"{'-' * w}  {'-' * 12}  {'-' * 12}  ------",
        ]
        for c in self.checks:
            mark = "pass" if c.passed else "FAIL"
            lines.append(
                f"{c.name.ljust(w)}  {c.statistic:>12.6g}  {c.threshold:>12.6g}  {mark}"
            )
        lines.append("")
        lines.append(f"arrival rate, {self.hist_bin_s:g}s bins  (S = source, R = replayed)")
        lines.append(_sparkbars(self.source_hist, self.replayed_hist))
        lines.append("")
        lines.append("PASS — the pipeline preserves the trace" if self.passed else self.explain())
        return "\n".join(lines)

    def explain(self) -> str:
        """Prose for every failed check. A printed number is not an explanation."""
        bad = [c for c in self.checks if not c.passed]
        if not bad:
            return "no mismatch to explain"
        out = ["FAIL — the replayed stream differs from the source:"]
        for c in bad:
            out.append(f"  * {c.name}: {c.statistic:.6g} exceeds {c.threshold:.6g}. {c.detail}")
        return "\n".join(out)


_BLOCKS = " ▁▂▃▄▅▆▇█"
_ASCII = " .:-=+*#@"


def _ramp() -> str:
    """Block characters when the console can encode them, ASCII when it cannot.

    A Windows console on cp1252 raises ``UnicodeEncodeError`` on U+2588, which
    would turn "show the comparison" into a crash at exactly the moment someone
    is looking at a failure.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        _BLOCKS.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return _ASCII
    return _BLOCKS


def _spark(counts: list[int], peak: int) -> str:
    ramp = _ramp()
    if peak <= 0:
        return " " * len(counts)
    return "".join(
        ramp[min(int(c / peak * (len(ramp) - 1) + 0.5), len(ramp) - 1)] for c in counts
    )


def _sparkbars(a: list[int], b: list[int]) -> str:
    peak = max([*a, *b, 1])
    return f"  S |{_spark(a, peak)}| peak {max(a, default=0)}\n  R |{_spark(b, peak)}| peak {max(b, default=0)}"


def ks_statistic(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic — the max CDF gap.

    Eight lines of numpy rather than a scipy dependency: only the statistic is
    needed, never the p-value, because the thresholds here are operational
    ("this replay is faithful enough to run on") and not a hypothesis test.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 1.0
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def _quantize(xs: list[float]) -> np.ndarray:
    """Snap arrival times to :data:`ARRIVAL_RESOLUTION_S` before comparing them."""
    return np.rint(np.asarray(xs, dtype=float) / ARRIVAL_RESOLUTION_S)


def _hist(trace: Trace, *, nbins: int, span: float) -> list[int]:
    if span <= 0 or not trace.requests:
        return [len(trace.requests)] + [0] * (nbins - 1)
    idx = np.minimum((np.asarray(trace.arrivals) / span * nbins).astype(int), nbins - 1)
    return [int(v) for v in np.bincount(idx, minlength=nbins)]


def _rel(a: float, b: float) -> float:
    """Relative error of ``b`` against ``a``, with a zero-safe denominator."""
    return abs(a - b) / abs(a) if a else (0.0 if not b else 1.0)


def compare(
    source: Trace,
    replayed: Trace,
    *,
    thresholds: Thresholds = REPLAY_THRESHOLDS,
    bin_s: float = DEFAULT_BIN_S,
    hist_bins: int = 60,
) -> ValidationReport:
    """Compare a replayed (or sampled) trace against its source."""
    checks: list[Check] = []

    def add(name: str, stat: float, thr: float, detail: str) -> None:
        checks.append(
            Check(name=name, statistic=stat, threshold=thr, passed=stat <= thr, detail=detail)
        )

    add(
        "request_count",
        _rel(len(source), len(replayed)),
        thresholds.count,
        f"{len(source)} source vs {len(replayed)} replayed — requests were lost or "
        "invented between the adapter and the emitted file.",
    )
    if thresholds.check_arrival_times:
        add(
            "arrival_ks",
            ks_statistic(_quantize(source.arrivals), _quantize(replayed.arrivals)),
            thresholds.ks,
            "the replayed arrival times do not follow the source's; timing fidelity "
            "is lost, so every burstiness-conditioned result is suspect.",
        )
    add(
        "input_len_ks",
        ks_statistic(source.input_tokens, replayed.input_tokens),
        thresholds.ks,
        "prompt lengths differ — check block coverage: vLLM expands hash_ids at "
        "--timed-trace-chunk-hash-size tokens each and truncates silently when the "
        "size is too small (Mooncake is 512, the vLLM default is 16).",
    )
    add(
        "output_len_ks",
        ks_statistic(source.output_tokens, replayed.output_tokens),
        thresholds.ks,
        "decode lengths differ, which moves the prefill/decode ratio the regime "
        "axis is defined on.",
    )
    add(
        "rate_rps",
        _rel(source.rate_rps(), replayed.rate_rps()),
        thresholds.rate,
        "mean offered rate differs from the source's.",
    )
    add(
        "burstiness",
        _rel(
            index_of_dispersion(source.arrivals, bin_s=bin_s, span_s=source.meta.span_s),
            index_of_dispersion(replayed.arrivals, bin_s=bin_s, span_s=replayed.meta.span_s),
        ),
        thresholds.burstiness,
        "the index of dispersion moved, so the two traces sit in different regime "
        "buckets even where their means agree.",
    )

    span = max(source.meta.span_s, replayed.meta.span_s)
    return ValidationReport(
        standard=thresholds.name,
        source=f"{source.meta.source}({source.meta.sha256[:12]})",
        replayed=f"{replayed.meta.source}({replayed.meta.sha256[:12]})",
        checks=checks,
        source_hist=_hist(source, nbins=hist_bins, span=span),
        replayed_hist=_hist(replayed, nbins=hist_bins, span=span),
        hist_bin_s=span / hist_bins if span > 0 else 0.0,
    )
