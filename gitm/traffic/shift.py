"""Has production traffic changed shape since the baseline was recorded? (ENG-98)

The loop trusts a floor measured under one traffic shape. When live traffic moves
off that shape the floor is stale and the loop should re-baseline. This module is
the test for "moved off": it compares a recent window of traffic to the recorded
baseline on three things vLLM itself exposes, the request rate and the prompt and
output length distributions, and answers ``shifted``, ``stable`` or ``inconclusive``.

Distance. Each component is in [0, 1] and the distance is the largest one, the same
convention as the distance metric v1 (so a threshold means the same kind of number):

* rate: ``1 - 2**-|log2(rate_a / rate_b)|``, the transform the T component uses;
* input and output length: total variation between the two histograms over fixed
  log2 length bins.

Noise. Two samples of the same traffic still differ, and a short window differs more.
A fixed threshold would fire on a short matched window and sleep through a long
shifted one. So the threshold actually used is the larger of the configured threshold
and the ``null_quantile`` of the distance a same-shaped window of the same size would
show by chance, simulated from the baseline (multinomial for the histograms, a normal
approximation to the request count scaled by the baseline's index of dispersion for the
rate). A window with fewer than ``min_requests`` requests is ``inconclusive``.

What it does not do. It does not look at arrival burstiness, prefix sharing, or the
joint distribution of input and output length. Two traces with the same marginals and
a different input/output pairing look identical here. The default threshold is a
starting point, not a calibrated value: it has to be set from real traffic. Validated
on synthetic replays only (tests/test_traffic_shift.py).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from gitm.traffic.regime import index_of_dispersion
from gitm.traffic.schema import Trace

__all__ = [
    "N_BINS", "ShiftConfig", "ShiftResult", "TrafficProfile",
    "detect_shift", "distance_components", "SHIFTED", "STABLE", "INCONCLUSIVE",
]

SHIFTED = "shifted"
STABLE = "stable"
INCONCLUSIVE = "inconclusive"

#: Length bins are half an octave wide: bin k holds lengths in [2**(k/2), 2**((k+1)/2)),
#: and the last bin takes everything above. Whole octaves were too coarse to see a 1.5x
#: change in prompt length at all.
BINS_PER_OCTAVE = 2
N_BINS = 21 * BINS_PER_OCTAVE


def _bin_of(length: float) -> int:
    if length < 1:
        return 0
    return min(int(math.floor(BINS_PER_OCTAVE * math.log2(length))), N_BINS - 1)


def _hist(lengths: list[int] | list[float]) -> tuple[int, ...]:
    counts = [0] * N_BINS
    for n in lengths:
        counts[_bin_of(n)] += 1
    return tuple(counts)


@dataclass(frozen=True)
class TrafficProfile:
    """What the detector compares: how many requests, over how long, and their lengths."""

    requests: int
    span_s: float
    input_hist: tuple[int, ...]
    output_hist: tuple[int, ...]
    #: Index of dispersion of arrival counts (1.0 = Poisson). Sets how much the request
    #: count is allowed to wander by chance; 1.0 when it cannot be measured.
    dispersion: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.input_hist) != N_BINS or len(self.output_hist) != N_BINS:
            raise ValueError(f"histograms must have {N_BINS} bins")
        if self.requests < 0 or self.span_s < 0:
            raise ValueError("requests and span_s must be non-negative")
        if min(self.input_hist + self.output_hist) < 0:
            raise ValueError("histogram counts must be non-negative")

    @property
    def rate_rps(self) -> float:
        return self.requests / self.span_s if self.span_s > 0 else 0.0

    @classmethod
    def from_trace(cls, trace: Trace, *, bin_s: float = 1.0) -> TrafficProfile:
        if not trace.requests:
            raise ValueError("cannot profile an empty trace")
        span = trace.meta.span_s
        return cls(
            requests=len(trace.requests),
            span_s=span,
            input_hist=_hist(trace.input_tokens),
            output_hist=_hist(trace.output_tokens),
            dispersion=index_of_dispersion(trace.arrivals, bin_s=bin_s, span_s=span),
            source=trace.meta.source,
        )

    @classmethod
    def from_vllm_histograms(
        cls,
        *,
        span_s: float,
        prompt_tokens: Mapping[float, float],
        generation_tokens: Mapping[float, float],
        dispersion: float = 1.0,
        source: str = "vllm-metrics",
    ) -> TrafficProfile:
        """From vLLM's cumulative Prometheus histograms, taken as a delta over ``span_s``.

        ``prompt_tokens`` and ``generation_tokens`` map each bucket's ``le`` upper edge
        (``math.inf`` for ``+Inf``) to its cumulative count over the window, i.e. the
        counter at the end minus the counter at the start.

        Each bucket's requests are placed in the half-octave bin that holds the bucket's upper
        edge. That is an approximation: it moves mass up within a bucket. It is the same
        approximation on both sides when the baseline and the window come from the same
        engine and bucket layout, which is the only way they should be compared.
        """
        inp = _from_cumulative(prompt_tokens)
        out = _from_cumulative(generation_tokens)
        n = sum(inp)
        if n != sum(out) and sum(out) > 0:
            raise ValueError(
                f"prompt and generation histograms count different requests ({n} vs {sum(out)})")
        return cls(requests=n, span_s=span_s, input_hist=tuple(inp), output_hist=tuple(out),
                   dispersion=dispersion, source=source)


def _from_cumulative(cum: Mapping[float, float]) -> list[int]:
    edges = sorted(cum)
    if not edges:
        raise ValueError("an empty histogram")
    counts = [0] * N_BINS
    prev = 0.0
    for le in edges:
        c = float(cum[le])
        if c < prev - 1e-9:
            raise ValueError("cumulative bucket counts must not decrease")
        step = c - prev
        prev = c
        if step <= 0:
            continue
        # Bucket (prev_le, le]: the upper edge decides the bin; +Inf goes to the top bin.
        b = N_BINS - 1 if math.isinf(le) else _bin_of(le)
        counts[b] += int(round(step))
    return counts


@dataclass(frozen=True)
class ShiftConfig:
    #: Distance above which the traffic counts as moved. A starting point: calibrate it
    #: on real traffic before trusting it.
    threshold: float = 0.25
    #: Fewer requests than this in the window and the answer is ``inconclusive``.
    min_requests: int = 200
    null_draws: int = 1000
    #: How much of the same-shape-by-chance distance the threshold must clear.
    null_quantile: float = 0.99
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if self.min_requests < 1 or self.null_draws < 1:
            raise ValueError("min_requests and null_draws must be at least 1")
        if not 0.5 <= self.null_quantile < 1:
            raise ValueError("null_quantile must be in [0.5, 1)")


@dataclass(frozen=True)
class ShiftResult:
    status: str
    distance: float | None
    components: dict[str, float] = field(default_factory=dict)
    limiting_axis: str | None = None
    #: The ``null_quantile`` of the distance a same-shape window would show by chance.
    noise_floor: float | None = None
    #: What the distance was compared against: ``max(threshold, noise_floor)``.
    threshold_used: float | None = None
    reasons: tuple[str, ...] = ()

    @property
    def shifted(self) -> bool:
        return self.status == SHIFTED


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _norm(h: tuple[int, ...] | np.ndarray) -> np.ndarray:
    arr = np.asarray(h, dtype=float)
    s = arr.sum()
    return arr / s if s > 0 else arr


def _rate_component(ra: float, rb: float) -> float:
    if ra == rb:
        return 0.0
    if ra <= 0 or rb <= 0:
        return 1.0
    return 1.0 - 2.0 ** (-abs(math.log2(ra / rb)))


def _quantile(hist: np.ndarray, q: float) -> float:
    """The q-quantile of lengths from a histogram, log-uniform within the bin it falls in."""
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    target = q * total
    cum = 0.0
    for k, c in enumerate(hist):
        if c > 0 and cum + c >= target:
            frac = (target - cum) / c
            return 2.0 ** ((k + frac) / BINS_PER_OCTAVE)
        cum += c
    return 2.0 ** (len(hist) / BINS_PER_OCTAVE)


def _components(rate_a: float, rate_b: float, in_a, in_b, out_a, out_b) -> dict[str, float]:
    """Every component, from rates and histograms. One function for the real comparison
    and for the noise simulation, so the two cannot drift apart."""
    in_a, in_b = np.asarray(in_a, dtype=float), np.asarray(in_b, dtype=float)
    comps = {
        "rate": _rate_component(rate_a, rate_b),
        "input_length": _tv(_norm(in_a), _norm(in_b)),
        "input_p50": _rate_component(_quantile(in_a, 0.5), _quantile(in_b, 0.5)),
        "input_p95": _rate_component(_quantile(in_a, 0.95), _quantile(in_b, 0.95)),
    }
    out_a, out_b = np.asarray(out_a, dtype=float), np.asarray(out_b, dtype=float)
    if out_a.sum() > 0 and out_b.sum() > 0:
        comps["output_length"] = _tv(_norm(out_a), _norm(out_b))
        comps["output_p50"] = _rate_component(_quantile(out_a, 0.5), _quantile(out_b, 0.5))
        comps["output_p95"] = _rate_component(_quantile(out_a, 0.95), _quantile(out_b, 0.95))
    return comps


def distance_components(a: TrafficProfile, b: TrafficProfile) -> dict[str, float]:
    return _components(a.rate_rps, b.rate_rps, a.input_hist, b.input_hist,
                       a.output_hist, b.output_hist)


def _noise_floor(baseline: TrafficProfile, n: int, span_s: float, config: ShiftConfig) -> float:
    """The ``null_quantile`` distance between the baseline and same-shape windows of size n."""
    rng = np.random.default_rng(config.seed)
    p_in, p_out = _norm(baseline.input_hist), _norm(baseline.output_hist)
    has_out = sum(baseline.output_hist) > 0
    mean_count = baseline.rate_rps * span_s
    sd_count = math.sqrt(max(baseline.dispersion, 1e-9) * mean_count)
    zeros = np.zeros(N_BINS)
    draws = []
    for _ in range(config.null_draws):
        count = max(rng.normal(mean_count, sd_count), 0.0)
        sim_in = rng.multinomial(n, p_in)
        sim_out = rng.multinomial(n, p_out) if has_out else zeros
        comps = _components(baseline.rate_rps, count / span_s, baseline.input_hist, sim_in,
                            baseline.output_hist, sim_out)
        draws.append(max(comps.values()))
    return float(np.quantile(draws, config.null_quantile))


def detect_shift(
    baseline: TrafficProfile, window: TrafficProfile, config: ShiftConfig | None = None
) -> ShiftResult:
    config = config or ShiftConfig()
    reasons = []
    if baseline.requests < config.min_requests:
        reasons.append(f"baseline_too_small:{baseline.requests}<{config.min_requests}")
    if window.requests < config.min_requests:
        reasons.append(f"window_too_small:{window.requests}<{config.min_requests}")
    if baseline.span_s <= 0 or window.span_s <= 0:
        reasons.append("zero_span")
    if reasons:
        return ShiftResult(INCONCLUSIVE, None, reasons=tuple(reasons))

    comps = distance_components(baseline, window)
    axis = max(comps, key=comps.__getitem__)  # first wins a tie
    d = comps[axis]
    floor = _noise_floor(baseline, window.requests, window.span_s, config)
    used = max(config.threshold, floor)
    status = SHIFTED if d > used else STABLE
    return ShiftResult(status, d, comps, axis, floor, used,
                       reasons=(f"{axis}={d:.3f}_vs_{used:.3f}",))
