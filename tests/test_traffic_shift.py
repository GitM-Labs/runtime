"""The traffic-shift detector (ENG-98): fires on a shifted replay, quiet on a matched one.

Replays are drawn from the Mooncake fixture's fitted envelope (bursty: D=6.7) with
gitm.traffic.sample_trace. Synthetic replays only; the threshold is not calibrated on
real traffic, and these tests do not claim it is.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gitm.traffic import fit, read_mooncake, sample_trace
from gitm.traffic.shift import (
    INCONCLUSIVE,
    N_BINS,
    SHIFTED,
    STABLE,
    ShiftConfig,
    TrafficProfile,
    detect_shift,
)

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/traffic_replay/fixtures/mooncake_slice.jsonl"
BASE_S, WINDOW_S = 600.0, 300.0


@pytest.fixture(scope="module")
def envelope():
    return fit(read_mooncake(FIXTURE))


@pytest.fixture(scope="module")
def baseline(envelope):
    trace, _ = sample_trace(envelope, duration_s=BASE_S, seed=0)
    return TrafficProfile.from_trace(trace)


def _window(envelope, seed, **kw):
    trace, _ = sample_trace(envelope, duration_s=WINDOW_S, seed=seed, **kw)
    return TrafficProfile.from_trace(trace)


def test_matched_replays_stay_quiet_at_a_low_false_alarm_rate(envelope, baseline):
    fired = [s for s in range(1, 101) if detect_shift(baseline, _window(envelope, s)).shifted]
    assert len(fired) <= 5, f"false alarms on matched replays at seeds {fired}"


@pytest.mark.parametrize("kw, axis", [
    (dict(rate_mult=2.0), "rate"),
    (dict(rate_mult=0.5), "rate"),
    (dict(input_scale=2.0), "input"),
    (dict(input_scale=1.5), "input"),
    (dict(output_scale=0.4), "output"),
    (dict(output_scale=1.5), "output"),
])
def test_a_deliberately_shifted_replay_fires_and_names_the_axis(envelope, baseline, kw, axis):
    res = detect_shift(baseline, _window(envelope, 1, **kw))
    assert res.status == SHIFTED, res
    assert res.limiting_axis.startswith(axis)
    assert res.distance > res.threshold_used


@pytest.mark.parametrize("seed", range(1, 11))
def test_every_seed_of_a_large_shift_fires(envelope, baseline, seed):
    assert detect_shift(baseline, _window(envelope, seed, rate_mult=2.0, input_scale=2.0)).shifted


def test_a_small_drift_stays_below_the_threshold(envelope, baseline):
    res = detect_shift(baseline, _window(envelope, 1, rate_mult=1.1, input_scale=1.05))
    assert res.status == STABLE
    assert res.distance < 0.25


def test_a_short_window_is_inconclusive_not_a_verdict(envelope, baseline):
    trace, _ = sample_trace(envelope, duration_s=10.0, seed=3)
    short = TrafficProfile.from_trace(trace)
    res = detect_shift(baseline, short, ShiftConfig(min_requests=200))
    assert short.requests < 200
    assert res.status == INCONCLUSIVE and res.distance is None
    assert any(r.startswith("window_too_small") for r in res.reasons)


def test_a_small_window_raises_the_bar_it_must_clear(envelope, baseline):
    big = detect_shift(baseline, _window(envelope, 1))
    trace, _ = sample_trace(envelope, duration_s=80.0, seed=1)
    small = detect_shift(baseline, TrafficProfile.from_trace(trace), ShiftConfig(min_requests=50))
    assert small.noise_floor > big.noise_floor


def test_the_known_blind_spot_burstiness_alone_is_not_seen(envelope, baseline):
    """Same marginals, different bunching: this detector does not look at it."""
    res = detect_shift(baseline, _window(envelope, 1, burstiness=1.0))
    assert res.status == STABLE


def test_the_result_is_reproducible(envelope, baseline):
    w = _window(envelope, 1, rate_mult=2.0)
    assert detect_shift(baseline, w) == detect_shift(baseline, w)


def _cumulative(counts_by_edge):
    out, run = {}, 0
    for le in sorted(counts_by_edge):
        run += counts_by_edge[le]
        out[le] = run
    return out


EDGES = (16, 64, 256, 1024, 4096, math.inf)


def _vllm(span, in_counts, out_counts):
    return TrafficProfile.from_vllm_histograms(
        span_s=span,
        prompt_tokens=_cumulative(dict(zip(EDGES, in_counts, strict=True))),
        generation_tokens=_cumulative(dict(zip(EDGES, out_counts, strict=True))))


def test_vllm_histograms_match_and_shift():
    base = _vllm(600, [100, 300, 400, 300, 100, 20], [300, 500, 300, 100, 20, 0])
    same = _vllm(300, [52, 148, 205, 147, 52, 9], [150, 251, 149, 51, 10, 2])
    moved = _vllm(300, [5, 20, 80, 300, 400, 100], [220, 370, 220, 75, 15, 5])
    assert detect_shift(base, same).status == STABLE
    res = detect_shift(base, moved)
    assert res.status == SHIFTED and res.limiting_axis.startswith("input")


def test_vllm_histogram_counts_land_in_the_bin_of_their_upper_edge():
    p = _vllm(60, [0, 0, 5, 0, 0, 0], [5, 0, 0, 0, 0, 0])
    assert p.input_hist[16] == 5         # edge 256 -> bin 16 (half octaves)
    assert p.output_hist[8] == 5         # edge 16 -> bin 8
    assert p.requests == 5 and len(p.input_hist) == N_BINS


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="must not decrease"):
        TrafficProfile.from_vllm_histograms(
            span_s=1, prompt_tokens={16: 5, 64: 3}, generation_tokens={16: 5})
    with pytest.raises(ValueError, match="different requests"):
        TrafficProfile.from_vllm_histograms(
            span_s=1, prompt_tokens={16: 5}, generation_tokens={16: 4})
    with pytest.raises(ValueError, match="threshold"):
        ShiftConfig(threshold=1.5)
    with pytest.raises(ValueError, match="bins"):
        TrafficProfile(requests=1, span_s=1, input_hist=(1,), output_hist=(1,))
