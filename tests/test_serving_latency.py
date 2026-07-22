"""Per-request serving latency: TTFT/TPOT/goodput and the wall-clock join."""

from __future__ import annotations

from types import SimpleNamespace

from gitm.tracer.vllm_stats import (
    RequestRecord,
    request_records_from_outputs,
    summarize_requests,
)


def _rec(arrival, first, finished, n_tokens):
    return RequestRecord(
        arrival_wall_s=arrival,
        first_token_wall_s=first,
        finished_wall_s=finished,
        n_output_tokens=n_tokens,
    )


# --------------------------------------------------------------------------- #
# per-request derivations                                                     #
# --------------------------------------------------------------------------- #
def test_ttft_and_tpot_from_timestamps():
    # arrival 0, first token at 0.2, finished at 1.2 with 11 tokens →
    # TTFT 0.2s; the 10 tokens after the first span 1.0s → TPOT 0.1s.
    r = _rec(0.0, 0.2, 1.2, 11)
    assert r.ttft_s == 0.2
    assert abs(r.tpot_s - 0.1) < 1e-9


def test_single_token_response_has_no_tpot():
    # One token means there is no inter-token interval to measure. It must read
    # as unknown, not as a rate derived from a zero-length gap.
    r = _rec(0.0, 0.2, 0.2, 1)
    assert r.ttft_s == 0.2
    assert r.tpot_s is None


def test_missing_timestamps_yield_none_not_zero():
    r = RequestRecord(n_output_tokens=8)
    assert r.ttft_s is None and r.tpot_s is None


# --------------------------------------------------------------------------- #
# summary: percentiles exclude unmeasurable requests                          #
# --------------------------------------------------------------------------- #
def test_unmeasurable_requests_drop_out_of_percentiles():
    records = [
        _rec(0.0, 0.1, 1.1, 11),
        _rec(0.0, 0.3, 1.3, 11),
        RequestRecord(n_output_tokens=5),  # no metrics on this build
    ]
    s = summarize_requests(records)
    assert s.n_requests == 3
    # Only the two with timestamps contribute — a missing sample must not be
    # imputed as a 0s TTFT, which would make latency look better than it was.
    assert s.n_ttft == 2
    assert s.ttft_p50_s in (0.1, 0.3)
    assert s.ttft_p50_s > 0.0


def test_empty_records_summarize_to_zeros_not_crash():
    s = summarize_requests([])
    assert s.n_requests == 0 and s.n_ttft == 0
    assert s.ttft_p50_s is None and s.goodput_rps is None


# --------------------------------------------------------------------------- #
# goodput                                                                     #
# --------------------------------------------------------------------------- #
def test_goodput_counts_only_slo_meeting_requests():
    # Two fast requests inside the SLO, one with a 5s TTFT that blows it. The
    # window spans arrival 0 → last finish 6s.
    records = [
        _rec(0.0, 0.1, 1.1, 11),
        _rec(0.0, 0.2, 1.2, 11),
        _rec(0.0, 5.0, 6.0, 11),
    ]
    s = summarize_requests(records, ttft_slo_s=1.0, tpot_slo_s=0.5)
    assert s.n_met_slo == 2
    assert s.window_s == 6.0
    assert abs(s.goodput_rps - 2 / 6.0) < 1e-9


def test_tpot_violation_alone_fails_the_slo():
    # TTFT is fine but the stream crawls: 11 tokens over 10s → TPOT 1.0s.
    records = [_rec(0.0, 0.1, 10.1, 11)]
    s = summarize_requests(records, ttft_slo_s=1.0, tpot_slo_s=0.05)
    assert s.n_met_slo == 0


def test_single_token_request_not_penalised_for_absent_tpot():
    # No TPOT to measure, TTFT well inside the SLO — this is a good request.
    records = [_rec(0.0, 0.1, 0.1, 1)]
    s = summarize_requests(records, ttft_slo_s=1.0, tpot_slo_s=0.05)
    assert s.n_met_slo == 1


# --------------------------------------------------------------------------- #
# vLLM RequestOutput adaptation                                               #
# --------------------------------------------------------------------------- #
def test_records_from_vllm_outputs():
    out = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
        metrics=SimpleNamespace(arrival_time=10.0, first_token_time=10.5, finished_time=12.5),
    )
    (rec,) = request_records_from_outputs([out])
    assert rec.n_output_tokens == 3
    assert rec.ttft_s == 0.5


def test_outputs_without_metrics_still_yield_a_record():
    # V1 builds may leave `metrics` unset. We still know a request ran and how
    # many tokens it produced — that is the honest partial record.
    out = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])], metrics=None)
    (rec,) = request_records_from_outputs([out])
    assert rec.n_output_tokens == 2
    assert rec.ttft_s is None


def test_records_from_empty_outputs():
    assert request_records_from_outputs([]) == []
    assert request_records_from_outputs(None) == []


# --------------------------------------------------------------------------- #
# the join: scheduler samples carry a wall-clock anchor                       #
# --------------------------------------------------------------------------- #
def test_sampler_summary_carries_wall_clock_anchor():
    from gitm.tracer.vllm_stats import sample_scheduler_stats

    engine = SimpleNamespace(scheduler=SimpleNamespace(running=[1], waiting=[]))
    with sample_scheduler_stats(engine, interval_s=0.002) as sampler:
        pass
    s = sampler.summary()
    # Without this anchor the scheduler series (monotonic t_ns) cannot be placed
    # on the same clock as request timestamps or the trace.
    assert s.t0_wall_ns > 0


def test_summary_without_engine_has_no_anchor():
    from gitm.tracer.vllm_stats import sample_scheduler_stats

    with sample_scheduler_stats(None) as sampler:
        pass
    assert sampler.summary().t0_wall_ns == 0
