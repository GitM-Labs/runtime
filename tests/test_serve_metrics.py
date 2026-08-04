"""Tests for the server-side serving account (gitm/serve/metrics.py).

In observe mode there is no client, so these numbers are the *only* description of
what the server did inside the capture window. That makes the difference between a
window-scoped figure and a lifetime-scoped one load-bearing: reading _sum/_count once
gives the mean since process start, which on a server that has been up for a day is
dominated by traffic that has nothing to do with the trace sitting next to it.
"""

from __future__ import annotations

from gitm.serve import metrics

# Shapes taken from a real vLLM /metrics: labelled families, a histogram with buckets
# that must not be folded into the family total, and a NaN gauge.
BEFORE = """\
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="Qwen/Qwen3-8B"} 1000.0
vllm:generation_tokens_total{model_name="Qwen/Qwen3-8B"} 2000.0
vllm:request_success_total{finished_reason="stop",model_name="Qwen/Qwen3-8B"} 8.0
vllm:request_success_total{finished_reason="length",model_name="Qwen/Qwen3-8B"} 2.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="Qwen/Qwen3-8B"} 5.0
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-8B"} 1.0
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-8B"} 10.0
vllm:time_per_output_token_seconds_sum{model_name="Qwen/Qwen3-8B"} 20.0
vllm:time_per_output_token_seconds_count{model_name="Qwen/Qwen3-8B"} 1000.0
vllm:num_requests_running{model_name="Qwen/Qwen3-8B"} 0.0
"""

AFTER = """\
vllm:prompt_tokens_total{model_name="Qwen/Qwen3-8B"} 6000.0
vllm:generation_tokens_total{model_name="Qwen/Qwen3-8B"} 4560.0
vllm:request_success_total{finished_reason="stop",model_name="Qwen/Qwen3-8B"} 14.0
vllm:request_success_total{finished_reason="length",model_name="Qwen/Qwen3-8B"} 6.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="Qwen/Qwen3-8B"} 9.0
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen3-8B"} 3.0
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen3-8B"} 20.0
vllm:time_per_output_token_seconds_sum{model_name="Qwen/Qwen3-8B"} 45.0
vllm:time_per_output_token_seconds_count{model_name="Qwen/Qwen3-8B"} 3560.0
vllm:num_requests_running{model_name="Qwen/Qwen3-8B"} 12.0
"""


def test_parse_sums_label_sets_and_skips_buckets():
    snap = metrics.parse_prometheus(BEFORE)
    # stop + length: a window's request count is every finished_reason added up.
    assert snap["vllm:request_success_total"] == 10.0
    # Buckets are cumulative counts of the same observations the _count already holds;
    # folding them into the family would double-count every request.
    assert "vllm:time_to_first_token_seconds_bucket" not in snap
    assert snap["vllm:prompt_tokens_total"] == 1000.0


def test_parse_ignores_comments_and_nan():
    snap = metrics.parse_prometheus(
        "# HELP x help\n# TYPE x gauge\nvllm:num_requests_waiting NaN\nvllm:num_requests_running 3\n"
    )
    # NaN is an untouched histogram/gauge, not a zero — recording it as 0 would make an
    # unobserved metric indistinguishable from an idle one.
    assert "vllm:num_requests_waiting" not in snap
    assert snap["vllm:num_requests_running"] == 3.0


def test_window_is_the_difference_not_the_lifetime():
    w = metrics.window_from_snapshots(BEFORE, AFTER, window_s=10.0)

    assert w.requests_finished == 10.0          # 20 - 10
    assert w.prompt_tokens == 5000.0
    assert w.generation_tokens == 2560.0
    assert w.output_tokens_per_s == 256.0

    # The lifetime TTFT mean is 3.0/20 = 150ms; the window's is (3-1)/(20-10) = 200ms.
    # Reporting the first would describe traffic that predates the trace.
    assert w.ttft_mean_s == 0.2
    assert w.tpot_mean_s == (45.0 - 20.0) / (3560.0 - 1000.0)


def test_counter_reset_reports_nothing_rather_than_a_small_number():
    """A counter that went backwards means the server restarted mid-window. The
    difference is meaningless, and a plausible-looking small positive would be read as
    a real measurement."""
    w = metrics.window_from_snapshots(AFTER, BEFORE, window_s=10.0)
    assert w.requests_finished is None
    assert w.generation_tokens is None


def test_idle_window_is_called_out(capsys):
    w = metrics.window_from_snapshots(BEFORE, BEFORE, window_s=30.0)
    assert w.requests_finished == 0.0
    assert any("idle" in n for n in w.notes)


def test_unreadable_metrics_degrade_to_a_note_not_an_exception():
    w = metrics.window_from_snapshots("# unavailable: timed out\n", AFTER, window_s=5.0)
    assert w.requests_finished is None
    assert any("unreadable" in n for n in w.notes)


def test_disabled_log_stats_is_distinguished_from_an_idle_server():
    plain = "python_gc_objects_collected_total 1.0\n"
    w = metrics.window_from_snapshots(plain, plain, window_s=5.0)
    assert any("--disable-log-stats" in n for n in w.notes)


def test_gauge_samples_become_queue_depth_percentiles():
    """Queue depth is what separates 'the GPU was busy' from 'requests were piling
    up', and two endpoint reads cannot see it."""
    samples = [
        {"running": r, "waiting": w, "kv_cache_usage": 0.5}
        for r, w in [(0, 0), (4, 0), (8, 12), (8, 30), (2, 0)]
    ]
    w = metrics.window_from_snapshots(BEFORE, AFTER, window_s=5.0, samples=samples)

    assert w.n_samples == 5
    assert w.running_p50 == 4.0
    assert w.waiting_p95 == 30.0
    assert w.kv_cache_usage_p95 == 0.5


def test_missing_window_length_leaves_throughput_unset():
    w = metrics.window_from_snapshots(BEFORE, AFTER)
    assert w.generation_tokens == 2560.0
    assert w.output_tokens_per_s is None


def test_fetch_never_raises_on_a_dead_endpoint():
    text = metrics.fetch_metrics("http://127.0.0.1:1", timeout=0.5)
    assert text.startswith("# unavailable")
