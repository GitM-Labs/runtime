"""Scheduler-stats adapter: duck-types engine forms and aggregates a series."""

from __future__ import annotations

from types import SimpleNamespace

from gitm.tracer.vllm_sched import SchedulerStatsTracker, sample_scheduler


def test_sample_from_mapping():
    s = sample_scheduler({"num_running": 6, "num_waiting": 4, "gpu_cache_usage": 0.8}, ts_ns=10)
    assert s.num_running == 6
    assert s.queue_depth == 4
    assert s.batch_occupancy == 6 / 10
    assert s.gpu_cache_usage == 0.8


def test_sample_from_scheduler_object():
    engine = SimpleNamespace(scheduler=SimpleNamespace(running=[1, 2, 3], waiting=[4], swapped=[]))
    s = sample_scheduler(engine, ts_ns=5)
    assert s.num_running == 3
    assert s.num_waiting == 1
    assert s.queue_depth == 1


def test_sample_from_get_stats_method():
    engine = SimpleNamespace(get_scheduler_stats=lambda: {"running": 2, "waiting": 8})
    s = sample_scheduler(engine, ts_ns=1)
    assert s.num_running == 2 and s.num_waiting == 8
    assert s.batch_occupancy == 0.2  # starved


def test_tracker_summarize():
    tr = SchedulerStatsTracker()
    tr.record({"num_running": 1, "num_waiting": 9}, ts_ns=1)  # starved
    tr.record({"num_running": 9, "num_waiting": 1}, ts_ns=2)  # healthy
    summary = tr.summarize()
    assert summary["n_samples"] == 2
    assert summary["max_queue_depth"] == 9
    assert summary["starved_fraction"] == 0.5
