"""Read scheduler stats from a vLLM V1-shaped engine.

V1 keeps the scheduler inside EngineCore and exposes a stats object via
`make_stats()` (num_running_reqs / num_waiting_reqs / kv_cache_usage) instead of
the V0 running/waiting deques. Fake that shape to pin the read path off-GPU; the
exact attr names still need GPU validation on the target vLLM build.
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.tracer.vllm_stats import read_scheduler_stats


def _v1_engine(running: int, waiting: int, cache: float, max_seqs: int = 256):
    stats = SimpleNamespace(
        num_running_reqs=running, num_waiting_reqs=waiting, kv_cache_usage=cache
    )
    scheduler = SimpleNamespace(make_stats=lambda: stats)
    # llm.llm_engine.engine_core.engine_core.scheduler (in-process V1)
    return SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler))
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
    )


def test_reads_v1_stats_object():
    s = read_scheduler_stats(_v1_engine(running=3, waiting=12, cache=0.87), t_ns=0)
    assert s is not None
    assert s.num_running == 3
    assert s.num_waiting == 12
    assert s.gpu_cache_usage == 0.87
    assert s.batch_occupancy == 3 / 256  # occupancy derived from V1 num_running


def test_none_engine_still_none():
    assert read_scheduler_stats(None) is None


def test_v1_engine_without_stats_is_none():
    # a V1-shaped engine whose scheduler exposes nothing readable -> None, no crash.
    empty = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=object()))
        )
    )
    assert read_scheduler_stats(empty) is None
