"""The planner has to know which GPU it is on, and how many are in the run.

``_query_nvml`` is pynvml. On a ROCm box it returns ``(None, None)``, so the
SKU falls through to ``HardwareSpec()`` — A100-SXM4-80GB — and every roofline
floor is priced against the wrong silicon. An 8x MI355X run reported
``"hardware": "A100-SXM4-80GB"`` for exactly that reason, with MI355X already
sitting in ``_PEAKS``.

The count is a separate question with a different right answer. It feeds
``has_collective``, which gates levers declaring ``requires_collective`` — so
what matters is the *run's* world size, not how many devices the box has. A
TP=1 job on an eight-GPU node has no collectives and must not be told it does.
"""

from __future__ import annotations

import gitm.planner.context as ctx


class _Parallel:
    def __init__(self, world_size):
        self.world_size = world_size


class _VllmConfig:
    def __init__(self, world_size):
        self.parallel_config = _Parallel(world_size)


class _Engine:
    """A live vLLM engine, as the loop holds it."""

    def __init__(self, world_size):
        self.llm_engine = type("E", (), {"vllm_config": _VllmConfig(world_size)})()


# ── the SKU ────────────────────────────────────────────────────────────────
def test_an_amd_sku_is_found_when_nvml_is_absent(monkeypatch):
    """The bug: no NVML, so no SKU, so A100 peaks on an MI355X."""
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 8))
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)

    pctx = ctx.build_planner_context(None, workload="vllm-decode")

    assert pctx.peak is not None, "fell through to the A100 default"
    assert "MI355X" in pctx.peak.name


def test_the_env_override_still_wins(monkeypatch):
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H100", 4))
    monkeypatch.setenv("GITM_GPU_SKU", "MI355X")

    assert "MI355X" in ctx.build_planner_context(None, workload="vllm-decode").peak.name


def test_nvml_still_wins_over_the_fallback(monkeypatch):
    """The fallback is a fallback — it must not displace a working NVML."""
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H200", 2))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("should not be asked", 99))
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)

    pctx = ctx.build_planner_context(None, workload="vllm-decode")

    assert "H200" in pctx.peak.name
    assert pctx.gate.num_gpus == 2


# ── the count ──────────────────────────────────────────────────────────────
def test_the_run_world_size_decides_collectives_not_the_box(monkeypatch):
    """A TP=1 job on an eight-GPU node has no collectives. Reporting 8 would
    admit levers that need a collective there is none of."""
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 8))
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)

    pctx = ctx.build_planner_context(_Engine(world_size=1), workload="vllm-decode")

    assert pctx.gate.num_gpus == 1
    assert pctx.gate.has_collective is False


def test_a_sharded_run_reports_its_collectives(monkeypatch):
    """TP=8 on the same node. Without this the run reads as single-GPU and
    every requires_collective lever is rejected — on a sharded MoE run, the
    ones worth testing."""
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 8))
    monkeypatch.delenv("GITM_GPU_SKU", raising=False)

    pctx = ctx.build_planner_context(_Engine(world_size=8), workload="vllm-decode")

    assert pctx.gate.num_gpus == 8
    assert pctx.gate.has_collective is True


def test_an_explicit_count_still_wins(monkeypatch):
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (None, None))
    monkeypatch.setattr(ctx, "_query_torch", lambda: ("AMD Instinct MI355X", 8))

    pctx = ctx.build_planner_context(_Engine(world_size=8), workload="vllm-decode", num_gpus=2)

    assert pctx.gate.num_gpus == 2


def test_nothing_is_probed_when_both_answers_are_already_known(monkeypatch):
    """The cluster's own configuration: GITM_GPU_SKU set and a live engine that
    knows its world size. Neither NVML nor torch has anything to contribute, and
    `get_device_name` initialises a CUDA/HIP context — a side effect the planner
    should not cause for a value it already has."""
    asked = []
    monkeypatch.setattr(ctx, "_query_nvml", lambda: (asked.append("nvml"), (None, None))[1])
    monkeypatch.setattr(ctx, "_query_torch", lambda: (asked.append("torch"), (None, None))[1])
    monkeypatch.setenv("GITM_GPU_SKU", "MI355X")

    pctx = ctx.build_planner_context(_Engine(world_size=8), workload="vllm-decode")

    assert asked == [], f"probed the device for nothing: {asked}"
    assert "MI355X" in pctx.peak.name
    assert pctx.gate.num_gpus == 8


def test_an_empty_override_does_not_count_as_an_answer(monkeypatch):
    """`GITM_GPU_SKU=` in a manifest arrives as "", not None. Treated as set it
    suppresses detection and leaves the A100 default on a box that could have
    been identified."""
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H200", 4))
    monkeypatch.setenv("GITM_GPU_SKU", "")

    pctx = ctx.build_planner_context(_Engine(world_size=8), workload="vllm-decode")

    assert pctx.peak is not None, "fell through to the A100 default"
    assert "H200" in pctx.peak.name


def test_a_whitespace_override_is_also_unset(monkeypatch):
    monkeypatch.setattr(ctx, "_query_nvml", lambda: ("NVIDIA H200", 4))
    monkeypatch.setenv("GITM_GPU_SKU", "   ")

    assert "H200" in ctx.build_planner_context(None, workload="vllm-decode").peak.name


def test_world_size_is_read_behind_an_engine_attribute():
    """The layout the scheduler lookup already read and this one did not: the
    config behind `.engine`. Missed, a TP=1 run on a multi-GPU node was counted
    as the whole node and admitted levers that need collectives."""
    from types import SimpleNamespace as NS

    engine = NS(engine=NS(vllm_config=NS(parallel_config=NS(world_size=1))))
    assert ctx._engine_world_size(engine) == 1


def test_world_size_and_max_num_seqs_read_the_same_places():
    """One list of places for every config value, so the two cannot drift."""
    from types import SimpleNamespace as NS

    from gitm.tracer.vllm_stats import _max_num_seqs

    for wrap in (lambda c: c, lambda c: NS(engine=c), lambda c: NS(llm_engine=c)):
        for cfg in (NS(vllm_config=NS(parallel_config=NS(world_size=2),
                                      scheduler_config=NS(max_num_seqs=64))),
                    NS(parallel_config=NS(world_size=2),
                       scheduler_config=NS(max_num_seqs=64))):
            engine = wrap(cfg)
            assert ctx._engine_world_size(engine) == 2
            assert _max_num_seqs(engine) == 64
