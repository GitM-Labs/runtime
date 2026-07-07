"""GITM_KNOBS_VIA_RESTART routes scheduling knobs through the rebuild.

If an engine ignores a live scheduler-config mutation (vLLM V1 reads it at
construction), force_restart makes the applicator apply scheduling knobs via the
restart path — a fresh engine with the knob baked in — instead of hot-swapping.
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.optimizer.apply import LiveEngineApplicator

_SCHED_KNOB = SimpleNamespace(knob="max_num_seqs", value=256)  # a scheduling knob


def test_force_restart_rebuilds_for_a_scheduling_knob():
    built: list[tuple[str, object]] = []

    def restart(_engine, knob, value):
        built.append((knob, value))
        return SimpleNamespace(name="candidate")

    app = LiveEngineApplicator(
        SimpleNamespace(name="baseline"),
        throughput_fn=lambda _e: 100.0,
        restart_fn=restart,
        force_restart=True,
    )
    app.snapshot()
    app.apply(_SCHED_KNOB)
    assert built == [("max_num_seqs", 256)]  # rebuilt, not hot-swapped
    assert app._prev[0] == "restart"


def test_default_hot_swaps_a_scheduling_knob():
    sets: list[tuple[str, object]] = []
    app = LiveEngineApplicator(
        SimpleNamespace(),
        throughput_fn=lambda _e: 100.0,
        getter=lambda _e, _k: None,
        setter=lambda _e, k, v: sets.append((k, v)),
        restart_fn=lambda _e, _k, _v: SimpleNamespace(),
    )
    app.snapshot()
    app.apply(_SCHED_KNOB)
    assert sets == [("max_num_seqs", 256)]  # hot-swapped in place
    assert app._prev[0] == "hotswap"
