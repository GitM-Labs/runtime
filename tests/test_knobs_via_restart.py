"""EngineArg knobs route through restart by default.

vLLM V1 reads many scheduler-looking EngineArgs at construction time. The
applicator should rebuild a fresh engine with those values baked in instead of
mutating scheduler_config fields that the live engine may ignore.
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.optimizer.apply import LiveEngineApplicator

_ENGINE_ARG_KNOB = SimpleNamespace(knob="max_num_seqs", value=256)


def test_engine_arg_rebuilds_by_default():
    built: list[dict] = []

    def restart(_engine, knob_values):
        built.append(dict(knob_values))
        return SimpleNamespace(name="candidate")

    app = LiveEngineApplicator(
        SimpleNamespace(name="baseline"),
        throughput_fn=lambda _e: 100.0,
        restart_fn=restart,
    )
    app.snapshot()
    app.apply(_ENGINE_ARG_KNOB)
    assert built == [{"max_num_seqs": 256}]
    assert app._prev[0] == "restart"


def test_unknown_knob_defaults_to_restart_not_hotswap():
    built: list[dict] = []
    sets: list[tuple[str, object]] = []

    def restart(_engine, knob_values):
        built.append(dict(knob_values))
        return SimpleNamespace(name="candidate")

    app = LiveEngineApplicator(
        SimpleNamespace(name="baseline"),
        throughput_fn=lambda _e: 100.0,
        getter=lambda _e, _k: None,
        setter=lambda _e, k, v: sets.append((k, v)),
        restart_fn=restart,
    )
    app.snapshot()
    app.apply(SimpleNamespace(knob="future_engine_arg", value=1))
    assert sets == []
    assert built == [{"future_engine_arg": 1}]
    assert app._prev[0] == "restart"
