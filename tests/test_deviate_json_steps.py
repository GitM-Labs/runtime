"""`gitm deviate --json` must not state a floor it could not scale.

The predicted graph prices ONE decode step. ``--steps`` says how many the
captured window holds, and nothing in the tool can derive it — the observation
always covers the whole capture either way. ``steps or 1`` left the floor at a
single step while ``observed_s`` beside it covered all of them, so a 100-step
window read exactly 100x over budget with nothing in the payload saying why.

``render_deviation`` already refuses that comparison, printing UNSCALED instead
of a ratio. This is the path a tool reads with no human present, and it is the
one that emitted the number.
"""

from __future__ import annotations

import json

from gitm.optimizer.deviation import main as deviate_main


def _trace(path, steps=100):
    with open(path, "w", encoding="utf-8") as fh:
        t = 0
        for _ in range(steps):
            for name, dur in (("flash_fwd_kernel", 4000), ("void cutlass_gemm", 6000)):
                fh.write(json.dumps({"kind": "kernel", "name": name, "start_ns": t,
                                     "end_ns": t + dur, "stream_id": 7,
                                     "device_id": 0}) + "\n")
                t += dur + 100
    return path


def _run(capsys, trace, *extra):
    """Through the real argv path the CLI uses, so the flag name is pinned too."""
    assert deviate_main([str(trace), "--json", "--model", "kimi-k2.5", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_without_steps_no_floor_is_stated(tmp_path, capsys):
    """A one-step floor beside a whole-window observation is not a ratio, so the
    payload says it has no floor rather than quoting one off by the step count."""
    doc = _run(capsys, _trace(tmp_path / "t.jsonl"))

    assert doc["steps"] is None
    assert doc["floors_scaled"] is False
    assert all(v["floor_s"] is None for v in doc["ops"].values())


def test_with_steps_the_floor_is_scaled_to_the_window(tmp_path, capsys):
    doc = _run(capsys, _trace(tmp_path / "t.jsonl"), "--steps", "100")

    assert doc["steps"] == 100
    assert doc["floors_scaled"] is True
    modeled = {op: v for op, v in doc["ops"].items() if op != "<unmodeled>"}
    assert modeled and all(v["floor_s"] > 0 for v in modeled.values())


def test_the_floor_scales_with_the_step_count(tmp_path, capsys):
    """Ten times the steps is ten times the floor — which is the factor the
    missing flag used to drop on the floor while leaving the observation whole."""
    trace = _trace(tmp_path / "t.jsonl")
    ten = _run(capsys, trace, "--steps", "10")["ops"]["attn_score_value"]["floor_s"]
    hundred = _run(capsys, trace, "--steps", "100")["ops"]["attn_score_value"]["floor_s"]

    assert abs(hundred - ten * 10) < 1e-12


def test_observed_time_does_not_depend_on_steps(tmp_path, capsys):
    """``--steps`` scales the prediction, never the measurement. If it moved the
    observation too, the unscaled comparison would have been self-consistent."""
    trace = _trace(tmp_path / "t.jsonl")
    without = _run(capsys, trace)["ops"]["attn_score_value"]["observed_s"]
    with_steps = _run(capsys, trace, "--steps", "100")["ops"]["attn_score_value"]["observed_s"]

    assert without == with_steps


def test_floors_scaled_distinguishes_the_two_reasons_a_floor_is_null(tmp_path, capsys):
    """``<unmodeled>`` has no floor because the graph does not price it. Every op
    has none when the window could not be scaled. A consumer has to tell those
    apart, and null alone cannot."""
    doc = _run(capsys, _trace(tmp_path / "t.jsonl"), "--steps", "100")

    assert doc["floors_scaled"] is True
    assert doc["ops"]["<unmodeled>"]["floor_s"] is None
    assert any(v["floor_s"] is not None for v in doc["ops"].values())
