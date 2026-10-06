"""Re-deciding what to try next, from what this run has already measured.

Phase 3 built an order before a single candidate had been measured and Phase 4
walked it to the end, so nothing a run learned could reach the next choice until
the following run read the export back. On a 24h budget that is a feedback loop
with a 24h period.

``rerank="recapture"`` traces the workload again after each applied candidate.
Coverage is measured per trace, so a fresh one re-ranks the rest on its own with
no new scoring rule: the region the last candidate fixed is no longer where the
time is, and the ranking follows.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import gitm.scheduler.loop as loop
from gitm.tracer.schema import KernelEvent, Trace


def _trace(names, *, run_id="r"):
    events = [
        KernelEvent(name=n, start_ns=i * 1000, end_ns=i * 1000 + 900, stream_id=7,
                    device_id=0, correlation_id=i)
        for i, n in enumerate(names)
    ]
    return Trace(workload_id="vllm-decode", fingerprint="f", run_id=run_id,
                 device_count=1, vendor="nvidia", captured_at_ns=0,
                 duration_ns=max(len(names), 1) * 1000, events=events)


class _Runner:
    workload_id = "vllm-decode"

    def __call__(self) -> dict:
        return {"events": 1}


@pytest.fixture
def captures(monkeypatch):
    """Every capture the run takes, in order. The first is Phase 1's; any after
    it are re-captures, which is the thing under test."""
    taken: list[Path] = []

    @contextmanager
    def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        taken.append(Path(out_path))
        yield _trace(["flash_fwd_kernel", "void cutlass_gemm"], run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fake_capture)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    return taken


def _run(tmp_path, **kw):
    from gitm import optimize

    return optimize(budget="30s", scratch=str(tmp_path),
                    workload_runner=_Runner(), **kw)


def _rerank_doc(tmp_path):
    hits = list(Path(tmp_path).glob("runs/*/rerank.json"))
    return json.loads(hits[0].read_text()) if hits else None


# --------------------------------------------------------------------------- #
# off is what the loop has always done                                         #
# --------------------------------------------------------------------------- #
def test_off_takes_one_capture_and_writes_no_rerank_record(tmp_path, captures):
    """The default. Merging this changes no run's behaviour."""
    _run(tmp_path)

    assert len(captures) == 1
    assert _rerank_doc(tmp_path) is None


def test_off_is_the_default_for_every_entry_point(tmp_path, captures):
    from gitm.scheduler.loop import LoopConfig

    assert LoopConfig().rerank == "off"


# --------------------------------------------------------------------------- #
# recapture                                                                    #
# --------------------------------------------------------------------------- #
def test_recapture_traces_again_after_each_applied_candidate(tmp_path, captures):
    _run(tmp_path, rerank="recapture")

    assert len(captures) > 1, "no re-capture was taken"
    # every re-capture writes its own trace, so one cannot overwrite another
    assert len(set(captures)) == len(captures)


def test_each_recapture_is_recorded_with_what_prompted_it(tmp_path, captures):
    """Without this a report says which candidates were tried but not that the
    order moved between them, nor on what."""
    _run(tmp_path, rerank="recapture")
    doc = _rerank_doc(tmp_path)

    assert doc is not None
    assert doc["mode"] == "recapture"
    assert doc["steps"]
    step = doc["steps"][0]
    assert set(step) >= {"after", "recaptured", "error", "order_before",
                         "order_after", "changed"}
    assert step["recaptured"] is True


def test_an_attempted_candidate_never_returns_to_the_queue(tmp_path, captures):
    """Re-ranking re-orders what is left. A candidate already applied or rejected
    has been popped, and re-ordering the remainder cannot bring it back — without
    that, a lever could be measured twice in one run and counted twice."""
    _run(tmp_path, rerank="recapture")
    doc = _rerank_doc(tmp_path)

    attempted = [s["after"] for s in doc["steps"]]
    for step in doc["steps"]:
        assert step["after"] not in step["order_after"]
    assert len(attempted) == len(set(attempted))


def test_the_queue_only_ever_shrinks(tmp_path, captures):
    """A re-rank re-orders; it must not add. Otherwise a run could never finish."""
    _run(tmp_path, rerank="recapture")
    doc = _rerank_doc(tmp_path)

    for step in doc["steps"]:
        assert len(step["order_after"]) <= len(step["order_before"])
        assert set(step["order_after"]) <= set(step["order_before"])


# --------------------------------------------------------------------------- #
# a failed re-measurement must not cost the run                                #
# --------------------------------------------------------------------------- #
def test_a_failed_recapture_keeps_the_order_it_had(tmp_path, monkeypatch):
    """The run has already paid for its trace and its A/Bs. Losing all of that to
    a failed re-measurement would be a worse outcome than not re-ranking."""
    calls = {"n": 0}

    @contextmanager
    def flaky(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("capture device went away")
        yield _trace(["flash_fwd_kernel", "void cutlass_gemm"], run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", flaky)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    result = _run(tmp_path, rerank="recapture")      # must not raise

    assert result.get("summary") is not None
    doc = _rerank_doc(tmp_path)
    if doc is not None:
        for step in doc["steps"]:
            assert step["recaptured"] is False
            assert step["order_before"] == step["order_after"]


def test_an_empty_recapture_keeps_the_order_it_had(tmp_path, monkeypatch):
    """A trace with no kernels would rank everything at zero coverage and shuffle
    the queue into alphabetical order on no evidence at all."""
    calls = {"n": 0}

    @contextmanager
    def empties(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        calls["n"] += 1
        names = ["flash_fwd_kernel", "void cutlass_gemm"] if calls["n"] == 1 else []
        yield _trace(names, run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", empties)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    _run(tmp_path, rerank="recapture")
    doc = _rerank_doc(tmp_path)

    if doc is not None:
        for step in doc["steps"]:
            assert step["order_before"] == step["order_after"]


# --------------------------------------------------------------------------- #
# the point: a changed trace changes what gets tried next                      #
# --------------------------------------------------------------------------- #
def test_a_moved_bottleneck_reorders_what_is_tried_next(tmp_path, monkeypatch):
    """The whole claim. The opening trace is all attention, so attention-scoped
    levers cover all of it. The re-capture is all MLP — those same levers now
    cover none of the work and have to sink. Without re-ranking the run would
    keep spending its budget on attention after attention stopped being where the
    time goes.

    Driven through ``run_loop`` rather than ``optimize`` to widen ``top_n``: the
    default five are all whole-step levers, which name every dense op and so
    cover both traces equally. Nothing could move, and a test that passed on that
    would be testing nothing.
    """
    from gitm.scheduler.loop import LoopConfig, run_loop

    calls = {"n": 0}

    @contextmanager
    def moving(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        calls["n"] += 1
        names = (["flash_fwd_kernel"] * 8 if calls["n"] == 1
                 else ["cutlass_down_proj_kernel"] * 8)
        yield _trace(names, run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", moving)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    run_loop(LoopConfig(budget="30s", scratch=str(tmp_path), workload_runner=_Runner(),
                        rerank="recapture", top_n_interventions=25))
    doc = _rerank_doc(tmp_path)

    assert doc is not None and doc["steps"]
    first = doc["steps"][0]
    assert first["recaptured"] is True
    assert first["changed"] is True, "the order did not move when the bottleneck did"

    # What actually decides the run: the next candidate off the queue.
    assert first["order_after"][0] != first["order_before"][0], (
        "the queue was re-ordered but the next candidate did not change")
    # and the re-ordering is confined to what was still unattempted
    assert set(first["order_after"]) == set(first["order_before"])


# --------------------------------------------------------------------------- #
# the re-capture spends budget, and says what went wrong                       #
# --------------------------------------------------------------------------- #
def test_the_budget_bounds_the_recapture_too(tmp_path, monkeypatch):
    """The re-capture runs the workload, so it spends wall clock. Checking the
    deadline only *before* it let a candidate cycle start on a budget the trace
    had already consumed — overrunning by a whole A/B on top of the trace.

    Asserted on what the run *did*, not on how long it took: the clock here is
    monkeypatched, so timing it would only measure the patch."""
    from gitm.scheduler.loop import LoopConfig, run_loop

    @contextmanager
    def fine(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        yield _trace(["flash_fwd_kernel"] * 4, run_id=run_id or "r")

    monkeypatch.setattr(loop, "capture", fine)
    monkeypatch.setattr(loop, "sync_device", lambda: None)

    # The first re-capture's workload run exhausts the budget, and nothing after
    # it should start. Any further candidate would add another rerank step.
    real_ns = loop.time.time_ns
    runs = {"n": 0, "spent": False}

    class _Burner(_Runner):
        def __call__(self):
            runs["n"] += 1
            if runs["n"] > 1:        # run 1 is Phase 1's capture, not a re-capture
                runs["spent"] = True
            return {"events": 1}

    monkeypatch.setattr(
        loop.time, "time_ns",
        lambda: real_ns() + 10 ** 12 if runs["spent"] else real_ns())

    # Counting candidate cycles directly: that is the unit the budget bounds, and
    # the step count cannot see the difference — an overrunning candidate is
    # applied and only then breaks, before it would record a step of its own.
    applies = {"n": 0}
    real_apply = loop.apply_intervention

    def counting(*a, **kw):
        applies["n"] += 1
        return real_apply(*a, **kw)

    monkeypatch.setattr(loop, "apply_intervention", counting)

    run_loop(LoopConfig(budget="30s", scratch=str(tmp_path),
                        workload_runner=_Burner(), rerank="recapture"))

    assert applies["n"] == 1, (
        "a candidate was applied after the re-capture had spent the budget")


def test_a_workload_failure_is_not_recorded_as_a_missing_trace(tmp_path, monkeypatch):
    """The workload dying during the extra run and the tracer being unavailable
    are different events. Recording both as "no trace" hides the first from
    whoever reads the run afterwards."""
    from gitm.scheduler.loop import LoopConfig, run_loop

    @contextmanager
    def fine(out_path, *, workload_id="w", fingerprint="f", run_id=None):
        yield _trace(["flash_fwd_kernel"] * 4, run_id=run_id or "r")

    calls = {"n": 0}

    class _Flaky(_Runner):
        def __call__(self):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("engine died mid-run")
            return {"events": 1}

    monkeypatch.setattr(loop, "capture", fine)
    monkeypatch.setattr(loop, "sync_device", lambda: None)
    run_loop(LoopConfig(budget="30s", scratch=str(tmp_path),
                        workload_runner=_Flaky(), rerank="recapture"))
    doc = _rerank_doc(tmp_path)

    assert doc is not None and doc["steps"]
    step = doc["steps"][0]
    assert step["recaptured"] is False
    assert step["error"] and "workload run failed" in step["error"]


# --------------------------------------------------------------------------- #
# a mode nobody implements must not read as "off"                              #
# --------------------------------------------------------------------------- #
def test_an_unknown_mode_is_refused_rather_than_ignored(tmp_path):
    """A typo used to run the whole budget without the thing it asked for, and
    say nothing. The CLI already refuses via choices=; the embedded entry point
    has to refuse too, or only one kind of caller is protected."""
    from gitm.scheduler.loop import LoopConfig, run_loop

    with pytest.raises(ValueError, match="rerank must be one of"):
        run_loop(LoopConfig(budget="1s", scratch=str(tmp_path),
                            workload_runner=_Runner(), rerank="recaptre"))


def test_the_cli_offers_exactly_the_modes_the_loop_knows(tmp_path):
    """One list. A mode the CLI accepts and the loop does not would now raise
    mid-run instead of reading as off — worse than the bug it replaced."""
    from gitm.cli import _parser
    from gitm.scheduler.loop import RERANK_MODES

    action = next(a for a in _parser()._subparsers._group_actions[0]
                  .choices["run"]._actions if a.dest == "rerank")

    assert tuple(action.choices) == RERANK_MODES
