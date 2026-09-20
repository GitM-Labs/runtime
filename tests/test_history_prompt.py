"""Asking, once, whether a run should be scored from what previous runs measured."""

from __future__ import annotations

import json
import os

from gitm.scheduler.loop import (
    LoopConfig,
    _ask_use_history,
    _prior_runs_with_results,
    _resolve_use_history,
)


def _pipe(text: str | None):
    """A real pipe, because the prompt waits on ``select`` and a StringIO is not
    selectable — a fake stream here would test something the run never does."""
    r, w = os.pipe()
    if text is not None:
        os.write(w, text.encode())
    os.close(w) if text is not None else None
    return os.fdopen(r)


def _runs(tmp_path, n, *, with_export=True):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = runs / f"run{i}"
        d.mkdir()
        if with_export:
            (d / "verification.json").write_text(json.dumps({"results": []}))
    return tmp_path


# --------------------------------------------------------------------------- #
# what there is to ask about                                                   #
# --------------------------------------------------------------------------- #
def test_only_runs_that_actually_recorded_something_count(tmp_path):
    _runs(tmp_path, 2)
    _runs(tmp_path / "other", 3, with_export=False)

    assert _prior_runs_with_results(str(tmp_path)) == 2
    assert _prior_runs_with_results(str(tmp_path / "other")) == 0


def test_no_previous_results_asks_nothing_and_uses_nothing(tmp_path):
    """There is no question to put, and no record to rank from."""
    assert _resolve_use_history(LoopConfig(scratch=str(tmp_path))) is False


# --------------------------------------------------------------------------- #
# the answer                                                                   #
# --------------------------------------------------------------------------- #
def test_yes_uses_the_previous_results():
    assert _ask_use_history(3, stream=_pipe("y\n"), tty=True) is True


def test_a_bare_enter_uses_them():
    """Capitalised [Y] in the prompt promises this."""
    assert _ask_use_history(3, stream=_pipe("\n"), tty=True) is True


def test_no_ignores_them():
    assert _ask_use_history(3, stream=_pipe("n\n"), tty=True) is False


def test_silence_uses_them_rather_than_waiting_forever():
    """An unattended run must not sit on a prompt. Of the two answers, using the
    record is the one that discards nothing."""
    assert _ask_use_history(3, timeout_s=0.2, stream=_pipe(None), tty=True) is True


def test_without_a_terminal_it_does_not_wait_at_all():
    """Nobody is there to answer, so it takes the same default immediately rather
    than burning the timeout against a pipe that will never reply."""
    assert _ask_use_history(3, timeout_s=30.0, stream=_pipe(None), tty=False) is True


# --------------------------------------------------------------------------- #
# the config still decides when it was told to                                 #
# --------------------------------------------------------------------------- #
def test_an_explicit_setting_is_not_second_guessed(tmp_path):
    """A caller that said which way it wants this is never prompted — that is
    what keeps scripted and scheduled runs deterministic."""
    _runs(tmp_path, 2)

    assert _resolve_use_history(LoopConfig(scratch=str(tmp_path), use_history=False)) is False
    assert _resolve_use_history(LoopConfig(scratch=str(tmp_path), use_history=True)) is True


def test_declining_deletes_nothing(tmp_path):
    """Answering no skips the record for this run. It does not throw away
    measurements that cost GPU time to produce."""
    _runs(tmp_path, 2)
    before = sorted(p.name for p in (tmp_path / "runs").iterdir())

    assert _ask_use_history(2, stream=_pipe("n\n"), tty=True) is False

    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == before
    assert all((tmp_path / "runs" / d / "verification.json").exists() for d in before)
