"""Getting the GPU back from workers that did not leave.

The model does not live in gitm's process. `VLLM_ENABLE_V1_MULTIPROCESSING`
defaults on, so the weights and the KV cache are held by a child process, and
until that child is gone its device memory is gone with it. The MI355X run hit
this as `Free memory 0.0/287.98 GiB` on a baseline rebuild, with five GPUs at
100% utilisation and 4-5% VRAM — workers holding the devices after a partial
teardown.

Calling `shutdown()` is not the same as the memory coming back. vLLM's
`MPClient.shutdown` no-ops once its finalizer has run, and gitm's teardown stops
at the first `shutdown` attribute it finds, which need not be the one owning the
workers. These use real processes, because the thing under test is whether a
process is actually gone.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from gitm.workloads import (
    ENGINE_WORKER_PREFIX,
    _shutdown_timeout,
    engine_worker_pids,
    reap_engine_workers,
)


def _sleep_forever():
    while True:
        time.sleep(0.2)


def _exit_soon():
    time.sleep(0.2)


@pytest.fixture
def spawned():
    """Start processes, and make sure none outlive the test whatever happens."""
    started: list[mp.Process] = []

    def start(target, name):
        p = mp.Process(target=target, name=name, daemon=True)
        p.start()
        started.append(p)
        return p

    yield start
    for p in started:
        if p.is_alive():
            p.kill()
            p.join(2)


def test_nothing_to_reap_is_not_an_error():
    assert reap_engine_workers(set(), 0.5) == []
    assert reap_engine_workers(None, 0.5) == []


def test_a_worker_that_exits_on_its_own_is_not_forced(spawned):
    proc = spawned(_exit_soon, f"{ENGINE_WORKER_PREFIX}_DP0")
    forced = reap_engine_workers({proc.pid}, 10.0)
    assert forced == [], forced
    assert not proc.is_alive()


def test_a_worker_that_will_not_leave_is_forced_down(spawned):
    """The run's alternative is a baseline rebuild into a device somebody else
    is still holding, which ends the run."""
    proc = spawned(_sleep_forever, ENGINE_WORKER_PREFIX)
    forced = reap_engine_workers({proc.pid}, 1.0)

    assert len(forced) == 1
    assert ENGINE_WORKER_PREFIX in forced[0]
    assert "did not exit within 1s" in forced[0]
    assert not proc.is_alive()


def test_only_vllm_workers_are_touched(spawned):
    """A run has other children. Reaping one of those to free a GPU would be a
    cure worse than the disease, so the match is on vLLM's own process name."""
    mine = spawned(_sleep_forever, "gitm-something-else")
    theirs = spawned(_sleep_forever, f"{ENGINE_WORKER_PREFIX}_DP1")

    forced = reap_engine_workers(engine_worker_pids(), 1.0)

    assert len(forced) == 1 and ENGINE_WORKER_PREFIX in forced[0]
    assert not theirs.is_alive()
    assert mine.is_alive(), "reaped a process that was not vLLM's"


def test_every_late_worker_is_reported_not_just_the_first(spawned):
    """A multi-GPU run has one per engine, and a report naming one of four reads
    as a tidier teardown than happened."""
    procs = [spawned(_sleep_forever, f"{ENGINE_WORKER_PREFIX}_DP{i}") for i in range(3)]
    forced = reap_engine_workers({p.pid for p in procs}, 1.0)

    assert len(forced) == 3
    assert {p.pid for p in procs} == {int(f.split("pid ")[1].split(")")[0]) for f in forced}
    assert not any(p.is_alive() for p in procs)


def test_waiting_is_bounded_by_the_timeout(spawned):
    proc = spawned(_sleep_forever, ENGINE_WORKER_PREFIX)
    started = time.monotonic()
    reap_engine_workers({proc.pid}, 1.0)
    # The wait, plus the bounded terminate/kill escalation after it.
    assert time.monotonic() - started < 9.0


# --- whose workers are these ------------------------------------------------


def test_another_engines_workers_survive_a_teardown(spawned):
    """The case that matters most. In parallel restart mode the baseline and the
    candidate are both up and their workers share a name, so tearing down the
    candidate must not touch the baseline's — the restore path reactivates that
    baseline engine, and it cannot run if its workers are gone."""
    baseline = spawned(_sleep_forever, ENGINE_WORKER_PREFIX)
    candidate = spawned(_sleep_forever, f"{ENGINE_WORKER_PREFIX}_DP0")

    forced = reap_engine_workers({candidate.pid}, 1.0)

    assert len(forced) == 1 and str(candidate.pid) in forced[0]
    assert not candidate.is_alive()
    assert baseline.is_alive(), "tearing down the candidate killed the baseline"


def test_the_pids_an_engine_owns_are_the_ones_it_added(spawned):
    """How ownership is established: sample either side of the build."""
    first = spawned(_sleep_forever, ENGINE_WORKER_PREFIX)
    before = engine_worker_pids()
    second = spawned(_sleep_forever, f"{ENGINE_WORKER_PREFIX}_DP1")

    owned = engine_worker_pids() - before
    assert owned == {second.pid}
    assert first.pid in before


# --- the timeout setting ----------------------------------------------------


def test_a_bad_timeout_warns_and_still_reaps(monkeypatch):
    """Parsing it inside the try meant a typo raised, the handler blanked the
    result, and the whole cleanup was skipped with nothing said."""
    monkeypatch.setenv("GITM_SHUTDOWN_TIMEOUT_S", "thirty")
    with pytest.warns(RuntimeWarning, match="not a number"):
        assert _shutdown_timeout() == 30.0


def test_the_timeout_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("GITM_SHUTDOWN_TIMEOUT_S", "5")
    assert _shutdown_timeout() == 5.0
    monkeypatch.setenv("GITM_SHUTDOWN_TIMEOUT_S", "-1")
    assert _shutdown_timeout() == 0.0          # clamped, never negative
    monkeypatch.delenv("GITM_SHUTDOWN_TIMEOUT_S")
    assert _shutdown_timeout() == 30.0
