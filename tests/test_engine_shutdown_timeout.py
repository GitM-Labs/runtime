"""An engine's shutdown is given gitm's timeout instead of vLLM's 5 s default."""

from __future__ import annotations

from gitm.workloads import _call_shutdown


def test_a_shutdown_that_takes_a_timeout_gets_gitms():
    seen = {}

    class MPClient:  # vLLM's shape: shutdown(self, timeout=None)
        def shutdown(self, timeout=None):
            seen["timeout"] = timeout

    _call_shutdown(MPClient().shutdown, 30.0)
    assert seen["timeout"] == 30.0          # not None, which vLLM turns into a 5 s kill


def test_a_shutdown_without_one_is_called_as_before():
    calls = []

    def shutdown():
        calls.append(1)

    _call_shutdown(shutdown, 30.0)
    assert calls == [1]


def test_the_timeout_honours_gitm_shutdown_timeout_s(monkeypatch):
    from gitm.workloads import _shutdown_timeout

    monkeypatch.setenv("GITM_SHUTDOWN_TIMEOUT_S", "90")
    assert _shutdown_timeout() == 90.0
