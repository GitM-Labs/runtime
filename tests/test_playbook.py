"""Playbook schema — the pytest face of ``python -m gitm.playbook --selftest``.

Same pattern as ``tests/test_traffic.py``: the assertions live in
:mod:`gitm.playbook._selftest` and are called from both places, so the runnable
check a reader is told about and the check CI runs cannot drift apart.
"""

from __future__ import annotations

import pytest

from gitm.playbook import _selftest


@pytest.mark.parametrize("check", _selftest.CHECKS, ids=lambda f: f.__name__)
def test_playbook_check(check) -> None:
    if not _selftest.EXAMPLES.exists():
        pytest.skip(f"examples not present at {_selftest.EXAMPLES}")
    check()


def test_every_check_is_registered() -> None:
    """A check that exists but is never run is worse than no check."""
    defined = {
        name
        for name in dir(_selftest)
        if name.startswith("check_") and callable(getattr(_selftest, name))
    }
    assert {f.__name__ for f in _selftest.CHECKS} == defined
