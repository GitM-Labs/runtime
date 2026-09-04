"""Traffic replay library — the pytest face of ``python -m gitm.traffic --selftest``.

The assertions live in :mod:`gitm.traffic._selftest` and are called from both
places, so the runnable check a reader is told about in the spec and the check CI
runs are the *same* check, not two that can drift apart.

Each case is one function from ``_selftest.CHECKS``; a failure names the check
that broke rather than the whole library.
"""

from __future__ import annotations

import pytest

from gitm.traffic import _selftest


@pytest.mark.parametrize("check", _selftest.CHECKS, ids=lambda f: f.__name__)
def test_traffic_check(check) -> None:
    if not _selftest.FIXTURES.exists():
        pytest.skip(f"fixtures not present at {_selftest.FIXTURES}")
    check()


def test_every_check_is_registered() -> None:
    """A check that exists but is never run is worse than no check."""
    defined = {
        name for name in dir(_selftest) if name.startswith("check_") and callable(getattr(_selftest, name))
    }
    assert {f.__name__ for f in _selftest.CHECKS} == defined
