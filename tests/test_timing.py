from __future__ import annotations

import math

import pytest

from gitm._timing import require_positive_duration, require_positive_work


@pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_nonpositive_or_nonfinite_timing_is_refused(duration):
    with pytest.raises(RuntimeError, match="timing unavailable"):
        require_positive_duration(duration, context="test workload")


def test_positive_timing_is_preserved():
    assert require_positive_duration(0.125, context="test workload") == 0.125


def test_empty_work_unit_is_refused():
    with pytest.raises(RuntimeError, match="work coverage unavailable"):
        require_positive_work(0, context="test workload")
