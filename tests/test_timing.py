from __future__ import annotations

import math

import pytest

from gitm._timing import require_positive_duration


@pytest.mark.parametrize("duration", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_nonpositive_or_nonfinite_timing_is_refused(duration):
    with pytest.raises(RuntimeError, match="timing unavailable"):
        require_positive_duration(duration, context="test workload")




