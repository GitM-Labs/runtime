from __future__ import annotations

import pytest

from gitm.routing.scorer_v0 import score_prospect


def test_unknown_company_tier_is_not_silently_scored_as_tier_three():
    with pytest.raises(ValueError, match="company_tier"):
        score_prospect(0.5, 0.5, 99, 1, 0.5, 0)
