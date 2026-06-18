import pandas as pd

from gitm.routing.scorer_v0 import score_dataframe, score_prospect


def test_score_prospect_basic():
    score = score_prospect(
        warmth=0.8,
        signal_recency=0.9,
        company_tier=1,
        pain_acknowledged=1,
        engagement_score=0.4,
        prior_engagement=0
    )
    assert score == 82.5

def test_score_prospect_cold():
    score = score_prospect(
        warmth=0.1,
        signal_recency=0.0,
        company_tier=3,
        pain_acknowledged=0,
        engagement_score=0.0,
        prior_engagement=0
    )
    assert score == 7.5

def test_score_prospect_unknown_tier():
    score = score_prospect(
        warmth=0.5,
        signal_recency=0.5,
        company_tier=99,
        pain_acknowledged=0,
        engagement_score=0.0,
        prior_engagement=0
    )
    # Unknown tier should default to tier 3 (0.2)
    assert score == 34.0

def test_score_dataframe_sorted():
    df = pd.DataFrame({
        "prospect_id": ["p001", "p002"],
        "sender_id": ["asmar", "jane"],
        "warmth": [0.8, 0.3],
        "signal_recency": [0.9, 0.2],
        "company_tier": [1, 2],
        "pain_acknowledged": [1, 0],
        "engagement_score": [0.4, 0.0],
        "prior_engagement": [0, 0]
    })
    results = score_dataframe(df)
    assert results.iloc[0]["prospect_id"] == "p001"
    assert results.iloc[0]["priority_score"] > results.iloc[1]["priority_score"]
