import pandas as pd
import numpy as np

def score_prospect(
    warmth: float,
    signal_recency: float,
    company_tier: int,
    pain_acknowledged: int,
    engagement_score: float,
    prior_engagement: int
) -> float:
    """
    Compute priority score for a single prospect.

    Inputs:
        warmth (float): 0-1, sender-specific connection strength
        signal_recency (float): 0-1, recency of GPU pain signal
        company_tier (int): 1 = orchestrator/managed GPU platform,
                            2 = non-text mid-market (biotech, robotics, edge, HPC),
                            3 = weak or unclear fit
        pain_acknowledged (int): 1 if prospect expressed GPU pain publicly, 0 if not
        engagement_score (float): 0-1, interaction with Git.M content
        prior_engagement (int): 1 if prospect previously replied to Git.M, 0 if not

    Returns:
        float: priority score between 0 and 100
    """

    # Weights
    W_WARMTH = 0.35
    W_SIGNAL_RECENCY = 0.25
    W_COMPANY_TIER = 0.20
    W_PAIN = 0.10
    W_ENGAGEMENT = 0.05
    W_PRIOR = 0.05

    # Company tier score
    tier_score_map = {1: 1.0, 2: 0.6, 3: 0.2}
    tier_score = tier_score_map.get(company_tier, 0.2)

    score = (
        warmth * W_WARMTH +
        signal_recency * W_SIGNAL_RECENCY +
        tier_score * W_COMPANY_TIER +
        pain_acknowledged * W_PAIN +
        engagement_score * W_ENGAGEMENT +
        prior_engagement * W_PRIOR
    )

    return round(score * 100, 2)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a full dataframe of prospects.
    Expects columns matching the input contract.
    Returns dataframe with priority_score column added, sorted descending.
    """
    df = df.copy()
    df["priority_score"] = df.apply(
        lambda row: score_prospect(
            warmth=row["warmth"],
            signal_recency=row["signal_recency"],
            company_tier=row["company_tier"],
            pain_acknowledged=row["pain_acknowledged"],
            engagement_score=row["engagement_score"],
            prior_engagement=row["prior_engagement"]
        ),
        axis=1
    )
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage with dummy data
    sample_data = {
        "prospect_id": ["p001", "p002", "p003"],
        "sender_id": ["asmar", "jane", "giancarlos"],
        "warmth": [0.8, 0.3, 0.5],
        "signal_recency": [0.9, 0.2, 0.7],
        "company_tier": [1, 2, 2],
        "pain_acknowledged": [1, 0, 1],
        "engagement_score": [0.4, 0.0, 0.2],
        "prior_engagement": [0, 0, 1]
    }

    df = pd.DataFrame(sample_data)
    results = score_dataframe(df)
    print(results[["prospect_id", "sender_id", "priority_score"]])