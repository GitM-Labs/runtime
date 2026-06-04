# Scoring Input Contract v0

**Owner:** Danny Le

**Track:** Track 4 - Routing & Priority Modeling

**Version:** v0

**Last updated:** June 3, 2026

## Purpose

This document defines the exact schema the scorer expects. Every field listed below must be present in Airtable before `scorer_v0.py` can run. Khoa and Arshad: if anything here is unclear or conflicts with how you're structuring your outputs, flag it before you finalize your schemas.

## Identifier Fields

These fields are not scorer inputs. They exist to uniquely identify records and join data across Airtable tables.

| Field | Type | Produced By | Description |
|---|---|---|---|
| prospect_id | string | Arshad | Unique identifier per prospect. Used to join enrichment, company, and outreach data. |
| sender_id | string | Danny | FDE this prospect is assigned to. Used to select the correct sender-specific warmth value at scoring time. Values: `asmar`, `jane`, `giancarlos`. |

## Scorer Input Fields

These are the six fields the scorer reads directly. All must be present and within the specified range or the scorer will error.

| Field | Type | Range | Produced By | Description |
|---|---|---|---|---|
| warmth | float | 0-1 | Khoa | Sender-specific connection strength. Must be computed per sender, not globally. A cold stranger = 0.1, a direct prior colleague = 0.9. |
| signal_recency | float | 0-1 | Khoa | Recency of the prospect's most recent public GPU pain signal. Computed as `np.exp(-days_since_signal / half_life)`, half_life tentatively 30 days. If no signal found, set to 0. |
| company_tier | int | 1, 2, or 3 | Arshad | 1 = orchestrator / managed GPU platform, 2 = non-text mid-market (biotech, robotics, edge, HPC), 3 = weak or unclear fit. |
| pain_acknowledged | int | 0 or 1 | Khoa | 1 if the prospect has publicly expressed a GPU pain signal (post, comment, article, talk). 0 otherwise. |
| engagement_score | float | 0-1 | Khoa | Continuous measure of how much this person has interacted with Git.M content, ex. liked posts, commented, connected with someone at Git.M. 0 = no interaction, 1 = high engagement. |
| prior_engagement | int | 0 or 1 | Jorge (via HeyReach reply pipeline) | 1 if this prospect has previously replied to or interacted with Git.M outreach. Defaults to 0 for all prospects at sprint start. Git.M has not yet contacted them. Will become meaningful as reply data flows in. |

## Notes

- `warmth` is sender-specific. Khoa should produce a warmth value per prospect per sender, not a single global value.
- `signal_recency` decay formula: `np.exp(-days_since_signal / half_life)`. Half-life is tentatively 30 days, pending confirmation from Jalon.
- `prior_engagement` defaults to 0 at sprint start. Jorge's HeyReach reply pipeline will update this field as reply data comes in.
- All float fields must be in range [0, 1]. Out-of-range values will not be caught silently, and the scorer will produce incorrect scores.
- `company_tier` must be exactly 1, 2, or 3. Any other value defaults to tier 3 (score = 0.2) in the current scorer logic.