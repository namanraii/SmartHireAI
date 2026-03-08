"""
ranker.py — Rank multiple candidates by composite score
"""


def compute_composite_score(ai_score: float, selection_prob: float,
                             ai_weight: float = 0.6, prob_weight: float = 0.4) -> float:
    """
    Composite ranking score = ai_weight * ai_score + prob_weight * selection_prob * 100
    """
    return round(ai_weight * ai_score + prob_weight * (selection_prob * 100), 2)


def rank_candidates(candidates: list) -> list:
    """
    Rank a list of candidate dicts by composite score.
    Each candidate dict must have:
        name: str
        ai_score: float (0-100)
        selection_prob: float (0-1)
    Optional:
        skill_match_percent, matched_skills, missing_skills, prediction
    Returns sorted list with rank and composite_score added.
    """
    if not candidates:
        return []

    for candidate in candidates:
        candidate["composite_score"] = compute_composite_score(
            candidate.get("ai_score", 0),
            candidate.get("selection_prob", 0),
        )

    ranked = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)
    for i, candidate in enumerate(ranked):
        candidate["rank"] = i + 1
        # Tier assignment
        score = candidate["composite_score"]
        if score >= 75:
            candidate["tier"] = "🥇 Strong Candidate"
        elif score >= 55:
            candidate["tier"] = "🥈 Good Candidate"
        elif score >= 35:
            candidate["tier"] = "🥉 Average Candidate"
        else:
            candidate["tier"] = "⚠️ Weak Candidate"

    return ranked
