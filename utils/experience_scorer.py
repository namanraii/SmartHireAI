"""
experience_scorer.py — Weighted score engine for candidates
Weights: Skill Match 40%, Experience 30%, Education 20%, Projects 10%
"""
import re
from utils.ner_extractor import extract_experience_years, extract_education, extract_skills
from utils.skill_gap import skill_match_fraction


WEIGHTS = {
    "skill_match": 0.40,
    "experience": 0.30,
    "education": 0.20,
    "projects": 0.10,
}

MAX_EXPERIENCE_YEARS = 10.0  # Normalize to this ceiling


def _education_score(resume_text: str) -> float:
    """
    Score education: PhD=1.0, Master=0.85, Bachelor=0.70, Diploma=0.50, Other=0.30
    """
    text = resume_text.lower()
    if any(kw in text for kw in ["phd", "ph.d", "doctorate"]):
        return 1.0
    if any(kw in text for kw in ["master", "m.tech", "m.sc", "mba", "m.e", "me "]):
        return 0.85
    if any(kw in text for kw in ["bachelor", "b.tech", "b.sc", "bca", "b.e", "be "]):
        return 0.70
    if any(kw in text for kw in ["diploma", "associate"]):
        return 0.50
    return 0.30


def _project_score(resume_text: str) -> float:
    """
    Score projects: count project indicators (github, project titles, etc.)
    """
    text = resume_text.lower()
    indicators = [
        "github.com", "gitlab.com", "project", "built", "developed", "implemented",
        "deployed", "created", "designed", "portfolio",
    ]
    hits = sum(1 for kw in indicators if kw in text)
    # Normalize: 5+ indicators → 1.0
    return min(hits / 5.0, 1.0)


def _experience_score(resume_text: str) -> float:
    """
    Score experience: years of experience normalized to MAX_EXPERIENCE_YEARS.
    Also considers internship keywords if no explicit years found.
    """
    years = extract_experience_years(resume_text)
    if years > 0:
        return min(years / MAX_EXPERIENCE_YEARS, 1.0)
    # Check for internship / fresher mentions
    text = resume_text.lower()
    if "internship" in text or "intern" in text:
        return 0.15
    return 0.05  # Fresher with no mention


def compute_weighted_score(resume_text: str, jd_text: str) -> dict:
    """
    Compute the final weighted AI score (0–100) for a candidate.

    Returns a dict with individual component scores and the final score.
    """
    skill_match = skill_match_fraction(resume_text, jd_text)
    experience = _experience_score(resume_text)
    education = _education_score(resume_text)
    projects = _project_score(resume_text)

    final_score = (
        WEIGHTS["skill_match"] * skill_match +
        WEIGHTS["experience"] * experience +
        WEIGHTS["education"] * education +
        WEIGHTS["projects"] * projects
    )

    return {
        "skill_match_score": round(skill_match * 100, 1),
        "experience_score": round(experience * 100, 1),
        "education_score": round(education * 100, 1),
        "project_score": round(projects * 100, 1),
        "final_score": round(final_score * 100, 1),
        "components": {
            "skill_match": {"raw": round(skill_match, 4), "weight": WEIGHTS["skill_match"]},
            "experience": {"raw": round(experience, 4), "weight": WEIGHTS["experience"]},
            "education": {"raw": round(education, 4), "weight": WEIGHTS["education"]},
            "projects": {"raw": round(projects, 4), "weight": WEIGHTS["projects"]},
        },
    }
