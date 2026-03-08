"""
skill_gap.py — Skill gap analysis between resume and job description
"""
import re
from utils.ner_extractor import extract_skills, TECH_SKILLS


def normalize_skill(skill: str) -> str:
    """Normalize skill string for comparison."""
    return skill.lower().strip()


def extract_jd_skills(jd_text: str) -> list:
    """Extract required skills from a job description."""
    return extract_skills(jd_text)


def compute_skill_gap(resume_text: str, jd_text: str) -> dict:
    """
    Compare resume skills vs JD skills.
    Returns:
        matched_skills: skills in both resume and JD
        missing_skills: skills in JD but not in resume
        extra_skills:   skills in resume but not in JD
        match_score:    float 0-1 representing match fraction
    """
    resume_skills = set(normalize_skill(s) for s in extract_skills(resume_text))
    jd_skills = set(normalize_skill(s) for s in extract_jd_skills(jd_text))

    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    match_score = len(matched) / len(jd_skills) if jd_skills else 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_score": round(match_score, 4),
        "match_percent": round(match_score * 100, 1),
    }


def skill_match_fraction(resume_text: str, jd_text: str) -> float:
    """Quick helper: returns skill match fraction [0, 1]."""
    gap = compute_skill_gap(resume_text, jd_text)
    return gap["match_score"]
