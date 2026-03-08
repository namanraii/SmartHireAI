"""
bias_detector.py — Fairness and bias detection in hiring predictions
"""
import re


# Keywords that may indicate subjective/biased data in a resume
GENDERED_TERMS = [
    "he", "she", "him", "her", "his", "hers", "man", "woman", "male",
    "female", "mr.", "mrs.", "ms.", "father", "mother",
]

BIAS_RISK_SCHOOLS = []  # Can be populated with known biased institution lists
RELIGION_MARKERS = ["christian", "muslim", "hindu", "jewish", "sikh", "buddhist", "church"]
AGE_MARKERS = [r"\b(born|dob|age|19\d{2}|20[01]\d)\b"]
PHOTO_MARKERS = ["photo", "photograph", "picture", "image attached"]
ADDRESS_MARKERS = [r"\b\d{1,5}\s\w+\s(street|st|avenue|ave|road|rd|lane|ln|blvd|boulevard)\b"]


def _flag(text: str, patterns: list, is_regex: bool = False) -> list:
    """Return a list of matched items from text."""
    text_lower = text.lower()
    found = []
    for p in patterns:
        if is_regex:
            if re.search(p, text_lower):
                found.append(p)
        else:
            if re.search(r"\b" + re.escape(p) + r"\b", text_lower):
                found.append(p)
    return found


def detect_bias_flags(resume_text: str) -> dict:
    """
    Scan resume for elements that could introduce bias into AI predictions.
    Returns a dict of flags and an overall fairness confidence score.
    """
    flags = {}
    risk_count = 0

    # Gendered language
    gender_hits = _flag(resume_text, GENDERED_TERMS)
    if gender_hits:
        flags["gender_indicators"] = {
            "found": gender_hits,
            "risk": "LOW",
            "note": "Gender-related terms found; model may inadvertently pick up gender signals.",
        }
        risk_count += 1

    # Religion
    religion_hits = _flag(resume_text, RELIGION_MARKERS)
    if religion_hits:
        flags["religion_indicators"] = {
            "found": religion_hits,
            "risk": "MEDIUM",
            "note": "Religious terms found; could introduce religious bias.",
        }
        risk_count += 2

    # Age information
    age_hits = _flag(resume_text, AGE_MARKERS, is_regex=True)
    if age_hits:
        flags["age_indicators"] = {
            "found": age_hits,
            "risk": "LOW",
            "note": "Age/date of birth detected; may introduce age bias.",
        }
        risk_count += 1

    # Photo mentions
    photo_hits = _flag(resume_text, PHOTO_MARKERS)
    if photo_hits:
        flags["photo_indicators"] = {
            "found": photo_hits,
            "risk": "MEDIUM",
            "note": "Photo mentioned; visual bias risk if images are processed.",
        }
        risk_count += 2

    # Address (location bias)
    addr_hits = _flag(resume_text, ADDRESS_MARKERS, is_regex=True)
    if addr_hits:
        flags["address_indicators"] = {
            "found": addr_hits,
            "risk": "LOW",
            "note": "Home address detected; may introduce geographic bias.",
        }
        risk_count += 1

    # Resume length bias
    word_count = len(resume_text.split())
    if word_count < 100:
        flags["resume_length"] = {
            "found": [f"{word_count} words"],
            "risk": "MEDIUM",
            "note": "Very short resume may disadvantage the candidate unfairly.",
        }
        risk_count += 2
    elif word_count > 1500:
        flags["resume_length"] = {
            "found": [f"{word_count} words"],
            "risk": "LOW",
            "note": "Very long resume; consider if length penalizes good candidates.",
        }
        risk_count += 1

    # Compute fairness confidence (lower risk → higher confidence)
    max_risk = 10
    fairness_score = max(0, round((1 - risk_count / max_risk) * 100, 1))
    fairness_score = min(100.0, fairness_score)

    overall_risk = "LOW"
    if risk_count >= 5:
        overall_risk = "HIGH"
    elif risk_count >= 3:
        overall_risk = "MEDIUM"

    return {
        "flags": flags,
        "risk_count": risk_count,
        "overall_risk": overall_risk,
        "fairness_confidence_score": fairness_score,
        "flag_count": len(flags),
        "summary": (
            "No bias indicators detected." if not flags
            else f"{len(flags)} potential bias indicator(s) detected. Review flagged fields."
        ),
    }
