"""
ner_extractor.py — Named Entity Recognition for resumes using spaCy + keyword lists
"""
import re


# ── Skill keyword database ───────────────────────────────────────────────────

TECH_SKILLS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "r",
    "go", "golang", "rust", "kotlin", "swift", "scala", "php", "ruby",
    "matlab", "perl", "bash", "shell", "powershell", "dart", "lua",
    # Web
    "html", "css", "react", "angular", "vue", "svelte", "nextjs", "nodejs",
    "express", "django", "flask", "fastapi", "spring", "laravel", "rails",
    "bootstrap", "tailwind", "jquery", "graphql", "rest", "restful",
    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "sklearn", "xgboost", "lightgbm", "catboost", "huggingface", "transformers",
    "bert", "gpt", "llm", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "tableau", "power bi", "spark", "pyspark", "hadoop", "hive",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "neo4j", "firebase",
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "google cloud", "ci/cd",
    "jenkins", "github actions", "terraform", "ansible", "linux", "git",
    "github", "gitlab", "bitbucket", "nginx", "apache",
    # Other
    "agile", "scrum", "jira", "microservices", "api", "oauth", "jwt",
    "blockchain", "iot", "arduino", "raspberry pi", "opencv", "unity",
]

EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "ph.d", "b.tech", "m.tech", "b.e", "m.e",
    "b.sc", "m.sc", "mba", "bba", "bca", "mca", "diploma", "associate",
    "computer science", "information technology", "electrical engineering",
    "mechanical engineering", "data science", "artificial intelligence",
    "machine learning", "software engineering", "electronics",
    "communication engineering", "civil engineering", "mathematics", "physics",
    "b.com", "m.com", "b.a", "m.a",
]

DEGREE_PATTERNS = [
    r"\b(b\.?tech|m\.?tech|be|me|b\.?sc|m\.?sc|phd|ph\.?d|mba|bba|bca|mca|b\.?e\.?|m\.?e\.?|bachelor|master|doctorate|diploma)\b",
    r"\b(computer science|information technology|data science|artificial intelligence|machine learning|software engineering)\b",
]

EXPERIENCE_PATTERNS = [
    r"(\d+\.?\d*)\s*\+?\s*years?\s*(of\s+)?(experience|exp\.?|work)",
    r"(experience|exp\.?)\s*:?\s*(\d+\.?\d*)\s*\+?\s*years?",
    r"(\d+\.?\d*)\s*\+?\s*yrs?\s*(of\s+)?(experience|exp\.?|work)?",
]


def _get_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


_spacy_nlp = None


def extract_skills(text: str) -> list:
    """Extract technical skills from text using keyword matching."""
    text_lower = text.lower()
    found = []
    for skill in TECH_SKILLS:
        # Use word boundary matching (handle multi-word skills)
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_education(text: str) -> list:
    """Extract education qualifications from text."""
    text_lower = text.lower()
    found = []
    for pattern in DEGREE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            if isinstance(m, tuple):
                found.extend([x.strip() for x in m if x.strip()])
            else:
                found.append(m.strip())

    # Also check keyword list
    for kw in EDUCATION_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            found.append(kw)

    return sorted(set([f.lower() for f in found if len(f) > 2]))


def extract_experience_years(text: str) -> float:
    """Extract years of experience from text using regex patterns."""
    text_lower = text.lower()
    years = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            for part in (m if isinstance(m, tuple) else [m]):
                try:
                    val = float(re.search(r"\d+\.?\d*", part).group())
                    years.append(val)
                except (ValueError, AttributeError):
                    pass
    return max(years) if years else 0.0


def extract_organizations(text: str) -> list:
    """Use spaCy NER to extract organization names."""
    try:
        nlp = _get_spacy_model()
        doc = nlp(text[:10000])
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ in ("ORG",)]
        return sorted(set(orgs))
    except Exception:
        return []


def extract_persons(text: str) -> list:
    """Use spaCy NER to extract person names."""
    try:
        nlp = _get_spacy_model()
        doc = nlp(text[:5000])
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        return sorted(set(persons))
    except Exception:
        return []


def extract_all_entities(text: str) -> dict:
    """
    Full NER extraction pass.
    Returns a dict with skills, education, experience_years, organizations.
    """
    return {
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "organizations": extract_organizations(text),
        "persons": extract_persons(text),
    }
