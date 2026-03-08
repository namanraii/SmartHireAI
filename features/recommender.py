"""
recommender.py — Skill recommendation engine based on missing skills
"""

# Skill → learning resource mapping
SKILL_RESOURCES = {
    "python": "https://docs.python.org/3/tutorial/",
    "machine learning": "https://www.coursera.org/learn/machine-learning",
    "deep learning": "https://www.deeplearning.ai/",
    "nlp": "https://www.nltk.org/book/",
    "natural language processing": "https://huggingface.co/learn/nlp-course",
    "tensorflow": "https://www.tensorflow.org/tutorials",
    "pytorch": "https://pytorch.org/tutorials/",
    "scikit-learn": "https://scikit-learn.org/stable/getting_started.html",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    "pandas": "https://pandas.pydata.org/getting_started.html",
    "numpy": "https://numpy.org/learn/",
    "sql": "https://www.w3schools.com/sql/",
    "mysql": "https://dev.mysql.com/doc/",
    "postgresql": "https://www.postgresql.org/docs/",
    "mongodb": "https://www.mongodb.com/docs/",
    "docker": "https://docs.docker.com/get-started/",
    "kubernetes": "https://kubernetes.io/docs/tutorials/",
    "aws": "https://aws.amazon.com/training/",
    "azure": "https://learn.microsoft.com/en-us/azure/",
    "gcp": "https://cloud.google.com/learn/training",
    "react": "https://react.dev/learn",
    "angular": "https://angular.io/tutorial",
    "vue": "https://vuejs.org/tutorial/",
    "nodejs": "https://nodejs.org/en/learn",
    "javascript": "https://javascript.info/",
    "typescript": "https://www.typescriptlang.org/docs/",
    "java": "https://docs.oracle.com/javase/tutorial/",
    "c++": "https://www.learncpp.com/",
    "git": "https://git-scm.com/book/en/v2",
    "linux": "https://linuxjourney.com/",
    "data science": "https://www.kaggle.com/learn",
    "tableau": "https://help.tableau.com/current/guides/get-started-tutorial/en-us/",
    "power bi": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi",
    "spark": "https://spark.apache.org/docs/latest/quick-start.html",
    "opencv": "https://docs.opencv.org/4.x/d9/df8/tutorial_root.html",
    "flutter": "https://flutter.dev/learn",
    "kotlin": "https://kotlinlang.org/docs/getting-started.html",
    "swift": "https://developer.apple.com/swift/resources/",
    "rust": "https://doc.rust-lang.org/book/",
    "go": "https://go.dev/learn/",
    "graphql": "https://graphql.org/learn/",
    "rest": "https://restfulapi.net/",
    "microservices": "https://microservices.io/patterns/index.html",
    "agile": "https://www.atlassian.com/agile",
    "scrum": "https://www.scrum.org/resources/what-scrum-module",
}

PRIORITY_MAP = {
    "machine learning": "HIGH",
    "deep learning": "HIGH",
    "nlp": "HIGH",
    "natural language processing": "HIGH",
    "python": "HIGH",
    "sql": "HIGH",
    "docker": "MEDIUM",
    "kubernetes": "MEDIUM",
    "aws": "MEDIUM",
    "azure": "MEDIUM",
    "tensorflow": "MEDIUM",
    "pytorch": "MEDIUM",
}


def get_recommendations(missing_skills: list, max_recs: int = 10) -> list:
    """
    Generate skill recommendations for a list of missing skills.
    Each recommendation includes: skill, priority, resource URL, description.
    """
    if not missing_skills:
        return []

    recommendations = []
    for skill in missing_skills[:max_recs]:
        skill_lower = skill.lower().strip()
        resource = SKILL_RESOURCES.get(skill_lower, f"https://www.google.com/search?q=learn+{skill_lower.replace(' ', '+')}")
        priority = PRIORITY_MAP.get(skill_lower, "LOW")
        recommendations.append({
            "skill": skill,
            "priority": priority,
            "resource_url": resource,
            "description": f"Learn {skill} to strengthen your profile for this role.",
        })

    # Sort by priority: HIGH > MEDIUM > LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return recommendations


def generate_learning_path(missing_skills: list) -> dict:
    """Generate a structured 3-phase learning roadmap."""
    recs = get_recommendations(missing_skills, max_recs=15)
    high = [r for r in recs if r["priority"] == "HIGH"]
    medium = [r for r in recs if r["priority"] == "MEDIUM"]
    low = [r for r in recs if r["priority"] == "LOW"]

    return {
        "phase_1_immediate": high[:3],
        "phase_2_short_term": medium[:3],
        "phase_3_long_term": low[:3],
        "total_skills_recommended": len(recs),
    }
