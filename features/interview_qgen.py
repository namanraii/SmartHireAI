"""
interview_qgen.py — Generate relevant interview questions based on resume content
"""
import random


# Question templates per skill / topic category
QUESTION_BANK = {
    "python": [
        "Explain the difference between a list and a tuple in Python.",
        "How does Python's GIL affect multithreaded programs?",
        "What are decorators in Python? Give an example use case.",
        "How would you manage memory efficiently in a large Python data pipeline?",
        "Describe the difference between @staticmethod and @classmethod.",
    ],
    "machine learning": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain overfitting and how you would address it.",
        "What cross-validation technique would you use for an imbalanced dataset?",
        "Compare Random Forest and XGBoost. When would you choose one over the other?",
        "Walk me through how you would approach a binary classification problem end-to-end.",
    ],
    "deep learning": [
        "What is the vanishing gradient problem and how do LSTMs solve it?",
        "Explain the architecture of a Convolutional Neural Network.",
        "What are attention mechanisms and how do they improve Transformers?",
        "How would you handle overfitting in a deep learning model?",
        "What is the purpose of batch normalization?",
    ],
    "nlp": [
        "How does TF-IDF differ from word embeddings like Word2Vec?",
        "Explain the architecture of BERT and what makes it bidirectional.",
        "How would you approach a named entity recognition task from scratch?",
        "What techniques would you use to handle class imbalance in a text classification task?",
        "Describe the process of fine-tuning a pre-trained language model.",
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "How would you optimize a slow SQL query?",
        "Explain window functions with an example.",
        "What is database normalization and why is it important?",
        "Describe the difference between HAVING and WHERE clauses.",
    ],
    "docker": [
        "What is the difference between a Docker image and a container?",
        "How do you use Docker Compose to orchestrate multi-container apps?",
        "Explain how Docker volumes work and why they are important.",
        "What is a multi-stage Docker build and when would you use it?",
        "How does Docker networking work between containers?",
    ],
    "kubernetes": [
        "What is the role of a Pod in Kubernetes?",
        "Explain the difference between a Deployment and a StatefulSet.",
        "How does Kubernetes handle service discovery?",
        "What is a Kubernetes Ingress controller?",
        "Describe how horizontal pod autoscaling works.",
    ],
    "react": [
        "Explain the concept of React hooks and give examples of useState and useEffect.",
        "What is the Virtual DOM and how does React use it for efficiency?",
        "How do you manage global state in a large React application?",
        "What is the difference between controlled and uncontrolled components?",
        "How would you optimize a React application for performance?",
    ],
    "aws": [
        "What is the difference between EC2, ECS, and Lambda?",
        "How do you design a highly available architecture on AWS?",
        "Explain the purpose of IAM roles and policies.",
        "What is S3 and what storage classes does it offer?",
        "How would you set up a CI/CD pipeline using AWS services?",
    ],
    "data science": [
        "Walk me through your data cleaning process for a messy real-world dataset.",
        "How do you decide which features to include in a model?",
        "What is the difference between correlation and causation?",
        "Describe a past project where your analysis led to a business decision.",
        "How would you communicate complex analytical results to a non-technical stakeholder?",
    ],
    "agile": [
        "Describe the Scrum ceremonies and the purpose of each.",
        "How do you handle scope creep in an Agile project?",
        "What metrics would you track in an Agile sprint?",
        "How do you prioritize a product backlog?",
        "Describe a situation where your team had to adapt to a major change mid-sprint.",
    ],
    "general": [
        "Tell me about yourself and your most significant technical achievement.",
        "Describe a challenging project you worked on and how you overcame obstacles.",
        "How do you keep up with rapidly evolving technologies in your field?",
        "Where do you see yourself in 3–5 years technically?",
        "Describe a time you disagreed with a technical decision and how you handled it.",
        "How do you approach debugging a complex, hard-to-reproduce bug?",
        "Tell me about a time you had to learn a new technology quickly under deadline pressure.",
    ],
    "project": [
        "Walk me through the most complex project listed on your resume.",
        "What technical decisions did you make in this project and what were the trade-offs?",
        "How did you measure the success of this project?",
        "What would you do differently if you were to rebuild this project from scratch?",
        "Did you work in a team on this project? How did you collaborate?",
    ],
}


def generate_questions(resume_skills: list, resume_text: str = "", num_questions: int = 10) -> list:
    """
    Generate interview questions based on skills found in the resume.
    Also adds general and project-specific questions.
    """
    questions = []
    used_topics = set()

    # Skill-based questions
    for skill in resume_skills:
        skill_lower = skill.lower()
        for category, qs in QUESTION_BANK.items():
            if category in skill_lower or skill_lower in category:
                if category not in used_topics:
                    selected = random.sample(qs, min(2, len(qs)))
                    questions.extend([{"question": q, "category": category.title(), "type": "Technical"} for q in selected])
                    used_topics.add(category)
                break

    # Always add general questions
    general_qs = random.sample(QUESTION_BANK["general"], min(3, len(QUESTION_BANK["general"])))
    questions.extend([{"question": q, "category": "Behavioral", "type": "Behavioral"} for q in general_qs])

    # Add project questions if projects are mentioned
    if any(kw in resume_text.lower() for kw in ["project", "github", "built", "developed"]):
        proj_qs = random.sample(QUESTION_BANK["project"], min(2, len(QUESTION_BANK["project"])))
        questions.extend([{"question": q, "category": "Project", "type": "Project-based"} for q in proj_qs])

    # Deduplicate and limit
    seen = set()
    unique = []
    for q in questions:
        if q["question"] not in seen:
            seen.add(q["question"])
            unique.append(q)

    random.shuffle(unique)
    return unique[:num_questions]
