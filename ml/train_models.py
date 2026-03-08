"""
train_models.py — Train Logistic Regression, SVM, Random Forest, XGBoost
and save models to ml/saved_models/

Run once before launching the app:
    python ml/train_models.py
"""
import os
import sys
import numpy as np
import joblib

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "skill_match_score",      # % of JD skills matched (0-100)
    "experience_years",       # years of experience (0-20)
    "education_score",        # encoded education level (0-4)
    "project_score",          # project count indicator (0-5)
    "bert_similarity",        # BERT cosine similarity (0-1)
    "tfidf_similarity",       # TF-IDF cosine similarity (0-1)
    "resume_length_score",    # resume word count normalized (0-1)
    "skills_count",           # total skills extracted
]

N_SAMPLES = 2000
np.random.seed(42)


def generate_synthetic_data(n: int = N_SAMPLES):
    """
    Generate synthetic training data with realistic feature distributions.
    Label is 1 (Selected) if a weighted combination exceeds a threshold.
    """
    skill_match = np.random.beta(2, 2, n)                  # 0-1
    experience_years = np.random.exponential(scale=3, size=n).clip(0, 20) / 20
    education = np.random.choice([0.3, 0.5, 0.7, 0.85, 1.0], n,
                                  p=[0.05, 0.10, 0.50, 0.25, 0.10])
    project = np.random.beta(1.5, 2, n)
    bert_sim = np.random.beta(2, 2, n)
    tfidf_sim = bert_sim * 0.8 + np.random.normal(0, 0.05, n)
    tfidf_sim = np.clip(tfidf_sim, 0, 1)
    resume_len = np.random.beta(2, 2, n)
    skills_count = np.random.randint(1, 30, n) / 30

    X = np.column_stack([
        skill_match * 100,
        experience_years * 20,
        education * 100,
        project * 100,
        bert_sim,
        tfidf_sim,
        resume_len,
        skills_count * 30,
    ])

    # Deterministic label: weighted formula + noise
    score = (
        0.35 * skill_match +
        0.25 * experience_years +
        0.20 * education +
        0.10 * project +
        0.10 * bert_sim +
        np.random.normal(0, 0.05, n)
    )
    y = (score >= 0.45).astype(int)
    return X, y


def train_and_save():
    print("=" * 60)
    print("SmartHire AI — Model Training")
    print("=" * 60)

    X, y = generate_synthetic_data(N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(probability=True, kernel="rbf", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name in ["logistic_regression", "svm"]:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        save_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"  Saved → {save_path}")
        trained[name] = {"accuracy": round(acc, 4), "precision": round(prec, 4),
                         "recall": round(rec, 4), "f1": round(f1, 4)}

    # Ensemble Voting Classifier
    print("\nTraining Ensemble Voting Classifier...")
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=10,
                                          random_state=42, n_jobs=-1)),
            ("xgb", xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=6, use_label_encoder=False,
                                       eval_metric="logloss", random_state=42)),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    y_ens = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test, y_ens)
    print(f"  Ensemble Accuracy: {ens_acc:.4f}")
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble.pkl"))
    print(f"  Saved → {os.path.join(MODEL_DIR, 'ensemble.pkl')}")

    # Save scaler and feature names
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(trained, os.path.join(MODEL_DIR, "model_metrics.pkl"))
    print("\n✅ All models saved successfully!")
    print("=" * 60)
    return trained


if __name__ == "__main__":
    train_and_save()
