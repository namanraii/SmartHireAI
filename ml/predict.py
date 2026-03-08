"""
predict.py — Load saved models and run ensemble prediction on feature vector
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

LABEL_MAP = {0: "Not Selected", 1: "Selected"}

_model_cache = {}


def _load(name: str):
    if name not in _model_cache:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Please run: python ml/train_models.py"
            )
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]


def get_feature_vector(scored_data: dict) -> np.ndarray:
    """
    Convert scored_data dict into a 1D numpy feature array.
    Keys expected (all floats):
        skill_match_score, experience_years, education_score, project_score,
        bert_similarity, tfidf_similarity, resume_length_score, skills_count
    """
    features = [
        scored_data.get("skill_match_score", 0.0),
        scored_data.get("experience_years", 0.0),
        scored_data.get("education_score", 50.0),
        scored_data.get("project_score", 0.0),
        scored_data.get("bert_similarity", 0.0),
        scored_data.get("tfidf_similarity", 0.0),
        scored_data.get("resume_length_score", 0.0),
        scored_data.get("skills_count", 0.0),
    ]
    return np.array(features, dtype=np.float64).reshape(1, -1)


def predict_all_models(feature_vector: np.ndarray) -> dict:
    """
    Run prediction using all individual models + ensemble.
    Returns dict with per-model label, probability, and ensemble result.
    """
    scaler = _load("scaler")
    fv_scaled = scaler.transform(feature_vector)

    results = {}

    model_names = ["logistic_regression", "svm", "random_forest", "xgboost"]
    for name in model_names:
        model = _load(name)
        if name in ["logistic_regression", "svm"]:
            fv_input = fv_scaled
        else:
            fv_input = feature_vector

        pred = int(model.predict(fv_input)[0])
        prob = model.predict_proba(fv_input)[0]
        results[name] = {
            "prediction": LABEL_MAP[pred],
            "selected": bool(pred),
            "probability_selected": round(float(prob[1]), 4),
            "probability_not_selected": round(float(prob[0]), 4),
        }

    # Ensemble
    ensemble = _load("ensemble")
    ens_pred = int(ensemble.predict(feature_vector)[0])
    ens_prob = ensemble.predict_proba(feature_vector)[0]
    results["ensemble"] = {
        "prediction": LABEL_MAP[ens_pred],
        "selected": bool(ens_pred),
        "probability_selected": round(float(ens_prob[1]), 4),
        "probability_not_selected": round(float(ens_prob[0]), 4),
    }

    return results


def predict_single_model(feature_vector: np.ndarray, model_name: str) -> dict:
    """Run prediction using a single named model."""
    scaler = _load("scaler")
    fv_scaled = scaler.transform(feature_vector)

    model = _load(model_name)
    if model_name in ["logistic_regression", "svm"]:
        fv_input = fv_scaled
    else:
        fv_input = feature_vector

    pred = int(model.predict(fv_input)[0])
    prob = model.predict_proba(fv_input)[0]
    return {
        "prediction": LABEL_MAP[pred],
        "selected": bool(pred),
        "probability_selected": round(float(prob[1]), 4),
        "probability_not_selected": round(float(prob[0]), 4),
    }


def get_model_metrics() -> dict:
    """Load saved model accuracy/metrics from training."""
    try:
        return _load("model_metrics")
    except Exception:
        return {}
