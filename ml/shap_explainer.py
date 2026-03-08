"""
shap_explainer.py — SHAP-based model explanation for Random Forest predictions
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_DISPLAY_NAMES = {
    "skill_match_score": "Skill Match %",
    "experience_years": "Years of Experience",
    "education_score": "Education Level",
    "project_score": "Projects Score",
    "bert_similarity": "BERT Semantic Similarity",
    "tfidf_similarity": "TF-IDF Keyword Match",
    "resume_length_score": "Resume Completeness",
    "skills_count": "Total Skills Listed",
}


def _load_rf():
    path = os.path.join(MODEL_DIR, "random_forest.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("random_forest.pkl not found. Run train_models.py first.")
    return joblib.load(path)


def compute_shap_values(feature_vector: np.ndarray) -> dict:
    """
    Compute SHAP values for a single feature vector using Random Forest.
    Returns dict with shap values, feature names, base value.
    """
    try:
        import shap
        rf = _load_rf()
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(feature_vector)

        # For binary: shap_values[1] = class=1 (Selected)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = np.array(shap_values)[0]
            if len(sv.shape) == 2: 
                # Shape (n_features, n_classes); slice to get class 1
                sv = sv[:, 1]

        feature_display = [
            FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names
        ]

        return {
            "shap_values": sv.tolist(),
            "feature_names": feature_display,
            "base_value": float(np.array(explainer.expected_value).flatten()[-1]),
            "feature_values": feature_vector[0].tolist(),
        }
    except Exception as e:
        return {"error": str(e), "shap_values": [], "feature_names": [], "base_value": 0.5}


def plot_shap_bar(shap_data: dict) -> bytes:
    """
    Generate a horizontal bar chart of SHAP values.
    Returns PNG bytes.
    """
    if not shap_data.get("shap_values"):
        return b""

    sv = np.array(shap_data["shap_values"]).flatten()
    names = shap_data["feature_names"]

    # Sort by absolute value
    indices = np.argsort(np.abs(sv))
    sv_sorted = sv[indices]
    names_sorted = [names[int(i)] for i in indices]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in sv_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1c1f2e")

    bars = ax.barh(names_sorted, sv_sorted, color=colors, edgecolor="none", height=0.6)

    # Add value labels
    for bar, val in zip(bars, sv_sorted):
        xpos = val + 0.002 if val >= 0 else val - 0.002
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha,
                color="white", fontsize=8)

    ax.set_xlabel("SHAP Value (impact on selection probability)", color="#aaaaaa", fontsize=9)
    ax.set_title("Feature Impact on Hiring Decision (SHAP)", color="white", fontsize=11, pad=12)
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.axvline(x=0, color="#555555", linewidth=0.8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_top_factors(shap_data: dict, top_n: int = 3) -> list:
    """Return top N most impactful features (by absolute SHAP value)."""
    if not shap_data.get("shap_values"):
        return []
    sv = np.array(shap_data["shap_values"]).flatten()
    names = shap_data["feature_names"]
    indices = np.argsort(np.abs(sv))[::-1][:top_n]
    return [
        {
            "feature": names[int(i)],
            "shap_value": round(float(sv[i]), 4),
            "direction": "positive" if sv[i] > 0 else "negative",
        }
        for i in indices
    ]
