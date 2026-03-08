"""
similarity_engine.py — TF-IDF + BERT semantic similarity between resume and JD
"""
import os
# Prevent Deep Learning Segment Faults / Deadlocks on macOS & Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from utils.preprocessor import preprocess


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using TF-IDF vectors."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        proc_a = preprocess(text_a)
        proc_b = preprocess(text_b)
        if not proc_a.strip() or not proc_b.strip():
            return 0.0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([proc_a, proc_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


_bert_model = None

def _get_bert_model():
    """Lazy-load the Sentence-BERT model."""
    global _bert_model
    if _bert_model is None:
        from sentence_transformers import SentenceTransformer
        _bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _bert_model


def bert_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using Sentence-BERT embeddings."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        model = _get_bert_model()
        # Truncate for speed
        emb = model.encode([text_a[:1024], text_b[:1024]], normalize_embeddings=True)
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        return 0.0


def combined_similarity(resume_text: str, jd_text: str,
                        tfidf_weight: float = 0.35,
                        bert_weight: float = 0.65) -> dict:
    """
    Compute a weighted combination of TF-IDF and BERT similarity.
    Returns a dict with individual scores and the combined score.
    """
    tfidf_score = tfidf_similarity(resume_text, jd_text)
    bert_score = bert_similarity(resume_text, jd_text)
    combined = tfidf_weight * tfidf_score + bert_weight * bert_score
    return {
        "tfidf_score": round(tfidf_score, 4),
        "bert_score": round(bert_score, 4),
        "combined_score": round(combined, 4),
        "combined_percent": round(combined * 100, 1),
    }
