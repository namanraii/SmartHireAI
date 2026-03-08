"""
preprocessor.py — NLP text cleaning and keyword extraction
"""
import re
import string


def clean_text(text: str) -> str:
    """
    Clean raw text: lowercase, remove URLs, emails, special characters,
    extra whitespace. Returns cleaned string.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove common English stopwords."""
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "i", "we", "you", "he", "she", "they", "it", "this", "that",
        "which", "who", "what", "when", "where", "how", "why",
        "not", "no", "as", "if", "so", "then", "than", "also",
        "up", "out", "about", "into", "through", "during"
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def lemmatize_spacy(text: str) -> str:
    """Lemmatize text using spaCy (if available), else return as-is."""
    try:
        import spacy
        nlp = _get_spacy_model()
        doc = nlp(text[:10000])
        return " ".join([token.lemma_ for token in doc if not token.is_punct])
    except Exception:
        return text


_spacy_model = None

def _get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        import spacy
        try:
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_model = spacy.load("en_core_web_sm")
    return _spacy_model


def preprocess(text: str, lemmatize: bool = False) -> str:
    """Full preprocessing pipeline: clean → tokenize → stopword removal → optional lemmatize."""
    cleaned = clean_text(text)
    if lemmatize:
        cleaned = lemmatize_spacy(cleaned)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def extract_noun_phrases(text: str) -> list:
    """Extract noun phrases using spaCy."""
    try:
        nlp = _get_spacy_model()
        doc = nlp(text[:10000])
        return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
    except Exception:
        return []
