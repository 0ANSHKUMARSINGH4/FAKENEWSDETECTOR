import re

DEFAULT_STOPWORDS = set(["the","and","is","in","to","of","for","a","on","with","as","by","at","from"])

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = DEFAULT_STOPWORDS

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower().strip()
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)
