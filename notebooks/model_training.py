import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.utils.preprocessor import clean_text

# Create model directory if not exists
MODEL_DIR = os.path.join("app", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Use a small demo dataset for fast training
data = {
    "text": [
        "Government passes new healthcare bill",
        "Aliens landed in New York City",
        "Stock markets hit all-time high",
        "Celebrity caught in scandal",
        "New technology promises clean energy",
        "Politician involved in bribery",
        "Scientists discover cure for disease",
        "Fake news about vaccines spreading",
        "Sports team wins championship",
        "Conspiracy theory about moon landing"
    ],
    "label": [
        "neutral", "biased", "neutral", "biased", "neutral",
        "biased", "neutral", "biased", "neutral", "biased"
    ]
}

df = pd.DataFrame(data)
df["text"] = df["text"].apply(clean_text)

# Vectorizer
vectorizer = TfidfVectorizer(max_df=0.85, ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "bias_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("✅ Quick training complete. Model saved in app/model/")
