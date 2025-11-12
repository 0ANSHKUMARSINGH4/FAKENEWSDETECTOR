# notebooks/model_training.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from app.utils.preprocessor import clean_text

# -----------------------------
# Prepare dataset (global + Indian)
# -----------------------------
# Small sample for demonstration; extend for better accuracy
data = pd.DataFrame({
    'headline': [
        "Henry Zeffman: Efforts to shore up Starmer's leadership may have backfired",
        "Trump pleads not guilty to 34 felony counts",
        "12 Days Before Blast, Suspects Got i20's Pollution Checked, Then Parked It",
        "State-sanctioned fuel smuggling cost Libya $20bn over three years – report",
        "Alfonsine, Jersey King, Ice Of Fire and Sunlit Path excel",
        "Xi’s Military Purges Show Unease About China’s Nuclear Forces"
    ],
    'label': ["neutral","neutral","biased","neutral","neutral","biased"]
})

# Clean headlines
data['clean_headline'] = data['headline'].apply(clean_text)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_headline'], data['label'], test_size=0.2, random_state=42
)

# -----------------------------
# Vectorizer + Model
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluate
preds = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, preds)
print(f"✅ Quick training complete. Accuracy: {acc*100:.2f}%")

# -----------------------------
# Save model & vectorizer
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "bias_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
print(f"Model saved in {MODEL_DIR}")
