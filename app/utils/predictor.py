import os
import joblib
import numpy as np

MODEL_PATH = os.path.join("app", "model", "bias_model.pkl")
VECT_PATH = os.path.join("app", "model", "vectorizer.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        raise FileNotFoundError("Model or vectorizer not found. Run notebooks/model_training.py first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

def predict_bias(text: str, model, vectorizer):
    """Predict bias label and confidence."""
    feat = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        idx = int(np.argmax(probs))
        label = model.classes_[idx]
        confidence = float(probs[idx])
    else:
        label = model.predict(feat)[0]
        confidence = 0.6
    return str(label), confidence
