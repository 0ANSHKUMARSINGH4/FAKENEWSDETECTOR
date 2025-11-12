import os
import joblib

# Get absolute path to the model folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "bias_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# Load model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
except FileNotFoundError:
    model = None
    vectorizer = None

def predict_headline(text):
    """
    Predict if a headline is biased/fake or neutral.
    Returns (prediction_label, confidence_percentage)
    """
    if model is None or vectorizer is None:
        return None, None

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    conf = max(model.predict_proba(X)[0]) * 100
    return pred, conf
