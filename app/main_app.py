import streamlit as st
from utils.preprocessor import clean_text
from utils.predictor import load_model, predict_bias
from utils.data_fetcher import fetch_latest_news

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("🌍 Fake/News Bias Detector")

# Load model
try:
    model, vectorizer = load_model()
except FileNotFoundError:
    model, vectorizer = None, None

st.subheader("Enter a headline for analysis:")
headline = st.text_input("Paste headline here...")
if st.button("Analyze"):
    if not headline.strip():
        st.warning("Please enter a headline!")
    else:
        text = clean_text(headline)
        if model and vectorizer:
            label, score = predict_bias(text, model, vectorizer)
            pct = score*100
            st.success(f"Prediction: {label} — Confidence: {pct:.1f}%")
        else:
            st.error("Model missing! Run notebooks/model_training.py first.")

st.subheader("Sample global & Indian headlines")
try:
    items = fetch_latest_news(limit=6)
except Exception:
    items = []

if items:
    for title, source in items:
        st.markdown(f"- **{title}** ({source})")
else:
    st.info("Could not fetch live headlines.")
