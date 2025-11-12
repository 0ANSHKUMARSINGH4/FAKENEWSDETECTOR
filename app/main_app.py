import streamlit as st
from utils.predictor import predict_headline, model, vectorizer

st.set_page_config(page_title="Fake/News Bias Detector 🌍", page_icon="📰", layout="centered")

st.title("🌍 Fake/News Bias Detector")
st.write("Enter a headline below to check if it is biased or neutral:")

# Check if model is loaded
if model is None or vectorizer is None:
    st.error("⚠️ Model not found! Please run the training script first: `python -m notebooks.model_training`")
else:
    # User input
    input_text = st.text_area("Enter a headline for analysis:", placeholder="Paste headline here...")
    
    if st.button("Check Headline") and input_text.strip():
        pred, conf = predict_headline(input_text)
        if pred is not None:
            st.success(f"Prediction: **{pred}** — Confidence: {conf:.1f}%")
        else:
            st.error("Prediction failed. Please try again.")

# Sample global & Indian headlines
st.markdown("---")
st.subheader("Sample global & Indian headlines")
st.markdown("""
- Henry Zeffman: Efforts to shore up Starmer's leadership may have backfired (BBC)  
- Trump pleads not guilty to 34 felony counts (CNN)  
- State-sanctioned fuel smuggling cost Libya $20bn over three years – report (The Guardian)  
- Xi’s Military Purges Show Unease About China’s Nuclear Forces (NYTimes)  
- 12 Days Before Blast, Suspects Got i20's Pollution Checked, Then Parked It (NDTV)  
- Alfonsine, Jersey King, Ice Of Fire and Sunlit Path excel (The Hindu)  
""")
