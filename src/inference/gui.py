"""
Streamlit GUI for Malicious URL Detection
File: src/inference/gui.py
Uses the EXACT same inference pipeline as the FastAPI backend.
"""
import sys
import os
import streamlit as st

# Ensure project root is in path so `src.` imports work reliably in Docker
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.predict import predict_url

st.set_page_config(
    page_title="🛡️ Malicious URL Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🔍 Malicious URL Detector")
st.markdown("Enter a URL to check if it's **benign**, **phishing**, **malware**, or **defacement**. Uses the exact same `predict_url()` pipeline as the API.")

with st.form("url_form", clear_on_submit=True):
    url_input = st.text_input(
        "🔗 Enter URL",
        placeholder="https://example.com",
        help="Paste any URL to analyze"
    )
    submitted = st.form_submit_button("🔎 Analyze", type="primary")

if submitted and url_input:
    with st.spinner("🔎 Running inference pipeline..."):
        try:
            # 🔑 Direct call to your existing pipeline
            result = predict_url(url_input)
            
            pred = result["predicted_class"].upper()
            icon = "✅" if pred == "BENIGN" else "🚨"
            
            # Main prediction card
            if pred == "BENIGN":
                st.success(f"{icon} **Prediction: {pred}**")
            else:
                st.error(f"{icon} **Prediction: {pred}**")
                
            st.caption(f"Class ID: {result['class_id']}")
            
            # Probabilities
            if isinstance(result["probabilities"], dict):
                st.subheader("📊 Confidence Scores")
                for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                    st.metric(label=label.title(), value=f"{prob:.2%}")
                    st.progress(prob)
            else:
                st.info("ℹ️ Model does not support probability output")
                
            # Technical details (collapsible)
            with st.expander("🔧 Technical Details"):
                st.json({
                    "input_url": result["input_url"],
                    "predicted_class": result["predicted_class"],
                    "class_id": result["class_id"]
                }, expanded=False)
                
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}")
            st.info("💡 Ensure `models/` directory contains `best_model.pkl` and `label_encoder.pkl`")
        except Exception as e:
            st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
            st.exception(e)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Pipeline Info")
    st.markdown("""
    - **Model**: RandomForest (via `predict_url`)
    - **Features**: 30+ lexical & structural
    - **Execution**: Direct Python import (no HTTP overhead)
    - **Consistency**: 100% identical to API backend
    """)
    st.divider()
    st.caption("🔐 All processing happens locally inside this container.")