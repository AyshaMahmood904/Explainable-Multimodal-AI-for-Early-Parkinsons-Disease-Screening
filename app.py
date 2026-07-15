import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Explainable Multimodal AI System")
st.subheader("Early Parkinson's Disease Detection")

st.markdown("---")

st.write("""
This application predicts Parkinson's Disease using:

✔ Vision-based gait features
✔ Speech features
✔ Multimodal Fusion
✔ Explainable AI (SHAP)
""")

st.markdown("---")

# -------------------------
# Load Models
# -------------------------

vision_model = joblib.load("model/parkinson_rf.pkl")
speech_model = joblib.load("model/speech_xgboost_model.pkl")
speech_scaler = joblib.load("model/speech_scaler.pkl")

# -------------------------
# File Upload
# -------------------------

vision_file = st.file_uploader(
    "Upload Vision CSV",
    type=["csv"]
)

speech_file = st.file_uploader(
    "Upload Speech CSV",
    type=["csv"]
)

if st.button("Predict"):

    if vision_file is None or speech_file is None:
        st.warning("Please upload both CSV files.")

    else:

        # Read CSVs
        vision_df = pd.read_csv(vision_file)
        speech_df = pd.read_csv(speech_file)

        # =====================================================
        # Vision Processing
        # =====================================================
        # Convert column names to string
        vision_df.columns = vision_df.columns.astype(str)

        # Remove unwanted columns
        vision_df = vision_df.drop(
            columns=["id", "class", "label", "1632"],
            errors="ignore"
        )

        # Expected feature names
        expected = [str(col) for col in vision_model.feature_names_in_]

        # Check missing columns
        missing = [c for c in expected if c not in vision_df.columns]

        if len(missing) > 0:
            st.error("Missing columns found")
            st.write(missing[:20])
            st.stop()

        # Keep only required columns
        vision_df = vision_df[expected]

        # Prediction
        vision_prob = vision_model.predict_proba(vision_df)[0]

        vision_healthy = vision_prob[0]
        vision_parkinson = vision_prob[1:].sum()

        # =====================================================
        # Speech Processing
        # =====================================================

        speech_df = speech_df.drop(
            columns=["id", "class", "label"],
            errors="ignore"
        )

        speech_scaled = speech_scaler.transform(speech_df)

        # Save for SHAP
        st.session_state["speech_scaled"] = speech_scaled

        speech_prob = speech_model.predict_proba(speech_scaled)[0]

        speech_healthy = speech_prob[0]
        speech_parkinson = speech_prob[1]

        # =====================================================
        # Fusion
        # =====================================================

        healthy = 0.4 * vision_healthy + 0.6 * speech_healthy
        parkinson = 0.4 * vision_parkinson + 0.6 * speech_parkinson

        st.success("Prediction Completed!")

        st.subheader("Results")

        st.write(f"Vision Parkinson Probability: **{vision_parkinson:.2%}**")
        st.write(f"Speech Parkinson Probability: **{speech_parkinson:.2%}**")

        st.write(f"Healthy Score: **{healthy:.2%}**")
        st.write(f"Parkinson Score: **{parkinson:.2%}**")

        if parkinson > healthy:
            st.error(f"🧠 Parkinson Detected\nConfidence: {parkinson:.2%}")
        else:
            st.success(f"✅ Healthy\nConfidence: {healthy:.2%}")

# =====================================================
# SHAP Explainability
# =====================================================

st.markdown("---")

if "speech_scaled" in st.session_state:

    if st.button("🔍 Explain Prediction (SHAP)"):

        st.subheader("AI Explanation for Current Prediction")

        explainer = shap.TreeExplainer(speech_model)

        explanation = explainer(st.session_state["speech_scaled"])

        plt.clf()

        shap.plots.bar(
            explanation,
            max_display=10,
            show=False
        )

        st.pyplot(plt.gcf())

        plt.clf()