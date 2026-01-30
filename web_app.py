import streamlit as st
import pandas as pd
import joblib

from src.preprocessing import preprocess
from src.explainability import explain_patient

# -------------------------
# Load Models
# -------------------------
base_model = joblib.load("models/logistic_base.pkl")
calibrated_model = joblib.load("models/logistic_calibrated.pkl")
feature_schema = joblib.load("models/feature_schema.pkl")
# -------------------------
# Helpers
# -------------------------
def risk_band(p):
    if p < 0.10:
        return "Low"
    elif p < 0.25:
        return "Moderate"
    else:
        return "High"

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Early Diabetes Risk Screening")
st.title("Early Diabetes Risk Screening System")

st.subheader("Patient Information")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
hba1c = st.number_input("HbA1c Level", 3.0, 12.0, 5.5)
glucose = st.number_input("Blood Glucose Level", 50, 400, 100)

smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "No Info"]
)

hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Risk"):

    patient = {
        "gender": gender,
        "age": age,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }

    df = pd.DataFrame([patient])
    df_p = preprocess(df)

    # Align schema
    for col in feature_schema:
        if col not in df_p.columns:
            df_p[col] = 0
    df_p = df_p[feature_schema]

    prob = calibrated_model.predict_proba(df_p)[0][1]
    band = risk_band(prob)

    explanation = explain_patient(
        base_model,
        df_p,    # use patient itself as background
        df_p
    )


    st.markdown("---")
    st.subheader("Result")

    st.metric("Risk Band", band)
    st.metric("Estimated Risk", f"{prob:.1%}")

    st.subheader("Key Risk Drivers")
    for f in explanation["feature"]:
        st.write("•", f)

    st.info(
        "This tool provides early risk estimation only and does not diagnose disease."
    )
