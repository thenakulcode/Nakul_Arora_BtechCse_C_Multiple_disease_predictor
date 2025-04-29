# -- coding: utf-8 --
"""
Created on Sat Apr 26 21:29:50 2025
@author: Nakul
"""

import os
import pickle
import numpy as np
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("🔬 Multiple Disease Prediction System")

# ─── PATHS ────────────────────────────────────────────────
BASE = r"C:\Users\nakul\OneDrive\Desktop\Multiple disease prediction system\savedmodels5"

DIABETES_MODEL_PATH = os.path.join(BASE, "diabetes_model (2).sav")
DIABETES_SCALER_PATH = os.path.join(BASE, "diabetes_scaler.sav")
HEART_MODEL_PATH = os.path.join(BASE, "heart_model.sav")
HEART_SCALER_PATH = os.path.join(BASE, "heart_scaler.sav")
PARKINSON_MODEL_PATH = os.path.join(BASE, "parkinson_model.sav")
PARKINSON_SCALER_PATH = os.path.join(BASE, "parkinson_scaler.sav")

# ─── CACHED MODEL LOADING ─────────────────────────────────
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

diabetes_model = load_model(DIABETES_MODEL_PATH)
diabetes_scaler = load_model(DIABETES_SCALER_PATH)
heart_model = load_model(HEART_MODEL_PATH)
heart_scaler = load_model(HEART_SCALER_PATH)
parkinson_model = load_model(PARKINSON_MODEL_PATH)
parkinson_scaler = load_model(PARKINSON_SCALER_PATH)

# ─── SIDEBAR NAVIGATION ──────────────────────────────────
options = ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"]
selected = st.sidebar.selectbox("Select Disease to Predict", options)

# ─── DIABETES ─────────────────────────────────────────────
if selected == "Diabetes Prediction":
    st.header("🩺 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

    if st.button("Predict Diabetes"):
        X = np.array([[pregnancies, glucose, blood_pressure,
                       skin_thickness, insulin, bmi, pedigree, age]])
        X_scaled = diabetes_scaler.transform(X)
        result = diabetes_model.predict(X_scaled)[0]
        st.success("🟢 Not Diabetic" if result == 0 else "🔴 Diabetic")

# ─── HEART ────────────────────────────────────────────────
elif selected == "Heart Disease Prediction":
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age", min_value=1)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Serum Cholesterol", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible)", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]])
        X_scaled = heart_scaler.transform(X)
        result = heart_model.predict(X_scaled)[0]
        st.success("🟢 No Heart Disease" if result == 0 else "🔴 Has Heart Disease")

# ─── PARKINSON ────────────────────────────────────────────
elif selected == "Parkinson's Prediction":
    st.header("🧠 Parkinson's Prediction")

    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
        "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    input_data = []
    for feature in features:
        val = st.number_input(feature, key=feature)
        input_data.append(val)

    if st.button("Predict Parkinson's"):
        X = np.array([input_data])
        X_scaled = parkinson_scaler.transform(X)
        result = parkinson_model.predict(X_scaled)[0]
        st.success("🟢 No Parkinson's" if result == 0 else "🔴 Parkinson's Detected")
