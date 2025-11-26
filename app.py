import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load(r"C:\Users\admin\Downloads\archive (3)\diabetes_random_forest_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)
hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=350, value=180)
systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
physical_activity = st.selectbox("Physical Activity Level",
                                 ["Low", "Moderate", "High"])

# Create raw input data (used by fallback one-hot encoding)
raw_input = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'glucose': [glucose],
    'hba1c': [hba1c],
    'cholesterol': [cholesterol],
    'systolic_bp': [systolic_bp],
    'diastolic_bp': [diastolic_bp],
    'gender': [gender],
    'smoking_status': [smoking],
    'physical_activity_level': [physical_activity]
})
# Attempt to align input columns with the features the model was trained on
selected_features_path = r"C:\Users\admin\Downloads\archive (3)\selected_features.txt"
try:
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
except Exception:
    selected_features = None

# Build an aligned input vector matching `selected_features` (fill missing with 0)
if selected_features:
    # Initialize all features to 0
    input_dict = {feat: 0 for feat in selected_features}

    # Map available user inputs into expected feature names where possible
    if 'age' in input_dict:
        input_dict['age'] = age
    if 'hba1c' in input_dict:
        input_dict['hba1c'] = hba1c
    # If training expected fasting/postprandial glucose, set both from single glucose input
    if 'glucose_fasting' in input_dict:
        input_dict['glucose_fasting'] = glucose
    if 'glucose_postprandial' in input_dict:
        input_dict['glucose_postprandial'] = glucose
    # Map a simple minutes-per-week estimate from the categorical activity level
    if 'physical_activity_minutes_per_week' in input_dict:
        activity_minutes = {'Low': 30, 'Moderate': 150, 'High': 300}
        input_dict['physical_activity_minutes_per_week'] = activity_minutes.get(physical_activity, 0)
    # If the model expects a precomputed risk score or family history, leave as 0 (unknown)

    # Create DataFrame in the correct column order
    input_aligned = pd.DataFrame([input_dict], columns=selected_features)
else:
    # Fallback: use simple one-hot encoding of the raw input (may still mismatch)
    input_aligned = pd.get_dummies(raw_input, drop_first=True)

# Prediction button
if st.button("Predict"):
    try:
        prediction = model.predict(input_aligned)
        result = "✔ Diabetes Detected" if prediction[0] == 1 else "❌ No Diabetes"
        st.subheader("Prediction Result:")
        st.info(result)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
