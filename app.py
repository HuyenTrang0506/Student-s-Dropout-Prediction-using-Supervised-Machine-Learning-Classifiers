import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# === Load model, PCA and scaler (currently commented out for UI preview) ===
# model = joblib.load("model.pkl")
# pca = joblib.load("pca.pkl")
# scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Dropout Prediction App")
st.markdown("Enter student information below for dropout risk prediction.")

# === INPUT FORM ===
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
application_mode = st.selectbox("Application Mode", ["Online", "In-person", "Other"])
application_order = st.slider("Application Order (preference)", 1, 10, 1)
course = st.selectbox("Course", ["Computer Science", "Economics", "Engineering", "Other"])
daytime_evening = st.selectbox("Attendance Time", ["Daytime", "Evening"])
prev_qualification = st.selectbox("Previous Qualification", ["High School", "College", "University", "Other"])
mother_qualification = st.selectbox("Mother's Qualification", ["Unknown", "High School", "University", "Postgraduate"])
father_occupation = st.selectbox("Father's Occupation", ["Laborer", "Employee", "Business", "Other"])
displaced = st.radio("Displaced Student?", ["Yes", "No"])
edu_special_needs = st.radio("Educational Special Needs?", ["Yes", "No"])
debtor = st.radio("Is Debtor?", ["Yes", "No"])
tuition_paid = st.radio("Tuition Fees Up to Date?", ["Yes", "No"])
gender = st.radio("Gender", ["Male", "Female"])
scholarship = st.radio("Scholarship Holder?", ["Yes", "No"])
age = st.number_input("Age at Enrollment", min_value=16, max_value=60, value=18)
is_international = st.radio("International Student?", ["Yes", "No"])

# === INPUTS FOR PCA ===
credit_1 = st.slider("1st Semester Credits", 0, 60, 30)
grade_1 = st.slider("1st Semester Grade", 0.0, 20.0, 12.0)
credit_2 = st.slider("2nd Semester Credits", 0, 60, 30)
grade_2 = st.slider("2nd Semester Grade", 0.0, 20.0, 13.0)

# === Placeholder for prediction logic ===
# Below is where the full model logic would go, including:
# - Encoding categorical variables
# - Applying PCA to semester input
# - Creating final input vector
# - Calling model.predict()

if st.button("Predict"):
    st.info("🚧 Model is currently disabled. This is a UI preview only. Uncomment model loading and prediction code when ready.")
