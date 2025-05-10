import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model, PCA and scaler
model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

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

# === ENCODING INPUTS ===
map_binary = lambda x: 1 if x in ["Yes", "Male", "Daytime"] else 0
encoded = [
    marital_status, application_mode, course, daytime_evening, prev_qualification,
    mother_qualification, father_occupation
]
categorical_encoded = pd.get_dummies(pd.DataFrame([encoded], columns=[
    "marital", "app_mode", "course", "daytime", "prev_q", "mom_q", "dad_job"])).reindex(columns=[
    "marital_Single", "marital_Married", "marital_Other",
    "app_mode_Online", "app_mode_In-person", "app_mode_Other",
    "course_Computer Science", "course_Economics", "course_Engineering", "course_Other",
    "daytime_Daytime", "daytime_Evening",
    "prev_q_High School", "prev_q_College", "prev_q_University", "prev_q_Other",
    "mom_q_Unknown", "mom_q_High School", "mom_q_University", "mom_q_Postgraduate",
    "dad_job_Laborer", "dad_job_Employee", "dad_job_Business", "dad_job_Other"
], fill_value=0)

binary_inputs = [
    map_binary(displaced),
    map_binary(edu_special_needs),
    map_binary(debtor),
    map_binary(tuition_paid),
    map_binary(gender),
    map_binary(scholarship),
    age,
    map_binary(is_international)
]

# PCA FEATURE
pca_input = pd.DataFrame([[credit_1, grade_1, credit_2, grade_2]], 
                         columns=["credit_1", "grade_1", "credit_2", "grade_2"])
pca_scaled = scaler.transform(pca_input)
pca_value = pca.transform(pca_scaled)[0][0]

# COMBINE FINAL INPUT
X_input = np.concatenate([categorical_encoded.values.flatten(), binary_inputs, [application_order, pca_value]])
X_input = X_input.reshape(1, -1)

# PREDICT
if st.button("Predict"):
    y_pred = model.predict(X_input)[0]
    label = "✅ Will Continue" if y_pred == 0 else "❌ At Risk of Dropout"
    st.subheader(f"Prediction Result: {label}")
