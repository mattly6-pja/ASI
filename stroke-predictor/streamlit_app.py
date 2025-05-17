import streamlit as st
import pandas as pd
import joblib

# Wczytaj model i preprocessor
model = joblib.load("data/06_models/stroke_model.pkl")
preprocessor = joblib.load("data/06_models/preprocessor.pkl")

st.title("Stroke Prediction App")

# Dane wejściowe
age = st.slider("Age", 0, 100, 30)
avg_glucose_level = st.number_input("Avg Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)
gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

# Tworzymy DataFrame jak w treningu
input_data = pd.DataFrame({
    "id": [0],  # placeholder, zostanie usunięty w preprocesingu
    "age": [age],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender": [gender],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "residence_type": [residence_type],
    "smoking_status": [smoking_status],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "stroke": [0],  # placeholder
})

transformed = preprocessor.transform(input_data)

if st.button("Predict Stroke Risk"):
    prediction = model.predict(transformed)[0]
    st.write("Prediction:", "Stroke" if prediction == 1 else "No Stroke")
