import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("data/06_models/stroke_model_raw.pkl")
preprocessor = joblib.load("data/06_models/preprocessor.pkl")

st.title("Sprawdź jakie jest Twoje ryzyko zawału (już dziś)!")

gender_dict = {"Mężczyzna": "Male", "Kobieta": "Female"}
work_type_dict = {
    "Nigdy nie pracowałem/am": "Never_worked",
    "Pracowałem/am lub pracuję obecnie w sektorze publicznym": "Govt_job",
    "Pracowałem/am lub pracuję obecnie w sektorze prywatnym": "Private",
    "Pracowałem/am lub pracuję obecnie na własny rachunek": "Self-employed",
    "Zajmowałem/am lub zajmuję się obecnie dziećmi": "children",
}
married_dict = {"Tak": "Yes", "Nie": "No"}
residence_dict = {"Mieszkam obecnie w mieście": "Urban", "Mieszkam obecnie na wsi": "Rural"}
smoking_dict = {
    "Nigdy nie paliłem/am": "never smoked",
    "Paliłem/am w przeszłości": "formerly smoked",
    "Palę obecnie": "smokes",
}
binary_dict = {"Nie": 0, "Tak": 1}

age = st.slider("Wiek (0 dla dzieci poniżej 1. roku życia)", 0, 100, 30)
avg_glucose_level = st.number_input("Średni poziom glukozy w mm/dL (pozostaw domyślne, jeśli nieznany)", 50, 300, 80)
height = st.number_input("Wzrost (cm)", 30.0, 250.0, 170.0)
weight = st.number_input("Waga (kg)", 2.5, 300.0, 70.0)
gender = st.selectbox("Płeć", gender_dict.keys())
ever_married = st.selectbox("Czy kiedykolwiek byłeś/aś żonaty/zamężna", married_dict.keys())
work_type = st.selectbox("Jaki był typ wykonywanej przez Ciebie ostatnio pracy", work_type_dict.keys())
residence_type = st.selectbox("Gdzie mieszkasz?", residence_dict.keys())
smoking_status = st.selectbox("Czy palisz lub paliłeś/aś papierosy?", smoking_dict.keys())
hypertension = st.selectbox("Czy masz nadciśnienie tętnicze?", binary_dict.keys())
heart_disease = st.selectbox("Czy chorujesz obecnie na chorobę sercowo-naczyniową?", binary_dict.keys())

bmi = weight / ((height / 100) ** 2)

input_data = pd.DataFrame({
    "age": [age],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "gender": [gender_dict[gender]],
    "ever_married": [married_dict[ever_married]],
    "work_type": [work_type_dict[work_type]],
    "residence_type": [residence_dict[residence_type]],
    "smoking_status": [smoking_status],
    "hypertension": [binary_dict[hypertension]],
    "heart_disease": [binary_dict[heart_disease]]
})

transformed = pd.DataFrame(
    preprocessor.transform(input_data),
    columns=[
        *preprocessor.named_transformers_["num"].get_feature_names_out(["age", "avg_glucose_level", "bmi"]),
        *preprocessor.named_transformers_["cat"].get_feature_names_out(
            ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]),
        "hypertension",
        "heart_disease"
    ],
    index=input_data.index
)

if st.button("Predict Stroke Risk"):
    proba = model.predict_proba(transformed)[0][1]
    percent = int(round(proba * 100))

    fig, ax = plt.subplots()
    size = 0.3
    ax.pie([proba, 1 - proba],
           colors=['red', 'lightgray'],
           radius=1, startangle=90, counterclock=False,
           wedgeprops=dict(width=size, edgecolor='white'))
    ax.text(0, 0, f'{percent}%', ha='center', va='center', fontsize=24, weight='bold')
    st.pyplot(fig)

    coefficients = model.coef_[0]
    feature_names = transformed.columns
    feature_values = transformed.iloc[0].values
    all_features = [(feature_names[i], coefficients[i] * feature_values[i]) for i in range(len(feature_names))]
    unchangeables = ["age", "ever_married", "gender"]
    features_without_unchangeables = [
        (name, value) for name, value in all_features
        if not any(blocked in name for blocked in unchangeables)
    ]
    top_features = sorted(features_without_unchangeables)[:3]

    st.markdown("## Najważniejsze czynniki wpływające na ryzyko:")
    for feature, value in top_features:
        if "avg_glucose_level" in feature:
            st.info("Dbaj o regularne kontrolowanie poziomu glukozy")
        elif "bmi" in feature:
            st.info(f"Skontroluj masę ciała – Twoje BMI wynosi {bmi:.2f}")
        elif "work_type" in feature:
            st.info("Rozważ zmianę pracy na mniej stresującą")
        elif "residence_type" in feature:
            st.info("Zastanów się nad zmianą miejsca zamieszkania na bardziej spokojne")
        elif "smoking_status" in feature:
            st.info("Powinieneś rzucić palenie")
        elif "hypertension" in feature:
            st.info("Monitoruj poziom ciśnienia krwi i regularnie odwiedzaj lekarza")
        elif "heart_disease" in feature:
            st.info("Zadbaj o zdrowie układu krążenia – regularne kontrole są kluczowe")
        else:
            continue

    st.markdown("##### *to nie jest porada medyczna")
