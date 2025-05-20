import numpy as np
import shap
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("data/06_models/best_model.pkl")
preprocessor = joblib.load("data/06_models/preprocessor.pkl")

st.title("Sprawdź jakie jest Twoje ryzyko zachorowania na cukrzycę!")

gender_dict = {"Kobieta": "Female", "Mężczyzna": "Male", }
smoking_dict = {
    "Nigdy nie paliłam/em": "non-smoker",
    "Paliłam/em w przeszłości, ale obecnie nie palę": "smoker",
    "Palę obecnie": "smoker",
}
binary_dict = {"Nie": 0, "Tak": 1}

age = st.slider("Wiek (0 dla dzieci poniżej 1. roku życia)", 0, 100, 30)
blood_glucose_level = st.number_input("Średni poziom glukozy w mm/dL (pozostaw domyślne, jeśli nieznany)", 50, 300, 80)
height = st.number_input("Wzrost (cm)", 30.0, 250.0, 170.0, 0.5)
weight = st.number_input("Waga (kg)", 2.5, 300.0, 60.0, 0.5)
gender = gender_dict[st.selectbox("Płeć", gender_dict.keys())]
smoking_history = st.selectbox("Czy palisz lub paliłeś/aś papierosy?", smoking_dict.keys())
hypertension = binary_dict[st.selectbox("Czy masz nadciśnienie tętnicze?", binary_dict.keys())]
heart_disease = binary_dict[st.selectbox("Czy chorujesz obecnie na chorobę sercowo-naczyniową?", binary_dict.keys())]

bmi = weight / ((height / 100) ** 2)

input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking_history],
    "bmi": [bmi],
    "blood_glucose_level": [blood_glucose_level],
})

transformed = pd.DataFrame(
    preprocessor.transform(input_data),
    columns=[
        *preprocessor.named_transformers_["num"].get_feature_names_out(["age", "blood_glucose_level", "bmi"]),
        *preprocessor.named_transformers_["cat"].get_feature_names_out(
            ["gender", "smoking_history"]),
        "hypertension",
        "heart_disease"
    ],
    index=input_data.index
)

if st.button("Przewiduj ryzyko cukrzycy"):
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_contrib = shap_values[1][0]  # dla klasy pozytywnej (diabetes = 1)
    else:
        shap_contrib = shap_values[0]

    shap_df = pd.DataFrame({
        'feature': transformed.columns,
        'value': transformed.iloc[0].values,
        'shap_value': shap_contrib
    })

    top_features = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index).head(4)

    st.markdown("## Najważniejsze czynniki wpływające na ryzyko:")

    positive_features = []
    negative_features = []

    for _, row in top_features.iterrows():
        feat = row["feature"]
        shap_val = row["shap_value"]
        is_positive = shap_val < 0

        if "blood_glucose_level" in feat:
            if is_positive and blood_glucose_level < 120:
                positive_features.append("Twój poziom glukozy **jest w normie i zmniejsza ryzyko cukrzycy**.")
            else:
                negative_features.append(
                    "Podwyższony poziom glukozy to **poważny czynnik ryzyka cukrzycy** – kontroluj go regularnie.")
        elif "bmi" in feat:
            if bmi < 18.5:
                negative_features.append(
                    f"Twoje BMI ({bmi:.2f}) **jest zbyt niskie** – warto skonsultować się z lekarzem.")
            elif is_positive:
                positive_features.append(
                    f"Twoje BMI ({bmi:.2f}) **wygląda w porządku i przyczynia się do zmniejszenia ryzyka cukrzycy**.")
            else:
                negative_features.append(
                    f"Wyższe BMI ({bmi:.2f}) to **czynnik ryzyka** – warto zadbać o prawidłową masę ciała.")
        elif "smoking" in feat:
            if smoking_history == "non-smoker":
                positive_features.append(
                    "Brak historii palenia tytoniu **działa ochronnie i obniża ryzyko cukrzycy** – świetna decyzja!")
            else:
                negative_features.append("Palenie to **jeden z czynników ryzyka cukrzycy**.")
        elif "hypertension" in feat:
            if is_positive:
                positive_features.append("Ciśnienie krwi w normie **zapobiega powstawaniu** cukrzycy.")
            else:
                negative_features.append("Nadciśnienie to **czynnik ryzyka** – kontroluj ciśnienie regularnie "
                                         "oraz pamiętaj o wizytach u lekarza.")
        elif "heart_disease" in feat:
            if is_positive:
                positive_features.append("Brak chorób serca **zmniejsza ryzyko cukrzycy**.")
            else:
                negative_features.append("Występujące choroby serca to **czynnik ryzyka** – "
                                         "zadbaj o układ krążenia.")
        elif "age" in feat:
            if is_positive:
                positive_features.append("Dla osób w Twoim wieku ryzyko cukrzycy **jest niższe**.")
            else:
                negative_features.append("Twój wiek to jeden z **czynników ryzyka** –"
                                         " warto zwiększyć czujność wraz z upływem lat i dbać o zdrowie.")
        elif "gender" in feat:
            if is_positive:
                positive_features.append(
                    "Twoja płeć, w zestawieniu z innymi parametrami **przeciwdziała ryzyku cukrzycy**.")
            else:
                negative_features.append("Twoja płeć, w zestawieniu z innymi parametrami "
                                         "może być **czynnikiem ryzyka** – warto mieć to na uwadze.")
        else:
            continue

    if positive_features:
        st.info("### Czynniki działające ochronnie:\n\n" + "\n\n".join(f"✅ {msg}" for msg in positive_features))

    if negative_features:
        st.error("### Czynniki ryzyka:\n\n" + "\n\n".join(f"⚠️ {msg}" for msg in negative_features))

    st.markdown("##### *to nie jest porada medyczna")
