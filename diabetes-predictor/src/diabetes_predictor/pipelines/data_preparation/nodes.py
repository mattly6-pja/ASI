import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data_copy = data.copy()
    data_copy.drop(columns=['HbA1c_level'], inplace=True)
    data_copy["smoking_history"] = data_copy["smoking_history"].apply(lambda x: "smoker" if x in ["former", "current"] else "non-smoker")

    numeric_cols = ["age", "blood_glucose_level", "bmi"]
    categorical_cols = ["gender", "smoking_history"]
    passthrough_cols = ["hypertension", "heart_disease"]

    X = data_copy[numeric_cols + categorical_cols + passthrough_cols]
    y = data_copy["diabetes"]

    preprocessor_diabetes = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough"
    )

    transformed = preprocessor_diabetes.fit_transform(X)

    feature_names = (
            numeric_cols
            + list(preprocessor_diabetes.named_transformers_["cat"].get_feature_names_out(categorical_cols))
            + passthrough_cols
    )

    df_out = pd.DataFrame(transformed, columns=feature_names, index=data_copy.index)
    df_out["diabetes"] = y.values

    joblib.dump(preprocessor_diabetes, "data/06_models/preprocessor.pkl")

    return df_out
