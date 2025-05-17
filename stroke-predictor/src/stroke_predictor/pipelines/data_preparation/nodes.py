import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.drop(columns=['id'], inplace=True)
    data = data[data['gender'] != 'Other']
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())

    for col in ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    return data


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    scaler = MinMaxScaler()

    cols_to_scale = ['age', 'avg_glucose_level', 'bmi']
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data[data["gender"] != "Other"]
    data.drop(columns=["id"], inplace=True)
    data["bmi"] = data["bmi"].fillna(data["bmi"].median())

    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    categorical_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough"
    )

    transformed = preprocessor.fit_transform(data)
    feature_names = (
        list(preprocessor.named_transformers_["num"].get_feature_names_out(numeric_cols))
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
        + ["hypertension", "heart_disease", "stroke"]
    )

    df_out = pd.DataFrame(transformed, columns=feature_names, index=data.index)

    joblib.dump(preprocessor, "data/06_models/preprocessor.pkl")

    return df_out
