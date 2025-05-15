import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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

