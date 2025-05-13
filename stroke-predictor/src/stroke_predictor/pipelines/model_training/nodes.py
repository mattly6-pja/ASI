import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    X = data.drop("stroke", axis=1)
    y = data["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)  # always set the same random state for reproducibility
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test


def train_model(X_res, X_test, y_res, y_test):
    from pycaret.classification import setup, compare_models, save_model
    train = pd.concat([X_res, y_res], axis=1)
    train.columns = list(X_res.columns) + ['stroke']
    setup(data=train, target='stroke', session_id=123, fix_imbalance=True, silent=True)  # set session_id for reproducibility
    model = compare_models()
    save_model(model, 'stroke_model')
    return model