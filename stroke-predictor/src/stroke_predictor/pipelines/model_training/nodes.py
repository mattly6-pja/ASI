import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pycaret.classification import setup, compare_models, predict_model, pull, save_model


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    X = data.drop("stroke", axis=1)
    y = data["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)  # always set the same random state for reproducibility
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return {
        "X_resampled": X_res,
        "X_test": X_test,
        "y_resampled": y_res,
        "y_test": y_test
    }


def train_model(split_output: dict, top_n: int = 5):

    X_res = split_output["X_resampled"]
    X_test = split_output["X_test"]
    y_res = split_output["y_resampled"]
    y_test = split_output["y_test"]

    train = pd.concat([X_res, y_res], axis=1)
    train.columns = list(X_res.columns) + ['stroke']

    setup(
        data=train,
        target='stroke',
        session_id=123,
        fix_imbalance=True,
        verbose=False,
    )

    top_models = compare_models(n_select=top_n)

    metrics_list = []

    for i, model in enumerate(top_models):
        model_name = type(model).__name__
        save_model(model, f"data/06_models/{model_name}")

        test_df = X_test.copy()
        test_df['stroke'] = y_test.values
        predict_model(model, data=test_df)
        metrics = pull()
        metrics['model_name'] = model_name
        metrics_list.append(metrics.to_dict())

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df["F1"] = pd.to_numeric(metrics_df["F1"], errors="coerce")
    metrics_df = metrics_df.sort_values(by="F1", ascending=False)
    metrics_df.to_csv("data/08_reporting/best_models_metrics.csv", index=False)

    return top_models[0]
