import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, predict_model, pull, save_model
import joblib


def split_data(data: pd.DataFrame, test_size: float = 0.2):
    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def train_model(split_output: dict, top_n: int = 15):
    X_train = split_output["X_train"]
    X_test = split_output["X_test"]
    y_train = split_output["y_train"]
    y_test = split_output["y_test"]

    train = pd.concat([X_train, y_train], axis=1)
    train.columns = list(X_train.columns) + ['diabetes']

    setup(
        data=train,
        target='diabetes',
        session_id=123,
        fix_imbalance=True,
        verbose=False,
    )

    top_models = compare_models(n_select=top_n)

    metrics_list = []
    model_objs = []

    for i, model in enumerate(top_models):
        model_name = type(model).__name__
        save_model(model, f"data/06_models/{model_name}_{i}")

        test_df = X_test.copy()
        test_df['diabetes'] = y_test.values
        predict_model(model, data=test_df)
        metrics = pull()
        metrics['model_name'] = model_name
        metrics_list.append(metrics)
        model_objs.append((model_name, model))

    metrics_df = pd.concat(metrics_list, ignore_index=True)
    metrics_df["F1"] = pd.to_numeric(metrics_df["F1"], errors="coerce")
    metrics_df["Recall"] = pd.to_numeric(metrics_df["Recall"], errors="coerce")
    f1 = metrics_df.pop("F1")
    metrics_df.insert(1, "F1", f1)
    metrics_df = metrics_df.sort_values(by="F1", ascending=False)
    metrics_df.to_csv("data/08_reporting/best_models_metrics.csv", index=False)

    top3 = metrics_df.head(3)
    best_model_name = top3.sort_values(by="Recall", ascending=False).iloc[0]["model_name"]
    best_model = dict(model_objs)[best_model_name]

    joblib.dump(best_model, "data/06_models/best_model.pkl")

    return best_model
