import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_training_data():

    data_path = Path("data/processed")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")

    y_train = pd.read_csv(data_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics


def train_models():

    X_train, X_test, y_train, y_test = load_training_data()

    mlflow.set_experiment("churn_prediction")

    # ---------------------
    # Logistic Regression
    # ---------------------

    with mlflow.start_run(run_name="logistic_regression"):

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params({
            "model": "logistic_regression"
        })

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")

        print("Logistic Regression Metrics:", metrics)


    # ---------------------
    # Random Forest
    # ---------------------

    with mlflow.start_run(run_name="random_forest"):

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params({
            "model": "random_forest",
            "n_estimators": 200,
            "max_depth": 10
        })

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model")

        print("Random Forest Metrics:", metrics)


if __name__ == "__main__":

    train_models()