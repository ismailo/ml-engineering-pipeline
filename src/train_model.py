import joblib
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------
# Load training datasets
# ---------------------------------------------------

def load_training_data():

    data_path = Path("data/processed")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")

    y_train = pd.read_csv(data_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Model evaluation
# ---------------------------------------------------

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    return metrics


# ---------------------------------------------------
# Train models
# ---------------------------------------------------

def train_models():

    X_train, X_test, y_train, y_test = load_training_data()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    }

    best_model = None
    best_score = 0

    for model_name, model in models.items():

        print(f"\nTraining {model_name}")

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        print(metrics)

        if metrics["f1"] > best_score:

            best_score = metrics["f1"]

            best_model = model


    print("\nBest model F1 score:", best_score)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")

    print("Best model saved to models/best_model.pkl")


# ---------------------------------------------------
# Run training
# ---------------------------------------------------

if __name__ == "__main__":

    train_models()