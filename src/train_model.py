import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def load_training_data():

    data_path = Path("data/processed")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")

    y_train = pd.read_csv(data_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    print("Confusion Matrix:")
    print(cm)

    print("\n")


def train_models():

    X_train, X_test, y_train, y_test = load_training_data()

    print("Training Logistic Regression")

    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train, y_train)

    evaluate_model(lr_model, X_test, y_test)

    print("Training Random Forest")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    evaluate_model(rf_model, X_test, y_test)


if __name__ == "__main__":

    train_models()