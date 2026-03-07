from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_validation import validate_and_clean_data
from src.data_loader import load_config


def build_features(config_path="config.yaml"):

    # Load cleaned dataset
    df = validate_and_clean_data(config_path)

    cfg = load_config(config_path)

    print("Dataset loaded:", df.shape)

    # -------------------------
    # Separate features and target
    # -------------------------

    X = df.drop(columns=[cfg.target_col, cfg.id_col])
    y = df[cfg.target_col]

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    # -------------------------
    # Identify numeric columns
    # -------------------------

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # -------------------------
    # Scale numeric features
    # -------------------------

    scaler = StandardScaler()

    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print("Numeric features scaled")

    # -------------------------
    # Train/Test Split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train/Test split complete")

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # -------------------------
    # Save datasets
    # -------------------------

    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)

    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    print("Training datasets saved")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = build_features()

    print("\nPipeline Step Complete")