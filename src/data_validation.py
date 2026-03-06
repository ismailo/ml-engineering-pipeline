from pathlib import Path
import pandas as pd

from src.data_loader import ingest_raw_data, load_config


def validate_and_clean_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Validate and clean the raw Telco churn dataset.
    """

    # Load configuration
    cfg = load_config(config_path)

    # Load raw dataset
    df = ingest_raw_data(config_path)

    print("Initial dataset shape:", df.shape)

    # -----------------------------
    # 1. Remove duplicate rows
    # -----------------------------
    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    print(f"Removed {before - after} duplicate rows")

    # -----------------------------
    # 2. Fix TotalCharges column
    # -----------------------------

    # Convert blank strings to NaN
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

    # Convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # -----------------------------
    # 3. Validate target column
    # -----------------------------

    valid_labels = {"Yes", "No"}

    if not set(df[cfg.target_col].unique()).issubset(valid_labels):
        raise ValueError("Target column contains invalid labels")

    # Convert to binary
    df[cfg.target_col] = df[cfg.target_col].map({"Yes": 1, "No": 0})

    # -----------------------------
    # 4. Convert categorical columns
    # -----------------------------

    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # Remove ID column from encoding
    if cfg.id_col in categorical_cols:
        categorical_cols.remove(cfg.id_col)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # -----------------------------
    # 5. Save cleaned dataset
    # -----------------------------

    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_path / "clean_telco_customer_churn.csv"

    df.to_csv(cleaned_file, index=False)

    print("Clean dataset saved to:", cleaned_file)

    print("Final dataset shape:", df.shape)

    return df


if __name__ == "__main__":

    df = validate_and_clean_data()

    print("\nPreview cleaned dataset:")

    print(df.head())