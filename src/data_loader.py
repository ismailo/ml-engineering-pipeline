from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import yaml


# -----------------------------
# 1) Config model (simple)
# -----------------------------
@dataclass
class DataConfig:
    raw_path: str
    target_col: str
    id_col: str
    required_cols: List[str]


def load_config(config_path: str = "config.yaml") -> DataConfig:
    """
    Loads config.yaml and returns a DataConfig object.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    return DataConfig(
        raw_path=data_cfg["raw_path"],
        target_col=data_cfg["target_col"],
        id_col=data_cfg["id_col"],
        required_cols=data_cfg["required_cols"],
    )


# -----------------------------
# 2) Ingestion helpers
# -----------------------------
def _assert_file_exists(file_path: str) -> Path:
    """
    Ensures the raw CSV file exists before we try to read it.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {file_path}\n"
            f"Put your dataset at: ml-engineering-pipeline/{file_path}"
        )
    return path


def _assert_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Ensures the dataset has the columns we expect.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


# -----------------------------
# 3) Main ingestion function
# -----------------------------
def ingest_raw_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Loads the raw Telco churn CSV into a DataFrame with basic validation.
    """
    cfg = load_config(config_path=config_path)

    raw_csv_path = _assert_file_exists(cfg.raw_path)

    # Read CSV into a DataFrame
    df = pd.read_csv(raw_csv_path)

    # Basic dataset sanity checks
    if df.empty:
        raise ValueError("Loaded dataset is empty. Check the CSV file content.")

    _assert_required_columns(df, cfg.required_cols)

    # Return raw df (no cleaning yet—cleaning belongs in the next step)
    return df


# -----------------------------
# 4) CLI entry point
# -----------------------------
if __name__ == "__main__":
    df = ingest_raw_data()
    print("✅ Data ingestion successful")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("First 3 rows:")
    print(df.head(3))