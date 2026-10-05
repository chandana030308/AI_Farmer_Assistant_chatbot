"""
Helpers for loading and understanding data/agriculture_data.csv.
The code detects the crop (target) column and numeric feature columns automatically.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "agriculture_data.csv"

TARGET_CANDIDATES = ["label", "crop", "crop_name", "recommended_crop", "target", "class"]


def load_data(path=DATA_PATH):
    """Read the CSV file and do basic cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty after cleaning.")
    return df


def get_target_column(df):
    """Find the column that holds the crop name."""
    lowered = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    # Otherwise use the first text column, or the last column
    for column in df.columns:
        if df[column].dtype == object or str(df[column].dtype).startswith("string"):
            return column
    return df.columns[-1]


def get_feature_columns(df, target=None):
    """Return the numeric columns used as model inputs."""
    if target is None:
        target = get_target_column(df)
    numeric = df.select_dtypes(include="number").columns.tolist()
    features = [column for column in numeric if column != target]
    if not features:
        raise ValueError("No numeric feature columns found in the dataset.")
    return features


def get_feature_stats(df):
    """Return min / max / mean for every feature (used for the input form)."""
    target = get_target_column(df)
    stats = {}
    for column in get_feature_columns(df, target):
        stats[column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "mean": float(df[column].mean()),
        }
    return stats