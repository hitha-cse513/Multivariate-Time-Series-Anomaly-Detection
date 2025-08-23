from __future__ import annotations
"""CSV IO utilities."""

from typing import List, Tuple
import pandas as pd

OUT_SCORE_COL = "Abnormality_score"
OUT_TOP_COLS = [f"top_feature_{i}" for i in range(1, 8)]
ALL_OUT_COLS = [OUT_SCORE_COL] + OUT_TOP_COLS

def read_csv_with_time(path: str, timestamp_col: str) -> pd.DataFrame:
    """
    Read CSV and parse datetime column.
    """
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found.")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df

def write_augmented_csv(df: pd.DataFrame, out_scores: pd.Series, top_features: pd.DataFrame, output_path: str) -> None:
    """
    Append required output columns and write CSV.

    Args:
        df: Original dataframe.
        out_scores: Series of 0-100 scores aligned to df index.
        top_features: DataFrame with columns OUT_TOP_COLS.
        output_path: Path to write CSV.
    """
    out = df.copy()
    out[OUT_SCORE_COL] = out_scores.values
    for c in OUT_TOP_COLS:
        out[c] = top_features[c].astype(str).values
    out.to_csv(output_path, index=False)
