from __future__ import annotations
"""Data loading, cleaning, splitting, and scaling."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

from .exceptions import TimestampError, DataQualityError
from .utils import check_regular_intervals, zscore

@dataclass
class SplitData:
    full_df: pd.DataFrame
    features_df: pd.DataFrame
    train_idx: np.ndarray
    analysis_idx: np.ndarray
    feature_names: List[str]

class DataProcessor:
    """Handles CSV loading, timestamp validation, cleaning, splitting, and scaling."""

    def __init__(self, timestamp_col: str = "Time") -> None:
        self.timestamp_col = timestamp_col
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names_: List[str] | None = None

    def load_and_prepare(
        self,
        input_csv: str,
        train_start: str,
        train_end: str,
        analysis_start: str,
        analysis_end: str,
    ) -> SplitData:
        df = pd.read_csv(input_csv)
        if self.timestamp_col not in df.columns:
            raise TimestampError(f"Missing timestamp column '{self.timestamp_col}'.")
        # Parse timestamps
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # Validate regular intervals
        freq, is_regular = check_regular_intervals(df[self.timestamp_col])
        if not is_regular:
            # Optionally resample if desired; for hackathon, raise a warning via exception message
            raise TimestampError("Timestamps are not at regular intervals. Please fix the source data.")

        # Keep only numeric columns as features (exclude timestamp)
        feature_cols = [c for c in df.columns if c != self.timestamp_col]
        features_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        # Handle missing values via linear interpolation, then forward/back fill
        features_df = features_df.interpolate(method="linear", limit_direction="both").ffill().bfill()

        # Identify constant features (zero variance) and keep them (std=1 later)
        nunique = features_df.nunique(dropna=False)
        if nunique.sum() == 0 or features_df.shape[1] == 0:
            raise DataQualityError("No usable numeric features found.")
        # training and analysis masks
        train_mask = (df[self.timestamp_col] >= pd.to_datetime(train_start)) & (df[self.timestamp_col] <= pd.to_datetime(train_end))
        analysis_mask = (df[self.timestamp_col] >= pd.to_datetime(analysis_start)) & (df[self.timestamp_col] <= pd.to_datetime(analysis_end))

        train_idx = np.where(train_mask.values)[0]
        analysis_idx = np.where(analysis_mask.values)[0]
        if len(train_idx) == 0 or len(analysis_idx) == 0:
            raise DataQualityError("Train/analysis windows select zero rows. Check your time bounds.")

        # Require minimum hours of training data (approx by count since intervals are regular)
        if len(train_idx) < 72:  # 72 time steps ~ 72 hours for hourly data
            raise DataQualityError("Insufficient training data: need at least 72 time steps.")

        # Standardize based on training only
        X_train = features_df.iloc[train_idx].to_numpy(dtype=float)
        Z_train, mean, std = zscore(X_train)
        self.mean_ = mean
        self.std_ = std
        self.feature_names_ = feature_cols

        # Standardize full features using training mean/std
        X_full = features_df.to_numpy(dtype=float)
        Z_full = (X_full - mean) / std

        Z_full_df = pd.DataFrame(Z_full, columns=feature_cols, index=features_df.index)

        return SplitData(
            full_df=df,
            features_df=Z_full_df,
            train_idx=train_idx,
            analysis_idx=analysis_idx,
            feature_names=feature_cols,
        )
