from __future__ import annotations
"""Utility functions used across the project."""

from typing import Tuple
import numpy as np
import pandas as pd

def check_regular_intervals(ts: pd.Series) -> Tuple[pd.Timedelta, bool]:
    """
    Check if a datetime series has regular intervals.

    Args:
        ts: A pandas Series of dtype datetime64.

    Returns:
        A tuple (freq, is_regular) where freq is the most common difference and
        is_regular is True if all diffs match the mode difference.
    """
    diffs = ts.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(0), True
    mode = diffs.mode().iloc[0]
    is_regular = bool((diffs == mode).all())
    return mode, is_regular

def safe_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """
    Convert values to percentile ranks in [0, 100].
    Ties handled by average rank; adds a tiny epsilon to avoid identical zeros.

    Args:
        values: 1-D array of raw anomaly scores.

    Returns:
        1-D array of percentile ranks in [0, 100].
    """
    x = np.asarray(values, dtype=float)
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 100, num=len(x), endpoint=True)
    # add tiny noise to avoid perfect 0s when required
    eps = np.random.default_rng(42).normal(0, 1e-6, size=len(x))
    return np.clip(ranks + eps, 0.0, 100.0)

def zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit variance.
    Handles zero-variance by keeping std as 1.0 for those columns.

    Returns:
        (Z, mean, std)
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    Z = (X - mean) / std_safe
    return Z, mean, std_safe
