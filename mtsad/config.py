from __future__ import annotations
"""
Default configuration values for time windows and processing.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    # Hackathon-defined windows
    TRAIN_START: str = "2004-01-01 00:00"
    TRAIN_END: str = "2004-01-05 23:59"
    ANALYSIS_START: str = "2004-01-01 00:00"
    ANALYSIS_END: str = "2004-01-19 07:59"

    # CSV timestamp column
    TIMESTAMP_COL: str = "Time"

    # Minimum hours required for training
    MIN_TRAINING_HOURS: int = 72

    # Random seed for reproducibility
    RANDOM_SEED: int = 42
