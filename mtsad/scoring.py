from __future__ import annotations
"""Score normalization to 0–100 scale using percentile ranks."""

import numpy as np
import pandas as pd
from .utils import safe_percentile_ranks

def to_0_100_percentiles(raw_scores: np.ndarray) -> np.ndarray:
    """
    Transform raw anomaly scores to [0, 100] using percentile ranks.
    """
    return safe_percentile_ranks(raw_scores)
