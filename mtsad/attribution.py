from __future__ import annotations
"""Feature attribution helpers."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from .io_utils import OUT_TOP_COLS

def top_contributors(contrib_matrix: np.ndarray, feature_names: list[str], threshold: float = 0.01, k: int = 7) -> pd.DataFrame:
    """Return top-k feature contributors per row with alphabetical tie-breaking."""
    rows = []
    for row in contrib_matrix:
        contribs = [(abs(val), fname) for val, fname in zip(row, feature_names)]
        contribs = [(v, f) for v, f in contribs if v > threshold]
        # Sort by value desc, then alphabetically asc
        contribs.sort(key=lambda x: (-x[0], x[1]))
        top_feats = [f for _, f in contribs[:k]]
        # Fill remaining with ""
        while len(top_feats) < k:
            top_feats.append("")
        rows.append(top_feats)
    return pd.DataFrame(rows, columns=OUT_TOP_COLS)

