from __future__ import annotations
import numpy as np

def to_custom_scaled_scores(
    raw_scores: np.ndarray,
    min_raw: float,
    mean_raw: float,
    max_raw: float,
) -> np.ndarray:
    """
    Scale raw anomaly scores to 0-100 scale with:
    - Scores <= mean_raw scaled 0 to 10 (min_raw to mean_raw)
    - Scores > mean_raw and <= max_raw scaled 10 to 25
    - Scores > max_raw scaled 25 to 100 (scaled linearly beyond max_raw, capped at 100)
    """
    scaled_scores = np.empty_like(raw_scores, dtype=float)

    # Clip scores below min_raw to min_raw to avoid negative scaling
    raw_scores_clipped = np.maximum(raw_scores, min_raw)

    # Masks for different regions
    mask_low = raw_scores_clipped <= mean_raw
    mask_mid = (raw_scores_clipped > mean_raw) & (raw_scores_clipped <= max_raw)
    mask_high = raw_scores_clipped > max_raw

    # 1) Scale scores <= mean_raw from 0 to 10
    if mean_raw > min_raw:
        scaled_scores[mask_low] = (
            (raw_scores_clipped[mask_low] - min_raw) / (mean_raw - min_raw)
        ) * 10
    else:
        scaled_scores[mask_low] = 0

    # 2) Scale scores > mean_raw and <= max_raw from 10 to 25
    if max_raw > mean_raw:
        scaled_scores[mask_mid] = 10 + (
            (raw_scores_clipped[mask_mid] - mean_raw) / (max_raw - mean_raw)
        ) * 15  # 25 - 10 = 15
    else:
        scaled_scores[mask_mid] = 25

    # 3) Scale scores > max_raw from 25 to 100
    # For values beyond max_raw, scale linearly with a reasonable assumed max limit
    # To avoid infinite scale, define an upper bound for raw scores (e.g., max_raw + delta)
    delta = max_raw - mean_raw if max_raw > mean_raw else 1.0  # fallback delta
    upper_bound = max_raw + 4 * delta  # arbitrary scaling window beyond max_raw

    raw_scores_high = raw_scores_clipped[mask_high]
    # Clip high scores to upper_bound
    raw_scores_high_clipped = np.minimum(raw_scores_high, upper_bound)

    scaled_scores[mask_high] = 25 + (
        (raw_scores_high_clipped - max_raw) / (upper_bound - max_raw)
    ) * 75  # 100 - 25 = 75

    # For values beyond upper_bound, cap at 100
    scaled_scores[mask_high & (raw_scores_clipped > upper_bound)] = 100

    return scaled_scores
