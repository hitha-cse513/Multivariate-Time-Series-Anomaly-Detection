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
    - Scores > max_raw scaled smoothly from 25 to 100,
      capped at 100 using a smooth logistic or capped linear scaling.

    Args:
        raw_scores: raw anomaly scores array
        min_raw: minimum raw score in training
        mean_raw: mean raw score in training
        max_raw: maximum raw score in training
        
    Returns:
        np.ndarray of scaled scores between 0 and 100
    """

    scaled_scores = np.empty_like(raw_scores, dtype=float)
    raw_scores_clipped = np.maximum(raw_scores, min_raw)

    mask_low = raw_scores_clipped <= mean_raw
    mask_mid = (raw_scores_clipped > mean_raw) & (raw_scores_clipped <= max_raw)
    mask_high = raw_scores_clipped > max_raw

    # Scale 0 to 10
    if mean_raw > min_raw:
        scaled_scores[mask_low] = ((raw_scores_clipped[mask_low] - min_raw) / (mean_raw - min_raw)) * 10
    else:
        scaled_scores[mask_low] = 0

    # Scale 10 to 25
    if max_raw > mean_raw:
        scaled_scores[mask_mid] = 10 + ((raw_scores_clipped[mask_mid] - mean_raw) / (max_raw - mean_raw)) * 15
    else:
        scaled_scores[mask_mid] = 25

    # Smooth scale beyond max_raw, capped at 100
    if np.any(mask_high):
        # Define how far above max_raw the raw_scores can go to be scaled from 25 to 100
        scale_range = max_raw * 5  # you can tweak this to control scale stretch

        high_vals = raw_scores_clipped[mask_high]
        # Linear scale from max_raw to max_raw + scale_range
        scaled = 25 + ((high_vals - max_raw) / scale_range) * 75
        # Cap at 100 max
        scaled_scores[mask_high] = np.clip(scaled, 25, 100)

    return scaled_scores
