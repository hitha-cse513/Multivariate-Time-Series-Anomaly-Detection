from __future__ import annotations
"""CLI entrypoint orchestrating MTSAD pipeline."""

from typing import Literal
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .config import Defaults
from .data import DataProcessor
from .model import PCAModel, IsolationForestModel, BaseModel
from .attribution import top_contributors
from .scoring import to_custom_scaled_scores
from .io_utils import write_augmented_csv, OUT_TOP_COLS, OUT_SCORE_COL


def build_model(name: Literal["pca", "iforest"]) -> BaseModel:
    """Return model instance based on name."""
    if name == "pca":
        return PCAModel()
    if name == "iforest":
        return IsolationForestModel()
    raise ValueError("Unknown model name. Choose from {'pca','iforest'}.")


def run_pipeline(
    input_csv: str,
    output_csv: str,
    model_name: Literal["pca", "iforest", "ensemble"] = "pca",
    train_start: str = Defaults.TRAIN_START,
    train_end: str = Defaults.TRAIN_END,
    analysis_start: str = Defaults.ANALYSIS_START,
    analysis_end: str = Defaults.ANALYSIS_END,
    timestamp_col: str = Defaults.TIMESTAMP_COL,
    validate_training: bool = False,
) -> None:
    """Run full anomaly detection pipeline with preprocessing, training, scoring, and attribution."""

    # 1-2. Data load & split
    dp = DataProcessor(timestamp_col=timestamp_col)
    split = dp.load_and_prepare(
        input_csv=input_csv,
        train_start=train_start,
        train_end=train_end,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )

    dt_start = datetime.fromisoformat(train_start)
    dt_end = datetime.fromisoformat(train_end)
    if dt_end - dt_start < timedelta(hours=72):
        raise ValueError(f"Training window too short (<72h): {train_start} → {train_end}")

    X = split.features_df.to_numpy(dtype=float)
    X_train = split.features_df.iloc[split.train_idx].to_numpy(dtype=float)
    analysis_mask = np.zeros(len(split.features_df), dtype=bool)
    analysis_mask[split.analysis_idx] = True

    def normalize_scores(raw: np.ndarray):
        if np.allclose(raw.std(), 0):
            raw = raw + np.random.normal(0, 1e-6, size=len(raw))
        mean = raw.mean()
        std = raw.std() if raw.std() > 1e-6 else 1.0
        norm = (raw - mean) / std
        return norm, mean, std

    if model_name == "ensemble":
        # Train both models
        model_pca = PCAModel()
        model_iforest = IsolationForestModel()
        model_pca.fit(X_train)
        model_iforest.fit(X_train)

        # Score training data and normalize
        raw_train_pca = model_pca.score_samples(X_train)
        raw_train_iforest = model_iforest.score_samples(X_train)
        norm_train_pca, mean_pca, std_pca = normalize_scores(raw_train_pca)
        norm_train_iforest, mean_iforest, std_iforest = normalize_scores(raw_train_iforest)

        # Combine normalized training scores (average)
        combined_train = (norm_train_pca + norm_train_iforest) / 2

        norm_train = np.clip(combined_train, a_min=0.0, a_max=None)
        if validate_training:
            print(f"[Training validation] mean={norm_train.mean():.2f}, max={norm_train.max():.2f}")
            if norm_train.mean() >= 10 or norm_train.max() >= 25:
                raise ValueError(
                    f"Training anomalies too high (mean={norm_train.mean():.2f}, max={norm_train.max():.2f})"
                )
        else:
            if norm_train.mean() >= 10 or norm_train.max() >= 25:
                print(f"[Warning] Training anomalies high: mean={norm_train.mean():.2f}, max={norm_train.max():.2f}")

        # Score all data and normalize per model
        raw_scores_pca = model_pca.score_samples(X)
        raw_scores_iforest = model_iforest.score_samples(X)
        norm_scores_pca = (raw_scores_pca - mean_pca) / std_pca
        norm_scores_iforest = (raw_scores_iforest - mean_iforest) / std_iforest

        # Combine normalized scores (average) and clip negatives
        combined_scores = (norm_scores_pca + norm_scores_iforest) / 2
        combined_scores = np.clip(combined_scores, a_min=0.0, a_max=None)

        scaled_all = to_custom_scaled_scores(combined_scores, norm_train.min(), norm_train.mean(), norm_train.max())

        contrib_pca = model_pca.feature_contributions(X)
        contrib_iforest = model_iforest.feature_contributions(X)
        contrib_all = (contrib_pca + contrib_iforest) / 2
        contrib_analysis = contrib_all[analysis_mask]

    else:
        # Original single-model flow
        model = build_model(model_name)
        model.fit(X_train)

        raw_train = model.score_samples(X_train)
        if np.allclose(raw_train.std(), 0):
            raw_train = raw_train + np.random.normal(0, 1e-6, size=len(raw_train))

        train_mean = raw_train.mean()
        train_std = raw_train.std() if raw_train.std() > 1e-6 else 1.0

        norm_train = (raw_train - train_mean) / train_std
        norm_train = np.clip(norm_train, a_min=0.0, a_max=None)
        if validate_training:
            print(f"[Training validation] mean={norm_train.mean():.2f}, max={norm_train.max():.2f}")
            if norm_train.mean() >= 10 or norm_train.max() >= 25:
                raise ValueError(
                    f"Training anomalies too high (mean={norm_train.mean():.2f}, max={norm_train.max():.2f})"
                )
        else:
            if norm_train.mean() >= 10 or norm_train.max() >= 25:
                print(f"[Warning] Training anomalies high: mean={norm_train.mean():.2f}, max={norm_train.max():.2f}")

        raw_scores_all = model.score_samples(X)
        scaled_all = to_custom_scaled_scores(raw_scores_all, norm_train.min(), norm_train.mean(), norm_train.max())
        contrib_all = model.feature_contributions(X)
        contrib_analysis = contrib_all[analysis_mask]

    top_feats_df = top_contributors(contrib_analysis, split.feature_names, threshold=0.01, k=7)

    empty_top = {col: [""] * len(split.features_df) for col in OUT_TOP_COLS}
    top_full = pd.DataFrame(empty_top)
    top_full.loc[analysis_mask, OUT_TOP_COLS] = top_feats_df.values

    write_augmented_csv(
        df=split.full_df,
        out_scores=pd.Series(scaled_all, index=split.full_df.index, name=OUT_SCORE_COL),
        top_features=top_full,
        output_path=output_csv,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multivariate Time Series Anomaly Detection (MTSAD)\n\n"
            "Example:\n"
            "  python -m mtsad.main --input_csv data.csv --output_csv out.csv --model pca\n"
            "  python -m mtsad.main --input_csv data.csv --output_csv out.csv --model ensemble\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input_csv", required=True, help="Path to input CSV file")
    p.add_argument("--output_csv", required=True, help="Path to write augmented CSV")
    p.add_argument("--model", default="pca", choices=["pca", "iforest", "ensemble"], help="Anomaly model")
    p.add_argument("--train_start", default=Defaults.TRAIN_START, help="Training window start (YYYY-MM-DD HH:MM)")
    p.add_argument("--train_end", default=Defaults.TRAIN_END, help="Training window end (YYYY-MM-DD HH:MM)")
    p.add_argument("--analysis_start", default=Defaults.ANALYSIS_START, help="Analysis window start (YYYY-MM-DD HH:MM)")
    p.add_argument("--analysis_end", default=Defaults.ANALYSIS_END, help="Analysis window end (YYYY-MM-DD HH:MM)")
    p.add_argument("--timestamp_col", default=Defaults.TIMESTAMP_COL, help="Name of timestamp column")
    p.add_argument("--validate_training", action="store_true", help="Strict check for training mean/max anomaly scores")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_name=args.model,
        train_start=args.train_start,
        train_end=args.train_end,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        timestamp_col=args.timestamp_col,
        validate_training=args.validate_training,
    )


if __name__ == "__main__":
    main()
