from __future__ import annotations
"""Model interfaces and implementations for anomaly detection."""

from dataclasses import dataclass
from typing import Protocol, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

class BaseModel(Protocol):
    def fit(self, X_train: np.ndarray) -> None: ...
    def score_samples(self, X: np.ndarray) -> np.ndarray: ...
    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """
        Return a (n_samples, n_features) non-negative contribution matrix.
        Higher values indicate stronger contribution to anomaly at that row.
        """
        ...

@dataclass
class PCAModel:
    """PCA-based reconstruction error model with per-feature contributions."""
    n_components: int | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        self.pca_: PCA | None = None
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> None:
        # Choose components via min(n_features-1, median heuristic) if not set
        n_features = X_train.shape[1]
        n_comp = self.n_components or max(1, min(n_features - 1, int(np.ceil(min(n_features, 10)))))
        self.pca_ = PCA(n_components=0.99, random_state=self.random_state)
        self.pca_.fit(X_train)
        self.components_ = self.pca_.components_
        self.mean_ = self.pca_.mean_

    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        assert self.pca_ is not None
        Z = self.pca_.transform(X)
        X_hat = self.pca_.inverse_transform(Z)
        return X_hat

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_hat = self._reconstruct(X)
        err = (X - X_hat) ** 2
        # Raw anomaly score is total reconstruction error
        return err.sum(axis=1)

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        X_hat = self._reconstruct(X)
        err = (X - X_hat) ** 2
        # Non-negative per-feature contributions based on squared error
        return err

@dataclass
class IsolationForestModel:
    """Isolation Forest model. Feature-level contributions approximated by z-score magnitude."""
    n_estimators: int = 200
    contamination: str | float = "auto"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.iforest_: IsolationForest | None = None

    def fit(self, X_train: np.ndarray) -> None:
        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.iforest_.fit(X_train)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        assert self.iforest_ is not None
        # sklearn IsolationForest: higher is less abnormal -> invert
        raw = -self.iforest_.score_samples(X)
        return raw

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        # Approximate contributions by absolute standardized deviations
        # Since X was standardized earlier, |X| acts like |z-score|.
        return np.abs(X)
