"""AI decision model: predict whether an underdog token reverses and wins.

The model is trained and evaluated strictly out-of-sample on real resolved
Polymarket BTC 5m markets using forward-chaining (time-series) splits, so a
prediction never sees data from its own future.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.ai.features import FEATURE_COLUMNS, features_to_frame


def _new_estimator() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=30,
        subsample=0.85,
        random_state=42,
    )


class ReversalDecisionModel:
    """Gradient boosting classifier over order-flow / price-path features."""

    def __init__(self) -> None:
        self.model = _new_estimator()
        self.feature_columns = list(FEATURE_COLUMNS)
        self._is_fitted = False
        self.metrics: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Fit on a dataset of real markets."""
        X = df[self.feature_columns]
        y = df["underdog_won"]
        self.model.fit(X, y)
        self._is_fitted = True

    def evaluate_walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> dict[str, float]:
        """
        Forward-chaining evaluation: train on the past, test on the future.

        Returns out-of-sample discrimination metrics. AUC near 0.50 means the
        model carries no predictive information beyond chance.
        """
        if len(df) < n_splits * 20:
            raise ValueError(f"Need at least {n_splits * 20} rows, got {len(df)}")

        df = df.sort_values("end_ts").reset_index(drop=True)
        X = df[self.feature_columns]
        y = df["underdog_won"].to_numpy()

        oos_prob = np.full(len(df), np.nan)
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            # Skip folds where one class is absent — AUC would be undefined.
            if len(np.unique(y[train_idx])) < 2:
                continue
            est = _new_estimator()
            est.fit(X.iloc[train_idx], y[train_idx])
            oos_prob[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]

        mask = ~np.isnan(oos_prob)
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            self.metrics = {"n_out_of_sample": int(mask.sum())}
            return self.metrics

        self.metrics = {
            "n_out_of_sample": int(mask.sum()),
            "roc_auc": float(roc_auc_score(y[mask], oos_prob[mask])),
            "brier_score": float(brier_score_loss(y[mask], oos_prob[mask])),
            "base_rate": float(y[mask].mean()),
            "mean_predicted": float(oos_prob[mask].mean()),
        }
        return self.metrics

    def out_of_sample_predictions(self, df: pd.DataFrame, n_splits: int = 5) -> pd.Series:
        """Out-of-sample probabilities aligned to ``df``, for honest backtesting."""
        df = df.sort_values("end_ts").reset_index(drop=True)
        X = df[self.feature_columns]
        y = df["underdog_won"].to_numpy()

        oos_prob = np.full(len(df), np.nan)
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            if len(np.unique(y[train_idx])) < 2:
                continue
            est = _new_estimator()
            est.fit(X.iloc[train_idx], y[train_idx])
            oos_prob[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]

        return pd.Series(oos_prob, index=df.index, name="reversal_probability")

    def predict_reversal_probability(self, features: dict[str, float]) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        return float(self.model.predict_proba(features_to_frame(features))[0, 1])

    def feature_importance(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        return pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "metrics": self.metrics, "features": self.feature_columns},
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.metrics = data.get("metrics", {})
        self.feature_columns = data.get("features", list(FEATURE_COLUMNS))
        self._is_fitted = True
