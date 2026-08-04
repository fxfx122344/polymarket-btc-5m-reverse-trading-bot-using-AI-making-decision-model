"""AI decision model for predicting price reversals on underdog tokens."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ai.features import FEATURE_COLUMNS, features_to_array, generate_training_data
from src.config import settings


class ReversalDecisionModel:
    """
    Gradient Boosting classifier trained to identify conditions where
    underdog tokens (price < 0.5) are likely to reverse and win.
    """

    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            min_samples_leaf=20,
            subsample=0.85,
            random_state=42,
        )
        self.feature_columns = FEATURE_COLUMNS
        self._is_fitted = False
        self.metrics: dict[str, float] = {}

    def train(self, n_samples: int = 5000) -> dict[str, float]:
        """Train on synthetic market data and return validation metrics."""
        df = generate_training_data(n_samples=n_samples)
        X = df[self.feature_columns]
        y = df["reversed_win"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "baseline_win_rate": float(y.mean()),
            "model_win_rate_at_threshold": float((y_prob >= settings.min_ai_confidence).mean()),
        }
        return self.metrics

    def predict_reversal_probability(self, features: dict[str, float]) -> float:
        """Return probability that underdog token reverses and wins."""
        if not self._is_fitted:
            self.load_or_train()
        X = features_to_array(features)
        return float(self.model.predict_proba(X)[0, 1])

    def should_enter(self, features: dict[str, float]) -> tuple[bool, float]:
        """Decide whether to enter a reverse trade."""
        prob = self.predict_reversal_probability(features)
        return prob >= settings.min_ai_confidence, prob

    def feature_importance(self) -> pd.DataFrame:
        """Return ranked feature importances."""
        if not self._is_fitted:
            self.load_or_train()
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, path: str | None = None) -> None:
        path = path or settings.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "metrics": self.metrics}, path)

    def load(self, path: str | None = None) -> None:
        path = path or settings.model_path
        data = joblib.load(path)
        self.model = data["model"]
        self.metrics = data.get("metrics", {})
        self._is_fitted = True

    def load_or_train(self) -> None:
        path = Path(settings.model_path)
        if path.exists():
            self.load()
        else:
            self.train()
            self.save()
