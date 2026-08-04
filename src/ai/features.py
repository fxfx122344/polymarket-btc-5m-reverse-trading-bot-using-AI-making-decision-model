"""Features used by the reversal decision model.

Every feature is computed from real Polymarket price data observed at or before
the decision moment. See ``src/data/dataset.py`` for how they are assembled.
"""

from __future__ import annotations

import pandas as pd

from src.data.dataset import REAL_FEATURE_COLUMNS

FEATURE_COLUMNS = REAL_FEATURE_COLUMNS


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    """Convert a feature dict into a single-row frame with the model's columns."""
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
