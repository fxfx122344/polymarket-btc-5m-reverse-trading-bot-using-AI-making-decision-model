"""Feature engineering for AI reversal prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.polymarket.client import OrderBookSnapshot
from src.polymarket.orderbook import compute_depth_ratio

FEATURE_COLUMNS = [
    "entry_price",
    "spread",
    "imbalance",
    "depth_ratio",
    "price_distance_from_50",
    "implied_payout",
    "bid_depth",
    "ask_depth",
    "momentum_1m",
    "momentum_5m",
    "volatility_5m",
    "time_to_expiry_min",
    "opposite_price",
    "book_pressure",
]


def extract_features(
    book: OrderBookSnapshot,
    outcome: str,
    opposite_price: float = 0.5,
    momentum_1m: float = 0.0,
    momentum_5m: float = 0.0,
    volatility_5m: float = 0.02,
    time_to_expiry_min: float = 2.5,
) -> dict[str, float]:
    """Extract feature vector from order book snapshot."""
    entry = book.best_ask if book.asks else book.mid_price
    implied_payout = 1.0 / entry if entry > 0 else 0.0

    return {
        "entry_price": entry,
        "spread": book.spread,
        "imbalance": book.imbalance,
        "depth_ratio": compute_depth_ratio(book),
        "price_distance_from_50": abs(entry - 0.5),
        "implied_payout": implied_payout,
        "bid_depth": book.bid_depth,
        "ask_depth": book.ask_depth,
        "momentum_1m": momentum_1m,
        "momentum_5m": momentum_5m,
        "volatility_5m": volatility_5m,
        "time_to_expiry_min": time_to_expiry_min,
        "opposite_price": opposite_price,
        "book_pressure": book.imbalance * (0.5 - entry),
    }


def features_to_array(features: dict[str, float]) -> pd.DataFrame:
    """Convert feature dict to model input DataFrame."""
    return pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def generate_training_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data mimicking BTC 5m market dynamics.

    Labels represent whether the underdog token reversed and won.
    Features are engineered to correlate with reversal probability.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        entry_price = rng.uniform(0.15, 0.48)
        spread = rng.uniform(0.01, 0.05)
        imbalance = rng.uniform(-0.6, 0.6)
        depth_ratio = rng.uniform(0.3, 2.5)
        momentum_1m = rng.normal(0, 0.03)
        momentum_5m = rng.normal(0, 0.05)
        volatility = rng.uniform(0.01, 0.08)
        time_left = rng.uniform(0.5, 4.5)
        opposite = 1.0 - entry_price + rng.normal(0, 0.02)

        # Reversal probability model — conditions that favor underdog wins
        logit = (
            -1.2
            + 2.5 * (0.5 - entry_price)
            + 1.8 * imbalance
            + 0.6 * (depth_ratio - 1.0)
            + 1.2 * momentum_1m
            + 0.8 * momentum_5m
            - 2.0 * volatility
            + 0.3 * (2.5 - time_left)
            + rng.normal(0, 0.4)
        )
        prob = 1 / (1 + np.exp(-logit))
        reversed_win = int(rng.random() < prob)

        rows.append(
            {
                "entry_price": entry_price,
                "spread": spread,
                "imbalance": imbalance,
                "depth_ratio": depth_ratio,
                "price_distance_from_50": abs(entry_price - 0.5),
                "implied_payout": 1.0 / entry_price,
                "bid_depth": rng.uniform(100, 5000),
                "ask_depth": rng.uniform(100, 5000),
                "momentum_1m": momentum_1m,
                "momentum_5m": momentum_5m,
                "volatility_5m": volatility,
                "time_to_expiry_min": time_left,
                "opposite_price": opposite,
                "book_pressure": imbalance * (0.5 - entry_price),
                "reversed_win": reversed_win,
            }
        )

    return pd.DataFrame(rows)
