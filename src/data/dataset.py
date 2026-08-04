"""Build a leakage-free dataset from real Polymarket BTC 5m price paths.

For each resolved market we freeze a decision moment at ``end_ts - decision_offset``
and use *only* price points at or before that moment. The label is the true
settled outcome, so nothing about the future leaks into the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.fetch import ResolvedMarket

# Features available at decision time. Deliberately excludes anything
# derived from the outcome or from prices after the decision moment.
REAL_FEATURE_COLUMNS = [
    "underdog_price",
    "price_move_in_window",
    "momentum_last_min",
    "path_volatility",
    "distance_from_50",
    "implied_payout",
    "seconds_to_expiry",
    "volume",
    "n_observations",
    "favorite_price",
    "max_underdog_drawdown",
    "underdog_trend",
]


def _price_at_or_before(path: list[dict], ts: int) -> float | None:
    """Most recent observed price at or before ``ts`` (no lookahead)."""
    eligible = [p for p in path if p["t"] <= ts]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p["t"])["p"]


def _series_up_to(path: list[dict], ts: int) -> list[dict]:
    return sorted((p for p in path if p["t"] <= ts), key=lambda p: p["t"])


def build_feature_row(
    underdog_price: float,
    favorite_price: float,
    underdog_prices: list[float],
    volume: float,
    seconds_to_expiry: float,
) -> dict[str, float]:
    """
    Assemble the model's feature vector.

    Shared by the backtest and the live paper trader so that training and
    serving compute features identically.
    """
    if len(underdog_prices) >= 2:
        price_move = underdog_prices[-1] - underdog_prices[0]
        momentum_last_min = underdog_prices[-1] - underdog_prices[-2]
        path_volatility = float(np.std(underdog_prices))
        running_max = np.maximum.accumulate(underdog_prices)
        max_drawdown = float(np.max(running_max - np.array(underdog_prices)))
        trend = float(np.polyfit(range(len(underdog_prices)), underdog_prices, 1)[0])
    else:
        price_move = 0.0
        momentum_last_min = 0.0
        path_volatility = 0.0
        max_drawdown = 0.0
        trend = 0.0

    return {
        "underdog_price": underdog_price,
        "favorite_price": favorite_price,
        "price_move_in_window": price_move,
        "momentum_last_min": momentum_last_min,
        "path_volatility": path_volatility,
        "distance_from_50": 0.5 - underdog_price,
        "implied_payout": 1.0 / underdog_price if underdog_price > 0 else 0.0,
        "seconds_to_expiry": float(seconds_to_expiry),
        "volume": volume,
        "n_observations": float(len(underdog_prices)),
        "max_underdog_drawdown": max_drawdown,
        "underdog_trend": trend,
    }


def build_dataset(
    markets: list[ResolvedMarket],
    paths: dict,
    decision_offset: int = 60,
    min_volume: float = 1000.0,
) -> pd.DataFrame:
    """
    Assemble one row per market describing the underdog side at decision time.

    Args:
        decision_offset: seconds before market close when the decision is made.
        min_volume: skip illiquid markets where prices are stale placeholders.
    """
    rows: list[dict] = []

    for market in markets:
        path_pair = paths.get(market.slug)
        if not path_pair:
            continue
        if market.volume < min_volume:
            continue

        decision_ts = market.end_ts - decision_offset
        up_path = path_pair.get("up") or []
        down_path = path_pair.get("down") or []

        up_price = _price_at_or_before(up_path, decision_ts)
        down_price = _price_at_or_before(down_path, decision_ts)

        # Binary market: the two sides are complementary, so one side is enough.
        if up_price is None and down_price is None:
            continue
        if up_price is None:
            up_price = 1.0 - down_price
        if down_price is None:
            down_price = 1.0 - up_price

        # Identify the underdog: the side trading below 0.50.
        if up_price < down_price:
            underdog_side = "Up"
            underdog_price = up_price
            favorite_price = down_price
            underdog_path = up_path
            underdog_won = market.up_won
        else:
            underdog_side = "Down"
            underdog_price = down_price
            favorite_price = up_price
            underdog_path = down_path
            underdog_won = not market.up_won

        if not (0.0 < underdog_price < 0.5):
            continue

        prices = [p["p"] for p in _series_up_to(underdog_path, decision_ts)]

        features = build_feature_row(
            underdog_price=underdog_price,
            favorite_price=favorite_price,
            underdog_prices=prices,
            volume=market.volume,
            seconds_to_expiry=float(decision_offset),
        )

        rows.append(
            {
                "slug": market.slug,
                "end_ts": market.end_ts,
                "underdog_side": underdog_side,
                **features,
                "underdog_won": int(underdog_won),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("end_ts").reset_index(drop=True)
    return df


def calibration_table(df: pd.DataFrame, bins: list[float] | None = None) -> pd.DataFrame:
    """
    Compare the market's implied probability against the realized win rate.

    This is the decisive test of the reverse-trading premise: an edge exists
    only where the realized win rate exceeds the entry price.
    """
    if df.empty:
        return pd.DataFrame()

    bins = bins or [0.0, 0.15, 0.25, 0.35, 0.42, 0.46, 0.50]
    df = df.copy()
    df["bucket"] = pd.cut(df["underdog_price"], bins=bins, include_lowest=True)

    grouped = df.groupby("bucket", observed=True).agg(
        n=("underdog_won", "size"),
        avg_entry_price=("underdog_price", "mean"),
        realized_win_rate=("underdog_won", "mean"),
    )
    grouped["implied_win_rate"] = grouped["avg_entry_price"]
    grouped["edge"] = grouped["realized_win_rate"] - grouped["implied_win_rate"]
    # Expected profit per $1 staked, before costs.
    grouped["ev_per_dollar"] = (
        grouped["realized_win_rate"] / grouped["avg_entry_price"] - 1.0
    )
    # Binomial standard error on the realized rate, to judge significance.
    grouped["std_error"] = np.sqrt(
        grouped["realized_win_rate"] * (1 - grouped["realized_win_rate"]) / grouped["n"]
    )
    return grouped.reset_index()
