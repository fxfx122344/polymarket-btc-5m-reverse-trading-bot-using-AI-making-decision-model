"""Tests for dataset construction, especially leakage protection."""

import pandas as pd
import pytest

from src.data.dataset import (
    _price_at_or_before,
    build_dataset,
    build_feature_row,
    calibration_table,
)
from src.data.fetch import ResolvedMarket


def make_market(slug="btc-updown-5m-1000000", end_ts=1000000, up_won=True, volume=50000.0):
    return ResolvedMarket(
        slug=slug,
        end_ts=end_ts,
        up_token="up",
        down_token="down",
        up_won=up_won,
        volume=volume,
    )


def test_price_at_or_before_ignores_future_points():
    path = [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.6}, {"t": 300, "p": 0.9}]
    assert _price_at_or_before(path, 250) == 0.6
    assert _price_at_or_before(path, 100) == 0.4
    assert _price_at_or_before(path, 50) is None


def test_build_dataset_excludes_post_decision_prices():
    """A price spike after the decision moment must not affect features."""
    market = make_market(end_ts=1000, up_won=True)
    # Decision at 1000 - 60 = 940. The 0.95 point at t=980 is in the future.
    paths = {
        market.slug: {
            "up": [{"t": 800, "p": 0.30}, {"t": 900, "p": 0.35}, {"t": 980, "p": 0.95}],
            "down": [{"t": 800, "p": 0.70}, {"t": 900, "p": 0.65}, {"t": 980, "p": 0.05}],
        }
    }
    df = build_dataset([market], paths, decision_offset=60, min_volume=0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["underdog_side"] == "Up"
    assert row["underdog_price"] == pytest.approx(0.35)
    # Only the two pre-decision points inform the features.
    assert row["n_observations"] == 2
    assert row["price_move_in_window"] == pytest.approx(0.05)


def test_underdog_is_the_side_below_half():
    market = make_market(end_ts=1000, up_won=False)
    paths = {
        market.slug: {
            "up": [{"t": 900, "p": 0.80}],
            "down": [{"t": 900, "p": 0.20}],
        }
    }
    df = build_dataset([market], paths, decision_offset=60, min_volume=0)
    row = df.iloc[0]
    assert row["underdog_side"] == "Down"
    assert row["underdog_price"] == pytest.approx(0.20)
    # Down won, and Down was the underdog.
    assert row["underdog_won"] == 1


def test_label_follows_actual_resolution():
    """The label must come from settlement, not from prices."""
    market = make_market(end_ts=1000, up_won=True)
    paths = {market.slug: {"up": [{"t": 900, "p": 0.10}], "down": [{"t": 900, "p": 0.90}]}}
    df = build_dataset([market], paths, decision_offset=60, min_volume=0)
    # Up was a heavy underdog at 0.10 but actually won.
    assert df.iloc[0]["underdog_side"] == "Up"
    assert df.iloc[0]["underdog_won"] == 1


def test_low_volume_markets_are_skipped():
    market = make_market(end_ts=1000, volume=10.0)
    paths = {market.slug: {"up": [{"t": 900, "p": 0.3}], "down": [{"t": 900, "p": 0.7}]}}
    assert build_dataset([market], paths, decision_offset=60, min_volume=1000).empty


def test_missing_side_is_derived_from_complement():
    market = make_market(end_ts=1000, up_won=True)
    paths = {market.slug: {"up": [{"t": 900, "p": 0.30}], "down": []}}
    df = build_dataset([market], paths, decision_offset=60, min_volume=0)
    assert df.iloc[0]["favorite_price"] == pytest.approx(0.70)


def test_non_underdog_markets_are_excluded():
    """Both sides at exactly 0.50 leaves no underdog to buy."""
    market = make_market(end_ts=1000)
    paths = {market.slug: {"up": [{"t": 900, "p": 0.50}], "down": [{"t": 900, "p": 0.50}]}}
    assert build_dataset([market], paths, decision_offset=60, min_volume=0).empty


def test_feature_row_handles_single_observation():
    features = build_feature_row(0.4, 0.6, [0.4], 1000.0, 60)
    assert features["price_move_in_window"] == 0.0
    assert features["path_volatility"] == 0.0
    assert features["implied_payout"] == pytest.approx(2.5)


def test_calibration_edge_is_realized_minus_implied():
    df = pd.DataFrame({
        "underdog_price": [0.30, 0.30, 0.30, 0.30],
        "underdog_won": [1, 1, 0, 0],
    })
    table = calibration_table(df, bins=[0.0, 0.5])
    row = table.iloc[0]
    assert row["realized_win_rate"] == pytest.approx(0.5)
    assert row["implied_win_rate"] == pytest.approx(0.3)
    assert row["edge"] == pytest.approx(0.2)
    # Buying at 0.30 and winning half the time returns 0.5/0.3 - 1.
    assert row["ev_per_dollar"] == pytest.approx(0.6667, abs=1e-3)
