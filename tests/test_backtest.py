"""Tests for the backtest engine's payout math and cost handling."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, run_backtest


def make_df(prices, outcomes):
    return pd.DataFrame({
        "end_ts": list(range(1000, 1000 + 300 * len(prices), 300)),
        "underdog_price": prices,
        "underdog_won": outcomes,
    })


def test_winning_trade_pays_one_dollar_per_share():
    """A $10 stake at an executed 0.50 buys 20 shares, returning $20."""
    df = make_df([0.49], [1])
    result = run_backtest(df, BacktestConfig(stake=10.0, slippage=0.01,
                                             max_entry_price=0.499,
                                             min_ai_confidence=None))
    assert result.n_trades == 1
    assert result.trades.iloc[0]["exec_price"] == pytest.approx(0.50)
    assert result.trades.iloc[0]["pnl"] == pytest.approx(10.0)


def test_losing_trade_loses_the_stake():
    df = make_df([0.40], [0])
    result = run_backtest(df, BacktestConfig(stake=10.0, slippage=0.0,
                                             min_ai_confidence=None))
    assert result.trades.iloc[0]["pnl"] == pytest.approx(-10.0)


def test_slippage_raises_the_breakeven_win_rate():
    df = make_df([0.40] * 10, [1, 0] * 5)
    free = run_backtest(df, BacktestConfig(slippage=0.0, min_ai_confidence=None))
    costly = run_backtest(df, BacktestConfig(slippage=0.05, min_ai_confidence=None))
    assert costly.stats["breakeven_win_rate"] > free.stats["breakeven_win_rate"]
    assert costly.stats["total_pnl"] < free.stats["total_pnl"]


def test_entry_band_filters_out_of_range_prices():
    df = make_df([0.10, 0.30, 0.49], [1, 1, 1])
    result = run_backtest(
        df, BacktestConfig(min_entry_price=0.15, max_entry_price=0.40,
                           min_ai_confidence=None)
    )
    assert result.n_trades == 1
    assert result.trades.iloc[0]["underdog_price"] == pytest.approx(0.30)


def test_outcomes_are_never_taken_from_model_probabilities():
    """High model confidence on losing markets must still lose money."""
    df = make_df([0.30] * 8, [0] * 8)
    prob = pd.Series([0.99] * 8, index=df.index)
    result = run_backtest(df, BacktestConfig(min_ai_confidence=0.5, stake=10.0),
                          probabilities=prob)
    assert result.n_trades == 8
    assert result.stats["win_rate"] == 0.0
    assert result.stats["total_pnl"] == pytest.approx(-80.0)


def test_rows_without_predictions_are_not_traded():
    df = make_df([0.30] * 4, [1, 1, 1, 1])
    prob = pd.Series([np.nan, 0.8, np.nan, 0.9], index=df.index)
    result = run_backtest(df, BacktestConfig(min_ai_confidence=0.5), probabilities=prob)
    assert result.n_trades == 2


def test_breakeven_win_rate_yields_roughly_zero_profit():
    """Winning exactly at the executed price should be about break-even."""
    outcomes = [1] * 40 + [0] * 60  # 40% win rate
    df = make_df([0.40] * 100, outcomes)
    result = run_backtest(df, BacktestConfig(slippage=0.0, stake=10.0,
                                             min_ai_confidence=None))
    assert result.stats["mean_return_per_dollar"] == pytest.approx(0.0, abs=1e-9)
    assert result.stats["edge_vs_breakeven"] == pytest.approx(0.0, abs=1e-9)


def test_confidence_interval_flags_insignificant_results():
    rng = np.random.default_rng(0)
    outcomes = (rng.random(200) < 0.40).astype(int)
    df = make_df([0.40] * 200, outcomes)
    result = run_backtest(df, BacktestConfig(slippage=0.0, min_ai_confidence=None))
    # A fair coin-flip strategy must not be reported as significant.
    assert result.stats["significant_at_95"] is False
    assert result.stats["ci95_low"] < 0 < result.stats["ci95_high"]


def test_empty_input_returns_no_trades():
    result = run_backtest(pd.DataFrame(), BacktestConfig())
    assert result.n_trades == 0
