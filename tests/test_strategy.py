"""Tests for strategy entry rules and the model's no-leakage guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.ai.decision_model import ReversalDecisionModel
from src.config import settings
from src.data.dataset import build_feature_row
from src.strategy.reverse_trading import ReverseTradingStrategy


def features(price: float) -> dict:
    return build_feature_row(price, 1 - price, [price, price], 50000.0, 60)


def test_rejects_prices_at_or_above_half():
    strategy = ReverseTradingStrategy(model=None)
    decision = strategy.evaluate(features(0.55), "Up", balance=1000)
    assert not decision.should_trade
    assert "underdog" in decision.reason


def test_rejects_prices_outside_band():
    strategy = ReverseTradingStrategy(model=None)
    below = strategy.evaluate(features(0.05), "Up", balance=1000)
    above = strategy.evaluate(features(0.495), "Up", balance=1000)
    assert not below.should_trade
    assert not above.should_trade
    assert "band" in above.reason


def test_accepts_price_inside_band_without_model():
    strategy = ReverseTradingStrategy(model=None)
    decision = strategy.evaluate(features(0.30), "Down", balance=1000)
    assert decision.should_trade
    assert decision.ai_confidence is None
    assert decision.exec_price == pytest.approx(0.30 + settings.slippage)


def test_breakeven_rate_equals_executed_price():
    strategy = ReverseTradingStrategy(model=None)
    decision = strategy.evaluate(features(0.25), "Up", balance=1000)
    assert decision.breakeven_win_rate == pytest.approx(decision.exec_price)


def test_rejects_when_balance_is_empty():
    strategy = ReverseTradingStrategy(model=None)
    decision = strategy.evaluate(features(0.30), "Up", balance=0)
    assert not decision.should_trade


def _training_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    prices = rng.uniform(0.2, 0.49, n)
    rows = [build_feature_row(p, 1 - p, [p, p + 0.01], 50000.0, 60) for p in prices]
    df = pd.DataFrame(rows)
    df["end_ts"] = np.arange(1000, 1000 + 300 * n, 300)
    df["underdog_won"] = (rng.random(n) < 0.5).astype(int)
    return df


def test_walk_forward_leaves_early_rows_unpredicted():
    """Forward chaining cannot predict the first fold, so it stays unscored."""
    df = _training_frame()
    model = ReversalDecisionModel()
    preds = model.out_of_sample_predictions(df, n_splits=5)
    assert preds.isna().sum() > 0
    assert preds.notna().sum() < len(df)
    # The unpredicted rows are the earliest ones.
    assert preds.iloc[0] != preds.iloc[0]  # NaN


def test_walk_forward_auc_near_chance_on_random_labels():
    df = _training_frame()
    model = ReversalDecisionModel()
    metrics = model.evaluate_walk_forward(df, n_splits=5)
    assert 0.35 < metrics["roc_auc"] < 0.65


def test_unfitted_model_refuses_to_predict():
    model = ReversalDecisionModel()
    with pytest.raises(RuntimeError):
        model.predict_reversal_probability(features(0.3))


def test_model_roundtrips_through_disk(tmp_path):
    df = _training_frame(n=120)
    model = ReversalDecisionModel()
    model.fit(df)
    expected = model.predict_reversal_probability(features(0.35))

    path = tmp_path / "model.pkl"
    model.save(str(path))

    reloaded = ReversalDecisionModel()
    reloaded.load(str(path))
    assert reloaded.predict_reversal_probability(features(0.35)) == pytest.approx(expected)
