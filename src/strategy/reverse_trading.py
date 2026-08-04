"""Reverse trading strategy for Polymarket BTC 5m markets."""

from __future__ import annotations

from dataclasses import dataclass

from src.ai.decision_model import ReversalDecisionModel
from src.ai.features import extract_features
from src.config import settings
from src.polymarket.client import OrderBookSnapshot
from src.polymarket.orderbook import analyze_underdog


@dataclass
class TradeDecision:
    """Output of strategy evaluation."""

    should_trade: bool
    outcome: str
    token_id: str
    entry_price: float
    size_usdc: float
    ai_confidence: float
    expected_payout: float
    reason: str


class ReverseTradingStrategy:
    """
    Buy underdog tokens (price < 0.5) when AI model predicts reversal.

    Profit math: buy at price P, win pays 1/P shares worth $1 each.
    Expected value = win_rate * (1/P - 1) - (1 - win_rate) * 1
    Profitable when win_rate > P (e.g., 40% WR at P=0.40 breaks even).
    """

    def __init__(self, model: ReversalDecisionModel | None = None) -> None:
        self.model = model or ReversalDecisionModel()

    def evaluate(
        self,
        book: OrderBookSnapshot,
        outcome: str,
        opposite_price: float,
        balance: float,
        momentum_1m: float = 0.0,
        momentum_5m: float = 0.0,
    ) -> TradeDecision:
        signal = analyze_underdog(book, outcome)
        if signal is None:
            return TradeDecision(
                should_trade=False,
                outcome=outcome,
                token_id=book.token_id,
                entry_price=book.mid_price,
                size_usdc=0,
                ai_confidence=0,
                expected_payout=0,
                reason="Price not in underdog range (< 0.5)",
            )

        entry = signal.entry_price
        if entry < settings.min_token_price or entry > settings.max_token_price:
            return TradeDecision(
                should_trade=False,
                outcome=outcome,
                token_id=book.token_id,
                entry_price=entry,
                size_usdc=0,
                ai_confidence=0,
                expected_payout=0,
                reason=f"Price {entry:.3f} outside bounds [{settings.min_token_price}, {settings.max_token_price}]",
            )

        features = extract_features(
            book,
            outcome,
            opposite_price=opposite_price,
            momentum_1m=momentum_1m,
            momentum_5m=momentum_5m,
        )
        should_enter, confidence = self.model.should_enter(features)

        size = min(settings.max_position_size, balance * 0.03)
        expected_payout = confidence * (1.0 / entry - 1.0) - (1.0 - confidence)

        reason = (
            f"AI reversal confidence {confidence:.1%} — enter underdog {outcome} @ {entry:.3f}"
            if should_enter
            else f"AI confidence {confidence:.1%} below threshold {settings.min_ai_confidence:.1%}"
        )

        return TradeDecision(
            should_trade=should_enter,
            outcome=outcome,
            token_id=book.token_id,
            entry_price=entry,
            size_usdc=size if should_enter else 0,
            ai_confidence=confidence,
            expected_payout=expected_payout,
            reason=reason,
        )
