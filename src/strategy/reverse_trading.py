"""Reverse trading strategy: buy the underdog when a reversal looks likely.

Profit mechanics: buying at price P and winning pays $1.00 per share, a payout
of 1/P. The trade is therefore profitable only when the realized win rate
exceeds the price actually paid, including the spread crossed on entry.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ai.decision_model import ReversalDecisionModel
from src.config import settings


@dataclass
class TradeDecision:
    """Result of evaluating one market."""

    should_trade: bool
    outcome: str
    entry_price: float
    exec_price: float
    size_usdc: float
    ai_confidence: float | None
    breakeven_win_rate: float
    reason: str


class ReverseTradingStrategy:
    """Applies the entry band and, when available, the AI confidence filter."""

    def __init__(self, model: ReversalDecisionModel | None = None) -> None:
        self.model = model

    def evaluate(
        self,
        features: dict[str, float],
        underdog_side: str,
        balance: float,
    ) -> TradeDecision:
        entry = features["underdog_price"]
        exec_price = entry + settings.slippage

        def reject(reason: str, confidence: float | None = None) -> TradeDecision:
            return TradeDecision(
                should_trade=False,
                outcome=underdog_side,
                entry_price=entry,
                exec_price=exec_price,
                size_usdc=0.0,
                ai_confidence=confidence,
                breakeven_win_rate=exec_price,
                reason=reason,
            )

        if not 0.0 < entry < 0.5:
            return reject(f"price {entry:.3f} is not an underdog (must be < 0.50)")
        if entry < settings.min_token_price or entry > settings.max_token_price:
            return reject(
                f"price {entry:.3f} outside band "
                f"[{settings.min_token_price:.2f}, {settings.max_token_price:.2f}]"
            )
        if exec_price >= 1.0:
            return reject("execution price would exceed $1.00")

        size = min(settings.stake, balance)
        if size <= 0:
            return reject("insufficient balance")

        confidence: float | None = None
        if self.model is not None:
            confidence = self.model.predict_reversal_probability(features)
            if confidence < settings.min_ai_confidence:
                return reject(
                    f"AI confidence {confidence:.1%} below threshold "
                    f"{settings.min_ai_confidence:.1%}",
                    confidence,
                )

        return TradeDecision(
            should_trade=True,
            outcome=underdog_side,
            entry_price=entry,
            exec_price=exec_price,
            size_usdc=size,
            ai_confidence=confidence,
            breakeven_win_rate=exec_price,
            reason=(
                f"underdog {underdog_side} at {entry:.3f} "
                f"(pays {1 / exec_price:.2f}x, needs {exec_price:.1%} win rate)"
            ),
        )
