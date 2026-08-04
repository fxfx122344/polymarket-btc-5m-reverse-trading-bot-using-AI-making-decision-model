"""Order book analysis utilities for reverse trading signals."""

from __future__ import annotations

from dataclasses import dataclass

from src.polymarket.client import OrderBookSnapshot


@dataclass
class ReverseSignal:
    """Signal derived from order book for reverse trading."""

    token_id: str
    outcome: str
    entry_price: float
    implied_payout: float
    edge: float
    spread: float
    imbalance: float
    is_underdog: bool


def analyze_underdog(book: OrderBookSnapshot, outcome: str) -> ReverseSignal | None:
    """
    Identify underdog tokens (price < 0.5) with favorable reverse-trading edge.

    Reverse trading edge = (1 / entry_price) * win_prob - 1
    At 50% win rate with price 0.40: payout = 1/0.40 = 2.5x, breakeven at 40% WR.
    """
    entry = book.best_ask if book.asks else book.mid_price
    if entry <= 0 or entry >= 0.5:
        return None

    implied_payout = 1.0 / entry
    # Raw edge assuming 50% baseline — AI model refines this
    edge = implied_payout * 0.5 - 1.0

    return ReverseSignal(
        token_id=book.token_id,
        outcome=outcome,
        entry_price=entry,
        implied_payout=implied_payout,
        edge=edge,
        spread=book.spread,
        imbalance=book.imbalance,
        is_underdog=True,
    )


def compute_depth_ratio(book: OrderBookSnapshot, levels: int = 5) -> float:
    """Ratio of bid to ask depth in top N levels."""
    bid_depth = sum(l.size for l in book.bids[-levels:])
    ask_depth = sum(l.size for l in book.asks[-levels:])
    if ask_depth == 0:
        return 1.0
    return bid_depth / ask_depth
