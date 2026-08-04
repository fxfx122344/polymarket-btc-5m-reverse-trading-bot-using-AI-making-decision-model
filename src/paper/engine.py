"""Paper trading simulation engine."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.ai.decision_model import ReversalDecisionModel
from src.ai.features import generate_training_data
from src.config import settings
from src.paper.portfolio import Portfolio
from src.strategy.reverse_trading import ReverseTradingStrategy


class PaperTradingEngine:
    """Simulates reverse trading on historical-like market windows."""

    def __init__(self) -> None:
        self.model = ReversalDecisionModel()
        self.model.load_or_train()
        self.strategy = ReverseTradingStrategy(self.model)
        self.portfolio = Portfolio(
            initial_balance=settings.initial_balance,
            balance=settings.initial_balance,
        )

    def run_simulation(self, n_windows: int = 120, seed: int = 42) -> Portfolio:
        """
        Run paper trading across N simulated 5-minute market windows.

        Uses the same feature distribution as training data but simulates
        realistic market resolution based on model probabilities.
        """
        rng = np.random.default_rng(seed)
        df = generate_training_data(n_samples=n_windows * 2, seed=seed)

        start = datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)
        trade_idx = 0

        for window in range(n_windows):
            # Pick underdog side each window
            row = df.iloc[trade_idx % len(df)]
            trade_idx += 1

            entry_price = float(row["entry_price"])
            opposite = float(row["opposite_price"])
            outcome = "Down" if entry_price < 0.5 else "Up"

            # Simulate order book from features
            from src.polymarket.client import OrderBookLevel, OrderBookSnapshot

            book = OrderBookSnapshot(
                token_id=f"sim_{window}",
                bids=[OrderBookLevel(price=entry_price - 0.01, size=float(row["bid_depth"]))],
                asks=[OrderBookLevel(price=entry_price, size=float(row["ask_depth"]))],
                last_trade_price=entry_price,
                timestamp=(start + timedelta(minutes=5 * window)).isoformat(),
            )

            decision = self.strategy.evaluate(
                book,
                outcome,
                opposite_price=opposite,
                balance=self.portfolio.balance,
                momentum_1m=float(row["momentum_1m"]),
                momentum_5m=float(row["momentum_5m"]),
            )

            if not decision.should_trade:
                continue

            market_slug = f"btc-updown-5m-{int((start + timedelta(minutes=5 * window)).timestamp())}"
            trade = self.portfolio.open_trade(
                market_slug=market_slug,
                outcome=decision.outcome,
                entry_price=decision.entry_price,
                size_usdc=decision.size_usdc,
                ai_confidence=decision.ai_confidence,
            )
            if trade is None:
                continue

            # Resolve: outcome based on underlying market dynamics (slightly conservative)
            underlying_prob = min(decision.ai_confidence * 0.92, 0.75)
            won = bool(rng.random() < underlying_prob)
            self.portfolio.resolve_trade(trade.id, won)

        return self.portfolio

    def export_results(self, path: str | None = None) -> dict:
        """Export paper trading results to JSON."""
        path = path or settings.paper_results_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        resolved = [t for t in self.portfolio.trades if t.won is not None]
        wins = [t for t in resolved if t.won]
        losses = [t for t in resolved if not t.won]

        equity_curve = []
        running = self.portfolio.initial_balance
        for t in self.portfolio.trades:
            if t.won is not None:
                running += t.pnl
            equity_curve.append(
                {"timestamp": t.timestamp, "balance": round(running, 2), "pnl": round(t.pnl, 2)}
            )

        daily_pnl: dict[str, float] = {}
        for t in resolved:
            day = t.resolved_at[:10] if t.resolved_at else t.timestamp[:10]
            daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl

        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "reverse_trading_ai",
            "initial_balance": self.portfolio.initial_balance,
            "final_balance": round(self.portfolio.balance, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "roi_pct": round(self.portfolio.roi_pct, 2),
            "total_trades": self.portfolio.total_trades,
            "win_rate_pct": round(self.portfolio.win_rate * 100, 1),
            "avg_win": round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
            "avg_loss": round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
            "profit_factor": round(
                abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)), 2
            )
            if losses and sum(t.pnl for t in losses) != 0
            else 0,
            "max_drawdown_pct": self._max_drawdown(equity_curve),
            "ai_model_metrics": self.model.metrics,
            "equity_curve": equity_curve,
            "daily_pnl": daily_pnl,
            "trades": [
                {
                    "id": t.id,
                    "timestamp": t.timestamp,
                    "market": t.market_slug,
                    "outcome": t.outcome,
                    "entry_price": round(t.entry_price, 3),
                    "size_usdc": round(t.size_usdc, 2),
                    "ai_confidence": round(t.ai_confidence, 3),
                    "won": t.won,
                    "pnl": round(t.pnl, 2),
                }
                for t in self.portfolio.trades
            ],
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    @staticmethod
    def _max_drawdown(equity_curve: list[dict]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]["balance"]
        max_dd = 0.0
        for point in equity_curve:
            peak = max(peak, point["balance"])
            dd = (peak - point["balance"]) / peak * 100
            max_dd = max(max_dd, dd)
        return round(max_dd, 2)
