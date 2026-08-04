"""Live paper trading against real Polymarket BTC 5m markets.

No orders are ever placed. Each paper trade is opened from the real quoted
price at the decision moment and settled from the market's actual resolution,
so recorded results reflect genuine market outcomes.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.ai.decision_model import ReversalDecisionModel
from src.config import settings
from src.data.dataset import build_feature_row
from src.paper.portfolio import Portfolio
from src.polymarket.client import LiveMarket, PolymarketClient
from src.strategy.reverse_trading import ReverseTradingStrategy


class LivePaperTrader:
    """Polls open markets, decides at the configured offset, settles on resolution."""

    def __init__(self, model: ReversalDecisionModel | None = None) -> None:
        self.client = PolymarketClient()
        self.model = model
        self.strategy = ReverseTradingStrategy(model)
        self.portfolio = Portfolio(
            initial_balance=settings.initial_balance,
            balance=settings.initial_balance,
        )
        self._pending: dict[str, int] = {}  # market slug -> paper trade id
        self._handled: set[str] = set()

    def observe_market(self, market: LiveMarket) -> dict | None:
        """
        Build the feature vector for a market from real quoted prices.

        Returns None when the market has no usable quotes yet.
        """
        up_path = self.client.get_price_path(market.up_token, market.start_ts, market.end_ts)
        down_path = self.client.get_price_path(market.down_token, market.start_ts, market.end_ts)

        up_price = up_path[-1]["p"] if up_path else None
        down_price = down_path[-1]["p"] if down_path else None
        if up_price is None and down_price is None:
            return None
        if up_price is None:
            up_price = 1.0 - down_price
        if down_price is None:
            down_price = 1.0 - up_price

        if up_price < down_price:
            side, underdog_price, favorite_price = "Up", up_price, down_price
            prices = [p["p"] for p in up_path]
        else:
            side, underdog_price, favorite_price = "Down", down_price, up_price
            prices = [p["p"] for p in down_path]

        if not 0.0 < underdog_price < 0.5:
            return None

        features = build_feature_row(
            underdog_price=underdog_price,
            favorite_price=favorite_price,
            underdog_prices=prices,
            volume=market.volume,
            seconds_to_expiry=max(market.seconds_remaining(), 0.0),
        )
        return {"side": side, "features": features}

    def try_enter(self, market: LiveMarket) -> dict | None:
        """Evaluate a market once and record a paper trade if it qualifies."""
        if market.slug in self._handled:
            return None

        observation = self.observe_market(market)
        if observation is None:
            return None

        decision = self.strategy.evaluate(
            observation["features"], observation["side"], self.portfolio.balance
        )
        self._handled.add(market.slug)

        record = {
            "slug": market.slug,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "side": decision.outcome,
            "entry_price": round(decision.entry_price, 4),
            "exec_price": round(decision.exec_price, 4),
            "ai_confidence": (
                round(decision.ai_confidence, 4) if decision.ai_confidence is not None else None
            ),
            "should_trade": decision.should_trade,
            "reason": decision.reason,
        }

        if decision.should_trade:
            trade = self.portfolio.open_trade(
                market_slug=market.slug,
                outcome=decision.outcome,
                entry_price=decision.exec_price,
                size_usdc=decision.size_usdc,
                ai_confidence=decision.ai_confidence or 0.0,
            )
            if trade is not None:
                self._pending[market.slug] = trade.id
                record["trade_id"] = trade.id

        return record

    def settle_pending(self) -> list[dict]:
        """Settle open paper trades whose markets have resolved."""
        settled = []
        for slug, trade_id in list(self._pending.items()):
            up_won = self.client.get_resolution(slug)
            if up_won is None:
                continue
            trade = next((t for t in self.portfolio.trades if t.id == trade_id), None)
            if trade is None:
                self._pending.pop(slug, None)
                continue
            won = up_won if trade.outcome == "Up" else not up_won
            pnl = self.portfolio.resolve_trade(trade_id, won)
            self._pending.pop(slug, None)
            settled.append({"slug": slug, "won": won, "pnl": round(pnl, 2)})
        return settled

    def run(self, max_markets: int = 5, poll_seconds: float = 15.0, verbose: bool = True) -> dict:
        """
        Trade the next ``max_markets`` markets as they close.

        Each 5-minute market takes real time to resolve, so this runs for
        roughly ``max_markets * 5`` minutes.
        """
        decisions: list[dict] = []
        deadline_markets = 0

        while deadline_markets < max_markets:
            open_markets = self.client.get_open_markets(limit=5)
            open_markets = [m for m in open_markets if m.slug not in self._handled]

            now = time.time()
            for market in open_markets:
                remaining = market.seconds_remaining(now)
                # Decide once inside the entry window, before the market closes.
                if 0 < remaining <= settings.decision_offset_seconds:
                    record = self.try_enter(market)
                    if record:
                        decisions.append(record)
                        deadline_markets += 1
                        if verbose:
                            status = "ENTER" if record["should_trade"] else "skip "
                            print(f"  [{status}] {record['slug']} {record['side']} "
                                  f"@ {record['entry_price']:.3f} — {record['reason']}")

            for s in self.settle_pending():
                if verbose:
                    outcome = "WIN " if s["won"] else "LOSS"
                    print(f"  [{outcome}] {s['slug']} pnl={s['pnl']:+.2f} "
                          f"balance={self.portfolio.balance:.2f}")

            time.sleep(poll_seconds)

        # Give the final markets a chance to resolve.
        for _ in range(8):
            if not self._pending:
                break
            time.sleep(poll_seconds)
            for s in self.settle_pending():
                if verbose:
                    outcome = "WIN " if s["won"] else "LOSS"
                    print(f"  [{outcome}] {s['slug']} pnl={s['pnl']:+.2f}")

        return self.export(decisions)

    def export(self, decisions: list[dict]) -> dict:
        """Persist paper trading state."""
        resolved = [t for t in self.portfolio.trades if t.won is not None]
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "live_paper_trading",
            "initial_balance": self.portfolio.initial_balance,
            "balance": round(self.portfolio.balance, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "roi_pct": round(self.portfolio.roi_pct, 2),
            "trades_opened": self.portfolio.total_trades,
            "trades_settled": len(resolved),
            "win_rate_pct": round(self.portfolio.win_rate * 100, 1) if resolved else None,
            "decisions": decisions,
            "trades": [
                {
                    "id": t.id,
                    "slug": t.market_slug,
                    "side": t.outcome,
                    "exec_price": round(t.entry_price, 4),
                    "size_usdc": round(t.size_usdc, 2),
                    "ai_confidence": round(t.ai_confidence, 4),
                    "won": t.won,
                    "pnl": round(t.pnl, 2),
                }
                for t in self.portfolio.trades
            ],
        }
        path = Path(settings.paper_trades_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def close(self) -> None:
        self.client.close()
