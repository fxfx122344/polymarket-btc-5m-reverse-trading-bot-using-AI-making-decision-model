"""Paper trading portfolio tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PaperTrade:
    """Single paper trade record."""

    id: int
    timestamp: str
    market_slug: str
    outcome: str
    entry_price: float
    size_usdc: float
    shares: float
    ai_confidence: float
    won: bool | None = None
    pnl: float = 0.0
    resolved_at: str | None = None


@dataclass
class Portfolio:
    """Paper trading portfolio state."""

    initial_balance: float
    balance: float
    trades: list[PaperTrade] = field(default_factory=list)
    _next_id: int = 1

    @property
    def total_pnl(self) -> float:
        return self.balance - self.initial_balance

    @property
    def win_rate(self) -> float:
        resolved = [t for t in self.trades if t.won is not None]
        if not resolved:
            return 0.0
        return sum(1 for t in resolved if t.won) / len(resolved)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def roi_pct(self) -> float:
        return (self.total_pnl / self.initial_balance) * 100

    def open_trade(
        self,
        market_slug: str,
        outcome: str,
        entry_price: float,
        size_usdc: float,
        ai_confidence: float,
    ) -> PaperTrade | None:
        if size_usdc <= 0 or size_usdc > self.balance:
            return None
        shares = size_usdc / entry_price
        trade = PaperTrade(
            id=self._next_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_slug=market_slug,
            outcome=outcome,
            entry_price=entry_price,
            size_usdc=size_usdc,
            shares=shares,
            ai_confidence=ai_confidence,
        )
        self._next_id += 1
        self.balance -= size_usdc
        self.trades.append(trade)
        return trade

    def resolve_trade(self, trade_id: int, won: bool) -> float:
        trade = next((t for t in self.trades if t.id == trade_id), None)
        if trade is None or trade.won is not None:
            return 0.0

        if won:
            payout = trade.shares * 1.0
            trade.pnl = payout - trade.size_usdc
            self.balance += payout
        else:
            trade.pnl = -trade.size_usdc

        trade.won = won
        trade.resolved_at = datetime.now(timezone.utc).isoformat()
        return trade.pnl
