"""Polymarket Gamma and CLOB API client."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from src.config import settings


@dataclass
class MarketOutcome:
    """Single outcome in a BTC 5m market."""

    token_id: str
    outcome: str  # "Up" or "Down"
    price: float
    market_slug: str
    end_date: str


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Parsed order book with computed metrics."""

    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    last_trade_price: float
    timestamp: str

    @property
    def best_bid(self) -> float:
        return self.bids[-1].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[-1].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2
        return self.last_trade_price

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)

    @property
    def imbalance(self) -> float:
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


class PolymarketClient:
    """Read-only client for Polymarket market and orderbook data."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0)

    def get_btc_5m_markets(self) -> list[MarketOutcome]:
        """Fetch active BTC 5-minute up/down markets."""
        try:
            resp = self._client.get(
                f"{settings.gamma_api_url}/events",
                params={"slug": settings.btc_series_slug, "closed": "false", "limit": 5},
            )
            resp.raise_for_status()
            events = resp.json()
        except httpx.HTTPError:
            return []

        outcomes: list[MarketOutcome] = []
        for event in events:
            for market in event.get("markets", []):
                token_ids = market.get("clobTokenIds") or []
                outcome_names = market.get("outcomes") or []
                prices = market.get("outcomePrices") or []

                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if isinstance(outcome_names, str):
                    outcome_names = json.loads(outcome_names)
                if isinstance(prices, str):
                    prices = json.loads(prices)

                for token_id, name, price in zip(token_ids, outcome_names, prices):
                    outcomes.append(
                        MarketOutcome(
                            token_id=str(token_id),
                            outcome=str(name),
                            price=float(price),
                            market_slug=market.get("slug", ""),
                            end_date=market.get("endDate", ""),
                        )
                    )
        return outcomes

    def get_orderbook(self, token_id: str) -> OrderBookSnapshot | None:
        """Fetch order book for a token."""
        try:
            resp = self._client.get(
                f"{settings.clob_api_url}/book",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return None

        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        return OrderBookSnapshot(
            token_id=token_id,
            bids=bids,
            asks=asks,
            last_trade_price=float(data.get("last_trade_price") or 0),
            timestamp=str(data.get("timestamp", "")),
        )

    def close(self) -> None:
        self._client.close()
