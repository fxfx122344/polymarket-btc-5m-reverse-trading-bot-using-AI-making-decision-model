"""Polymarket Gamma + CLOB client for live BTC 5m markets.

Read-only. This project never places orders.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from src.config import settings

WINDOW_SECONDS = 300


def _decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


@dataclass
class LiveMarket:
    """An open BTC 5m market."""

    slug: str
    end_ts: int
    up_token: str
    down_token: str
    volume: float

    @property
    def start_ts(self) -> int:
        return self.end_ts - WINDOW_SECONDS

    def seconds_remaining(self, now: float | None = None) -> float:
        return self.end_ts - (now if now is not None else time.time())


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Order book with the metrics the strategy cares about."""

    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    last_trade_price: float

    @property
    def best_bid(self) -> float:
        # Gamma returns bids ascending, so the best bid is last.
        return self.bids[-1].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        # Asks are returned descending, so the best ask is last.
        return self.asks[-1].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2
        return self.last_trade_price

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0


class PolymarketClient:
    """Minimal read-only client with retries."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3) -> None:
        self._client = httpx.Client(timeout=timeout)
        self._max_retries = max_retries

    def _get(self, url: str, params: dict):
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(url, params=params)
            except httpx.HTTPError:
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        return None

    def get_market_by_slug(self, slug: str) -> LiveMarket | None:
        """Look up a single market by its slug."""
        data = self._get(f"{settings.gamma_api_url}/markets", {"slug": slug, "limit": 1})
        if not data:
            return None
        market = data[0] if isinstance(data, list) else data
        tokens = _decode(market.get("clobTokenIds"))
        if not tokens or len(tokens) != 2:
            return None
        try:
            end_ts = int(str(slug).rsplit("-", 1)[1])
        except (ValueError, IndexError):
            return None
        return LiveMarket(
            slug=slug,
            end_ts=end_ts,
            up_token=str(tokens[0]),
            down_token=str(tokens[1]),
            volume=float(market.get("volumeNum") or 0.0),
        )

    def get_open_markets(self, limit: int = 3) -> list[LiveMarket]:
        """
        The next BTC 5m markets due to close, soonest first.

        Window boundaries are deterministic (every 300s) and slugs follow
        ``btc-updown-5m-<unix_close>``, so the imminent windows are addressed
        directly. Listing endpoints are unreliable here: ``closed=false`` also
        returns stale windows that were never settled, and paging by end date
        surfaces markets created up to 24 hours ahead.
        """
        now = int(time.time())
        next_close = ((now // WINDOW_SECONDS) + 1) * WINDOW_SECONDS

        markets: list[LiveMarket] = []
        for i in range(limit):
            end_ts = next_close + i * WINDOW_SECONDS
            market = self.get_market_by_slug(f"btc-updown-5m-{end_ts}")
            if market is not None:
                markets.append(market)
        return markets

    def get_price_path(self, token_id: str, start_ts: int, end_ts: int) -> list[dict]:
        """Observed price points for a token within a time window."""
        data = self._get(
            f"{settings.clob_api_url}/prices-history",
            {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": "1"},
        )
        if not data:
            return []
        return [
            {"t": int(p["t"]), "p": float(p["p"])}
            for p in data.get("history", [])
            if start_ts <= int(p["t"]) <= end_ts
        ]

    def get_orderbook(self, token_id: str) -> OrderBookSnapshot | None:
        data = self._get(f"{settings.clob_api_url}/book", {"token_id": token_id})
        if not data:
            return None
        return OrderBookSnapshot(
            token_id=token_id,
            bids=[OrderBookLevel(float(b["price"]), float(b["size"]))
                  for b in data.get("bids", [])],
            asks=[OrderBookLevel(float(a["price"]), float(a["size"]))
                  for a in data.get("asks", [])],
            last_trade_price=float(data.get("last_trade_price") or 0.0),
        )

    def get_resolution(self, slug: str) -> bool | None:
        """True if Up won, False if Down won, None if not settled yet."""
        data = self._get(f"{settings.gamma_api_url}/markets", {"slug": slug, "limit": 1})
        if not data:
            return None
        market = data[0] if isinstance(data, list) else data
        prices = _decode(market.get("outcomePrices"))
        if not prices or len(prices) != 2:
            return None
        try:
            up, down = float(prices[0]), float(prices[1])
        except (TypeError, ValueError):
            return None
        if {up, down} != {0.0, 1.0}:
            return None
        return up == 1.0

    def close(self) -> None:
        self._client.close()
