"""Fetch real resolved Polymarket BTC 5m markets and their price paths.

All data here comes from Polymarket's public APIs:
  - Gamma  : resolved market metadata + the true outcome
  - CLOB   : historical price path inside each 5-minute window

Results are cached on disk so a backtest can be reproduced without re-downloading.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
BTC_5M_SERIES_ID = "10684"

RAW_DIR = Path("data/raw")
MARKETS_FILE = RAW_DIR / "markets.json"
PATHS_FILE = RAW_DIR / "price_paths.json"

WINDOW_SECONDS = 300


@dataclass
class ResolvedMarket:
    """A resolved BTC 5m market with its true outcome."""

    slug: str
    end_ts: int
    up_token: str
    down_token: str
    up_won: bool
    volume: float

    @property
    def start_ts(self) -> int:
        return self.end_ts - WINDOW_SECONDS


def _decode(value):
    """Gamma returns some fields as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


class HistoryFetcher:
    """Downloads resolved markets and price paths, with disk caching."""

    def __init__(self, timeout: float = 60.0, max_retries: int = 4) -> None:
        self._client = httpx.Client(timeout=timeout)
        self._max_retries = max_retries

    def _get(self, url: str, params: dict) -> httpx.Response | None:
        """GET with backoff on rate limits and transient errors."""
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(url, params=params)
            except httpx.HTTPError:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        return None

    def fetch_resolved_markets(self, max_markets: int = 3000) -> list[ResolvedMarket]:
        """Page through closed BTC 5m events and extract resolved markets."""
        markets: list[ResolvedMarket] = []
        seen: set[str] = set()
        offset = 0
        page_size = 100

        while len(markets) < max_markets:
            resp = self._get(
                f"{GAMMA_URL}/events",
                {
                    "series_id": BTC_5M_SERIES_ID,
                    "closed": "true",
                    "limit": page_size,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
            )
            if resp is None:
                break
            events = resp.json()
            if not events:
                break

            for event in events:
                for market in event.get("markets") or []:
                    slug = market.get("slug")
                    if not slug or slug in seen:
                        continue

                    outcomes = _decode(market.get("outcomes"))
                    prices = _decode(market.get("outcomePrices"))
                    tokens = _decode(market.get("clobTokenIds"))

                    # Need a clean binary Up/Down market with a settled outcome.
                    if not (outcomes and prices and tokens):
                        continue
                    if len(tokens) != 2 or len(prices) != 2:
                        continue
                    if outcomes[0] != "Up" or outcomes[1] != "Down":
                        continue

                    try:
                        up_price, down_price = float(prices[0]), float(prices[1])
                        end_ts = int(str(slug).rsplit("-", 1)[1])
                    except (ValueError, IndexError):
                        continue

                    # Settled markets pay exactly 0 or 1; anything else is unresolved.
                    if {up_price, down_price} != {0.0, 1.0}:
                        continue

                    seen.add(slug)
                    markets.append(
                        ResolvedMarket(
                            slug=slug,
                            end_ts=end_ts,
                            up_token=str(tokens[0]),
                            down_token=str(tokens[1]),
                            up_won=up_price == 1.0,
                            volume=float(market.get("volumeNum") or 0.0),
                        )
                    )

            offset += page_size

        return markets[:max_markets]

    def fetch_price_path(self, token_id: str, start_ts: int, end_ts: int) -> list[dict]:
        """Price points for a token inside [start_ts, end_ts]."""
        resp = self._get(
            f"{CLOB_URL}/prices-history",
            {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": "1"},
        )
        if resp is None:
            return []
        history = resp.json().get("history", [])
        return [
            {"t": int(p["t"]), "p": float(p["p"])}
            for p in history
            if start_ts <= int(p["t"]) <= end_ts
        ]

    def close(self) -> None:
        self._client.close()


def download(max_markets: int = 3000, verbose: bool = True) -> tuple[list[ResolvedMarket], dict]:
    """Download markets + price paths for both outcome tokens and cache to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = HistoryFetcher()

    if verbose:
        print(f"Fetching resolved BTC 5m markets (target {max_markets})...")
    markets = fetcher.fetch_resolved_markets(max_markets=max_markets)
    if verbose:
        print(f"  got {len(markets)} resolved markets")

    paths: dict[str, dict] = {}
    for i, market in enumerate(markets, start=1):
        # Fetch both sides: the underdog is whichever side is below 0.50.
        up_path = fetcher.fetch_price_path(market.up_token, market.start_ts, market.end_ts)
        down_path = fetcher.fetch_price_path(market.down_token, market.start_ts, market.end_ts)
        if up_path or down_path:
            paths[market.slug] = {"up": up_path, "down": down_path}
        if verbose and i % 200 == 0:
            print(f"  price paths: {i}/{len(markets)}")

    fetcher.close()

    with open(MARKETS_FILE, "w") as f:
        json.dump([asdict(m) for m in markets], f)
    with open(PATHS_FILE, "w") as f:
        json.dump(paths, f)

    if verbose:
        print(f"  cached {len(paths)} price paths to {PATHS_FILE}")

    return markets, paths


def load_cached() -> tuple[list[ResolvedMarket], dict]:
    """Load previously downloaded data."""
    if not MARKETS_FILE.exists() or not PATHS_FILE.exists():
        raise FileNotFoundError(
            "No cached data. Run `python scripts/fetch_history.py` first."
        )
    with open(MARKETS_FILE) as f:
        markets = [ResolvedMarket(**m) for m in json.load(f)]
    with open(PATHS_FILE) as f:
        paths = json.load(f)
    return markets, paths
