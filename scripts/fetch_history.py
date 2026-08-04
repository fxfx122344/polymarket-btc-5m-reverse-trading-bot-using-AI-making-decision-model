#!/usr/bin/env python3
"""Download real resolved Polymarket BTC 5m market history."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetch import download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-markets", type=int, default=3000)
    args = parser.parse_args()

    markets, paths = download(max_markets=args.max_markets)

    if markets:
        span_start = min(m.end_ts for m in markets)
        span_end = max(m.end_ts for m in markets)
        hours = (span_end - span_start) / 3600
        print(f"\nMarkets: {len(markets)} spanning {hours:.1f} hours")
        print(f"Price paths cached: {len(paths)}")


if __name__ == "__main__":
    main()
