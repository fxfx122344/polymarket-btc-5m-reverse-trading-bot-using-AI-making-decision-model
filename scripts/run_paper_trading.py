#!/usr/bin/env python3
"""Run live paper trading against real Polymarket BTC 5m markets.

Places no orders. Each market takes real time to resolve, so trading N markets
takes roughly N * 5 minutes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.decision_model import ReversalDecisionModel
from src.config import settings
from src.paper.engine import LivePaperTrader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=5,
                        help="how many markets to trade before stopping")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--no-ai", action="store_true",
                        help="take every underdog in the band, skipping the AI filter")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate the current market once and exit")
    args = parser.parse_args()

    model = None
    if not args.no_ai:
        model_path = Path(settings.model_path)
        if model_path.exists():
            model = ReversalDecisionModel()
            model.load(str(model_path))
            print(f"Loaded AI model from {model_path}")
        else:
            print(f"No model at {model_path}; run scripts/train_model.py "
                  f"or pass --no-ai. Continuing without the AI filter.")

    trader = LivePaperTrader(model=model)
    print(f"Balance: ${trader.portfolio.balance:,.2f} | stake ${settings.stake:.2f} | "
          f"band [{settings.min_token_price:.2f}, {settings.max_token_price:.2f}]")

    try:
        if args.dry_run:
            markets = trader.client.get_open_markets(limit=3)
            if not markets:
                print("No open BTC 5m markets found.")
                return
            print(f"\nOpen markets: {len(markets)}")
            for market in markets:
                print(f"\n  {market.slug} closes in {market.seconds_remaining():.0f}s "
                      f"(volume ${market.volume:,.0f})")
                observation = trader.observe_market(market)
                if observation is None:
                    print("    no underdog below 0.50 quoted yet")
                    continue
                features = observation["features"]
                decision = trader.strategy.evaluate(
                    features, observation["side"], trader.portfolio.balance
                )
                print(f"    underdog {observation['side']} @ {features['underdog_price']:.3f} "
                      f"(favorite {features['favorite_price']:.3f})")
                print(f"    decision: {'ENTER' if decision.should_trade else 'skip'} "
                      f"— {decision.reason}")
            return

        print(f"\nTrading the next {args.markets} markets "
              f"(~{args.markets * 5} minutes). Ctrl-C to stop.\n")
        summary = trader.run(max_markets=args.markets, poll_seconds=args.poll_seconds)

        print("\n" + "=" * 60)
        print("PAPER TRADING SUMMARY")
        print("=" * 60)
        print(f"  Trades opened : {summary['trades_opened']}")
        print(f"  Trades settled: {summary['trades_settled']}")
        if summary["win_rate_pct"] is not None:
            print(f"  Win rate      : {summary['win_rate_pct']:.1f}%")
        print(f"  Balance       : ${summary['balance']:,.2f}")
        print(f"  Total P&L     : ${summary['total_pnl']:+,.2f}")
        print(f"\n  Saved to {settings.paper_trades_path}")
    except KeyboardInterrupt:
        print("\nStopped. Saving state...")
        trader.export([])
    finally:
        trader.close()


if __name__ == "__main__":
    main()
