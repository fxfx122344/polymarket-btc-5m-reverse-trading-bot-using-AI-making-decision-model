#!/usr/bin/env python3
"""Backtest the reverse-trading strategy on real resolved Polymarket BTC 5m markets.

Reports out-of-sample results with statistical significance. Outcomes come from
how markets actually settled, so the numbers are whatever the data says.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.ai.decision_model import ReversalDecisionModel
from src.backtest.engine import BacktestConfig, run_backtest, summarize
from src.data.dataset import build_dataset, calibration_table
from src.data.fetch import load_cached

RESULTS_PATH = Path("data/backtest_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-offset", type=int, default=60,
                        help="seconds before close when the decision is made")
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--slippage", type=float, default=0.01,
                        help="cost of crossing the spread, in probability units")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("=" * 72)
    print("REVERSE TRADING BACKTEST — real Polymarket BTC 5m markets")
    print("=" * 72)

    markets, paths = load_cached()
    df = build_dataset(markets, paths,
                       decision_offset=args.decision_offset,
                       min_volume=args.min_volume)

    if df.empty:
        print("No usable rows. Run scripts/fetch_history.py first.")
        return

    hours = (df["end_ts"].max() - df["end_ts"].min()) / 3600
    print(f"\nMarkets downloaded : {len(markets)}")
    print(f"Usable underdog rows: {len(df)} spanning {hours:.1f} hours")
    print(f"Decision point      : {args.decision_offset}s before close")
    print(f"Assumed slippage    : {args.slippage:.3f} (one tick)")

    print("\n" + "-" * 72)
    print("STEP 1 — Is the market mispricing underdogs? (calibration)")
    print("-" * 72)
    print("An edge exists only where realized win rate > entry price.\n")
    calib = calibration_table(df)
    print(calib.to_string(index=False))

    print("\n" + "-" * 72)
    print("STEP 2 — Can the AI model predict reversals out-of-sample?")
    print("-" * 72)
    model = ReversalDecisionModel()
    metrics = model.evaluate_walk_forward(df, n_splits=5)
    print("Forward-chaining validation (train on past, test on future):")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.4f}" if isinstance(v, float) else f"  {k:<18}: {v}")
    auc = metrics.get("roc_auc")
    if auc is not None:
        verdict = ("no better than chance" if abs(auc - 0.5) < 0.02
                   else "weak signal" if abs(auc - 0.5) < 0.06 else "signal present")
        print(f"  -> AUC {auc:.3f}: {verdict}")

    oos_prob = model.out_of_sample_predictions(df, n_splits=5)

    print("\n" + "-" * 72)
    print("STEP 3 — Backtest with real outcomes and execution costs")
    print("-" * 72)

    results = []

    # Baseline: buy every underdog in the bot's configured band, no AI filter.
    results.append(run_backtest(
        df,
        BacktestConfig(min_entry_price=0.15, max_entry_price=0.48,
                       min_ai_confidence=None, slippage=args.slippage),
        label="baseline band 0.15-0.48 (no AI)",
    ))

    # Wider band uses the bulk of the sample, where statistics are meaningful.
    results.append(run_backtest(
        df,
        BacktestConfig(min_entry_price=0.15, max_entry_price=0.499,
                       min_ai_confidence=None, slippage=args.slippage),
        label="all underdogs < 0.50 (no AI)",
    ))

    # Zero-cost variant isolates how much the spread matters.
    results.append(run_backtest(
        df,
        BacktestConfig(min_entry_price=0.15, max_entry_price=0.499,
                       min_ai_confidence=None, slippage=0.0),
        label="all underdogs < 0.50, zero slippage",
    ))

    # AI-filtered runs at several confidence thresholds.
    for threshold in (0.50, 0.55, 0.60):
        results.append(run_backtest(
            df,
            BacktestConfig(min_entry_price=0.15, max_entry_price=0.499,
                           min_ai_confidence=threshold, slippage=args.slippage),
            probabilities=oos_prob,
            label=f"AI filter p>={threshold:.2f}",
        ))

    print(summarize(results).to_string(index=False))
    print("\n  edge_pp   = realized win rate minus breakeven (executed price), in points")
    print("  ev_per_$  = average profit per $1 staked")
    print("  ci95      = bootstrap CI for ev_per_$; profitable only if entirely above 0")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    significant = [r for r in results if r.stats.get("significant_at_95")]
    if significant:
        print("Configurations with a statistically significant positive edge:")
        for r in significant:
            print(f"  - {r.label}: EV {r.stats['mean_return_per_dollar']:+.4f}/$ "
                  f"over {r.stats['n_trades']} trades")
    else:
        print("No configuration shows a statistically significant positive edge.")
        print("Every 95% confidence interval includes zero or is negative, which means")
        print("the results are indistinguishable from chance on this sample.")

    best = max(results, key=lambda r: r.stats.get("n_trades", 0))
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data": {
            "markets_downloaded": len(markets),
            "usable_rows": int(len(df)),
            "hours_covered": round(hours, 1),
            "decision_offset_seconds": args.decision_offset,
            "slippage": args.slippage,
        },
        "calibration": json.loads(calib.astype(str).to_json(orient="records")),
        "model_metrics": metrics,
        "runs": [
            {"label": r.label,
             "stats": {k: v for k, v in r.stats.items() if k != "equity_curve"}}
            for r in results
        ],
        "equity_curve": best.stats.get("equity_curve", []),
        "equity_curve_label": best.label,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
