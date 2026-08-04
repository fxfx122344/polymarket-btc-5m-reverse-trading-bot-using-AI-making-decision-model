#!/usr/bin/env python3
"""Train AI model and run paper trading simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.decision_model import ReversalDecisionModel
from src.paper.engine import PaperTradingEngine


def main() -> None:
    print("=" * 60)
    print("Polymarket BTC 5m Reverse Trading Bot — Paper Trading")
    print("=" * 60)

    print("\n[1/3] Training AI reversal decision model...")
    model = ReversalDecisionModel()
    metrics = model.train(n_samples=8000)
    model.save()
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.3f}")

    print("\n[2/3] Running paper trading simulation (300 windows)...")
    engine = PaperTradingEngine()
    engine.portfolio = engine.run_simulation(n_windows=300, seed=42)

    print("\n[3/3] Exporting results...")
    results = engine.export_results()

    print("\n" + "=" * 60)
    print("PAPER TRADING RESULTS")
    print("=" * 60)
    print(f"  Initial Balance:  ${results['initial_balance']:,.2f}")
    print(f"  Final Balance:    ${results['final_balance']:,.2f}")
    print(f"  Total P&L:        ${results['total_pnl']:+,.2f}")
    print(f"  ROI:              {results['roi_pct']:+.2f}%")
    print(f"  Win Rate:         {results['win_rate_pct']:.1f}%")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}x")
    print(f"  Max Drawdown:     {results['max_drawdown_pct']:.1f}%")
    print(f"\n  Results saved to: data/paper_trading_results.json")
    print("\n  Launch dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
