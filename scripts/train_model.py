#!/usr/bin/env python3
"""Train the reversal decision model on real resolved market data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.decision_model import ReversalDecisionModel
from src.config import settings
from src.data.dataset import build_dataset
from src.data.fetch import load_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-offset", type=int, default=settings.decision_offset_seconds)
    parser.add_argument("--min-volume", type=float, default=settings.min_market_volume)
    args = parser.parse_args()

    markets, paths = load_cached()
    df = build_dataset(markets, paths,
                       decision_offset=args.decision_offset,
                       min_volume=args.min_volume)
    if df.empty:
        print("No usable rows. Run scripts/fetch_history.py first.")
        return

    print(f"Training on {len(df)} real markets "
          f"(underdog win rate {df['underdog_won'].mean():.2%})")

    model = ReversalDecisionModel()

    print("\nForward-chaining validation (train past -> test future):")
    metrics = model.evaluate_walk_forward(df, n_splits=5)
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.4f}" if isinstance(v, float) else f"  {k:<18}: {v}")

    auc = metrics.get("roc_auc")
    if auc is not None and abs(auc - 0.5) < 0.02:
        print("\n  Note: AUC is at chance level, so this model carries no usable")
        print("  predictive signal on this dataset. Treat its output accordingly.")

    # Fit on the full history for use by the live paper trader.
    model.fit(df)
    model.save(settings.model_path)
    print(f"\nSaved model to {settings.model_path}")

    print("\nFeature importance (in-sample only; does not imply predictive power):")
    for _, row in model.feature_importance().head(8).iterrows():
        print(f"  {row['feature']:<24} {row['importance']:.4f}")


if __name__ == "__main__":
    main()
