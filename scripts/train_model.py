#!/usr/bin/env python3
"""Train and evaluate the AI reversal decision model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.decision_model import ReversalDecisionModel


def main() -> None:
    print("Training AI reversal decision model...")
    model = ReversalDecisionModel()
    metrics = model.train(n_samples=10000)
    model.save()

    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nTop Feature Importances:")
    importance = model.feature_importance()
    for _, row in importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
