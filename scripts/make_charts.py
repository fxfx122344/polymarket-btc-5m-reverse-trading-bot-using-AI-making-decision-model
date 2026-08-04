#!/usr/bin/env python3
"""Render backtest charts from real results for the README."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings

DOCS_DIR = Path("docs")


def load_results() -> dict:
    path = Path(settings.backtest_results_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/run_backtest.py first.")
    with open(path) as f:
        return json.load(f)


def plot_calibration(data: dict, out: Path) -> None:
    """Realized win rate vs the rate implied by price, with error bars."""
    calib = pd.DataFrame(data["calibration"])
    for col in ["realized_win_rate", "implied_win_rate", "std_error", "n"]:
        calib[col] = pd.to_numeric(calib[col], errors="coerce")

    x = range(len(calib))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], calib["realized_win_rate"], width,
           yerr=calib["std_error"], capsize=4, label="Realized win rate", color="#2ecc71")
    ax.bar([i + width / 2 for i in x], calib["implied_win_rate"], width,
           label="Implied by price (breakeven)", color="#95a5a6")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\nn={int(n)}" for b, n in zip(calib["bucket"], calib["n"])],
                       fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Underdog entry price bucket")
    ax.set_title("Do underdogs win more often than their price implies?\n"
                 "(error bars = 1 standard error)", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_equity(data: dict, out: Path) -> None:
    curve = pd.DataFrame(data.get("equity_curve", []))
    if curve.empty:
        return
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve["timestamp"], curve["balance"], color="#2980b9", linewidth=1.6)
    ax.axhline(curve["balance"].iloc[0], linestyle="--", color="gray",
               linewidth=1, label="Starting balance")
    ax.set_ylabel("Balance ($)")
    ax.set_xlabel("Market close time (UTC)")
    ax.set_title(f"Paper equity curve — {data.get('equity_curve_label', 'strategy')}\n"
                 "$10 flat stake, spread costs applied", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    data = load_results()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    plot_calibration(data, DOCS_DIR / "calibration.png")
    print(f"wrote {DOCS_DIR / 'calibration.png'}")

    plot_equity(data, DOCS_DIR / "equity_curve.png")
    print(f"wrote {DOCS_DIR / 'equity_curve.png'}")


if __name__ == "__main__":
    main()
