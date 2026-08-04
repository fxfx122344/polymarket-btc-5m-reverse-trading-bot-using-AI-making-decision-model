"""Backtest the reverse-trading strategy against real market outcomes.

Trade results come from how each market actually settled, never from the
model's own confidence. Execution costs are applied because the observed
book on these markets is roughly 0.495 / 0.505, so buying the underdog
means crossing a one-tick spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Strategy and cost parameters."""

    min_entry_price: float = 0.15
    max_entry_price: float = 0.48
    min_ai_confidence: float | None = 0.55
    # Cost of crossing the spread, in probability units (one tick = $0.01).
    slippage: float = 0.01
    # Proportional fee on notional, if any.
    fee_rate: float = 0.0
    stake: float = 10.0
    initial_balance: float = 1000.0


@dataclass
class BacktestResult:
    """Outcome of a backtest run, including significance statistics."""

    label: str
    config: BacktestConfig
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def _bootstrap_ci(returns: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean per-trade return."""
    if len(returns) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(returns, size=(n_boot, len(returns)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
    probabilities: pd.Series | None = None,
    label: str = "strategy",
) -> BacktestResult:
    """
    Apply the strategy to real markets and measure realized performance.

    Args:
        df: dataset of real markets, one row per market, sorted by time.
        probabilities: out-of-sample model probabilities aligned to ``df``.
            If omitted, every candidate in the price band is taken (baseline).
    """
    if df.empty:
        return BacktestResult(label=label, config=config)

    df = df.sort_values("end_ts").reset_index(drop=True)

    # Execution price includes the cost of crossing the spread.
    exec_price = df["underdog_price"] + config.slippage

    eligible = (
        (df["underdog_price"] >= config.min_entry_price)
        & (df["underdog_price"] <= config.max_entry_price)
        & (exec_price < 1.0)
    )

    if probabilities is not None:
        prob = probabilities.reindex(df.index)
        # Rows without an out-of-sample prediction cannot be traded honestly.
        eligible &= prob.notna()
        if config.min_ai_confidence is not None:
            eligible &= prob >= config.min_ai_confidence

    taken = df[eligible].copy()
    if taken.empty:
        return BacktestResult(
            label=label,
            config=config,
            stats={"n_trades": 0, "note": "no trades met the entry criteria"},
        )

    taken["exec_price"] = exec_price[eligible]
    if probabilities is not None:
        taken["model_probability"] = probabilities.reindex(taken.index)

    # Payout: a winning share settles at $1.00; a loser is worth $0.
    shares = config.stake / taken["exec_price"]
    gross = np.where(taken["underdog_won"] == 1, shares * 1.0, 0.0)
    fees = config.stake * config.fee_rate
    taken["pnl"] = gross - config.stake - fees
    # Return per dollar staked, the unit for statistical tests.
    taken["return_per_dollar"] = taken["pnl"] / config.stake

    returns = taken["return_per_dollar"].to_numpy()
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    t_stat = (
        float(mean_ret / (std_ret / np.sqrt(len(returns))))
        if len(returns) > 1 and std_ret > 0
        else float("nan")
    )
    ci_low, ci_high = _bootstrap_ci(returns)

    # Compounded equity curve, for display only.
    balance = config.initial_balance
    curve = []
    for _, row in taken.iterrows():
        balance += row["pnl"]
        curve.append(
            {
                "end_ts": int(row["end_ts"]),
                "timestamp": datetime.fromtimestamp(int(row["end_ts"]), tz=timezone.utc).isoformat(),
                "balance": round(balance, 2),
                "pnl": round(float(row["pnl"]), 2),
            }
        )
    equity = pd.DataFrame(curve)

    peak = equity["balance"].cummax()
    max_dd = float(((peak - equity["balance"]) / peak).max() * 100) if not equity.empty else 0.0

    wins = taken[taken["underdog_won"] == 1]
    losses = taken[taken["underdog_won"] == 0]
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(abs(losses["pnl"].sum()))

    stats = {
        "n_trades": int(len(taken)),
        "win_rate": float(taken["underdog_won"].mean()),
        "mean_entry_price": float(taken["underdog_price"].mean()),
        "mean_exec_price": float(taken["exec_price"].mean()),
        "breakeven_win_rate": float(taken["exec_price"].mean()),
        "total_pnl": float(taken["pnl"].sum()),
        "final_balance": float(balance),
        "roi_pct": float((balance - config.initial_balance) / config.initial_balance * 100),
        "mean_return_per_dollar": mean_ret,
        "return_std": std_ret,
        "t_statistic": t_stat,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": max_dd,
        "avg_win": float(wins["pnl"].mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if not losses.empty else 0.0,
        # Positive edge requires the win rate to exceed the executed price.
        "edge_vs_breakeven": float(taken["underdog_won"].mean() - taken["exec_price"].mean()),
        "significant_at_95": bool(ci_low > 0),
    }

    result = BacktestResult(label=label, config=config, trades=taken, stats=stats)
    result.stats["equity_curve"] = equity.to_dict("records")
    return result


def summarize(results: list[BacktestResult]) -> pd.DataFrame:
    """Compact comparison table across runs."""
    rows = []
    for r in results:
        s = r.stats
        if not s or s.get("n_trades", 0) == 0:
            rows.append({"strategy": r.label, "n_trades": 0})
            continue
        rows.append(
            {
                "strategy": r.label,
                "n_trades": s["n_trades"],
                "win_rate": round(s["win_rate"] * 100, 2),
                "breakeven": round(s["breakeven_win_rate"] * 100, 2),
                "edge_pp": round(s["edge_vs_breakeven"] * 100, 2),
                "roi_pct": round(s["roi_pct"], 2),
                "ev_per_$": round(s["mean_return_per_dollar"], 4),
                "t_stat": round(s["t_statistic"], 2) if s["t_statistic"] == s["t_statistic"] else None,
                "ci95": f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]",
                "significant": s["significant_at_95"],
            }
        )
    return pd.DataFrame(rows)
