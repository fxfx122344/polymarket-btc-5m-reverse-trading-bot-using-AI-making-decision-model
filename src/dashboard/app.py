"""Streamlit dashboard for paper trading results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import settings


def load_results(path: str | None = None) -> dict:
    path = path or settings.paper_results_path
    with open(path) as f:
        return json.load(f)


def render_dashboard(results: dict) -> None:
    st.set_page_config(
        page_title="Reverse Trading Bot — Paper Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("Polymarket BTC 5m Reverse Trading Bot")
    st.caption("AI-powered paper trading dashboard — underdog reversal strategy")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final Balance", f"${results['final_balance']:,.2f}", f"{results['roi_pct']:+.1f}% ROI")
    c2.metric("Total P&L", f"${results['total_pnl']:+,.2f}")
    c3.metric("Win Rate", f"{results['win_rate_pct']:.1f}%")
    c4.metric("Total Trades", results["total_trades"])
    c5.metric("Max Drawdown", f"{results['max_drawdown_pct']:.1f}%")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Equity Curve")
        eq_df = pd.DataFrame(results["equity_curve"])
        if not eq_df.empty:
            fig = px.line(
                eq_df,
                x="timestamp",
                y="balance",
                title="Portfolio Balance Over Time",
                labels={"balance": "Balance ($)", "timestamp": "Time"},
            )
            fig.add_hline(
                y=results["initial_balance"],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial",
            )
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Daily P&L")
        daily = results.get("daily_pnl", {})
        if daily:
            daily_df = pd.DataFrame(
                [{"date": k, "pnl": v} for k, v in sorted(daily.items())]
            )
            colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in daily_df["pnl"]]
            fig_bar = go.Figure(
                go.Bar(x=daily_df["date"], y=daily_df["pnl"], marker_color=colors)
            )
            fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    c_a, c_b = st.columns(2)

    with c_a:
        st.subheader("Trade Statistics")
        stats = pd.DataFrame(
            {
                "Metric": [
                    "Initial Balance",
                    "Final Balance",
                    "ROI",
                    "Win Rate",
                    "Avg Win",
                    "Avg Loss",
                    "Profit Factor",
                    "Max Drawdown",
                ],
                "Value": [
                    f"${results['initial_balance']:,.2f}",
                    f"${results['final_balance']:,.2f}",
                    f"{results['roi_pct']:+.2f}%",
                    f"{results['win_rate_pct']:.1f}%",
                    f"${results['avg_win']:+.2f}",
                    f"${results['avg_loss']:+.2f}",
                    f"{results['profit_factor']:.2f}x",
                    f"{results['max_drawdown_pct']:.1f}%",
                ],
            }
        )
        st.dataframe(stats, hide_index=True, use_container_width=True)

    with c_b:
        st.subheader("AI Model Metrics")
        ai = results.get("ai_model_metrics", {})
        if ai:
            ai_df = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "ROC AUC", "Baseline Win Rate"],
                    "Value": [
                        f"{ai.get('accuracy', 0):.1%}",
                        f"{ai.get('roc_auc', 0):.3f}",
                        f"{ai.get('baseline_win_rate', 0):.1%}",
                    ],
                }
            )
            st.dataframe(ai_df, hide_index=True, use_container_width=True)

        st.info(
            "The AI decision model uses Gradient Boosting on order book features "
            "(imbalance, depth ratio, momentum, volatility) to filter high-probability reversals."
        )

    st.subheader("Recent Trades")
    trades_df = pd.DataFrame(results["trades"])
    if not trades_df.empty:
        trades_df["won"] = trades_df["won"].map({True: "✅ Win", False: "❌ Loss"})
        st.dataframe(
            trades_df[
                ["timestamp", "market", "outcome", "entry_price", "size_usdc", "ai_confidence", "won", "pnl"]
            ].tail(20),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    results_path = Path(settings.paper_results_path)
    if not results_path.exists():
        st.error(
            f"No paper trading results found at `{results_path}`. "
            "Run `python scripts/run_paper_trading.py` first."
        )
        st.stop()

    render_dashboard(load_results())


if __name__ == "__main__":
    main()
