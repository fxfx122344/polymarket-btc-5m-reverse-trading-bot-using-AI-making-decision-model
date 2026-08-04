"""Streamlit dashboard for real backtest and paper trading results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import settings


def load_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def render_header(data: dict) -> None:
    meta = data["data"]
    st.title("Polymarket BTC 5m Reverse Trading Bot")
    st.caption(
        "Backtest on real resolved markets. Outcomes come from actual market "
        "resolutions, not from model predictions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Markets analyzed", f"{meta['markets_downloaded']:,}")
    c2.metric("Usable underdog rows", f"{meta['usable_rows']:,}")
    c3.metric("History covered", f"{meta['hours_covered']:.0f} h")
    c4.metric("Decision point", f"{meta['decision_offset_seconds']}s before close")


def render_verdict(data: dict) -> None:
    runs = data["runs"]
    significant = [r for r in runs if r["stats"].get("significant_at_95")]
    if significant:
        st.success(
            f"{len(significant)} configuration(s) show a statistically significant "
            "positive edge at 95% confidence."
        )
    else:
        st.warning(
            "**No statistically significant edge found.** Every configuration's 95% "
            "confidence interval for profit per dollar includes zero, so the results "
            "are indistinguishable from chance on this sample."
        )


def render_calibration(data: dict) -> None:
    st.subheader("Is the market mispricing underdogs?")
    st.caption(
        "An edge exists only where the realized win rate exceeds the entry price. "
        "Compare each bar against its implied rate, and note the sample size."
    )

    calib = pd.DataFrame(data["calibration"])
    if calib.empty:
        st.info("No calibration data available.")
        return

    for col in ["avg_entry_price", "realized_win_rate", "implied_win_rate", "edge", "std_error"]:
        calib[col] = pd.to_numeric(calib[col], errors="coerce")
    calib["n"] = pd.to_numeric(calib["n"], errors="coerce")

    fig = go.Figure()
    fig.add_bar(
        x=calib["bucket"].astype(str),
        y=calib["realized_win_rate"],
        name="Realized win rate",
        marker_color="#2ecc71",
        error_y=dict(type="data", array=calib["std_error"], visible=True),
    )
    fig.add_bar(
        x=calib["bucket"].astype(str),
        y=calib["implied_win_rate"],
        name="Implied by price (breakeven)",
        marker_color="#95a5a6",
    )
    fig.update_layout(
        barmode="group",
        height=380,
        yaxis_title="Probability",
        xaxis_title="Underdog entry price bucket",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = calib.copy()
    display["n"] = display["n"].astype(int)
    for col in ["avg_entry_price", "realized_win_rate", "implied_win_rate", "edge", "std_error"]:
        display[col] = display[col].round(4)
    st.dataframe(
        display[["bucket", "n", "avg_entry_price", "realized_win_rate", "edge", "std_error"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Where the standard error is as large as the edge, the apparent edge is noise."
    )


def render_runs(data: dict) -> None:
    st.subheader("Backtest configurations")
    rows = []
    for run in data["runs"]:
        s = run["stats"]
        if s.get("n_trades", 0) == 0:
            rows.append({"Strategy": run["label"], "Trades": 0})
            continue
        rows.append(
            {
                "Strategy": run["label"],
                "Trades": s["n_trades"],
                "Win rate %": round(s["win_rate"] * 100, 2),
                "Breakeven %": round(s["breakeven_win_rate"] * 100, 2),
                "Edge (pp)": round(s["edge_vs_breakeven"] * 100, 2),
                "ROI %": round(s["roi_pct"], 2),
                "EV per $": round(s["mean_return_per_dollar"], 4),
                "95% CI": f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]",
                "Significant": "yes" if s["significant_at_95"] else "no",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Breakeven % is the executed price including the spread crossed on entry. "
        "A strategy is profitable only when the win rate clears it."
    )


def render_equity(data: dict) -> None:
    curve = pd.DataFrame(data.get("equity_curve", []))
    if curve.empty:
        return
    st.subheader(f"Equity curve — {data.get('equity_curve_label', 'strategy')}")
    fig = px.line(curve, x="timestamp", y="balance",
                  labels={"balance": "Balance ($)", "timestamp": "Market close time"})
    fig.add_hline(y=curve["balance"].iloc[0], line_dash="dash", line_color="gray")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "A rising curve over a sample this size is not proof of edge; check the "
        "confidence interval above."
    )


def render_model(data: dict) -> None:
    st.subheader("AI decision model — out-of-sample")
    metrics = data.get("model_metrics", {})
    if not metrics:
        st.info("No model metrics available.")
        return

    auc = metrics.get("roc_auc")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC", f"{auc:.4f}" if auc else "n/a",
              help="0.50 means no predictive power")
    c2.metric("Brier score", f"{metrics.get('brier_score', 0):.4f}")
    c3.metric("Out-of-sample rows", f"{metrics.get('n_out_of_sample', 0):,}")

    if auc is not None:
        if abs(auc - 0.5) < 0.02:
            st.error(
                f"AUC of {auc:.4f} is at chance level (0.50). On this dataset the model "
                "carries no usable predictive signal, so its confidence scores should "
                "not be trusted as an edge."
            )
        elif abs(auc - 0.5) < 0.06:
            st.warning(f"AUC of {auc:.4f} indicates only a weak signal.")
        else:
            st.success(f"AUC of {auc:.4f} indicates measurable predictive signal.")


def render_paper_trades() -> None:
    paper = load_json(settings.paper_trades_path)
    if not paper:
        st.info(
            "No live paper trading session recorded yet. Run "
            "`python scripts/run_paper_trading.py` to trade live markets."
        )
        return

    st.subheader("Live paper trading session")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance", f"${paper['balance']:,.2f}")
    c2.metric("Total P&L", f"${paper['total_pnl']:+,.2f}")
    c3.metric("Opened", paper["trades_opened"])
    c4.metric("Settled", paper["trades_settled"])

    trades = pd.DataFrame(paper.get("trades", []))
    if not trades.empty:
        st.dataframe(trades, hide_index=True, use_container_width=True)

    decisions = pd.DataFrame(paper.get("decisions", []))
    if not decisions.empty:
        with st.expander("Every decision, including skips"):
            st.dataframe(decisions, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Reverse Trading Bot — Results", page_icon="📊", layout="wide")

    data = load_json(settings.backtest_results_path)
    if data is None:
        st.error(
            f"No backtest results at `{settings.backtest_results_path}`.\n\n"
            "Run:\n```\npython scripts/fetch_history.py\npython scripts/run_backtest.py\n```"
        )
        st.stop()

    render_header(data)
    render_verdict(data)
    st.divider()
    render_calibration(data)
    st.divider()
    render_runs(data)
    render_equity(data)
    st.divider()
    render_model(data)
    st.divider()
    render_paper_trades()


if __name__ == "__main__":
    main()
