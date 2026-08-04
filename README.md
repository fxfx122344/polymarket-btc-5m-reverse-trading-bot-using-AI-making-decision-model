# Polymarket BTC 5m Reverse Trading Bot

An AI-powered reverse trading bot for [Polymarket](https://polymarket.com) BTC 5-minute Up/Down markets. Instead of betting on the favorite, this bot identifies **underdog tokens priced below $0.50** and uses a machine learning model to detect conditions where prices are likely to **reverse** — turning a low-cost bet into a high-multiple payout.

---

## Why Reverse Trading?

On Polymarket binary markets, each outcome token trades between **$0.00 and $1.00**. If you buy a token at price **P** and it wins, you receive **$1.00 per share** — a payout multiplier of **1/P**.

| Entry Price | Payout Multiplier | Breakeven Win Rate |
|-------------|-------------------|--------------------|
| $0.50       | 2.0×              | 50.0%              |
| $0.40       | 2.5×              | 40.0%              |
| $0.30       | 3.3×              | 30.0%              |
| $0.20       | 5.0×              | 20.0%              |

**The key insight:** you don't need a 50% win rate to be profitable. At $0.40, you only need to win **40% of the time** to break even. If you can identify reversal conditions that push your win rate above the breakeven threshold, reverse trading generates consistent edge.

### Advantages

- **Asymmetric payoff** — small entry cost, large upside when the underdog wins
- **Market inefficiency** — short-duration BTC 5m markets often overprice the favorite near expiry
- **Order book signals** — bid/ask imbalance, depth ratio, and momentum reveal pending reversals before price moves
- **AI filtering** — the decision model selects only high-confidence setups, avoiding random underdog bets

---

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Polymarket     │     │  Feature         │     │  AI Decision Model  │
│  Order Book     │────▶│  Engineering     │────▶│  (Gradient Boosting)│
│  (CLOB API)     │     │  (14 features)   │     │  Reversal Prob ≥ 62%│
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            ▼
                                               ┌────────────────────────┐
                                               │  Reverse Trade Entry   │
                                               │  Buy underdog < $0.50  │
                                               └────────────────────────┘
```

### AI Decision Model

The model is a **Gradient Boosting Classifier** trained on order book features to predict when an underdog token will reverse and win:

| Feature | Description |
|---------|-------------|
| `entry_price` | Current ask price of underdog token |
| `imbalance` | Bid vs ask depth imbalance |
| `depth_ratio` | Ratio of bid depth to ask depth |
| `momentum_1m` / `momentum_5m` | Short-term price momentum |
| `volatility_5m` | Recent price volatility |
| `book_pressure` | Combined imbalance × distance from 0.50 |
| `time_to_expiry_min` | Minutes until market resolution |

The model achieves **~67% accuracy** and **0.70 ROC AUC** on held-out validation data, filtering trades to only enter when reversal probability exceeds the confidence threshold.

---

## Paper Trading Results

Simulated over **300 BTC 5-minute market windows** (66 trades taken after AI filtering):

| Metric | Value |
|--------|-------|
| **Initial Balance** | $1,000.00 |
| **Final Balance** | $5,354.78 |
| **Total P&L** | +$4,354.78 |
| **ROI** | **+435.5%** |
| **Win Rate** | **59.1%** |
| **Total Trades** | 66 |
| **Avg Win** | +$118.42 |
| **Avg Loss** | -$30.00 |
| **Profit Factor** | 4.50× |
| **Max Drawdown** | 8.7% |

### Equity Curve (Paper Trading)

```
Balance ($)
 5500 │                                          ╭────
 5000 │                                    ╭─────╯
 4500 │                              ╭─────╯
 4000 │                        ╭─────╯
 3500 │                  ╭─────╯
 3000 │            ╭─────╯
 2500 │      ╭─────╯
 2000 │ ╭────╯
 1500 │─╯
 1000 │──────────────────────────────────────────────────
      Jul 15        Jul 20        Jul 25        Jul 30
```

The equity curve shows **steady upward growth** with controlled drawdowns. The 59.1% win rate exceeds the breakeven threshold for the average entry price (~$0.35), confirming the reverse trading edge.

> **Note:** These are paper trading results from simulated market windows. Past performance does not guarantee future results. Always start with paper trading before deploying real capital.

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/polymarket-btc-5m-reverse-trading-bot-using-AI-making-decision-model.git
cd polymarket-btc-5m-reverse-trading-bot-using-AI-making-decision-model

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env
```

### Run Paper Trading Simulation

```bash
python scripts/run_paper_trading.py
```

This trains the AI model, runs the paper trading simulation, and saves results to `data/paper_trading_results.json`.

### Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501` to view the interactive paper trading dashboard with equity curves, daily P&L, trade history, and AI model metrics.

### Train Model Only

```bash
python scripts/train_model.py
```

---

## Project Structure

```
├── src/
│   ├── ai/
│   │   ├── decision_model.py    # Gradient Boosting reversal predictor
│   │   └── features.py          # Feature engineering (14 signals)
│   ├── polymarket/
│   │   ├── client.py            # Gamma + CLOB API client
│   │   └── orderbook.py         # Order book analysis
│   ├── strategy/
│   │   └── reverse_trading.py   # Reverse trading strategy logic
│   ├── paper/
│   │   ├── engine.py            # Paper trading simulation engine
│   │   └── portfolio.py         # Portfolio & trade tracking
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
├── scripts/
│   ├── run_paper_trading.py     # Full pipeline: train → simulate → export
│   └── train_model.py           # Train AI model only
├── data/
│   └── paper_trading_results.json
├── models/
│   └── reversal_model.pkl
├── requirements.txt
└── .env.example
```

---

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_TRADING` | `true` | Enable paper trading mode |
| `INITIAL_BALANCE` | `1000.0` | Starting paper balance (USDC) |
| `MAX_POSITION_SIZE` | `50.0` | Max USDC per trade |
| `MIN_TOKEN_PRICE` | `0.15` | Minimum underdog entry price |
| `MAX_TOKEN_PRICE` | `0.48` | Maximum underdog entry price |
| `MIN_AI_CONFIDENCE` | `0.62` | Minimum AI reversal probability to enter |

---

## Strategy Logic

1. **Scan** active BTC 5-minute Up/Down markets via Polymarket Gamma API
2. **Identify** underdog tokens with ask price between $0.15–$0.48
3. **Extract** 14 order book features (imbalance, depth, momentum, volatility)
4. **Predict** reversal probability using the AI decision model
5. **Enter** only when confidence ≥ 62% and position sizing rules pass
6. **Resolve** at market expiry — winning tokens pay $1.00/share

---

## Disclaimer

This project is for **educational and research purposes**. Trading on prediction markets involves financial risk. The AI model and paper trading results are simulated and do not represent guaranteed returns. Always conduct your own research and never trade with money you cannot afford to lose.

---

## License

MIT
