"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Bot, backtest, and dashboard settings loaded from environment."""

    # Paper mode never sends orders. Live trading is intentionally not implemented.
    paper_trading: bool = True
    initial_balance: float = 1000.0
    stake: float = 10.0

    # Entry band for underdog tokens.
    min_token_price: float = 0.15
    max_token_price: float = 0.48
    min_ai_confidence: float = 0.55

    # Cost of crossing the spread, in probability units (one tick = $0.01).
    slippage: float = 0.01

    # Seconds before market close when the entry decision is made.
    decision_offset_seconds: int = 60
    # Markets below this traded volume have stale placeholder quotes.
    min_market_volume: float = 1000.0

    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    btc_series_slug: str = "btc-up-or-down-5m"
    btc_series_id: str = "10684"

    model_path: str = "models/reversal_model.pkl"
    backtest_results_path: str = "data/backtest_results.json"
    paper_trades_path: str = "data/paper_trades.json"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
