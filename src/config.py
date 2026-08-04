"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Bot and dashboard settings loaded from environment."""

    paper_trading: bool = True
    initial_balance: float = 1000.0
    max_position_size: float = 50.0
    min_token_price: float = 0.15
    max_token_price: float = 0.48
    min_ai_confidence: float = 0.62

    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    btc_series_slug: str = "btc-up-or-down-5m"

    model_path: str = "models/reversal_model.pkl"
    paper_results_path: str = "data/paper_trading_results.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
