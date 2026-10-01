"""
Settings coming from environment variables or .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    data_path: str = "data/raw/winequality-red.csv"
    model_path: str = "models/model.joblib"
    metrics_path: str = "reports/metrics.json"
    cors_origins: list[str] = ["*"]
    api_url: str = "http://localhost:8000"


settings = Settings()
