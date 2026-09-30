"""
Settings coming from environment variables or .env"
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    data_path: str = "data/raw/winequality-red.csv"
    model_path: str = "models/model.joblib"
    metrics_path: str = "reports/metrics.json"

settings = Settings()
