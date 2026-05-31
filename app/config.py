from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://apex:apex@localhost:5432/apex"
    log_level: str = "INFO"
    app_env: str = "local"

    stale_feed_threshold_seconds: int = 600          # /health STALE_FEED if last event > 10m old
    queue_warn_depth: int = 5
    queue_critical_depth: int = 8
    queue_sustained_seconds: int = 120
    dead_zone_window_seconds: int = 1800             # 30 min
    conversion_drop_ratio: float = 0.7               # alert if today < 0.7 * baseline
    pos_correlation_window_seconds: int = 1800       # 30 min — used by /metrics potential_conversion_rate only
    pos_strict_window_seconds: int = 300             # 5 min — used by /funnel PURCHASE + abandon detection (brief)
    ingest_max_batch: int = 500


settings = Settings()
