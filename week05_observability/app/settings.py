"""Runtime configuration for most things, a bit ugly but works"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_prefix="ML520_")
    # GENAI COMMENT:
    #   Where the trainer writes the artifact and inferapi reads it back.
    #   Both containers mount the same named volume, so this path is identical for both.
    model_path: str = "/model/model.joblib"

    # GENAI COMMENT:
    #   Trainer-only: the bind-mounted, read-only dataset.
    data_path: str = "/data/bank_marketing.parquet"

    # GENAI COMMENT:
    #   Baked into the model artifact by train.py, then read back at serving time.
    #   The version travels with the artifact instead of living in an env var that
    #   could drift from what was actually trained.
    model_version: str = "1.0.0"

    # GENAI COMMENT:
    #   Set at image build time via `--build-arg BUILD_SHA`, see app/Dockerfile.
    build_sha: str = "dev"

    # GENAI COMMENT:
    #   "otlp" pushes to the Collector, "prometheus"
    #   serves /metrics in-process.
    #   See telemetry.py.
    metrics_export: Literal["otlp","prometheus"] = "otlp"

    # GENAI COMMENT:
    #   Milliseconds of busy CPU work per request, on top of the model call itself.
    #   Zero by default; the compose stack sets it so latency panels have shape.
    simulate_work_ms: int = 0

    log_level: str = "INFO"


settings = Settings()
