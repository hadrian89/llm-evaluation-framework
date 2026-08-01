"""Runtime configuration loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EVAL_CONFIG_PATH = Path(__file__).parent / "eval_config.yaml"


class Settings(BaseSettings):
    """Environment-driven settings. See .env.example for the full list."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers (all optional — evaluators fall back to heuristic
    # scoring when no key is configured, so the framework runs offline).
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_judge_provider: str = Field(default="heuristic", alias="LLM_JUDGE_PROVIDER")
    llm_judge_model: str = Field(default="gpt-4o-mini", alias="LLM_JUDGE_MODEL")

    # Integrations
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="llm-evaluation-framework", alias="LANGSMITH_PROJECT")
    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="llm-evaluation-framework", alias="MLFLOW_EXPERIMENT")

    # Pipeline
    eval_config_path: str = Field(default=str(DEFAULT_EVAL_CONFIG_PATH), alias="EVAL_CONFIG_PATH")
    report_output_dir: str = Field(default="reports", alias="REPORT_OUTPUT_DIR")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_eval_config(path: str | None = None) -> dict[str, Any]:
    """Load evaluation thresholds/params from YAML."""
    config_path = Path(path or get_settings().eval_config_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
