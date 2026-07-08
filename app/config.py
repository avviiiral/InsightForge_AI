"""Central, environment-driven configuration for InsightForge-AI.

All runtime configuration is read from environment variables (optionally
loaded from a `.env` file). Nothing in the application should hard-code
credentials, model names, or file paths — everything flows through the
`Settings` object returned by `get_settings()`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLES_DIR = DATA_DIR / "samples"


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # General
    app_env: str = "development"
    log_level: str = "INFO"
    app_secret_key: str = "change-me-in-production"

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_base_url: str = "http://localhost:8000"

    # LLM provider selection
    llm_provider: str = "none"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-06-01"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # LM Studio
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "local-model"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # Mistral
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"

    # Databases
    postgres_uri: str = ""
    mysql_uri: str = ""

    # Upload limits
    max_upload_rows: int = 2_000_000
    max_preview_rows: int = 500


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide `Settings` instance."""
    return Settings()


for _d in (DATA_DIR, CACHE_DIR, SAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
