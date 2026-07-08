"""LLM factory and router.

`LLMRouter` is the single object every agent talks to. It:
  1. Builds the configured provider from environment settings.
  2. Exposes `.generate(system_prompt, user_prompt)` which NEVER raises —
     on any failure (missing key, network error, package not installed) it
     returns `None`, and the calling agent falls back to deterministic
     analytics. This is what makes the "no API key required" promise real.
"""
from __future__ import annotations

from typing import Optional

from app.config import Settings, get_settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base_provider import LLMProvider, LLMProviderError
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_compatible_provider import AzureOpenAIProvider, OpenAICompatibleProvider
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PROVIDER_BUILDERS = {
    "openai": lambda s: OpenAICompatibleProvider(
        base_url="", api_key=s.openai_api_key, model=s.openai_model, provider_name="openai"
    ),
    "azure_openai": lambda s: AzureOpenAIProvider(
        endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        deployment=s.azure_openai_deployment,
        api_version=s.azure_openai_api_version,
    ),
    "anthropic": lambda s: AnthropicProvider(api_key=s.anthropic_api_key, model=s.anthropic_model),
    "gemini": lambda s: GeminiProvider(api_key=s.gemini_api_key, model=s.gemini_model),
    "ollama": lambda s: OllamaProvider(base_url=s.ollama_base_url, model=s.ollama_model),
    "lmstudio": lambda s: OpenAICompatibleProvider(
        base_url=s.lmstudio_base_url, api_key="lm-studio", model=s.lmstudio_model, provider_name="lmstudio"
    ),
    "openrouter": lambda s: OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=s.openrouter_api_key,
        model=s.openrouter_model,
        provider_name="openrouter",
    ),
    "deepseek": lambda s: OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        api_key=s.deepseek_api_key,
        model=s.deepseek_model,
        provider_name="deepseek",
    ),
    "mistral": lambda s: OpenAICompatibleProvider(
        base_url="https://api.mistral.ai/v1",
        api_key=s.mistral_api_key,
        model=s.mistral_model,
        provider_name="mistral",
    ),
}


def build_provider(settings: Optional[Settings] = None) -> Optional[LLMProvider]:
    """Instantiate the LLM provider selected by `LLM_PROVIDER`, or None."""
    settings = settings or get_settings()
    provider_key = (settings.llm_provider or "none").lower().strip()
    if provider_key in ("", "none"):
        return None
    builder = _PROVIDER_BUILDERS.get(provider_key)
    if builder is None:
        logger.warning(f"Unknown LLM_PROVIDER='{provider_key}', falling back to deterministic mode.")
        return None
    try:
        provider = builder(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to construct provider '{provider_key}': {exc}")
        return None
    return provider


class LLMRouter:
    """Fail-safe wrapper around a single `LLMProvider`.

    Agents should depend only on this class, never on a concrete provider.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._provider = build_provider(self.settings)

    @property
    def provider_name(self) -> str:
        return self._provider.name if self._provider else "deterministic-fallback"

    def is_enabled(self) -> bool:
        return self._provider is not None and self._provider.is_available()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Return an LLM completion, or None if unavailable/failed (never raises)."""
        if not self.is_enabled():
            return None
        try:
            text = self._provider.complete(system_prompt, user_prompt, max_tokens=max_tokens)
            return text.strip() if text else None
        except LLMProviderError as exc:
            logger.warning(f"LLM generation failed, falling back to deterministic logic: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Unexpected LLM error, falling back to deterministic logic: {exc}")
            return None


_ROUTER_SINGLETON: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Return a process-wide singleton `LLMRouter`."""
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        _ROUTER_SINGLETON = LLMRouter()
    return _ROUTER_SINGLETON
