"""Generic OpenAI-compatible chat-completions provider.

A huge fraction of the LLM ecosystem (OpenAI itself, Azure OpenAI,
OpenRouter, DeepSeek, Mistral's `/v1/chat/completions`, LM Studio, and most
self-hosted local model servers) all speak the same wire protocol. Rather
than duplicating a client per vendor, this single class is parameterized
with a `base_url`, `api_key`, and `model`, and is reused by
`llm_factory.py` for every one of those backends.

Azure OpenAI is the one exception with a slightly different URL shape, so
it gets a thin subclass below.
"""
from __future__ import annotations

from app.llm.base_provider import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions client for any OpenAI-wire-protocol-compatible backend."""

    def __init__(self, base_url: str, api_key: str, model: str, provider_name: str = "openai_compatible"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.name = provider_name

    def is_available(self) -> bool:
        # Local servers (LM Studio, Ollama-as-OpenAI) don't require a key.
        local = any(host in self.base_url for host in ("localhost", "127.0.0.1"))
        return bool(self.model) and (bool(self.api_key) or local)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("`openai` package not installed") from exc

        try:
            client = OpenAI(base_url=self.base_url or None, api_key=self.api_key or "not-needed")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{self.name}] completion failed: {exc}")
            raise LLMProviderError(str(exc)) from exc


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI uses a deployment-based URL and api-version query param."""

    name = "azure_openai"

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version

    def is_available(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("`openai` package not installed") from exc

        try:
            client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
            response = client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[azure_openai] completion failed: {exc}")
            raise LLMProviderError(str(exc)) from exc
