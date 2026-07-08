"""Anthropic (Claude) provider."""
from __future__ import annotations

from app.llm.base_provider import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("`anthropic` package not installed") from exc

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
            return "".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[anthropic] completion failed: {exc}")
            raise LLMProviderError(str(exc)) from exc
