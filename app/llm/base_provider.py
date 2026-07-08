"""Abstract interface every LLM provider must implement.

Keeping this interface tiny (`complete`) is deliberate: it makes it trivial
to add new providers (OpenAI, Anthropic, Gemini, Ollama, OpenRouter,
DeepSeek, Mistral, LM Studio, or any future local model) without touching
any agent code. Agents only depend on `LLMProvider`, never on a concrete
vendor SDK.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Raised when a provider fails to produce a completion."""


class LLMProvider(ABC):
    """Common contract for all LLM backends."""

    name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        """Return a text completion for the given prompts.

        Implementations must raise `LLMProviderError` on failure so the
        caller (usually `LLMRouter`) can fall back to deterministic logic.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Cheap, side-effect-free check of whether this provider is usable."""
        return True
