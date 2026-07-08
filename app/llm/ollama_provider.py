"""Ollama provider — talks to a local Ollama daemon over its native HTTP API."""
from __future__ import annotations

import httpx

from app.llm.base_provider import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        if not self.model:
            return False
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            }
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ollama] completion failed: {exc}")
            raise LLMProviderError(str(exc)) from exc
