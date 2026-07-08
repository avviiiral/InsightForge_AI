"""Google Gemini provider."""
from __future__ import annotations

from app.llm.base_provider import LLMProvider, LLMProviderError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError("`google-generativeai` package not installed") from exc

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
            response = model.generate_content(
                user_prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
            )
            return response.text or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[gemini] completion failed: {exc}")
            raise LLMProviderError(str(exc)) from exc
