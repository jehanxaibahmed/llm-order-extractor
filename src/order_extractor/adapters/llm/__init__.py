"""LLM client adapters and the factory that picks one from settings."""

from order_extractor.adapters.llm.fake import FakeLLM
from order_extractor.adapters.llm.openai_client import (
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)
from order_extractor.application.ports import LLMClient
from order_extractor.config import ConfigError, Settings

_CLIENTS = {"openai": OpenAIClient, "openrouter": OpenRouterClient, "ollama": OllamaClient}
_KEY_VARS = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def create_llm_client(settings: Settings) -> LLMClient:
    extra: dict[str, str] = {}
    if settings.llm_provider != "ollama":  # Ollama is local and needs no key
        if not settings.api_key:
            raise ConfigError(
                f"{_KEY_VARS[settings.llm_provider]} is not set "
                f"(LLM_PROVIDER={settings.llm_provider})."
            )
        extra["api_key"] = settings.api_key
    if settings.llm_base_url:
        extra["base_url"] = settings.llm_base_url
    return _CLIENTS[settings.llm_provider](
        model=settings.llm_model,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        temperature=settings.temperature,
        structured_output=settings.structured_output,
        **extra,
    )


__all__ = ["FakeLLM", "OllamaClient", "OpenAIClient", "OpenRouterClient", "create_llm_client"]
