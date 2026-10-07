"""LLM client adapters and the factory that picks one from settings."""

from order_extractor.adapters.llm.synthetic import SyntheticLLM
from order_extractor.adapters.llm.litellm_client import LiteLLMClient
from order_extractor.application.ports import LLMClient
from order_extractor.config import ConfigError, Settings

def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider != "ollama" and not settings.api_key:
        raise ConfigError(
            f"API key is not set for provider "
            f"(LLM_PROVIDER={settings.llm_provider})."
        )
        
    return LiteLLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        temperature=settings.temperature,
        structured_output=settings.structured_output,
    )

__all__ = ["SyntheticLLM", "LiteLLMClient", "create_llm_client"]
