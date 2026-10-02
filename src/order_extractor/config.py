"""Settings read from the environment. Entrypoints load ``.env`` before calling ``from_env``."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

Provider = Literal["openai", "openrouter", "ollama"]
PROVIDERS: tuple[str, ...] = get_args(Provider)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0


class ConfigError(ValueError):
    """An environment variable is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_provider: Provider = "openai"
    llm_model: str = DEFAULT_MODEL
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float | None = DEFAULT_TEMPERATURE
    """``None`` omits the parameter, for models that reject it (e.g. reasoning models)."""
    structured_output: bool = True
    """Use JSON-schema structured outputs. Off by default for OpenRouter and Ollama."""
    llm_base_url: str | None = None
    """Override the provider's API base URL (any OpenAI-compatible endpoint)."""

    @property
    def api_key(self) -> str | None:
        if self.llm_provider == "openai":
            return self.openai_api_key
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return None  # ollama needs no key

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Settings":
        env = _Env(environ)
        provider = (env.text("LLM_PROVIDER") or "openai").lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, got {provider!r}."
            )

        return cls(
            llm_provider=provider,
            llm_model=env.text("LLM_MODEL") or DEFAULT_MODEL,
            openai_api_key=env.text("OPENAI_API_KEY"),
            openrouter_api_key=env.text("OPENROUTER_API_KEY"),
            timeout_seconds=env.number("LLM_TIMEOUT_SECONDS", float, DEFAULT_TIMEOUT_SECONDS),
            max_retries=env.number("LLM_MAX_RETRIES", int, DEFAULT_MAX_RETRIES),
            temperature=None
            if env.text("LLM_TEMPERATURE") == "none"
            else env.number("LLM_TEMPERATURE", float, DEFAULT_TEMPERATURE),
            structured_output=env.flag("LLM_STRUCTURED_OUTPUT", default=provider == "openai"),
            llm_base_url=env.text("LLM_BASE_URL"),
        )


class _Env:
    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ

    def text(self, name: str) -> str | None:
        return self._environ.get(name, "").strip() or None

    def number(self, name: str, kind: type[int] | type[float], default):
        value = self.text(name)
        if value is None:
            return default
        try:
            number = kind(value)
        except ValueError:
            raise ConfigError(f"{name} must be a number, got {value!r}.") from None
        if number < 0:
            raise ConfigError(f"{name} must not be negative, got {value!r}.")
        return number

    def flag(self, name: str, default: bool) -> bool:
        value = self.text(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{name} must be true or false, got {value!r}.")
