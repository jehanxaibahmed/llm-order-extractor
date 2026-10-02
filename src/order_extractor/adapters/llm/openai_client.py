"""LLM clients for OpenAI, OpenRouter and Ollama (all speak the OpenAI API, at different URLs).

Retries with exponential backoff (connection errors, 408/409/429/5xx) and timeouts come from
the ``openai`` SDK itself via ``max_retries`` and ``timeout``.
"""

import json
import re
from typing import Any

import openai
from openai import AsyncOpenAI

from order_extractor.adapters.llm.schema import to_strict_schema
from order_extractor.application.errors import LLMError
from order_extractor.application.ports import LLMResponse

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
SCHEMA_NAME = "order"

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```$", re.DOTALL)


class OpenAICompatibleClient:
    """Implements the ``LLMClient`` port for any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        temperature: float | None = 0.0,
        structured_output: bool = True,
        default_headers: dict[str, str] | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.structured_output = structured_output
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            default_headers=default_headers,
        )

    async def extract(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        request: dict[str, Any] = {"model": self.model}
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.structured_output:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_NAME,
                    "schema": to_strict_schema(schema),
                    "strict": True,
                },
            }
        else:
            # No structured outputs: describe the schema in the prompt instead.
            system = f"{system}\n\nJSON schema:\n{json.dumps(schema)}"
        request["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            completion = await self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if not completion.choices:
            raise LLMError("The model returned no choices.")
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError(f"The model refused: {message.refusal}")
        if not message.content:
            raise LLMError("The model returned an empty response.")

        usage = completion.usage
        return LLMResponse(
            text=strip_code_fences(message.content),
            model=completion.model or self.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class OpenAIClient(OpenAICompatibleClient):
    def __init__(self, *, api_key: str, model: str, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, model=model, **kwargs)


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(self, *, api_key: str, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
        kwargs.setdefault("default_headers", {"X-Title": "llm-order-extractor"})
        super().__init__(api_key=api_key, model=model, **kwargs)


class OllamaClient(OpenAICompatibleClient):
    """Local Ollama server. It ignores the API key, but the SDK requires a non-empty one."""

    def __init__(self, *, model: str, api_key: str = "ollama", **kwargs: Any) -> None:
        kwargs.setdefault("base_url", OLLAMA_BASE_URL)
        super().__init__(api_key=api_key, model=model, **kwargs)


def strip_code_fences(text: str) -> str:
    """Models without structured outputs often wrap JSON in ```json fences."""
    text = text.strip()
    match = _CODE_FENCE.match(text)
    return match.group(1).strip() if match else text
