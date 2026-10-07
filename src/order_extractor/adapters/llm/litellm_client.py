import json
import re
from typing import Any

import litellm

from order_extractor.adapters.llm.schema import to_strict_schema
from order_extractor.application.errors import LLMError
from order_extractor.application.ports import LLMClient, LLMResponse

litellm._turn_on_debug()

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    match = _CODE_FENCE.match(text)
    return match.group(1).strip() if match else text


class LiteLLMClient(LLMClient):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        temperature: float | None = 0.0,
        structured_output: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.structured_output = structured_output

    async def extract(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "order",
                    "schema": to_strict_schema(schema),
                    "strict": True,
                },
            }
        else:
            system = f"{system}\n\nJSON schema:\n{json.dumps(schema)}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        # Prefix the model with provider if necessary for litellm.
        # OpenRouter and others might need explicit routing if the model name is ambiguous.
        litellm_model = self.model
        if self.provider == "openrouter" and not litellm_model.startswith("openrouter/"):
            litellm_model = f"openrouter/{litellm_model}"
        elif self.provider == "ollama" and not litellm_model.startswith("ollama/"):
            litellm_model = f"ollama/{litellm_model}"
        elif self.provider == "gemini" and not litellm_model.startswith("gemini/"):
            litellm_model = f"gemini/{litellm_model}"
        elif self.provider == "anthropic" and not litellm_model.startswith("anthropic/"):
            litellm_model = f"anthropic/{litellm_model}"
        elif self.provider == "deepseek" and not litellm_model.startswith("deepseek/"):
            litellm_model = f"deepseek/{litellm_model}"

        try:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                timeout=self.timeout_seconds,
                num_retries=self.max_retries,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if not response.choices:
            raise LLMError("The model returned no choices.")
        message = response.choices[0].message

        # In litellm, message is often a litellm.Message object.
        content = message.content
        if not content:
            raise LLMError("The model returned an empty response.")

        usage = response.usage
        return LLMResponse(
            text=strip_code_fences(content),
            model=response.model or self.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
