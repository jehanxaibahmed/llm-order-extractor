import json

import openai
import pytest
from openai.types.chat import ChatCompletion

from order_extractor.adapters.llm import (
    FakeLLM,
    OpenAIClient,
    OpenRouterClient,
    create_llm_client,
)
from order_extractor.adapters.llm.openai_client import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OllamaClient,
    strip_code_fences,
)
from order_extractor.application.errors import LLMError
from order_extractor.application.ports import LLMResponse
from order_extractor.config import ConfigError, Settings
from order_extractor.domain.models import Order

SCHEMA = Order.model_json_schema()
ORDER_JSON = json.dumps({"customer_name": "Green Leaf Café", "lines": []})


def completion(content=ORDER_JSON, refusal=None, usage=True, model="gpt-4o-mini-2024-07-18"):
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }
    if usage:
        data["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    return ChatCompletion.model_validate(data)


class StubSDK:
    """Stands in for AsyncOpenAI: records requests, returns or raises a canned result."""

    def __init__(self, result):
        self.result = result
        self.requests = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client_with(result, **kwargs):
    sdk = StubSDK(result)
    return OpenAIClient(api_key="sk-test", model="gpt-4o-mini", client=sdk, **kwargs), sdk


# --- OpenAI-compatible client ---------------------------------------------------------------


async def test_structured_output_request():
    client, sdk = client_with(completion())
    response = await client.extract("SYSTEM", "USER", SCHEMA)

    request = sdk.requests[0]
    assert request["model"] == "gpt-4o-mini"
    assert request["temperature"] == 0.0
    assert request["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False

    assert response == LLMResponse(
        text=ORDER_JSON, model="gpt-4o-mini-2024-07-18", input_tokens=120, output_tokens=30
    )


async def test_json_only_fallback_puts_schema_in_prompt():
    client, sdk = client_with(completion(), structured_output=False)
    await client.extract("SYSTEM", "USER", SCHEMA)

    request = sdk.requests[0]
    assert "response_format" not in request
    system = request["messages"][0]["content"]
    assert system.startswith("SYSTEM\n\nJSON schema:\n")
    assert json.loads(system.split("JSON schema:\n", 1)[1]) == SCHEMA


async def test_temperature_none_is_omitted():
    client, sdk = client_with(completion(), temperature=None)
    await client.extract("S", "U", SCHEMA)
    assert "temperature" not in sdk.requests[0]


async def test_code_fences_are_stripped():
    client, _ = client_with(completion(content=f"```json\n{ORDER_JSON}\n```"))
    assert (await client.extract("S", "U", SCHEMA)).text == ORDER_JSON


async def test_missing_usage_gives_none_tokens():
    client, _ = client_with(completion(usage=False))
    response = await client.extract("S", "U", SCHEMA)
    assert (response.input_tokens, response.output_tokens) == (None, None)


async def test_refusal_raises_llm_error():
    client, _ = client_with(completion(content=None, refusal="I can't help with that."))
    with pytest.raises(LLMError, match="refused"):
        await client.extract("S", "U", SCHEMA)


async def test_empty_content_raises_llm_error():
    client, _ = client_with(completion(content=""))
    with pytest.raises(LLMError, match="empty"):
        await client.extract("S", "U", SCHEMA)


async def test_sdk_errors_become_llm_error():
    client, _ = client_with(openai.OpenAIError("connection reset"))
    with pytest.raises(LLMError, match="connection reset") as info:
        await client.extract("S", "U", SCHEMA)
    assert isinstance(info.value.__cause__, openai.OpenAIError)


def test_sdk_is_configured_with_timeout_and_retries():
    client = OpenAIClient(api_key="sk-test", model="m", timeout_seconds=12, max_retries=2)
    assert client._client.timeout == 12
    assert client._client.max_retries == 2
    assert str(client._client.base_url).startswith("https://api.openai.com")


def test_openrouter_uses_its_base_url():
    client = OpenRouterClient(api_key="sk-or", model="openai/gpt-4o-mini")
    assert str(client._client.base_url).rstrip("/") == OPENROUTER_BASE_URL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('  \n```json\n{"a": 1}```  ', '{"a": 1}'),
        ('Here you go: {"a": 1}', 'Here you go: {"a": 1}'),
    ],
)
def test_strip_code_fences(raw, expected):
    assert strip_code_fences(raw) == expected


# --- FakeLLM --------------------------------------------------------------------------------


async def test_fake_llm_returns_responses_in_order_then_repeats_last():
    fake = FakeLLM({"lines": []}, "not json")
    first = await fake.extract("S", "U1", SCHEMA)
    second = await fake.extract("S", "U2", SCHEMA)
    third = await fake.extract("S", "U3", SCHEMA)
    assert json.loads(first.text) == {"lines": []}
    assert second.text == third.text == "not json"
    assert [c.user for c in fake.calls] == ["U1", "U2", "U3"]
    assert first.model == "fake-llm"


async def test_fake_llm_raises_canned_exception():
    fake = FakeLLM(LLMError("rate limited"))
    with pytest.raises(LLMError, match="rate limited"):
        await fake.extract("S", "U", SCHEMA)


def test_fake_llm_needs_a_response():
    with pytest.raises(ValueError):
        FakeLLM()


# --- factory --------------------------------------------------------------------------------


def test_factory_builds_openai_client():
    settings = Settings(openai_api_key="sk-test", llm_model="gpt-4o-mini", max_retries=1)
    client = create_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"
    assert client._client.max_retries == 1


def test_factory_builds_openrouter_client():
    settings = Settings(
        llm_provider="openrouter", openrouter_api_key="sk-or", structured_output=False
    )
    client = create_llm_client(settings)
    assert isinstance(client, OpenRouterClient)
    assert client.structured_output is False


@pytest.mark.parametrize(
    ("provider", "variable"), [("openai", "OPENAI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")]
)
def test_factory_requires_the_provider_key(provider, variable):
    # A key for the *other* provider does not count.
    keys = {"openai": {"openrouter_api_key": "sk-or"}, "openrouter": {"openai_api_key": "sk-oa"}}
    settings = Settings(llm_provider=provider, **keys[provider])
    with pytest.raises(ConfigError, match=variable):
        create_llm_client(settings)


def test_factory_builds_ollama_client_without_a_key():
    settings = Settings(
        llm_provider="ollama", llm_model="qwen2.5:14b-instruct", structured_output=False
    )
    client = create_llm_client(settings)
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:14b-instruct"
    assert client.structured_output is False
    assert str(client._client.base_url).rstrip("/") == OLLAMA_BASE_URL


def test_factory_passes_base_url_through():
    settings = Settings(llm_provider="ollama", llm_base_url="http://gpu-box:11434/v1")
    assert str(create_llm_client(settings)._client.base_url).startswith("http://gpu-box:11434")
    settings = Settings(openai_api_key="sk", llm_base_url="http://proxy.local/v1")
    assert str(create_llm_client(settings)._client.base_url).startswith("http://proxy.local")
    settings = Settings(
        llm_provider="openrouter", openrouter_api_key="sk", llm_base_url="http://x/v1"
    )
    assert str(create_llm_client(settings)._client.base_url).startswith("http://x")


def test_factory_without_base_url_keeps_provider_defaults():
    settings = Settings(llm_provider="openrouter", openrouter_api_key="sk")
    client = create_llm_client(settings)
    assert str(client._client.base_url).rstrip("/") == OPENROUTER_BASE_URL


async def test_ollama_client_puts_schema_in_prompt():
    sdk = StubSDK(completion())
    client = OllamaClient(model="qwen2.5:14b-instruct", structured_output=False, client=sdk)
    await client.extract("sys", "user", {"type": "object"})
    request = sdk.requests[0]
    assert "response_format" not in request
    assert "JSON schema" in request["messages"][0]["content"]
