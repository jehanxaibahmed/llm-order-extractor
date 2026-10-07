import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from order_extractor.adapters.llm import create_llm_client
from order_extractor.adapters.llm.litellm_client import LiteLLMClient, strip_code_fences
from order_extractor.adapters.llm.synthetic import SyntheticLLM
from order_extractor.application.errors import LLMError
from order_extractor.config import Settings

ORDER_JSON = '{"lines": [{"product": "Widgets", "quantity": 10}]}'
SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array"}},
    "additionalProperties": False,
}


@pytest.fixture
def simulator_acompletion():
    with patch("litellm.acompletion", new_callable=AsyncMock) as simulator:
        yield simulator


def make_completion(content=ORDER_JSON, input_tokens=120, output_tokens=30, refusal=None):
    simulator_resp = MagicMock()
    simulator_resp.model = "gpt-4o-mini-2024-07-18"
    simulator_message = MagicMock()
    simulator_message.content = content
    simulator_message.refusal = refusal
    simulator_resp.choices = [MagicMock(message=simulator_message)]
    if input_tokens is not None:
        simulator_resp.usage = MagicMock(
            prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    else:
        simulator_resp.usage = None
    return simulator_resp


async def test_structured_output_request(simulator_acompletion):
    simulator_acompletion.return_value = make_completion()
    client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
    response = await client.extract("SYSTEM", "USER", SCHEMA)

    simulator_acompletion.assert_called_once()
    kwargs = simulator_acompletion.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
    fmt = kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True

    assert response.text == ORDER_JSON
    assert response.model == "gpt-4o-mini-2024-07-18"
    assert response.input_tokens == 120
    assert response.output_tokens == 30


async def test_json_only_fallback_puts_schema_in_prompt(simulator_acompletion):
    simulator_acompletion.return_value = make_completion()
    client = LiteLLMClient(provider="openai", model="gpt-4o-mini", structured_output=False)
    await client.extract("SYSTEM", "USER", SCHEMA)

    kwargs = simulator_acompletion.call_args.kwargs
    assert "response_format" not in kwargs
    system = kwargs["messages"][0]["content"]
    assert system.startswith("SYSTEM\n\nJSON schema:\n")


async def test_temperature_none_is_omitted(simulator_acompletion):
    simulator_acompletion.return_value = make_completion()
    client = LiteLLMClient(provider="openai", model="gpt-4o-mini", temperature=None)
    await client.extract("S", "U", SCHEMA)
    assert "temperature" not in simulator_acompletion.call_args.kwargs


async def test_empty_content_raises_llm_error(simulator_acompletion):
    simulator_acompletion.return_value = make_completion(content="")
    client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
    with pytest.raises(LLMError, match="empty"):
        await client.extract("S", "U", SCHEMA)


async def test_litellm_exception_raises_llm_error(simulator_acompletion):
    simulator_acompletion.side_effect = Exception("connection reset")
    client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
    with pytest.raises(LLMError, match="connection reset"):
        await client.extract("S", "U", SCHEMA)


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


# --- SyntheticLLM ---


async def test_fake_llm_returns_responses_in_order_then_repeats_last():
    fake = SyntheticLLM({"lines": []}, "not json")
    first = await fake.extract("S", "U1", SCHEMA)
    second = await fake.extract("S", "U2", SCHEMA)
    third = await fake.extract("S", "U3", SCHEMA)
    assert json.loads(first.text) == {"lines": []}
    assert second.text == third.text == "not json"


async def test_fake_llm_raises_canned_exception():
    fake = SyntheticLLM(LLMError("rate limited"))
    with pytest.raises(LLMError, match="rate limited"):
        await fake.extract("S", "U", SCHEMA)


# --- factory ---


def test_factory_builds_openai_client():
    settings = Settings(openai_api_key="sk-test", llm_model="gpt-4o-mini", max_retries=1)
    client = create_llm_client(settings)
    assert isinstance(client, LiteLLMClient)
    assert client.provider == "openai"
    assert client.api_key == "sk-test"
    assert client.max_retries == 1


def test_factory_builds_anthropic_client():
    settings = Settings(
        llm_provider="anthropic", anthropic_api_key="sk-ant", structured_output=False
    )
    client = create_llm_client(settings)
    assert isinstance(client, LiteLLMClient)
    assert client.provider == "anthropic"
    assert client.api_key == "sk-ant"


def test_factory_builds_ollama_client_without_a_key():
    settings = Settings(
        llm_provider="ollama", llm_model="qwen2.5:14b-instruct", structured_output=False
    )
    client = create_llm_client(settings)
    assert isinstance(client, LiteLLMClient)
    assert client.provider == "ollama"
    assert client.api_key is None
    assert client.model == "qwen2.5:14b-instruct"
