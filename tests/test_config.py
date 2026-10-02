import pytest

from order_extractor.config import ConfigError, Settings


def test_defaults():
    s = Settings.from_env({})
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o-mini"
    assert s.timeout_seconds == 30.0
    assert s.max_retries == 2
    assert s.temperature == 0.0
    assert s.structured_output is True
    assert s.api_key is None


def test_reads_environment():
    s = Settings.from_env(
        {
            "LLM_PROVIDER": "OpenRouter",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "OPENAI_API_KEY": "sk-openai",
            "OPENROUTER_API_KEY": " sk-or ",
            "LLM_TIMEOUT_SECONDS": "12.5",
            "LLM_MAX_RETRIES": "0",
            "LLM_TEMPERATURE": "0.2",
        }
    )
    assert s.llm_provider == "openrouter"
    assert s.llm_model == "openai/gpt-4o-mini"
    assert s.api_key == "sk-or"
    assert s.timeout_seconds == 12.5
    assert s.max_retries == 0
    assert s.temperature == 0.2


def test_blank_values_fall_back_to_defaults():
    s = Settings.from_env({"LLM_MODEL": "  ", "OPENAI_API_KEY": "", "LLM_MAX_RETRIES": ""})
    assert s.llm_model == "gpt-4o-mini"
    assert s.api_key is None
    assert s.max_retries == 2


def test_structured_output_default_depends_on_provider():
    assert Settings.from_env({"LLM_PROVIDER": "openai"}).structured_output is True
    assert Settings.from_env({"LLM_PROVIDER": "openrouter"}).structured_output is False
    env = {"LLM_PROVIDER": "openrouter", "LLM_STRUCTURED_OUTPUT": "yes"}
    assert Settings.from_env(env).structured_output is True


def test_temperature_none_omits_it():
    assert Settings.from_env({"LLM_TEMPERATURE": "none"}).temperature is None


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"LLM_PROVIDER": "anthropic"}, "LLM_PROVIDER"),
        ({"LLM_TIMEOUT_SECONDS": "soon"}, "LLM_TIMEOUT_SECONDS"),
        ({"LLM_MAX_RETRIES": "-1"}, "LLM_MAX_RETRIES"),
        ({"LLM_MAX_RETRIES": "1.5"}, "LLM_MAX_RETRIES"),
        ({"LLM_STRUCTURED_OUTPUT": "maybe"}, "LLM_STRUCTURED_OUTPUT"),
    ],
)
def test_invalid_values(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


def test_suite_runs_without_api_keys():
    # Guard from conftest.py: a key in the shell must not leak into tests.
    import os

    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENROUTER_API_KEY" not in os.environ


def test_ollama_needs_no_key_and_defaults_to_prompt_schema():
    s = Settings.from_env({"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen2.5:14b-instruct"})
    assert s.llm_provider == "ollama"
    assert s.api_key is None
    assert s.structured_output is False
    assert s.llm_base_url is None
    env = {"LLM_PROVIDER": "ollama", "LLM_STRUCTURED_OUTPUT": "true"}
    assert Settings.from_env(env).structured_output is True


def test_base_url_is_read_and_blank_means_unset():
    assert (
        Settings.from_env({"LLM_BASE_URL": " http://host:1/v1 "}).llm_base_url == "http://host:1/v1"
    )
    assert Settings.from_env({"LLM_BASE_URL": " "}).llm_base_url is None
