import re

with open("tests/entrypoints/test_cli.py", "r") as f:
    content = f.read()
content = content.replace('"OPENAI_API_KEY is not set"', '"API key is not set for provider"')
with open("tests/entrypoints/test_cli.py", "w") as f:
    f.write(content)

with open("tests/test_evaluate.py", "r") as f:
    content = f.read()
content = content.replace('"OPENAI_API_KEY is not set"', '"API key is not set for provider"')
with open("tests/test_evaluate.py", "w") as f:
    f.write(content)

with open("tests/test_config.py", "r") as f:
    content = f.read()
content = content.replace('{"LLM_PROVIDER": "anthropic"}, "LLM_PROVIDER"', '{"LLM_PROVIDER": "invalid"}, "LLM_PROVIDER"')
with open("tests/test_config.py", "w") as f:
    f.write(content)
